"""Phase 2: counterfactual teacher mining on stationary boundary states.

Per ChatGPT 2026-05-23 plan:
  - Sample stationary boundary anchors (empty 32-50) from V13 selfplay
  - Stratify by current LEC and n_components buckets
  - For each anchor, evaluate top-K policy candidate moves via common-RNG
    rollouts (R replicates, H horizon)
  - Label each candidate by stationarity-risk objective:
      primary: die_rate (rollouts that hit game_over within H)
      secondary: P(min_empty_H < 30) — leaves stationary band
      tertiary: P10(score_rate_H) — floor of score-rate
      quaternary: mean score
  - Save dataset (anchor state + per-candidate rollout stats + soft weights)
    suitable for distillation into pillar3c policy.

Outputs:
  <out>.pt  — per-anchor records (state obs + 10-candidate labels)

This is the slow phase. Uses multiprocessing.Pool with model loaded per
worker (CPU device for stable multiprocessing on macOS).
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from collections import defaultdict
from multiprocessing import Pool, set_start_method
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.rng import SimpleRng
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation


# ── Per-worker globals ──────────────────────────────────────────────────────
_W_NET = None
_W_DEVICE = None
_W_DTYPE = None


def _init_worker(model_path, device_str):
    import torch as _t
    global _W_NET, _W_DEVICE, _W_DTYPE
    if device_str == 'cuda' and _t.cuda.is_available():
        _W_DEVICE = _t.device('cuda')
    elif device_str == 'mps' and _t.backends.mps.is_available():
        _W_DEVICE = _t.device('mps')
    else:
        _W_DEVICE = _t.device('cpu')
    _W_NET, _ = load_model(model_path, _W_DEVICE,
                            fp16=(_W_DEVICE.type != 'cpu'))
    _W_DTYPE = next(_W_NET.parameters()).dtype
    _t.set_num_threads(1)


def policy_topk(net, device, net_dtype, game, k=10):
    """Run policy, return list of (move, prob) for top-k legal moves."""
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    obs = torch.from_numpy(
        build_observation(game.board, nr, nc, ncol, nn)
    ).unsqueeze(0).to(device=device, dtype=net_dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()

    source_mask = game.get_source_mask()
    candidates = []
    for sr in range(9):
        for sc in range(9):
            if source_mask[sr, sc] == 0:
                continue
            tmask = game.get_target_mask((sr, sc))
            for tr in range(9):
                for tc in range(9):
                    if tmask[tr, tc] > 0:
                        idx = (sr * 9 + sc) * 81 + tr * 9 + tc
                        candidates.append((logits[idx], ((sr, sc), (tr, tc))))
    if not candidates:
        return []
    candidates.sort(key=lambda x: -x[0])
    arr = np.array([c[0] for c in candidates], dtype=np.float64)
    arr -= arr.max()
    e = np.exp(arr)
    p = e / e.sum()
    return [(candidates[i][1], float(p[i])) for i in range(min(k, len(candidates)))]


def policy_argmax(net, device, net_dtype, game):
    out = policy_topk(net, device, net_dtype, game, k=1)
    return out[0][0] if out else None


def largest_empty_component(board):
    visited = np.zeros_like(board, dtype=bool)
    best = 0
    for r0 in range(9):
        for c0 in range(9):
            if board[r0, c0] != 0 or visited[r0, c0]:
                continue
            sz = 0
            stack = [(r0, c0)]
            visited[r0, c0] = True
            while stack:
                r, c = stack.pop()
                sz += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr, nc]
                            and board[nr, nc] == 0):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if sz > best:
                best = sz
    return int(best)


def restore_game(anchor_state):
    g = ColorLinesGame()
    board = np.array(anchor_state['board'], dtype=np.int8)
    nb = [(tuple(p), int(c)) for p, c in anchor_state['next_balls']]
    g.reset(board=board, next_balls=nb)
    g.score = int(anchor_state['score'])
    g.turns = int(anchor_state['turn'])
    return g


def rollout(net, device, net_dtype, anchor_state, first_move,
            rollout_seed, H=100):
    """Common-RNG rollout. Returns dict of outcomes."""
    g = restore_game(anchor_state)
    g.rng = SimpleRng(rollout_seed)
    start_score = g.score
    res = g.move(*first_move)
    if not res['valid']:
        return None
    min_empty = int((g.board == 0).sum())
    min_lec = largest_empty_component(g.board)
    died = False
    for h in range(H):
        if g.game_over:
            died = True
            break
        mv = policy_argmax(net, device, net_dtype, g)
        if mv is None:
            died = True
            break
        r = g.move(*mv)
        if not r['valid']:
            died = True
            break
        e = int((g.board == 0).sum())
        if e < min_empty:
            min_empty = e
        l = largest_empty_component(g.board)
        if l < min_lec:
            min_lec = l
    survived = not g.game_over
    return {
        'score_gained': int(g.score - start_score),
        'min_empty': min_empty,
        'min_lec': min_lec,
        'died': died,
        'survived': survived,
        'turns_played': int(g.turns - int(anchor_state['turn'])),
    }


def _worker_run(args):
    """Process one (anchor_idx, candidate_rank). Runs R rollouts.
    Returns (anchor_idx, rank, list[rollout dicts | None])."""
    anchor_idx, rank, move, anchor_state, R, H = args
    out = []
    for rs in range(R):
        r = rollout(_W_NET, _W_DEVICE, _W_DTYPE,
                     anchor_state, move, rs, H=H)
        out.append(r)
    return (anchor_idx, rank, out)


# ── Anchor sampling ─────────────────────────────────────────────────────────


def sample_anchors_from_selfplay(games_dir, max_anchors,
                                   empty_min=32, empty_max=50,
                                   min_start_turn=100, stride=50,
                                   rng_seed=42):
    """Walk V13 selfplay games and sample stationary boundary states.

    Stratify by (lec_bucket, n_components_bucket) and return up to
    max_anchors total, distributed across strata.
    """
    files = sorted(glob.glob(os.path.join(games_dir, 'game_seed*.json')))
    print(f"  scanning {len(files)} games in {games_dir} for anchors...",
          flush=True)
    rng = np.random.default_rng(rng_seed)
    candidates_by_strata = defaultdict(list)

    def lec_bucket(lec):
        if lec < 10: return '<10'
        if lec < 15: return '10-14'
        if lec < 20: return '15-19'
        if lec < 25: return '20-24'
        if lec < 30: return '25-29'
        return '>=30'

    def nc_bucket(nc):
        if nc == 1: return '1'
        if nc == 2: return '2'
        if nc <= 4: return '3-4'
        return '>=5'

    t0 = time.time()
    for i, path in enumerate(files):
        try:
            with open(path) as f:
                g = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        moves = g.get('moves', [])
        if len(moves) < min_start_turn + 120:
            continue
        seed = g.get('seed', 0)
        for t in range(min_start_turn, len(moves) - 110, stride):
            board = np.array(moves[t]['board'], dtype=np.int8)
            empties = int((board == 0).sum())
            if not (empty_min <= empties <= empty_max):
                continue
            lec = largest_empty_component(board)
            # cheap n_components
            visited = np.zeros_like(board, dtype=bool)
            nc = 0
            for r0 in range(9):
                for c0 in range(9):
                    if board[r0, c0] != 0 or visited[r0, c0]:
                        continue
                    nc += 1
                    stk = [(r0, c0)]
                    visited[r0, c0] = True
                    while stk:
                        r, c = stk.pop()
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, ncc = r + dr, c + dc
                            if (0 <= nr < 9 and 0 <= ncc < 9
                                    and not visited[nr, ncc]
                                    and board[nr, ncc] == 0):
                                visited[nr, ncc] = True
                                stk.append((nr, ncc))
            strata_key = (lec_bucket(lec), nc_bucket(nc))
            candidates_by_strata[strata_key].append({
                'seed': seed,
                'turn': t,
                'score': 0,  # score timeline isn't tracked here;
                              # only board+next_balls matters for restoration
                'board': moves[t]['board'],
                'next_balls': [(
                    (nb['row'], nb['col']) if isinstance(nb, dict)
                    else (nb[0][0], nb[0][1]),
                    nb['color'] if isinstance(nb, dict) else nb[1])
                    for nb in moves[t]['next_balls']],
                'empties_now': empties,
                'lec_now': lec,
                'n_components_now': nc,
            })
        if (i + 1) % 200 == 0:
            n_cands = sum(len(v) for v in candidates_by_strata.values())
            print(f"    [{i+1}/{len(files)}] {n_cands} candidates "
                  f"({time.time() - t0:.0f}s)", flush=True)

    print(f"  candidates by strata:", flush=True)
    for k in sorted(candidates_by_strata.keys()):
        print(f"    {k}: {len(candidates_by_strata[k])}", flush=True)

    # Sample evenly across strata. If a stratum has fewer than per_stratum
    # we take all of it; remaining budget redistributes.
    n_strata = len(candidates_by_strata)
    if n_strata == 0:
        return []
    per_stratum = max(1, max_anchors // n_strata)
    sampled = []
    used = set()  # (strata_key, idx) to avoid duplicates
    for k, vs in sorted(candidates_by_strata.items()):
        if not vs:
            continue
        n_take = min(per_stratum, len(vs))
        picks = rng.choice(len(vs), size=n_take, replace=False)
        for p in picks:
            sampled.append(vs[p])
            used.add((k, int(p)))
    # Fill remaining budget by redrawing from larger strata
    remaining = max_anchors - len(sampled)
    if remaining > 0:
        pool = []
        for k, vs in candidates_by_strata.items():
            for i, v in enumerate(vs):
                if (k, i) not in used:
                    pool.append(v)
        if pool:
            extra = rng.choice(len(pool),
                                size=min(remaining, len(pool)),
                                replace=False)
            sampled.extend(pool[e] for e in extra)
    rng.shuffle(sampled)
    return sampled[:max_anchors]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                         default='alphatrain/data/pillar3b_epoch_20.pt')
    parser.add_argument('--selfplay-dir', default='data/selfplay_v13')
    parser.add_argument('--out',
                         default='alphatrain/data/stationary_counterfactuals_v1.pt')
    parser.add_argument('--max-anchors', type=int, default=1500)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--R', type=int, default=32,
                         help='Common-RNG rollouts per (anchor, candidate)')
    parser.add_argument('--H', type=int, default=100,
                         help='Rollout horizon')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--device', default='cpu',
                         choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # === Sample anchors (sequential, fast) ===
    print(f"\n=== Phase 2 anchor sampling ===", flush=True)
    anchors = sample_anchors_from_selfplay(
        args.selfplay_dir, args.max_anchors)
    n_anchors = len(anchors)
    print(f"  Sampled {n_anchors} anchors", flush=True)
    if n_anchors == 0:
        print("No anchors — aborting")
        return

    # === Top-K candidate precompute (sequential, main process) ===
    print(f"\n=== Phase 2 top-{args.top_k} candidates per anchor ===",
          flush=True)
    device = torch.device('cpu')  # main process — use CPU; cheap precompute
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    net, _ = load_model(args.model, device,
                         fp16=(device.type != 'cpu'))
    net_dtype = next(net.parameters()).dtype

    anchor_topks = []
    t0 = time.time()
    for i, anchor in enumerate(anchors):
        g = restore_game(anchor)
        topk = policy_topk(net, device, net_dtype, g, k=args.top_k)
        anchor_topks.append(topk)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{n_anchors}] {time.time() - t0:.0f}s",
                  flush=True)
    print(f"  Top-K precompute done: {time.time() - t0:.0f}s", flush=True)

    # Free main-process model (workers re-load)
    del net
    if device.type == 'mps':
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    # Build work units: (anchor_idx, rank, move, anchor_state, R, H)
    units = []
    for ai, (anchor, topk) in enumerate(zip(anchors, anchor_topks)):
        for rank, (move, _prob) in enumerate(topk):
            units.append((ai, rank, move, anchor, args.R, args.H))
    print(f"\n=== Phase 2 rollouts: {len(units)} (anchor, rank) units × "
          f"R={args.R} × H={args.H} = "
          f"{len(units) * args.R} rollouts ===", flush=True)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # results[anchor_idx][rank] = [rollout dicts]
    results = defaultdict(dict)
    t0 = time.time()
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.model, args.device)) as pool:
        n_done = 0
        for anchor_idx, rank, rollouts in pool.imap_unordered(
                _worker_run, units, chunksize=2):
            results[anchor_idx][rank] = rollouts
            n_done += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_done * (len(units) - n_done)
                print(f"  [{n_done}/{len(units)}] elapsed={elapsed:.0f}s  "
                      f"eta={eta:.0f}s", flush=True)

    # === Aggregate per anchor ===
    print(f"\n=== Phase 2 aggregation ===", flush=True)
    records = []
    for ai, (anchor, topk) in enumerate(zip(anchors, anchor_topks)):
        candidate_stats = []
        for rank, (move, prior) in enumerate(topk):
            rollouts = [r for r in results[ai].get(rank, []) if r is not None]
            if not rollouts:
                continue
            n_died = sum(1 for r in rollouts if r['died'])
            n_left_band = sum(1 for r in rollouts if r['min_empty'] < 30)
            scores = np.array([r['score_gained'] for r in rollouts])
            min_empties = np.array([r['min_empty'] for r in rollouts])
            min_lecs = np.array([r['min_lec'] for r in rollouts])
            candidate_stats.append({
                'rank': rank + 1,
                'move': move,
                'prior_p': prior,
                'valid': len(rollouts),
                'die_rate': n_died / len(rollouts),
                'leave_band_rate': n_left_band / len(rollouts),
                'mean_score': float(scores.mean()),
                'p10_score': float(np.percentile(scores, 10)),
                'mean_min_empty': float(min_empties.mean()),
                'mean_min_lec': float(min_lecs.mean()),
                'p10_min_empty': float(np.percentile(min_empties, 10)),
                'p10_min_lec': float(np.percentile(min_lecs, 10)),
            })
        if not candidate_stats:
            continue

        # Lex objective: minimize die_rate, then leave_band_rate, then -p10_score
        # Pick the best candidate (the "teacher target")
        def floor_key(c):
            return (c['die_rate'], c['leave_band_rate'],
                    -c['p10_score'], -c['mean_score'])
        floor_winner = min(candidate_stats, key=floor_key)
        records.append({
            'anchor': anchor,
            'candidates': candidate_stats,
            'floor_winner_rank': floor_winner['rank'],
        })

    print(f"  {len(records)} anchors with valid records "
          f"(from {n_anchors} sampled)", flush=True)

    # Save
    torch.save({
        'records': records,
        'config': {
            'max_anchors': args.max_anchors,
            'top_k': args.top_k,
            'R': args.R,
            'H': args.H,
            'model': args.model,
            'selfplay_dir': args.selfplay_dir,
        },
    }, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"  saved {args.out} ({sz:.1f} MB)", flush=True)

    # Verdict distribution
    rank_counts = defaultdict(int)
    for r in records:
        rank_counts[r['floor_winner_rank']] += 1
    print(f"\n=== Floor-winner rank distribution ===")
    for rank in range(1, args.top_k + 1):
        c = rank_counts.get(rank, 0)
        pct = 100 * c / max(1, len(records))
        print(f"  top-{rank}: {c} ({pct:.1f}%)")


if __name__ == '__main__':
    main()
