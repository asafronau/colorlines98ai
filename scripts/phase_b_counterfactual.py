"""Phase B counterfactual rollout diagnostic for risk-blind move selection.

Plan: docs/phase_b_counterfactual_plan.md

For each decision point (DP) flagged in failure trajectories AND in
matched-density success trajectories:
  1. Load the saved game state at that turn.
  2. Take top-5 policy moves.
  3. For each move × K rollout_seeds: common-RNG continuation with H=100 turns.
  4. Per branch aggregate survival_50, survival_100, P10/P25/mean score gained,
     median turns survived.
  5. Lexicographic floor metric: survival_100 → P10 → mean.
  6. Cross-tab: for failures vs successes, who wins (top-1 / top-2 / ... / top-5).
  7. Verdict per decision rules in plan.

Outputs:
  logs/phase_b/per_dp.csv         — one row per DP with all top-5 stats
  logs/phase_b/summary.txt        — failure vs success cross-tab + verdict
  logs/phase_b/correction.json    — if hypothesis confirmed, training labels
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


# ── Per-worker globals (set by _init_worker, used by _worker_run) ──
_W_NET = None
_W_DEVICE = None
_W_DTYPE = None


def _init_worker(model_path, device_str):
    """Initializer: load model once per process."""
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


def _worker_run(args):
    """Worker: one (DP, candidate_rank) — runs K rollouts.
    Returns: (dp_idx, rank, list[rollout_dict|None]).
    """
    dp_idx, rank, move, dp_state, K, H = args
    out = []
    for rs in range(K):
        r = rollout(_W_NET, _W_DEVICE, _W_DTYPE, dp_state, move, rs, H=H)
        out.append(r)
    return (dp_idx, rank, out)


def policy_topk(net, device, net_dtype, game, k=5):
    """Run policy, return [(move, prob), ...] for top-k legal moves."""
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
    # Softmax over legal moves for probabilities
    arr = np.array([c[0] for c in candidates], dtype=np.float64)
    arr -= arr.max()
    e = np.exp(arr)
    p = e / e.sum()
    return [(candidates[i][1], float(p[i])) for i in range(min(k, len(candidates)))]


def policy_argmax(net, device, net_dtype, game):
    """Same as policy_topk(k=1) but returns just the best move (or None)."""
    out = policy_topk(net, device, net_dtype, game, k=1)
    return out[0][0] if out else None


def restore_game(dp_state):
    """Build a ColorLinesGame from a saved DP snapshot."""
    g = ColorLinesGame()
    board = np.array(dp_state['board'], dtype=np.int8)
    nb = [(tuple(p), int(c)) for p, c in dp_state['next_balls']]
    g.reset(board=board, next_balls=nb)
    g.score = int(dp_state['score'])
    g.turns = int(dp_state['turn'])
    return g


def rollout(net, device, net_dtype, dp_state, first_move,
            rollout_seed, H=100):
    """Common-RNG rollout from a DP, applying first_move then playing policy."""
    g = restore_game(dp_state)
    g.rng = SimpleRng(rollout_seed)
    start_score = g.score
    res = g.move(*first_move)
    if not res['valid']:
        return None
    survived_50 = False
    survived_100 = False
    for h in range(H):
        if g.game_over:
            break
        mv = policy_argmax(net, device, net_dtype, g)
        if mv is None:
            break
        r = g.move(*mv)
        if not r['valid']:
            break
        if h + 1 == 50 and not g.game_over:
            survived_50 = True
    if not g.game_over:
        survived_100 = True
    if not survived_50:  # set if survived_50 was bypassed because H<50 but alive
        if not g.game_over:
            survived_50 = True
    return {
        'score_gained': int(g.score - start_score),
        'turns_played': int(g.turns - int(dp_state['turn'])),
        'survived_50': bool(survived_50),
        'survived_100': bool(survived_100),
        'final_empties': int((g.board == 0).sum()),
    }


def identify_dps(traj, min_turn_before_death=20, max_empties=30,
                  max_dps_per_traj=30):
    """Return list of decision point indices into traj['metrics']."""
    m = traj['metrics']
    final_turn = traj['final_turn']
    if not m:
        return []
    candidates = []
    for i, x in enumerate(m):
        # Need full state present (lec field is in extended schema)
        if 'board' not in x:
            continue
        # Require minimum runway after the DP
        if x['turn'] > final_turn - min_turn_before_death:
            continue
        # Density signal: empties low or trending down OR lec shrinking
        empties_now = x['empties']
        lec_now = x.get('lec', empties_now)
        condA = empties_now < max_empties
        condB = False
        condC = False
        if i >= 5:
            prev = m[i - 5]
            condB = empties_now < prev['empties']
            condC = lec_now < prev.get('lec', empties_now)
        if not (condA or condB or condC):
            continue
        candidates.append(i)
    # Spread the DPs across the trajectory rather than clustering early
    if len(candidates) <= max_dps_per_traj:
        return candidates
    step = len(candidates) / max_dps_per_traj
    return [candidates[int(j * step)] for j in range(max_dps_per_traj)]


def density_bucket(empties, lec):
    """Bucket DP density profile for matched-success sampling."""
    e_b = min(3, empties // 10)  # 0: <10, 1: 10-19, 2: 20-29, 3: 30+
    l_b = min(3, lec // 10)
    return (e_b, l_b)


def collect_dps(traj_dir, label, max_dps_per_traj=30):
    """Walk a trajectory dir, return list of (seed, traj, dp_index, state)."""
    out = []
    files = sorted(glob.glob(os.path.join(traj_dir, 'traj_seed*.json')))
    for f in files:
        with open(f) as fp:
            traj = json.load(fp)
        idxs = identify_dps(traj, max_dps_per_traj=max_dps_per_traj)
        for i in idxs:
            state = traj['metrics'][i]
            out.append({
                'label': label,
                'seed': traj['seed'],
                'traj_final_score': traj['final_score'],
                'turn_idx_in_traj': i,
                'state': state,
                'bucket': density_bucket(state['empties'],
                                          state.get('lec', state['empties'])),
            })
    return out


def match_success_to_failures(failure_dps, success_dps, n_per_failure=1):
    """Pick success DPs that match failure DPs by density bucket.

    For each failure DP, find success DPs in the same bucket and pick one.
    """
    s_by_bucket = defaultdict(list)
    for dp in success_dps:
        s_by_bucket[dp['bucket']].append(dp)
    rng = np.random.default_rng(0)
    matched = []
    for f_dp in failure_dps:
        candidates = s_by_bucket.get(f_dp['bucket'], [])
        if not candidates:
            continue
        picks = rng.choice(len(candidates),
                           size=min(n_per_failure, len(candidates)),
                           replace=False)
        for p in picks:
            matched.append(candidates[p])
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                         default='alphatrain/data/pillar3b_epoch_20.pt')
    parser.add_argument('--fail-dir',
                         default='logs/instrumented_failures_full')
    parser.add_argument('--succ-dir',
                         default='logs/instrumented_success_full')
    parser.add_argument('--out-dir', default='logs/phase_b')
    parser.add_argument('--K', type=int, default=64)
    parser.add_argument('--H', type=int, default=100)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max-dps-per-traj', type=int, default=12,
                         help='Cap DPs per trajectory to avoid one game '
                              'dominating the analysis. Spread across turns.')
    parser.add_argument('--max-failure-dps', type=int, default=150)
    parser.add_argument('--workers', type=int, default=12,
                         help='Number of multiprocessing workers for rollouts.')
    parser.add_argument('--device', default='cpu',
                         choices=['cpu', 'mps', 'cuda'],
                         help='Per-worker device. CPU is most stable with '
                              'multiprocessing on macOS; MPS works with spawn.')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Main-process device for top-k precompute (small, ~300 forwards total)
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Main device: {device}, worker device: {args.device}, "
          f"workers: {args.workers}", flush=True)
    net, _ = load_model(args.model, device,
                         fp16=(device.type != 'cpu'))
    net_dtype = next(net.parameters()).dtype

    # === Collect DPs from both pools ===
    print(f"Collecting DPs from {args.fail_dir}...", flush=True)
    fail_dps = collect_dps(args.fail_dir, 'failure',
                             max_dps_per_traj=args.max_dps_per_traj)
    print(f"  {len(fail_dps)} failure DPs", flush=True)
    print(f"Collecting DPs from {args.succ_dir}...", flush=True)
    succ_dps_raw = collect_dps(args.succ_dir, 'success',
                                 max_dps_per_traj=args.max_dps_per_traj)
    print(f"  {len(succ_dps_raw)} success DPs (pre-match)", flush=True)

    # Cap failure DPs
    if len(fail_dps) > args.max_failure_dps:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(fail_dps), args.max_failure_dps, replace=False)
        fail_dps = [fail_dps[i] for i in sel]
        print(f"  capped to {len(fail_dps)} failure DPs", flush=True)
    succ_dps = match_success_to_failures(fail_dps, succ_dps_raw,
                                           n_per_failure=1)
    print(f"  {len(succ_dps)} matched success DPs", flush=True)

    all_dps = [(dp, 'failure') for dp in fail_dps] + \
              [(dp, 'success') for dp in succ_dps]
    n_total = len(all_dps)
    print(f"\n=== Phase 1: precompute top-{args.top_k} per DP (n={n_total}) ===",
          flush=True)

    # Per-DP top-k in main process (cheap; sequential)
    dp_meta = []  # list of dicts: {dp, label, state, topk}
    for dp_idx, (dp, label) in enumerate(all_dps):
        state = dp['state']
        g = restore_game(state)
        topk = policy_topk(net, device, net_dtype, g, k=args.top_k)
        dp_meta.append({'dp_idx': dp_idx, 'dp': dp, 'label': label,
                        'state': state, 'topk': topk})
    # Free main-process model — workers will load their own
    del net
    if device.type in ('cuda', 'mps'):
        try:
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            torch.mps.empty_cache() if device.type == 'mps' else None
        except Exception:
            pass

    # Build work units: one per (DP, rank)
    units = []
    for m in dp_meta:
        for rank, (move, _prob) in enumerate(m['topk']):
            units.append((m['dp_idx'], rank, move, m['state'],
                           args.K, args.H))
    print(f"  {len(units)} (DP, rank) work units (each = K={args.K} rollouts)",
          flush=True)

    print(f"\n=== Phase 2: rollouts via {args.workers} workers ({args.device}) ===",
          flush=True)
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # results[dp_idx][rank] = list[rollout dicts or None]
    results = defaultdict(dict)
    t0 = time.time()
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.model, args.device)) as pool:
        n_done = 0
        for dp_idx, rank, rollouts in pool.imap_unordered(
                _worker_run, units, chunksize=2):
            results[dp_idx][rank] = rollouts
            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t0
                eta = elapsed / n_done * (len(units) - n_done)
                print(f"  [{n_done}/{len(units)}] elapsed={elapsed:.0f}s "
                      f"eta={eta:.0f}s", flush=True)

    print(f"\n=== Phase 3: aggregate ===", flush=True)
    rows = []
    for m in dp_meta:
        per_branch = []
        for rank, (move, prob) in enumerate(m['topk']):
            if rank not in results[m['dp_idx']]:
                continue
            rollouts = [r for r in results[m['dp_idx']][rank] if r is not None]
            if not rollouts:
                continue
            scores = np.asarray([r['score_gained'] for r in rollouts])
            turns = np.asarray([r['turns_played'] for r in rollouts])
            surv50 = sum(1 for r in rollouts if r['survived_50'])
            surv100 = sum(1 for r in rollouts if r['survived_100'])
            valid = len(rollouts)
            per_branch.append({
                'rank': rank + 1,
                'move': move,
                'prior_p': prob,
                'valid': valid,
                'mean_score': float(scores.mean()),
                'median_score': float(np.median(scores)),
                'p10_score': float(np.percentile(scores, 10)),
                'p25_score': float(np.percentile(scores, 25)),
                'mean_turns': float(turns.mean()),
                'median_turns': float(np.median(turns)),
                'surv50_rate': surv50 / valid,
                'surv100_rate': surv100 / valid,
            })
        if not per_branch:
            continue

        # Lexicographic floor winner: survival_100 → p10 → mean
        def floor_key(b):
            return (b['surv100_rate'], b['p10_score'], b['mean_score'])
        floor_winner = max(per_branch, key=floor_key)
        ev_winner = max(per_branch, key=lambda b: b['mean_score'])

        state = m['state']
        dp = m['dp']
        rows.append({
            'label': m['label'],
            'seed': dp['seed'],
            'turn_in_traj': state['turn'],
            'traj_final_score': dp['traj_final_score'],
            'empties': state['empties'],
            'lec': state.get('lec', state['empties']),
            'n_components': state.get('n_components', 1),
            'top1_p': state.get('top1_p', 0.0),
            'top1_top2_gap': state.get('top1_top2_gap', 0.0),
            'turns_since_clear': state.get('tslc', 0),
            'floor_winner_rank': floor_winner['rank'],
            'ev_winner_rank': ev_winner['rank'],
            'top1_surv100': per_branch[0]['surv100_rate'],
            'top1_p10': per_branch[0]['p10_score'],
            'top1_mean': per_branch[0]['mean_score'],
            'floor_winner_surv100': floor_winner['surv100_rate'],
            'floor_winner_p10': floor_winner['p10_score'],
            'floor_winner_mean': floor_winner['mean_score'],
            'branches': per_branch,
        })

    # === Save raw rows ===
    out_json = os.path.join(args.out_dir, 'per_dp.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=1)
    print(f"\nSaved {len(rows)} DPs to {out_json}")

    # === Cross-tabulate ===
    fail_rows = [r for r in rows if r['label'] == 'failure']
    succ_rows = [r for r in rows if r['label'] == 'success']

    def tabulate(label, rs):
        rank_counts = defaultdict(int)
        floor_gains = []  # (floor_winner_p10 - top1_p10) when floor != top1
        surv_gains = []
        for r in rs:
            rank_counts[r['floor_winner_rank']] += 1
            if r['floor_winner_rank'] != 1:
                floor_gains.append(r['floor_winner_p10'] - r['top1_p10'])
                surv_gains.append(r['floor_winner_surv100'] - r['top1_surv100'])
        total = len(rs)
        print(f"\n=== {label.upper()} — floor winner distribution "
              f"(N={total}) ===")
        for rank in range(1, 6):
            c = rank_counts.get(rank, 0)
            pct = 100 * c / max(1, total)
            print(f"  top-{rank}: {c} ({pct:.1f}%)")
        if floor_gains:
            print(f"  When floor winner != top-1:")
            print(f"    median P10 gap (floor - top1): "
                  f"{np.median(floor_gains):+.0f}")
            print(f"    median survival_100 gap: "
                  f"{100*np.median(surv_gains):+.1f}pp")
        return rank_counts, floor_gains, surv_gains

    f_ranks, f_gains, f_surv = tabulate('failure', fail_rows)
    s_ranks, s_gains, s_surv = tabulate('success', succ_rows)

    # Verdict
    f_top1_win_pct = 100 * f_ranks.get(1, 0) / max(1, len(fail_rows))
    s_top1_win_pct = 100 * s_ranks.get(1, 0) / max(1, len(succ_rows))
    print(f"\n=== VERDICT ===")
    print(f"  Failure DPs:  top-1 wins floor in {f_top1_win_pct:.1f}%, "
          f"top-2..5 wins in {100-f_top1_win_pct:.1f}%")
    print(f"  Success DPs:  top-1 wins floor in {s_top1_win_pct:.1f}%, "
          f"top-2..5 wins in {100-s_top1_win_pct:.1f}%")
    print(f"  Differential: failures lose {(100-f_top1_win_pct) - (100-s_top1_win_pct):+.1f}pp "
          f"more often than successes")
    if f_gains and s_gains:
        med_f_gain = np.median(f_gains)
        med_s_gain = np.median(s_gains) if s_gains else 0
        med_f_surv = 100 * np.median(f_surv)
        med_s_surv = 100 * np.median(s_surv) if s_surv else 0
        print(f"  Median P10 gap (floor>top1): failures {med_f_gain:+.0f} "
              f"vs successes {med_s_gain:+.0f}")
        print(f"  Median survival_100 gap (floor>top1): failures "
              f"{med_f_surv:+.1f}pp vs successes {med_s_surv:+.1f}pp")
        # Decision
        if (med_f_surv >= 10 or med_f_gain >= 100) and \
                (med_f_surv - med_s_surv >= 5 or med_f_gain - med_s_gain >= 50):
            print(f"  ✓ STRONG GO — risk-blind hypothesis CONFIRMED")
        elif (med_f_surv >= 5 or med_f_gain >= 50):
            print(f"  ~ WEAK GO — signal present but marginal")
        else:
            print(f"  ✗ NO-GO — policy is not systematically risk-blind")

    # Save summary
    summary_path = os.path.join(args.out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Phase B counterfactual diagnostic summary\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Params: K={args.K}, H={args.H}, top-{args.top_k}\n\n")
        f.write(f"Failure DPs: {len(fail_rows)}\n")
        f.write(f"Success DPs: {len(succ_rows)}\n\n")
        for label, ranks in [('failure', f_ranks), ('success', s_ranks)]:
            total = sum(ranks.values())
            f.write(f"\n{label} floor-winner-rank distribution:\n")
            for rank in range(1, 6):
                c = ranks.get(rank, 0)
                pct = 100 * c / max(1, total)
                f.write(f"  top-{rank}: {c} ({pct:.1f}%)\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == '__main__':
    main()
