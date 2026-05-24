"""Phase A: mine bottom-tail pillar3b games for failure modes.

Premise (user, 2026-05-22): humans can always score 1K. Therefore every
<1000 game is a policy bug, not RNG inevitability. This script generates
games, identifies failures, logs per-turn danger metrics, and buckets
the failure mode.

Output:
  - logs/mine_failures_summary.txt — bucketed counts + medians
  - logs/mine_failures/game_<seed>_score<S>.json — failed game traces

Per-turn metrics:
  - empties: 81 − balls_on_board
  - largest_empty_component (LEC): BFS over empty cells
  - num_legal_moves: sum of legal (src, tgt) pairs
  - available_clears: count of moves that immediately clear ≥ a line
  - turns_since_last_clear
  - policy_top1_prob, policy_top1_top2_gap
  - score_delta_this_turn

Bucketing (priority order — first matching wins):
  A_early_rng:     death_turn < 30
  C_tactical_miss: available_clears>0 declined at any turn, and within
                   next 15 turns metrics collapse (empties↓ ≥3 OR
                   num_legal_moves halved)
  B_density_spiral: empties trend monotonically down for ≥20 turns
                   ending in death
  D_slow_drift:    survived ≥100 turns but score/turn < 1.5
  U_unbucketed:    everything else
"""
from __future__ import annotations
import argparse, json, os, sys, time
from collections import deque, Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.config import BOARD_SIZE
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation

# ── Configurable ──────────────────────────────────────────────────────────
FAIL_THRESH = 2000  # score < FAIL_THRESH is "bottom-tail failure"


def largest_empty_component(board):
    """BFS over empty cells, return size of largest connected region."""
    visited = np.zeros_like(board, dtype=bool)
    best = 0
    for r0 in range(9):
        for c0 in range(9):
            if board[r0, c0] != 0 or visited[r0, c0]:
                continue
            sz = 0
            q = deque([(r0, c0)])
            visited[r0, c0] = True
            while q:
                r, c = q.popleft()
                sz += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr, nc]
                            and board[nr, nc] == 0):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if sz > best:
                best = sz
    return best


def policy_argmax_with_stats(net, device, net_dtype, game):
    """Run policy, return (best_move, top1_prob, top1_top2_gap, top5_logits).
    Returns (None, ...) if no legal move."""
    board = game.board
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    obs = torch.from_numpy(
        build_observation(board, nr, nc, ncol, nn)
    ).unsqueeze(0).to(device=device, dtype=net_dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()

    source_mask = game.get_source_mask()
    legal_logits = []
    legal_moves = []
    for sr in range(9):
        for sc in range(9):
            if source_mask[sr, sc] == 0:
                continue
            tmask = game.get_target_mask((sr, sc))
            for tr in range(9):
                for tc in range(9):
                    if tmask[tr, tc] > 0:
                        idx = (sr * 9 + sc) * 81 + tr * 9 + tc
                        legal_logits.append(logits[idx])
                        legal_moves.append(((sr, sc), (tr, tc)))
    if not legal_moves:
        return None, 0.0, 0.0, 0

    lg = np.asarray(legal_logits)
    lg = lg - lg.max()
    e = np.exp(lg)
    p = e / e.sum()
    order = np.argsort(p)[::-1]
    top1_p = float(p[order[0]])
    top2_p = float(p[order[1]]) if len(order) > 1 else 0.0
    return legal_moves[order[0]], top1_p, top1_p - top2_p, len(legal_moves)


def count_available_clears(game):
    """Number of legal moves that immediately clear ≥ one line."""
    source_mask = game.get_source_mask()
    n = 0
    for sr in range(9):
        for sc in range(9):
            if source_mask[sr, sc] == 0:
                continue
            tmask = game.get_target_mask((sr, sc))
            for tr in range(9):
                for tc in range(9):
                    if tmask[tr, tc] > 0:
                        # Use clone to simulate without mutating
                        g2 = game.clone()
                        r = g2.move((sr, sc), (tr, tc))
                        if r.get('cleared', 0) > 0:
                            n += 1
    return n


def play_one(net, device, net_dtype, seed, max_turns=10000,
              track_clears=False):
    """Play one game, log per-turn metrics. Returns trajectory dict."""
    g = ColorLinesGame(seed=seed)
    g.reset()
    metrics = []
    turns_since_last_clear = 0
    prev_score = 0
    while not g.game_over and g.turns < max_turns:
        # Pre-move stats
        empties = int((g.board == 0).sum())
        lec = largest_empty_component(g.board)
        # Policy
        mv, top1_p, gap, nlegal = policy_argmax_with_stats(
            net, device, net_dtype, g)
        if mv is None:
            break
        # Available-clears (expensive — only compute when track_clears or
        # cheap-empty signal suggests danger). Default OFF for first pass;
        # turned ON for bottom-tail games in second pass.
        avail_clears = (count_available_clears(g)
                         if track_clears else -1)
        # Execute
        res = g.move(*mv)
        if not res['valid']:
            break
        cleared = res.get('cleared', 0)
        if cleared > 0:
            turns_since_last_clear = 0
        else:
            turns_since_last_clear += 1
        score_delta = g.score - prev_score
        prev_score = g.score

        metrics.append({
            'turn': g.turns,
            'empties': empties,
            'lec': lec,
            'nlegal': nlegal,
            'top1_p': top1_p,
            'top1_top2_gap': gap,
            'avail_clears': avail_clears,
            'tslc': turns_since_last_clear if cleared == 0
                     else 0,
            'score_delta': score_delta,
            'move_cleared': cleared,
        })
    return {
        'seed': seed,
        'final_score': g.score,
        'final_turn': g.turns,
        'game_over': g.game_over,
        'metrics': metrics,
    }


def classify(traj):
    """Bucket the failure mode. Priority order."""
    m = traj['metrics']
    final_turn = traj['final_turn']
    final_score = traj['final_score']

    if not traj['game_over']:
        return 'survived_to_cap'

    # A. early RNG disaster
    if final_turn < 30:
        return 'A_early_rng'

    # C. tactical miss: clear declined and metrics collapse within 15 turns
    for i, met in enumerate(m):
        if met['avail_clears'] is None or met['avail_clears'] <= 0:
            continue
        # Was a clear available AND policy didn't pick it?
        if met['move_cleared'] > 0:
            continue  # policy took the clear, no miss
        # Look forward 15 turns: did metrics collapse?
        future = m[i + 1: i + 16]
        if not future:
            continue
        emp_drop = met['empties'] - future[-1]['empties']
        legal_drop_ratio = (1.0 - future[-1]['nlegal'] / max(1, met['nlegal']))
        if emp_drop >= 3 or legal_drop_ratio >= 0.5:
            return 'C_tactical_miss'

    # B. density spiral — empties trend negative for ≥20 turns ending in death
    if len(m) >= 20:
        last20 = m[-20:]
        emp_traj = [x['empties'] for x in last20]
        # monotonic-ish down: more decreases than increases
        decs = sum(1 for i in range(1, len(emp_traj))
                   if emp_traj[i] < emp_traj[i - 1])
        incs = sum(1 for i in range(1, len(emp_traj))
                   if emp_traj[i] > emp_traj[i - 1])
        if decs >= incs * 2 and emp_traj[0] - emp_traj[-1] >= 5:
            return 'B_density_spiral'

    # D. slow drift
    if final_turn >= 100 and final_score / max(1, final_turn) < 1.5:
        return 'D_slow_drift'

    return 'U_unbucketed'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',
                   default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--N', type=int, default=500)
    p.add_argument('--seed-start', type=int, default=800000)
    p.add_argument('--fail-thresh', type=int, default=FAIL_THRESH)
    p.add_argument('--save-dir', default='logs/mine_failures')
    p.add_argument('--seeds-list', default=None,
                    help='Skip Pass 1 and replay only these seeds. '
                         'Path to a text file with one seed per line '
                         '(optionally followed by score). Already-known '
                         'failures from a fast eval_parallel scan.')
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('mps' if torch.backends.mps.is_available()
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}", flush=True)
    # IMPORTANT: fp16 to match eval_parallel.py's inference server (line 253
    # of eval_parallel.py uses fp16=(device != 'cpu')). Using fp32 here would
    # reproduce a DIFFERENT game per seed than eval_parallel did, because
    # argmax of policy logits differs under different floating-point precision.
    net, _ = load_model(args.model, device, fp16=(device.type != 'cpu'))
    net_dtype = next(net.parameters()).dtype

    # If a seeds-list is provided, skip Pass 1 — we already know the failures.
    if args.seeds_list:
        with open(args.seeds_list) as f:
            fail_seeds = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                fail_seeds.append(int(line.split()[0]))
        print(f"Skipping Pass 1; loaded {len(fail_seeds)} failure seeds "
              f"from {args.seeds_list}", flush=True)
    else:
        # Pass 1: cheap play to find failures
        print(f"\n=== Pass 1: playing {args.N} games (no avail_clears) ===",
              flush=True)
        t0 = time.time()
        trajs = []
        scores = []
        for i in range(args.N):
            seed = args.seed_start + i
            traj = play_one(net, device, net_dtype, seed, track_clears=False)
            trajs.append(traj)
            scores.append(traj['final_score'])
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (args.N - i - 1)
                print(f"  [{i+1}/{args.N}] score={traj['final_score']}, "
                      f"turn={traj['final_turn']}  "
                      f"({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)
        pass1_time = time.time() - t0
        print(f"Pass 1 done in {pass1_time:.0f}s", flush=True)

        s = np.asarray(scores)
        print(f"\nOverall distribution (N={args.N}):")
        print(f"  mean={s.mean():.0f}  P5={np.percentile(s, 5):.0f}  "
              f"P10={np.percentile(s, 10):.0f}  P25={np.percentile(s, 25):.0f}  "
              f"P50={np.percentile(s, 50):.0f}  min={s.min()}")
        print(f"  <500: {(s < 500).sum()}  <1000: {(s < 1000).sum()}  "
              f"<{args.fail_thresh}: {(s < args.fail_thresh).sum()}",
              flush=True)
        fail_seeds = [t['seed'] for t in trajs
                      if t['final_score'] < args.fail_thresh]
    print(f"\n=== Pass 2: re-running {len(fail_seeds)} failures "
          f"with avail_clears tracking ===", flush=True)
    failed_trajs = []
    t1 = time.time()
    for i, seed in enumerate(fail_seeds):
        traj = play_one(net, device, net_dtype, seed, track_clears=True)
        failed_trajs.append(traj)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (i + 1) * (len(fail_seeds) - i - 1)
            print(f"  [{i+1}/{len(fail_seeds)}] seed={seed} "
                  f"score={traj['final_score']}  "
                  f"({elapsed:.0f}s, eta {eta:.0f}s)", flush=True)

    # Classify + report
    bucket_counts = Counter()
    bucket_examples = defaultdict(list)
    for traj in failed_trajs:
        b = classify(traj)
        bucket_counts[b] += 1
        bucket_examples[b].append(traj)
        # Save each failed trajectory to disk
        fname = f"game_{traj['seed']}_score{traj['final_score']}.json"
        with open(os.path.join(args.save_dir, fname), 'w') as f:
            json.dump({
                'seed': traj['seed'],
                'final_score': traj['final_score'],
                'final_turn': traj['final_turn'],
                'game_over': traj['game_over'],
                'bucket': b,
                'metrics': traj['metrics'],
            }, f)

    print(f"\n{'=' * 80}")
    print(f"  FAILURE BUCKETS (model={os.path.basename(args.model)}, "
          f"N={args.N}, threshold<{args.fail_thresh})")
    print(f"{'=' * 80}")
    if not failed_trajs:
        print("  No failures found.")
        return
    print(f"  {'bucket':<22s} | {'count':<6s} | "
          f"{'%-of-N':<7s} | {'%-of-fail':<10s} | "
          f"{'median death turn':<18s} | {'median score':<12s} | "
          f"{'median avail_clears at end':<26s}")
    print("  " + "-" * 122)
    for bucket in ['A_early_rng', 'C_tactical_miss', 'B_density_spiral',
                    'D_slow_drift', 'U_unbucketed', 'survived_to_cap']:
        c = bucket_counts.get(bucket, 0)
        if c == 0:
            continue
        examples = bucket_examples[bucket]
        med_turn = int(np.median([t['final_turn'] for t in examples]))
        med_score = int(np.median([t['final_score'] for t in examples]))
        clears_end = []
        for t in examples:
            if t['metrics'] and t['metrics'][-1]['avail_clears'] is not None:
                clears_end.append(t['metrics'][-1]['avail_clears'])
        med_clears_end = (f"{np.median(clears_end):.0f}" if clears_end
                           else "n/a")
        print(f"  {bucket:<22s} | {c:>5d}  | "
              f"{100*c/args.N:>5.1f}% | "
              f"{100*c/len(failed_trajs):>8.1f}% | "
              f"{med_turn:>16d}   | {med_score:>10d}   | "
              f"{med_clears_end:>23s}")

    # Top-5 representative failed games per bucket
    print(f"\n  Top-5 reps per bucket (sample for replay):")
    for bucket in ['A_early_rng', 'C_tactical_miss', 'B_density_spiral',
                    'D_slow_drift', 'U_unbucketed']:
        examples = bucket_examples.get(bucket, [])
        if not examples:
            continue
        # Sort by ascending score (worst first)
        examples.sort(key=lambda t: t['final_score'])
        print(f"    {bucket}:")
        for t in examples[:5]:
            print(f"      seed={t['seed']}  score={t['final_score']}  "
                  f"turn={t['final_turn']}")


if __name__ == '__main__':
    main()
