"""Phase 1 / step 1: Analyze stationary-window outcomes in selfplay corpora.

Per ChatGPT's diagnosis: V-corpus teachers provide weak signal in the
stationary regime (turn >= 100, empty ≈ 35-45). This script samples windows
of horizon H from game trajectories and reports distributions of forward
outcomes (min_empty, empty_delta, LEC delta, score_rate, clear_rate) by
start-state empty/LEC bucket.

Read-only / report-only.

Usage:
    python scripts/analyze_stationary_windows_v13.py \\
        --games-dir data/selfplay_v13 \\
        --stride 50 \\
        --min-start-turn 100      # 0 for crisis (no fill-up phase)
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from collections import defaultdict

import numpy as np

# Score formula: n_balls * (n_balls - 4); 5-line = 5, 6-line = 12, etc.
def score_from_clear(n):
    return n * (n - 4) if n >= 5 else 0


def load_game(path):
    """Returns (moves, final_score, capped, seed). Each move has 'board', 'next_balls'."""
    with open(path) as f:
        g = json.load(f)
    return g['moves'], g['score'], g.get('capped', False), g['seed']


def largest_empty_component(board_2d):
    """BFS over empty cells, return (largest_component_size, n_components)."""
    visited = [[False] * 9 for _ in range(9)]
    best = 0
    n_comp = 0
    for r0 in range(9):
        for c0 in range(9):
            if board_2d[r0][c0] != 0 or visited[r0][c0]:
                continue
            n_comp += 1
            sz = 0
            stack = [(r0, c0)]
            visited[r0][c0] = True
            while stack:
                r, c = stack.pop()
                sz += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr][nc]
                            and board_2d[nr][nc] == 0):
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            if sz > best:
                best = sz
    return best, n_comp


def derive_timeline(moves):
    """For each move t, compute:
      - empties[t]: count of 0 cells in board[t]
      - lec[t]: largest connected empty component
      - n_components[t]: number of distinct empty components
      - cleared_balls[t]: number of balls removed by move t (0 if no clear)
      - score_step[t]: score gained by move t
    Returns numpy arrays of length len(moves).
    """
    n = len(moves)
    empties = np.empty(n, dtype=np.int32)
    lec = np.empty(n, dtype=np.int32)
    n_comp = np.empty(n, dtype=np.int32)
    occ = np.empty(n, dtype=np.int32)
    for t in range(n):
        board = moves[t]['board']
        c = 0
        for row in board:
            for v in row:
                if v == 0:
                    c += 1
        empties[t] = c
        occ[t] = 81 - c
        l, nc = largest_empty_component(board)
        lec[t] = l
        n_comp[t] = nc

    cleared_balls = np.zeros(n, dtype=np.int32)
    score_step = np.zeros(n, dtype=np.int32)
    for t in range(n - 1):
        delta = occ[t + 1] - occ[t]
        if delta < 0:
            cleared = -delta
            cleared_balls[t] = cleared
            score_step[t] = score_from_clear(cleared)
    return empties, lec, n_comp, cleared_balls, score_step


def windows_from_game(empties, lec, n_comp, cleared_balls, score_step,
                       stride, H_values, min_start_turn):
    """Yield (start_turn, empties_now, lec_now, n_comp_now, H, win_metrics)."""
    n = len(empties)
    if n <= min_start_turn:
        return
    for t in range(min_start_turn, n, stride):
        for H in H_values:
            if t + H >= n:
                continue
            window_empties = empties[t:t + H]
            window_lec = lec[t:t + H]
            min_empty = int(window_empties.min())
            min_lec = int(window_lec.min())
            empty_delta = int(empties[t + H - 1] - empties[t])
            lec_delta = int(lec[t + H - 1] - lec[t])
            n_clears = int((cleared_balls[t:t + H] > 0).sum())
            score_in_win = int(score_step[t:t + H].sum())
            yield (t, int(empties[t]), int(lec[t]), int(n_comp[t]), H, {
                'min_empty': min_empty,
                'min_lec': min_lec,
                'empty_delta': empty_delta,
                'lec_delta': lec_delta,
                'score_rate': score_in_win / H,
                'clear_rate': n_clears / H,
            })


def empty_bucket(e):
    """5-cell-wide buckets centered on stationary band."""
    if e < 25: return '<25'
    if e < 30: return '25-29'
    if e < 35: return '30-34'
    if e < 40: return '35-39'
    if e < 45: return '40-44'
    if e < 50: return '45-49'
    if e < 55: return '50-54'
    return '>=55'


def percentile(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/selfplay_v13')
    p.add_argument('--stride', type=int, default=50)
    p.add_argument('--min-start-turn', type=int, default=100,
                   help='Only sample windows starting at turn >= this '
                        '(skip the fill-up phase).')
    p.add_argument('--H-values', type=int, nargs='+',
                   default=[50, 100, 200])
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.games_dir,
                                            'game_seed*.json')))
    print(f"Loaded {len(files)} game files from {args.games_dir}",
          flush=True)

    # Aggregate per (bucket, H): list of window metrics
    buckets_H = defaultdict(list)
    t0 = time.time()
    score_check_ok = 0
    score_check_diff = []
    for i, path in enumerate(files):
        try:
            moves, final_score, capped, seed = load_game(path)
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        if not moves:
            continue
        empties, lec, n_comp, cleared, score_step = derive_timeline(moves)
        computed = int(score_step.sum())
        score_check_diff.append((final_score, computed))

        for t, e_now, l_now, nc_now, H, m in windows_from_game(
                empties, lec, n_comp, cleared, score_step,
                args.stride, args.H_values, args.min_start_turn):
            buckets_H[(empty_bucket(e_now), H)].append(m)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(files)}] {elapsed:.0f}s", flush=True)

    # Sanity check on score derivation
    scd = np.array(score_check_diff)
    final = scd[:, 0]
    computed = scd[:, 1]
    print(f"\n=== Score-derivation sanity check ===")
    print(f"  Final vs computed (n={len(final)}):")
    print(f"    final  mean={final.mean():.0f}  median={np.median(final):.0f}")
    print(f"    computed mean={computed.mean():.0f}  median={np.median(computed):.0f}")
    print(f"    abs diff: mean={np.abs(final-computed).mean():.0f}  "
          f"max={np.abs(final-computed).max():.0f}")
    print(f"  (Mismatches expected from multi-line clears; should be <5% off)")

    # Per-bucket-per-H summary
    print(f"\n=== Stationary window outcomes (start turn >= {args.min_start_turn}, "
          f"stride={args.stride}) ===")
    bucket_order = ['<25', '25-29', '30-34', '35-39', '40-44',
                     '45-49', '50-54', '>=55']
    for H in sorted(set(k[1] for k in buckets_H.keys())):
        print(f"\n--- H = {H} turns ---")
        print(f"  {'bucket':<8s} | {'n':>6s} | "
              f"{'min_empty':>22s} | "
              f"{'empty_delta':>22s} | "
              f"{'score_rate':>22s} | "
              f"{'clear_rate':>22s}")
        print('  ' + '-' * 116)
        for bucket in bucket_order:
            wins = buckets_H.get((bucket, H), [])
            n = len(wins)
            if n == 0:
                continue
            min_e = np.array([w['min_empty'] for w in wins])
            d_e = np.array([w['empty_delta'] for w in wins])
            s_r = np.array([w['score_rate'] for w in wins])
            c_r = np.array([w['clear_rate'] for w in wins])
            print(f"  {bucket:<8s} | {n:>6d} | "
                  f"P10={percentile(min_e, 10):>3.0f} P50={percentile(min_e, 50):>3.0f} P90={percentile(min_e, 90):>3.0f} | "
                  f"P10={percentile(d_e, 10):>4.0f} P50={percentile(d_e, 50):>4.0f} P90={percentile(d_e, 90):>4.0f} | "
                  f"P10={percentile(s_r, 10):>4.2f} P50={percentile(s_r, 50):>4.2f} P90={percentile(s_r, 90):>4.2f} | "
                  f"P10={percentile(c_r, 10):>4.2f} P50={percentile(c_r, 50):>4.2f} P90={percentile(c_r, 90):>4.2f}")

    # Risk thresholds — at each H, how often do "bad" outcomes occur?
    print(f"\n=== Risk event rates ===")
    thresholds = [
        ('future_min_empty<30', lambda w: w['min_empty'] < 30),
        ('future_min_empty<25', lambda w: w['min_empty'] < 25),
        ('future_min_lec<20', lambda w: w['min_lec'] < 20),
        ('future_min_lec<15', lambda w: w['min_lec'] < 15),
        ('future_min_lec<10', lambda w: w['min_lec'] < 10),
        ('future_empty_delta<=-8', lambda w: w['empty_delta'] <= -8),
        ('future_lec_delta<=-10', lambda w: w['lec_delta'] <= -10),
        ('score_rate<1.8', lambda w: w['score_rate'] < 1.8),
        ('clear_rate<0.30', lambda w: w['clear_rate'] < 0.30),
    ]
    print(f"  {'H':>4s} | {'condition':<26s} | {'rate':>8s}  ({'count / total':>20s})")
    print('  ' + '-' * 76)
    for H in sorted(set(k[1] for k in buckets_H.keys())):
        all_wins = [w for k, wins in buckets_H.items() if k[1] == H for w in wins]
        for name, fn in thresholds:
            hit = sum(1 for w in all_wins if fn(w))
            n = len(all_wins)
            print(f"  {H:>4d} | {name:<26s} | {100*hit/max(1,n):>6.1f}%  ({hit:>9d} / {n:>9d})")


if __name__ == '__main__':
    main()
