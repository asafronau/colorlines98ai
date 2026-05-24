"""How long do balls sit untouched on the board?

User intuition: in healthy long games, no ball stays in the same cell for
very long — the board churns. In dying games, stuck balls accumulate.

For each game, walk through moves and track per-cell "tenure":
  tenure[t][r,c] = (t - last_change_turn[r,c])   for occupied cells

A cell's "change" event is anything that alters its contents:
  - ball moved into / out of it
  - cleared (becomes empty)
  - spawn (was empty, became occupied)

Aggregate distribution by game-final-score bucket. Question:
"Do failing games have visibly higher ball-tenure than capped/long games?"

Usage:
    python scripts/analyze_ball_tenure.py --games-dir data/selfplay_v13
"""
from __future__ import annotations
import argparse, glob, json, os, time
from collections import defaultdict

import numpy as np


def analyze_game(moves):
    """For each move t in the game, compute tenure-distribution stats over
    occupied cells. Returns dict: turn → (n_occupied, tenure_max,
    tenure_p50, tenure_p90, frac_ten_ge_K for K in {20,50,100}).
    """
    n = len(moves)
    if n < 2:
        return {}
    last_change = np.full((9, 9), 0, dtype=np.int32)
    prev_board = np.asarray(moves[0]['board'], dtype=np.int8)
    out = {}
    # Turn 0 stats — all cells "just changed"
    occ_mask = prev_board != 0
    n_occ = int(occ_mask.sum())
    if n_occ:
        out[0] = {
            'n_occ': n_occ,
            'tenure_max': 0, 'tenure_p50': 0, 'tenure_p90': 0,
            'frac_te_ge_20': 0.0, 'frac_te_ge_50': 0.0, 'frac_te_ge_100': 0.0,
        }
    for t in range(1, n):
        board = np.asarray(moves[t]['board'], dtype=np.int8)
        diff = board != prev_board
        if diff.any():
            last_change[diff] = t
        prev_board = board

        occ_mask = board != 0
        n_occ = int(occ_mask.sum())
        if n_occ == 0:
            continue
        tenures = t - last_change[occ_mask]
        out[t] = {
            'n_occ': n_occ,
            'tenure_max': int(tenures.max()),
            'tenure_p50': int(np.percentile(tenures, 50)),
            'tenure_p90': int(np.percentile(tenures, 90)),
            'frac_te_ge_20': float((tenures >= 20).sum() / n_occ),
            'frac_te_ge_50': float((tenures >= 50).sum() / n_occ),
            'frac_te_ge_100': float((tenures >= 100).sum() / n_occ),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/selfplay_v13')
    p.add_argument('--sample-turns', type=int, nargs='+',
                   default=[100, 200, 500, 1000, 2000, 5000],
                   help='Turns at which to snapshot tenure stats.')
    p.add_argument('--min-game-length', type=int, default=50,
                   help='Skip games shorter than this.')
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.games_dir, 'game_seed*.json')))
    print(f"Loaded {len(files)} files from {args.games_dir}", flush=True)

    # Aggregate by (turn, score_bucket): list of tenure stats
    def score_bucket(s):
        if s < 1000:    return 'A_dead_<1K'
        if s < 5000:    return 'B_low_1-5K'
        if s < 10000:   return 'C_mid_5-10K'
        if s < 18000:   return 'D_high_10-18K'
        return 'E_cap_>=18K'

    buckets = defaultdict(list)  # (turn, bucket) → list of stat-dicts

    t0 = time.time()
    for i, path in enumerate(files):
        try:
            with open(path) as f:
                g = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        moves = g.get('moves', [])
        if len(moves) < args.min_game_length:
            continue
        final_score = g['score']
        bucket = score_bucket(final_score)
        stats = analyze_game(moves)
        for t_sample in args.sample_turns:
            if t_sample in stats:
                buckets[(t_sample, bucket)].append(stats[t_sample])
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(files)}] {elapsed:.0f}s", flush=True)

    # Report per-turn per-bucket distributions
    print(f"\n=== Ball-tenure stats by (turn, final-score bucket) ===")
    print(f"  Definition: 'tenure' = how many turns the cell has held the same")
    print(f"  color without ANY change (move-in, move-out, clear, spawn-into).")
    print(f"  frac_te_ge_K = fraction of currently occupied cells with tenure >= K.\n")
    bucket_order = ['A_dead_<1K', 'B_low_1-5K', 'C_mid_5-10K',
                     'D_high_10-18K', 'E_cap_>=18K']
    for t_sample in args.sample_turns:
        print(f"\n--- Turn {t_sample} ---")
        header = f"  {'bucket':<14s} | {'n':>4s} | "
        header += f"{'n_occ':>6s} | {'ten_max':>14s} | {'ten_p50':>14s} | "
        header += f"{'te≥20%':>14s} | {'te≥50%':>14s} | {'te≥100%':>14s}"
        print(header)
        print('  ' + '-' * (len(header) - 2))
        for bucket in bucket_order:
            samples = buckets.get((t_sample, bucket), [])
            if not samples:
                continue
            n = len(samples)
            n_occ = np.median([s['n_occ'] for s in samples])
            tm = np.array([s['tenure_max'] for s in samples])
            tp50 = np.array([s['tenure_p50'] for s in samples])
            f20 = np.array([s['frac_te_ge_20'] for s in samples]) * 100
            f50 = np.array([s['frac_te_ge_50'] for s in samples]) * 100
            f100 = np.array([s['frac_te_ge_100'] for s in samples]) * 100
            print(f"  {bucket:<14s} | {n:>4d} | "
                  f"{n_occ:>5.0f}  | "
                  f"med={np.median(tm):>3.0f} P90={np.percentile(tm, 90):>3.0f} | "
                  f"med={np.median(tp50):>3.0f} P90={np.percentile(tp50, 90):>3.0f} | "
                  f"med={np.median(f20):>3.0f} P90={np.percentile(f20, 90):>3.0f}  | "
                  f"med={np.median(f50):>3.0f} P90={np.percentile(f50, 90):>3.0f}  | "
                  f"med={np.median(f100):>3.0f} P90={np.percentile(f100, 90):>3.0f}")


if __name__ == '__main__':
    main()
