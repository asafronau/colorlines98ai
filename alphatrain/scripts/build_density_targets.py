"""Build future-state-density value targets (Pillar 3a — Steady-State Sovereign).

Per Gemini diagnosis: replace saturated survival classification with a continuous,
bounded regression target — number of empty squares at state t+H, normalized
to [0, 1] (divide by 81). Doesn't saturate as the policy strengthens; gives
high-resolution signal between "surviving ugly" (~40 empty) and "surviving
elegantly" (~55 empty) — the exact distinction MCTS needs for optionality.

Multiple horizons capture different timescales:
  H=5:   immediate consequence of next moves
  H=15:  short-term board health
  H=50:  medium-term state quality (past Markov mixing time)

For game-over before t+H: target = 0 (dying = no future empty squares).

Output schema (compatible with train_value_head.py once it learns 'density' mode):
  boards, next_pos, next_col, n_next   - same as survival tensors
  density_targets    (N, 3)  float32   - [0, 1] normalized empty counts
  is_train           (N,)    bool      - 90/10 game-level split
  horizons           [5, 15, 50]
  target_type        'density'

Usage:
    python -m alphatrain.scripts.build_density_targets \\
        --games-dir data/selfplay_v12 data/crisis_v12 \\
        --output alphatrain/data/value_targets_v12_density.pt
"""

import os
import json
import glob
import time
import argparse
import numpy as np
import torch


DENSITY_HORIZONS = (5, 15, 50)


def density_targets_for_game(moves, horizons=DENSITY_HORIZONS):
    """Per-position density targets from a single game trajectory.

    For each state t in [0, num_moves):
      For each horizon h in horizons:
        - If t+h < num_moves: target = empty_at(t+h) / 81
        - Else (game ended before t+h): target = 0 (game-over fill state)

    Returns:
      targets: (num_moves, len(horizons)) float32 — [0, 1] normalized
    """
    n = len(moves)
    H = len(horizons)
    targets = np.zeros((n, H), dtype=np.float32)
    if n == 0:
        return targets

    # Precompute empty count at every step
    empty_per_step = np.zeros(n, dtype=np.float32)
    for ti, move in enumerate(moves):
        board = np.asarray(move['board'], dtype=np.int8)
        empty_per_step[ti] = float((board == 0).sum()) / 81.0

    for ti in range(n):
        for hi, h in enumerate(horizons):
            future_t = ti + h
            if future_t < n:
                targets[ti, hi] = empty_per_step[future_t]
            else:
                # Game over before reaching t+h; treat as max-badness (0 empty)
                targets[ti, hi] = 0.0
    return targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--val-frac', type=float, default=0.1,
                   help='Fraction of games for val (game-level split by seed mod).')
    args = p.parse_args()

    files = []
    for d in args.games_dir:
        d_files = sorted(glob.glob(os.path.join(d, 'game_seed*.json')))
        print(f"Found {len(d_files)} files in {d}", flush=True)
        files.extend(d_files)
    print(f"Total: {len(files)} games across {len(args.games_dir)} dirs",
          flush=True)

    boards_l = []
    next_pos_l = []
    next_col_l = []
    n_next_l = []
    density_targets_l = []
    is_train_l = []

    val_modulus = max(2, int(round(1.0 / args.val_frac)))

    t0 = time.time()
    total_states = 0
    n_capped = 0
    for fi, fpath in enumerate(files):
        with open(fpath) as fp:
            game = json.load(fp)
        moves = game.get('moves', [])
        if not moves:
            continue
        if game.get('capped', False):
            n_capped += 1

        seed = int(game.get('seed', 0))
        is_train = (seed % val_modulus) != 0

        density = density_targets_for_game(moves)

        for ti, move in enumerate(moves):
            boards_l.append(np.asarray(move['board'], dtype=np.int8))

            npos = np.zeros((3, 2), dtype=np.int8)
            ncol = np.zeros(3, dtype=np.int8)
            nn = min(int(move['num_next']), 3)
            for i in range(nn):
                nb = move['next_balls'][i]
                npos[i, 0] = nb['row']
                npos[i, 1] = nb['col']
                ncol[i] = nb['color']
            next_pos_l.append(npos)
            next_col_l.append(ncol)
            n_next_l.append(nn)

            density_targets_l.append(density[ti])
            is_train_l.append(is_train)
            total_states += 1

        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(files)}] {total_states:,} states "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"\nTotal: {total_states:,} states from {len(files)} games "
          f"({n_capped} capped) in {time.time()-t0:.0f}s", flush=True)

    targets_arr = np.stack(density_targets_l)
    print(f"\nPer-horizon density stats (normalized empty count, range [0,1]):")
    for hi, h in enumerate(DENSITY_HORIZONS):
        col = targets_arr[:, hi]
        print(f"  H=+{h:>3}: mean={col.mean():.3f}  std={col.std():.3f}  "
              f"P10={np.percentile(col, 10):.3f}  P50={np.percentile(col, 50):.3f}  "
              f"P90={np.percentile(col, 90):.3f}  "
              f"zero-rate={(col == 0).mean()*100:.1f}%", flush=True)

    is_train_arr = np.array(is_train_l, dtype=bool)
    n_train = int(is_train_arr.sum())
    n_val = total_states - n_train
    print(f"\nSplit: train={n_train:,} ({100*n_train/total_states:.1f}%), "
          f"val={n_val:,} ({100*n_val/total_states:.1f}%)", flush=True)

    data = {
        'boards': torch.tensor(np.stack(boards_l), dtype=torch.int8),
        'next_pos': torch.tensor(np.stack(next_pos_l), dtype=torch.int8),
        'next_col': torch.tensor(np.stack(next_col_l), dtype=torch.int8),
        'n_next': torch.tensor(n_next_l, dtype=torch.int8),
        'density_targets': torch.tensor(targets_arr, dtype=torch.float32),
        'is_train': torch.tensor(is_train_arr, dtype=torch.bool),
        'horizons': list(DENSITY_HORIZONS),
        'target_type': 'density',
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
