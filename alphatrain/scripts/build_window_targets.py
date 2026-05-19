"""Build per-WINDOW survival labels from game JSONs (Experiment 1 — hazard relabel).

Difference from build_value_targets.py (cumulative survival):
  - Old labels:  P(survive >= H from state t) — CUMULATIVE
  - New labels:  P(survive window_i | alive at start of window_i) — PER-WINDOW
                 with windows [0, 25), [25, 50), [50, 100), [100, 200).

Why this change (per ChatGPT diagnosis):
As the policy improves, cumulative survival probabilities saturate near 1.0
for most reachable states — the head loses discrimination. Per-window
factors out the "alive so far" condition, putting probability mass into
each window's discrimination boundary instead.

Output schema is IDENTICAL to build_value_targets.py, so train_value_head.py
works unchanged — the labels just have different semantics. Combined
scalar V at inference still uses sum(w_i * p_i) form; weights may need
re-tuning but the framework is the same.

Usage:
    python -m alphatrain.scripts.build_window_targets \\
        --games-dir data/selfplay_v12 data/crisis_v12 \\
        --output alphatrain/data/value_targets_v12_window.pt
"""

import os
import json
import glob
import time
import argparse
import numpy as np
import torch


SURVIVAL_HORIZONS = (25, 50, 100, 200)
WINDOWS = ((0, 25), (25, 50), (50, 100), (100, 200))


def window_survival_labels(num_moves, capped, windows=WINDOWS):
    """Compute per-position, per-WINDOW survival labels.

    For each state at position t with remaining = num_moves - t:
      For each window i with bounds [start_i, end_i):
        - If remaining < start_i:   state died before reaching this window.
                                    No signal — mask=0.
        - If start_i <= remaining < end_i:
          - NOT capped:  state died within this window. label=0 (didn't
                         survive), mask=1.
          - Capped:      we don't know — censored. mask=0.
        - If remaining >= end_i:    state survived past window. label=1, mask=1.

    Returns:
      labels: (num_moves, num_windows) int8 — 1 = survived window, 0 = didn't
      masks:  (num_moves, num_windows) int8 — 1 = usable, 0 = excluded
    """
    W = len(windows)
    labels = np.zeros((num_moves, W), dtype=np.int8)
    masks = np.ones((num_moves, W), dtype=np.int8)
    for ti in range(num_moves):
        remaining = num_moves - ti
        for wi, (start, end) in enumerate(windows):
            if remaining < start:
                labels[ti, wi] = 0
                masks[ti, wi] = 0
            elif remaining < end:
                if capped:
                    labels[ti, wi] = 0
                    masks[ti, wi] = 0
                else:
                    labels[ti, wi] = 0
                    masks[ti, wi] = 1
            else:
                labels[ti, wi] = 1
                masks[ti, wi] = 1
    return labels, masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', nargs='+', required=True,
                   help='One or more directories of game_seed*.json files')
    p.add_argument('--output', required=True,
                   help='Path to write the .pt tensor file')
    p.add_argument('--val-frac', type=float, default=0.1,
                   help='Fraction of games reserved for val. '
                        '0.1 = 10%% of games, deterministic by seed mod 10.')
    args = p.parse_args()

    files = []
    for d in args.games_dir:
        d_files = sorted(glob.glob(os.path.join(d, 'game_seed*.json')))
        print(f"Found {len(d_files)} game files in {d}", flush=True)
        files.extend(d_files)
    print(f"Total: {len(files)} games across {len(args.games_dir)} dir(s)",
          flush=True)

    boards_l = []
    next_pos_l = []
    next_col_l = []
    n_next_l = []
    survive_labels_l = []
    survive_masks_l = []
    is_train_l = []

    val_modulus = max(2, int(round(1.0 / args.val_frac)))

    t0 = time.time()
    total_states = 0
    total_capped = 0
    n_natural = 0
    for fi, fpath in enumerate(files):
        game = json.load(open(fpath))
        moves = game['moves']
        n = len(moves)
        capped = bool(game.get('capped', False))
        if not capped:
            n_natural += 1
        else:
            total_capped += 1

        seed = int(game.get('seed', 0))
        is_train = (seed % val_modulus) != 0

        labels, masks = window_survival_labels(n, capped)

        for ti, move in enumerate(moves):
            board = np.array(move['board'], dtype=np.int8)
            boards_l.append(board)

            npos = np.zeros((3, 2), dtype=np.int8)
            ncol = np.zeros(3, dtype=np.int8)
            nn = min(move['num_next'], 3)
            for i in range(nn):
                nb = move['next_balls'][i]
                npos[i, 0] = nb['row']
                npos[i, 1] = nb['col']
                ncol[i] = nb['color']
            next_pos_l.append(npos)
            next_col_l.append(ncol)
            n_next_l.append(nn)

            survive_labels_l.append(labels[ti])
            survive_masks_l.append(masks[ti])
            is_train_l.append(is_train)

            total_states += 1

        if (fi + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(files)}] {total_states:,} states "
                  f"({elapsed:.0f}s)", flush=True)

    print(f"\nTotal: {total_states:,} states from {len(files)} games "
          f"({n_natural} natural, {total_capped} capped) "
          f"in {time.time()-t0:.0f}s", flush=True)

    masks_arr = np.stack(survive_masks_l)
    labels_arr = np.stack(survive_labels_l)
    print(f"\nPer-window coverage (target_type=window, conditional on alive at window start):")
    for wi, (start, end) in enumerate(WINDOWS):
        n_usable = int(masks_arr[:, wi].sum())
        n_pos = int((labels_arr[:, wi] * masks_arr[:, wi]).sum())
        n_neg = n_usable - n_pos
        pct_usable = 100 * n_usable / total_states
        pct_pos = 100 * n_pos / max(n_usable, 1)
        print(f"  W=[{start:>3},{end:>3}): usable={n_usable:>10,} ({pct_usable:5.1f}%), "
              f"survived={n_pos:>10,} ({pct_pos:5.1f}% of usable), "
              f"died={n_neg:>10,}", flush=True)

    is_train_arr = np.array(is_train_l, dtype=bool)
    n_train = int(is_train_arr.sum())
    n_val = total_states - n_train
    print(f"\nSplit: train={n_train:,} ({100*n_train/total_states:.1f}%), "
          f"val={n_val:,} ({100*n_val/total_states:.1f}%)", flush=True)

    print("\nStacking tensors...", flush=True)
    data = {
        'boards': torch.tensor(np.stack(boards_l), dtype=torch.int8),
        'next_pos': torch.tensor(np.stack(next_pos_l), dtype=torch.int8),
        'next_col': torch.tensor(np.stack(next_col_l), dtype=torch.int8),
        'n_next': torch.tensor(n_next_l, dtype=torch.int8),
        'survive_labels': torch.tensor(labels_arr, dtype=torch.int8),
        'survive_masks': torch.tensor(masks_arr, dtype=torch.int8),
        'is_train': torch.tensor(is_train_arr, dtype=torch.bool),
        'horizons': list(SURVIVAL_HORIZONS),  # window end-points, for ValueHead compat
        'target_type': 'window',
        'window_bounds': [list(w) for w in WINDOWS],
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
