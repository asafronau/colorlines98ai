"""Build a survive_H label tensor from V11 self-play / crisis JSONs.

Walks game JSONs and emits per-position multi-horizon survival labels
for ValueHead training. For each position at turn `t` in a game with
final length `T` (or cap `C`), per horizon H ∈ {25, 50, 100, 200}:

    label=1, mask=1   if  T - t >= H               (definitely survives)
    label=0, mask=1   if  T - t <  H  AND not capped  (definitely dies)
    label=0, mask=0   if  capped AND C - t < H    (censored — excluded)

Censoring is critical: at pillar2y2 strength, 40% of V11 anchor games
hit the cap. Treating capped positions as "deaths" near the end would
teach the head false negatives.

Output: one .pt file with stacked tensors:
  boards            (N, 9, 9)        int8
  next_pos          (N, 3, 2)        int8
  next_col          (N, 3)           int8
  n_next            (N,)             int8
  survive_labels    (N, 4)           int8   — 0/1 per horizon
  survive_masks     (N, 4)           int8   — 1=use in loss, 0=censored
  is_train          (N,)             bool   — game-level 80/20 split
  horizons          [25, 50, 100, 200]

Usage:
    python -m alphatrain.scripts.build_value_targets \\
        --games-dir data/selfplay_v11_s600 data/crisis_v11 \\
        --output alphatrain/data/value_targets_v11.pt
"""

import os
import json
import glob
import time
import argparse

import numpy as np
import torch

from alphatrain.value_head import SURVIVAL_HORIZONS, NUM_HORIZONS

BOARD_SIZE = 9


def survive_labels_for_game(num_moves, capped, horizons=SURVIVAL_HORIZONS):
    """Compute per-position (label, mask) arrays for one game.

    Args:
        num_moves: int, number of moves recorded in the game (length of
            the trajectory). Position t in [0, num_moves) is the board
            state BEFORE the player's t-th move; t = num_moves means
            "end of game". So `T - t = num_moves - t` is the count of
            remaining moves from position t.
        capped: bool, True if the game ended via the cap rather than
            by natural death.
        horizons: tuple of int, survival horizons.

    Returns:
        labels: (num_moves, len(horizons)) int8
        masks:  (num_moves, len(horizons)) int8
    """
    H = len(horizons)
    labels = np.zeros((num_moves, H), dtype=np.int8)
    masks = np.ones((num_moves, H), dtype=np.int8)
    for ti in range(num_moves):
        remaining = num_moves - ti
        for hi, h in enumerate(horizons):
            if remaining >= h:
                labels[ti, hi] = 1
                masks[ti, hi] = 1
            else:
                if capped:
                    # Truncated trajectory: we don't know if the game
                    # would have survived h more moves past the cap.
                    labels[ti, hi] = 0
                    masks[ti, hi] = 0
                else:
                    # Natural death within h moves — definitely dies.
                    labels[ti, hi] = 0
                    masks[ti, hi] = 1
    return labels, masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', nargs='+', required=True,
                   help='One or more directories of game_seed*.json files')
    p.add_argument('--output', required=True,
                   help='Path to write the .pt tensor file')
    p.add_argument('--val-frac', type=float, default=0.1,
                   help='Fraction of GAMES (not positions) reserved for val. '
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
        # Game-level split: deterministic by seed mod val_modulus
        is_train = (seed % val_modulus) != 0

        labels, masks = survive_labels_for_game(n, capped, SURVIVAL_HORIZONS)

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

    # Per-horizon coverage report (how many usable labels per horizon)
    masks_arr = np.stack(survive_masks_l)
    labels_arr = np.stack(survive_labels_l)
    print(f"\nPer-horizon coverage:")
    for hi, h in enumerate(SURVIVAL_HORIZONS):
        n_usable = int(masks_arr[:, hi].sum())
        n_pos = int((labels_arr[:, hi] * masks_arr[:, hi]).sum())
        n_neg = n_usable - n_pos
        pct_usable = 100 * n_usable / total_states
        pct_pos = 100 * n_pos / max(n_usable, 1)
        print(f"  H={h:>4}: usable={n_usable:>10,} ({pct_usable:.1f}% of states), "
              f"positives={n_pos:>10,} ({pct_pos:.1f}% of usable)", flush=True)

    is_train_arr = np.array(is_train_l, dtype=bool)
    n_train = int(is_train_arr.sum())
    n_val = total_states - n_train
    print(f"\nSplit: train={n_train:,} ({100*n_train/total_states:.1f}%), "
          f"val={n_val:,} ({100*n_val/total_states:.1f}%) "
          f"— deterministic by seed mod {val_modulus}", flush=True)

    print("\nStacking tensors...", flush=True)
    data = {
        'boards': torch.tensor(np.stack(boards_l), dtype=torch.int8),
        'next_pos': torch.tensor(np.stack(next_pos_l), dtype=torch.int8),
        'next_col': torch.tensor(np.stack(next_col_l), dtype=torch.int8),
        'n_next': torch.tensor(n_next_l, dtype=torch.int8),
        'survive_labels': torch.tensor(labels_arr, dtype=torch.int8),
        'survive_masks': torch.tensor(masks_arr, dtype=torch.int8),
        'is_train': torch.tensor(is_train_arr, dtype=torch.bool),
        'horizons': list(SURVIVAL_HORIZONS),
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.0f} MB)", flush=True)
    print("Shapes:", flush=True)
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)


if __name__ == '__main__':
    main()
