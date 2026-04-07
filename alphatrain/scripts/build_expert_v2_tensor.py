"""Build pairwise training tensor from Rust-generated expert game JSONs.

Converts data/expert_v2/game_*.json into alphatrain/data/expert_v2_pairwise.pt
matching the TensorDatasetGPU format.

For each game position:
  - board + next_balls → compact storage (obs built on-the-fly during training)
  - chosen_move → sparse policy (index into 6561 flat space)
  - game_score → two-hot categorical value target
  - top_moves[0] vs top_moves[-1] → good/bad afterstate pair with margin

Usage:
    python -m alphatrain.scripts.build_expert_v2_tensor
"""

import os
import json
import glob
import time
import numpy as np
import torch

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561
MAX_SCORE = 30000.0
NUM_VALUE_BINS = 64


def move_to_flat(sr, sc, tr, tc):
    """Convert (sr, sc, tr, tc) to flat index in 6561 space."""
    return sr * 9 * 9 * 9 + sc * 9 * 9 + tr * 9 + tc


def score_to_twohot(score, max_score, num_bins):
    """Convert scalar score to two-hot categorical target."""
    bins = np.linspace(0, max_score, num_bins)
    score = np.clip(score, 0, max_score)
    idx = np.searchsorted(bins, score, side='right') - 1
    idx = np.clip(idx, 0, num_bins - 2)
    frac = (score - bins[idx]) / (bins[idx + 1] - bins[idx] + 1e-8)
    frac = np.clip(frac, 0, 1)
    target = np.zeros(num_bins, dtype=np.float32)
    target[idx] = 1.0 - frac
    target[idx + 1] = frac
    return target


def make_afterstate(board, sr, sc, tr, tc):
    """Make afterstate board (move ball, no line clearing)."""
    b = np.array(board, dtype=np.int8)
    color = b[sr, sc]
    b[sr, sc] = 0
    b[tr, tc] = color
    return b


def main():
    games_dir = 'data/expert_v2'
    output = 'alphatrain/data/expert_v2_pairwise.pt'
    top_k_policy = 5  # store top-5 policy entries

    files = sorted(glob.glob(os.path.join(games_dir, 'game_seed*.json')))
    print(f"Found {len(files)} game files in {games_dir}", flush=True)

    all_boards = []
    all_next_pos = []
    all_next_col = []
    all_n_next = []
    all_pol_indices = []
    all_pol_values = []
    all_pol_nnz = []
    all_val_targets = []
    all_good_boards = []
    all_bad_boards = []
    all_margins = []
    all_pair_base_idx = []

    t0 = time.time()
    total_states = 0
    total_pairs = 0

    for fi, fpath in enumerate(files):
        game = json.load(open(fpath))

        for mi, move in enumerate(game['moves']):
            board = np.array(move['board'], dtype=np.int8)
            all_boards.append(board)

            # Next balls
            npos = np.zeros((3, 2), dtype=np.int8)
            ncol = np.zeros(3, dtype=np.int8)
            nn = min(move['num_next'], 3)
            for i in range(nn):
                nb = move['next_balls'][i]
                npos[i, 0] = nb['row']
                npos[i, 1] = nb['col']
                ncol[i] = nb['color']
            all_next_pos.append(npos)
            all_next_col.append(ncol)
            all_n_next.append(nn)

            # Policy: top moves as sparse indices + softmax values
            top = move['top_moves'][:top_k_policy]
            top_scores = move['top_scores'][:top_k_policy]
            n_top = len(top)

            indices = np.zeros(top_k_policy, dtype=np.int64)
            values = np.zeros(top_k_policy, dtype=np.float32)
            for i in range(n_top):
                m = top[i]
                indices[i] = move_to_flat(m['sr'], m['sc'], m['tr'], m['tc'])
                values[i] = top_scores[i]

            # Normalize values to probabilities (softmax with temp=1)
            if n_top > 0:
                v = values[:n_top]
                v_max = v.max()
                exp_v = np.exp(v - v_max)
                exp_v /= exp_v.sum()
                values[:n_top] = exp_v

            all_pol_indices.append(indices)
            all_pol_values.append(values)
            all_pol_nnz.append(n_top)

            # Value target: two-hot categorical from game score
            val = score_to_twohot(move['game_score'], MAX_SCORE, NUM_VALUE_BINS)
            all_val_targets.append(val)

            # Pairwise: good (best move) vs bad (worst of top-5)
            if n_top >= 2:
                best = top[0]
                worst = top[n_top - 1]
                good = make_afterstate(board, best['sr'], best['sc'], best['tr'], best['tc'])
                bad = make_afterstate(board, worst['sr'], worst['sc'], worst['tr'], worst['tc'])
                margin = top_scores[0] - top_scores[n_top - 1]
                all_good_boards.append(good)
                all_bad_boards.append(bad)
                all_margins.append(margin)
                all_pair_base_idx.append(total_states)
                total_pairs += 1

            total_states += 1

        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {fi+1}/{len(files)} games, {total_states:,} states, "
                  f"{total_pairs:,} pairs ({elapsed:.0f}s)", flush=True)

    print(f"\nTotal: {total_states:,} states, {total_pairs:,} pairs from "
          f"{len(files)} games ({time.time()-t0:.0f}s)", flush=True)

    # Stack into tensors
    print("Stacking tensors...", flush=True)
    data = {
        'boards': torch.tensor(np.stack(all_boards), dtype=torch.int8),
        'next_pos': torch.tensor(np.stack(all_next_pos), dtype=torch.int8),
        'next_col': torch.tensor(np.stack(all_next_col), dtype=torch.int8),
        'n_next': torch.tensor(all_n_next, dtype=torch.int8),
        'pol_indices': torch.tensor(np.stack(all_pol_indices), dtype=torch.int64),
        'pol_values': torch.tensor(np.stack(all_pol_values), dtype=torch.float32),
        'pol_nnz': torch.tensor(all_pol_nnz, dtype=torch.int64),
        'val_targets': torch.tensor(np.stack(all_val_targets), dtype=torch.float32),
        'good_boards': torch.tensor(np.stack(all_good_boards), dtype=torch.int8),
        'bad_boards': torch.tensor(np.stack(all_bad_boards), dtype=torch.int8),
        'margins': torch.tensor(all_margins, dtype=torch.float32),
        'pair_base_idx': torch.tensor(all_pair_base_idx, dtype=torch.int64),
        'num_value_bins': NUM_VALUE_BINS,
        'max_score': MAX_SCORE,
        'num_channels': 18,
        'value_mode': 'pairwise',
        'gamma': 0.99,
        'n_pairs': total_pairs,
    }

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    torch.save(data, output)
    size_mb = os.path.getsize(output) / 1e6
    print(f"\nSaved to {output} ({size_mb:.0f} MB)", flush=True)

    # Summary
    print(f"\nShapes:", flush=True)
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)


if __name__ == '__main__':
    main()
