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
DEFAULT_MAX_SCORE = 2000.0
NUM_VALUE_BINS = 64
DEFAULT_GAMMA = 0.99


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


def compute_td_returns(moves, final_score, gamma=DEFAULT_GAMMA,
                       survival_bonus=0.0):
    """Compute discounted TD returns for each position.

    Reconstructs per-move rewards from board states (line clears at target),
    then computes discounted returns: V(t) = sum_{k=0}^{T-t} gamma^k * reward(t+k).

    If survival_bonus > 0, each turn gets a base reward of survival_bonus
    plus score_delta / C, where C = 1/survival_bonus. This creates a hybrid
    "survival + scoring" signal: r(t) = survival_bonus + score_delta / C.
    With survival_bonus=1.0 and C=10: r(t) = 1.0 + points/10.
    """
    from game.board import _find_lines_at, calculate_score as calc_score

    # Step 1: compute per-move score rewards from board snapshots
    score_rewards = []
    for m in moves:
        board = np.array(m['board'], dtype=np.int8)
        mv = m['chosen_move']
        sr, sc, tr, tc = mv['sr'], mv['sc'], mv['tr'], mv['tc']
        color = board[sr, sc]
        board[sr, sc] = 0
        board[tr, tc] = color
        cleared = _find_lines_at(board, tr, tc)
        reward = calc_score(cleared) if cleared > 0 else 0
        score_rewards.append(reward)

    # Step 2: adjust last reward to account for spawn clears (error correction)
    total_from_clears = sum(score_rewards)
    missing = final_score - total_from_clears
    if missing > 0 and len(score_rewards) > 0:
        nonzero = [i for i, r in enumerate(score_rewards) if r > 0]
        if nonzero:
            per_turn = missing / len(nonzero)
            for i in nonzero:
                score_rewards[i] += per_turn

    # Step 3: build per-turn rewards
    if survival_bonus > 0:
        # Hybrid: survival base + scoring bonus
        # r(t) = survival_bonus + score_delta / C, where C = 10 / survival_bonus
        C = 10.0
        rewards = [survival_bonus + s / C for s in score_rewards]
    else:
        # Pure score-based TD returns (original behavior)
        rewards = score_rewards

    # Step 4: compute discounted returns (backward pass)
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = rewards[t] + gamma * running
        returns[t] = running

    return returns.astype(np.float32)


def make_afterstate(board, sr, sc, tr, tc):
    """Make afterstate board (move ball, no line clearing)."""
    b = np.array(board, dtype=np.int8)
    color = b[sr, sc]
    b[sr, sc] = 0
    b[tr, tc] = color
    return b


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-dir', default='data/expert_v2')
    parser.add_argument('--output', default=None,
                        help='Output path (default: auto-named with gamma)')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--max-score', type=float, default=None,
                        help='Max score for two-hot encoding (default: auto from data)')
    parser.add_argument('--survival-bonus', type=float, default=0.0,
                        help='Per-turn survival reward (0=pure score, 1.0=hybrid survival+score/10)')
    parser.add_argument('--num-bins', type=int, default=NUM_VALUE_BINS,
                        help='Number of categorical value bins (default 64)')
    args = parser.parse_args()

    gamma = args.gamma
    num_bins = args.num_bins
    top_k_policy = 5

    if args.output is None:
        g_str = str(gamma).replace('.', '')
        surv_str = f'_surv{args.survival_bonus}' if args.survival_bonus > 0 else ''
        args.output = f'alphatrain/data/expert_v2_pairwise_g{g_str}{surv_str}.pt'

    files = sorted(glob.glob(os.path.join(args.games_dir, 'game_seed*.json')))
    print(f"Found {len(files)} game files in {args.games_dir}", flush=True)
    print(f"Gamma: {gamma}, survival_bonus: {args.survival_bonus}, "
          f"bins: {num_bins}, output: {args.output}", flush=True)

    all_boards = []
    all_next_pos = []
    all_next_col = []
    all_n_next = []
    all_pol_indices = []
    all_pol_values = []
    all_pol_nnz = []
    all_td_scalars = []  # raw TD returns (for distribution analysis)
    all_turns_remaining = []  # turns until game over
    all_good_boards = []
    all_bad_boards = []
    all_margins = []
    all_pair_base_idx = []

    t0 = time.time()
    total_states = 0
    total_pairs = 0

    for fi, fpath in enumerate(files):
        game = json.load(open(fpath))
        n_moves = len(game['moves'])

        # Compute TD returns for this game
        td_returns = compute_td_returns(game['moves'], game['score'], gamma=gamma,
                                        survival_bonus=args.survival_bonus)

        for mi, move in enumerate(game['moves']):
            board = np.array(move['board'], dtype=np.int8)
            all_boards.append(board)

            # Turns remaining until game over
            all_turns_remaining.append(n_moves - mi)

            # Raw TD return scalar
            all_td_scalars.append(td_returns[mi])

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

    # Analyze TD return distribution to determine max_score
    td_arr = np.array(all_td_scalars, dtype=np.float32)
    tr_arr = np.array(all_turns_remaining, dtype=np.int32)
    print(f"\nTD returns (gamma={gamma}):", flush=True)
    print(f"  Min:    {td_arr.min():.1f}", flush=True)
    print(f"  P25:    {np.percentile(td_arr, 25):.1f}", flush=True)
    print(f"  Median: {np.median(td_arr):.1f}", flush=True)
    print(f"  Mean:   {td_arr.mean():.1f}", flush=True)
    print(f"  P75:    {np.percentile(td_arr, 75):.1f}", flush=True)
    print(f"  P95:    {np.percentile(td_arr, 95):.1f}", flush=True)
    print(f"  P99:    {np.percentile(td_arr, 99):.1f}", flush=True)
    print(f"  P99.9:  {np.percentile(td_arr, 99.9):.1f}", flush=True)
    print(f"  Max:    {td_arr.max():.1f}", flush=True)
    print(f"\nTurns remaining:", flush=True)
    print(f"  Min: {tr_arr.min()}, Max: {tr_arr.max()}, "
          f"Mean: {tr_arr.mean():.0f}, Median: {np.median(tr_arr):.0f}", flush=True)
    endgame = (tr_arr <= 100).sum()
    print(f"  Last 100 turns: {endgame:,} ({100*endgame/len(tr_arr):.1f}%)", flush=True)

    # Set max_score: round up P99.9 to nice number with headroom
    if args.max_score is not None:
        max_score = args.max_score
    else:
        p999 = np.percentile(td_arr, 99.9)
        # Round up to next 50 with 20% headroom
        max_score = float(np.ceil(p999 * 1.2 / 50) * 50)
    print(f"\nUsing max_score={max_score} (bin width={max_score/(num_bins-1):.2f})",
          flush=True)
    n_clipped = (td_arr > max_score).sum()
    print(f"  Clipped: {n_clipped:,} ({100*n_clipped/len(td_arr):.3f}%)", flush=True)

    # Encode value targets as two-hot categorical
    print("Encoding two-hot targets...", flush=True)
    all_val_targets = []
    for v in all_td_scalars:
        all_val_targets.append(score_to_twohot(v, max_score, num_bins))

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
        'turns_remaining': torch.tensor(all_turns_remaining, dtype=torch.int32),
        'good_boards': torch.tensor(np.stack(all_good_boards), dtype=torch.int8),
        'bad_boards': torch.tensor(np.stack(all_bad_boards), dtype=torch.int8),
        'margins': torch.tensor(all_margins, dtype=torch.float32),
        'pair_base_idx': torch.tensor(all_pair_base_idx, dtype=torch.int64),
        'num_value_bins': num_bins,
        'max_score': max_score,
        'num_channels': 18,
        'value_mode': 'pairwise',
        'gamma': gamma,
        'n_pairs': total_pairs,
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.0f} MB)", flush=True)

    # Summary
    print(f"\nShapes:", flush=True)
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v.shape} {v.dtype}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)


if __name__ == '__main__':
    main()
