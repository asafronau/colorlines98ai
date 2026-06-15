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


K_CAND = 10   # candidates stored per state for prior/Q/visit targets (Gumbel/advantage)


def move_to_flat(sr, sc, tr, tc):
    """Convert (sr, sc, tr, tc) to flat index in 6561 space."""
    return sr * 9 * 9 * 9 + sc * 9 * 9 + tr * 9 + tc


def extract_candidates(move):
    """Return (flat[K_CAND], visits[K], prior[K], q[K], n, root_value, q_min, q_max).

    New schema (selfplay/crisis with Q): move has cand_moves (flat) / cand_visits /
    cand_prior (clean pre-Dirichlet log-prob) / cand_q (root Q) / root_value / q_min/q_max.
    Old schema (visit-only): top_moves (sr/sc/tr/tc dicts) + top_scores (log visit prob);
    prior := top_scores (no clean prior available), q := 0 (no Q → Gumbel unavailable).
    """
    flat = np.zeros(K_CAND, dtype=np.int64)
    visits = np.zeros(K_CAND, dtype=np.float32)
    prior = np.zeros(K_CAND, dtype=np.float32)
    q = np.zeros(K_CAND, dtype=np.float32)
    if 'cand_moves' in move:
        cm = move['cand_moves'][:K_CAND]
        n = len(cm)
        flat[:n] = np.asarray(cm, dtype=np.int64)
        visits[:n] = np.asarray(move['cand_visits'][:n], dtype=np.float32)
        prior[:n] = np.asarray(move['cand_prior'][:n], dtype=np.float32)
        q[:n] = np.asarray(move['cand_q'][:n], dtype=np.float32)
        return (flat, visits, prior, q, n, float(move.get('root_value', 0.0)),
                float(move.get('q_min', 0.0)), float(move.get('q_max', 0.0)))
    top = move['top_moves'][:K_CAND]
    ts = move['top_scores'][:K_CAND]
    n = len(top)
    for i in range(n):
        m = top[i]
        flat[i] = move_to_flat(m['sr'], m['sc'], m['tr'], m['tc'])
        prior[i] = ts[i]
        visits[i] = np.exp(ts[i])         # top_scores are log visit-probs
    return flat, visits, prior, q, n, 0.0, 0.0, 0.0


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
                       survival_bonus=0.0, bootstrap_value=0.0,
                       density_reward=False):
    """Compute discounted TD returns for each position.

    Reconstructs per-move rewards from board states (line clears at target),
    then computes discounted returns: V(t) = sum_{k=0}^{T-t} gamma^k * reward(t+k).

    If survival_bonus > 0:
        r(t) = survival_bonus + score_delta / 10 [+ empty_squares / 81 if density_reward]
    The density reward adds board occupancy to each turn's reward, providing
    continuous gradient signal that breaks the "value blob" for long games.
    """
    from game.board import _find_lines_at, calculate_score as calc_score

    # Step 1: compute per-move score rewards and board density
    score_rewards = []
    empty_fractions = []
    for m in moves:
        board = np.array(m['board'], dtype=np.int8)
        empty_fractions.append(np.sum(board == 0) / 81.0)
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
        C = 10.0
        if density_reward:
            # r(t) = 1.0 + empty/81 + score/10
            rewards = [survival_bonus + empty_fractions[i] + score_rewards[i] / C
                       for i in range(len(score_rewards))]
        else:
            # r(t) = 1.0 + score/10
            rewards = [survival_bonus + s / C for s in score_rewards]
    else:
        rewards = score_rewards

    # Step 4: compute discounted returns (backward pass)
    # bootstrap_value > 0 for capped games (not natural death):
    # V(T) = bootstrap instead of 0, so the model doesn't learn "fake death"
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    running = float(bootstrap_value)
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
    parser.add_argument('--games-dir', nargs='+', default=['data/expert_v2'],
                        help='One or more directories containing game JSONs. '
                             'V10 typically passes regular + openings + crisis.')
    parser.add_argument('--output', default=None,
                        help='Output path (default: auto-named with gamma)')
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    parser.add_argument('--max-score', type=float, default=None,
                        help='Max score for two-hot encoding (default: auto from data)')
    parser.add_argument('--survival-bonus', type=float, default=0.0,
                        help='Per-turn survival reward (0=pure score, 1.0=hybrid survival+score/10)')
    parser.add_argument('--density-reward', action='store_true',
                        help='Add empty_squares/81 to per-turn reward (breaks value blob)')
    parser.add_argument('--sqrt-turns', action='store_true',
                        help='Use sqrt(remaining_turns) as value target instead of TD returns. '
                             'Gives 485x stronger SNR for survival prediction.')
    parser.add_argument('--sqrt-turns-bonus', type=int, default=2000,
                        help='Estimated bonus turns beyond cap for capped games (default 2000)')
    parser.add_argument('--num-bins', type=int, default=NUM_VALUE_BINS,
                        help='Number of categorical value bins (default 64)')
    parser.add_argument('--policy-only-data', action='store_true',
                        help='Acknowledge that the input self-play data was '
                             'produced for policy-only training (V10+ writes '
                             'bootstrap_value=0 for capped games). Without '
                             'this flag, the builder refuses such data when '
                             'producing TD-return value targets.')
    args = parser.parse_args()

    gamma = args.gamma
    num_bins = args.num_bins
    top_k_policy = 5

    if args.output is None:
        g_str = str(gamma).replace('.', '')
        surv_str = f'_surv{args.survival_bonus}' if args.survival_bonus > 0 else ''
        args.output = f'alphatrain/data/expert_v2_pairwise_g{g_str}{surv_str}.pt'

    files = []
    for d in args.games_dir:
        d_files = sorted(glob.glob(os.path.join(d, 'game_seed*.json')))
        print(f"Found {len(d_files)} game files in {d}", flush=True)
        files.extend(d_files)
    print(f"Total: {len(files)} games across {len(args.games_dir)} dir(s)", flush=True)
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
    # NEW: candidate-level prior/Q for Gumbel/advantage targets (the trunk recipe)
    all_cand_idx = []
    all_cand_visit = []
    all_cand_prior = []
    all_cand_q = []
    all_cand_nnz = []
    all_root_value = []
    all_q_min = []
    all_q_max = []

    # SLIM: policy-only data → skip value-target (val_targets) + pairwise (good/bad
    # boards) machinery entirely. Drops ~4GB of fields train_path_b never reads
    # (project_slim_tensor_3j) AND sidesteps the capped-bootstrap value misuse.
    slim = args.policy_only_data

    t0 = time.time()
    total_states = 0
    total_pairs = 0
    capped_zero_bootstrap = 0  # count of capped games with bootstrap_value=0

    for fi, fpath in enumerate(files):
        game = json.load(open(fpath))
        n_moves = len(game['moves'])

        if slim:
            td_returns = None   # value targets not built for policy-only tensors
        elif args.sqrt_turns:
            # sqrt(remaining_turns) value target — loud survival signal
            is_capped = game.get('capped', False)
            bonus = args.sqrt_turns_bonus if is_capped else 0
            effective_length = n_moves + bonus
            td_returns = np.array([np.sqrt(effective_length - t)
                                   for t in range(n_moves)], dtype=np.float32)
        else:
            # Compute TD returns for this game
            # Capped games use bootstrap value instead of 0 at terminal state
            bootstrap = float(game.get('bootstrap_value', 0.0))
            if game.get('capped', False) and bootstrap == 0.0:
                # V10+ policy-only data writes bootstrap=0 because value
                # targets are not consumed during training. If this builder
                # is being used to *create* value targets, that's a misuse
                # — capped games would be labeled as deaths.
                capped_zero_bootstrap += 1
            td_returns = compute_td_returns(game['moves'], game['score'], gamma=gamma,
                                            survival_bonus=args.survival_bonus,
                                            bootstrap_value=bootstrap,
                                            density_reward=args.density_reward)

        for mi, move in enumerate(game['moves']):
            board = np.array(move['board'], dtype=np.int8)
            all_boards.append(board)

            # Turns remaining until game over
            all_turns_remaining.append(n_moves - mi)

            # Raw TD return scalar (skipped for slim/policy-only)
            if not slim:
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

            # Candidates: flat moves + visits + CLEAN prior + root Q (Gumbel/advantage)
            flat, visits, prior, q, n_cand, rootv, qmin, qmax = extract_candidates(move)
            all_cand_idx.append(flat)
            all_cand_visit.append(visits)
            all_cand_prior.append(prior)
            all_cand_q.append(q)
            all_cand_nnz.append(n_cand)
            all_root_value.append(rootv)
            all_q_min.append(qmin)
            all_q_max.append(qmax)

            # Visit-distribution policy target (top-5, normalized) — kept for the
            # current/fallback distillation target and apples-to-apples with V13.
            indices = np.zeros(top_k_policy, dtype=np.int64)
            values = np.zeros(top_k_policy, dtype=np.float32)
            k5 = min(n_cand, top_k_policy)
            indices[:k5] = flat[:k5]
            if k5 > 0:
                vv = visits[:k5].astype(np.float64)
                s = vv.sum()
                values[:k5] = (vv / s if s > 0 else np.ones(k5) / k5).astype(np.float32)
            all_pol_indices.append(indices)
            all_pol_values.append(values)
            all_pol_nnz.append(k5)

            # Pairwise good/bad afterstate (value-head training only) — skip in slim.
            if not slim and n_cand >= 2:
                bsf, wsf = int(flat[0]), int(flat[k5 - 1])
                good = make_afterstate(board, bsf // 729, (bsf // 81) % 9,
                                       (bsf // 9) % 9, bsf % 9)
                bad = make_afterstate(board, wsf // 729, (wsf // 81) % 9,
                                      (wsf // 9) % 9, wsf % 9)
                all_good_boards.append(good)
                all_bad_boards.append(bad)
                all_margins.append(float(prior[0] - prior[k5 - 1]))
                all_pair_base_idx.append(total_states)
                total_pairs += 1

            total_states += 1

        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {fi+1}/{len(files)} games, {total_states:,} states, "
                  f"{total_pairs:,} pairs ({elapsed:.0f}s)", flush=True)

    print(f"\nTotal: {total_states:,} states, {total_pairs:,} pairs from "
          f"{len(files)} games ({time.time()-t0:.0f}s)", flush=True)

    # Crash on policy-only data being consumed for value training. V10+
    # self-play writes bootstrap_value=0 for capped games because the
    # data is intended for policy-only distillation. If this builder is
    # producing value targets (not sqrt_turns) from such data, capped
    # games would be labeled as deaths — the value head would learn that
    # turn-6000 boards are losses. Refuse unless explicitly opted in.
    if capped_zero_bootstrap > 0 and not args.sqrt_turns and \
            not args.policy_only_data:
        raise SystemExit(
            f"\nFound {capped_zero_bootstrap} capped games with "
            f"bootstrap_value=0 (V10+ policy-only data convention). The "
            f"builder is producing TD-return value targets, which would "
            f"label those games as deaths. Either:\n"
            f"  - pass --policy-only-data to acknowledge that value "
            f"targets won't be used at training time, or\n"
            f"  - regenerate self-play with non-zero bootstrap values, or\n"
            f"  - use --sqrt-turns to compute targets from turn counts.")

    if slim:
        max_score = args.max_score or 0.0
        print(f"\nSLIM (policy-only): skipping value targets + pairwise boards "
              f"(~4GB of fields train_path_b ignores).", flush=True)
        # Guard: old-schema (no-Q) games yield all-zero cand_q and would corrupt
        # the Gumbel target. Real root Q is never exactly 0 for every candidate.
        _cq = np.stack(all_cand_q)
        _noq = int((_cq == 0).all(axis=1).sum())
        if _noq:
            print(f"  *** WARNING: {_noq}/{len(_cq)} states "
                  f"({100*_noq/len(_cq):.1f}%) have ALL-ZERO Q — these are "
                  f"old-schema/no-Q games and WILL corrupt the Gumbel target. "
                  f"Remove non-Q games from the input dirs and rebuild. ***",
                  flush=True)
    else:
        # Analyze TD return distribution to determine max_score
        td_arr = np.array(all_td_scalars, dtype=np.float32)
        tr_arr = np.array(all_turns_remaining, dtype=np.int32)
        print(f"\nTD returns (gamma={gamma}):", flush=True)
        print(f"  Min:    {td_arr.min():.1f}", flush=True)
        print(f"  Median: {np.median(td_arr):.1f}  Mean: {td_arr.mean():.1f}  "
              f"P99.9: {np.percentile(td_arr, 99.9):.1f}  Max: {td_arr.max():.1f}",
              flush=True)
        if args.max_score is not None:
            max_score = args.max_score
        else:
            max_score = float(np.ceil(np.percentile(td_arr, 99.9) * 1.2 / 50) * 50)
        n_clipped = (td_arr > max_score).sum()
        print(f"Using max_score={max_score}  clipped {n_clipped:,} "
              f"({100*n_clipped/len(td_arr):.3f}%)", flush=True)
        print("Encoding two-hot targets...", flush=True)
        all_val_targets = [score_to_twohot(v, max_score, num_bins)
                           for v in all_td_scalars]

    # Stack into tensors. Candidate prior/Q (Gumbel/advantage) + visit policy
    # target are ALWAYS written; value/pairwise fields only when not slim.
    print("Stacking tensors...", flush=True)
    data = {
        'boards': torch.tensor(np.stack(all_boards), dtype=torch.int8),
        'next_pos': torch.tensor(np.stack(all_next_pos), dtype=torch.int8),
        'next_col': torch.tensor(np.stack(all_next_col), dtype=torch.int8),
        'n_next': torch.tensor(all_n_next, dtype=torch.int8),
        'pol_indices': torch.tensor(np.stack(all_pol_indices), dtype=torch.int64),
        'pol_values': torch.tensor(np.stack(all_pol_values), dtype=torch.float32),
        'pol_nnz': torch.tensor(all_pol_nnz, dtype=torch.int64),
        # candidate-level improvement-target fields
        'cand_idx': torch.tensor(np.stack(all_cand_idx), dtype=torch.int64),
        'cand_visit': torch.tensor(np.stack(all_cand_visit), dtype=torch.float32),
        'cand_prior': torch.tensor(np.stack(all_cand_prior), dtype=torch.float32),
        'cand_q': torch.tensor(np.stack(all_cand_q), dtype=torch.float32),
        'cand_nnz': torch.tensor(all_cand_nnz, dtype=torch.int64),
        'root_value': torch.tensor(all_root_value, dtype=torch.float32),
        'q_min': torch.tensor(all_q_min, dtype=torch.float32),
        'q_max': torch.tensor(all_q_max, dtype=torch.float32),
        'turns_remaining': torch.tensor(all_turns_remaining, dtype=torch.int32),
        'num_value_bins': num_bins,
        'max_score': max_score,
        'num_channels': 18,
        'value_mode': 'policy_slim' if slim else 'pairwise',
        'gamma': gamma,
        'n_pairs': total_pairs,
        'k_cand': K_CAND,
    }
    if not slim:
        data['val_targets'] = torch.tensor(np.stack(all_val_targets), dtype=torch.float32)
        data['good_boards'] = torch.tensor(np.stack(all_good_boards), dtype=torch.int8)
        data['bad_boards'] = torch.tensor(np.stack(all_bad_boards), dtype=torch.int8)
        data['margins'] = torch.tensor(all_margins, dtype=torch.float32)
        data['pair_base_idx'] = torch.tensor(all_pair_base_idx, dtype=torch.int64)

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
