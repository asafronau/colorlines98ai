"""Diagnose why MCTS at high sim count underperforms low sim count on a seed.

Plays the same seed in parallel with two sim counts. At the first turn
where they pick different moves, prints both PUCT tables (top-N by
visits) so we can see what changed: did the higher-sim search pick
a lower-prior, higher-Q move? Did it commit harder to a value-bias trap?

Usage:
    python -m alphatrain.scripts.diagnose_sims \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --feature-value-weights alphatrain/data/feature_value_weights.npz \
        --seed 0 --sims-low 400 --sims-high 1600 --device mps
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import time
import math
import argparse
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def fmt_action(flat):
    s = flat // 81
    t = flat % 81
    return f"({s//9},{s%9})->({t//9},{t%9})"


def policy_top_k(net, device, game, k=30, fp16=False):
    """Get top-K policy priors for a board, sorted descending."""
    obs_np = _build_obs_for_game(game)
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    if fp16:
        obs = obs.half()
    with torch.inference_mode():
        pol_logits, _ = net(obs)
    pol_np = pol_logits[0].float().cpu().numpy()
    priors = _get_legal_priors_flat(game.board, pol_np, k)
    return sorted(priors.items(), key=lambda x: -x[1])


def print_puct_table(label, mcts, sims, root, min_q, max_q,
                      policy_ranks, top_n=12):
    """Print PUCT diagnostic for a search result."""
    q_range = max_q - min_q
    sqrt_parent = math.sqrt(root.visit_count)
    children = []
    for act, child in root.children.items():
        vc = child.visit_count
        if vc > 0:
            q_raw = child.value_sum / vc
            q_norm = (q_raw - min_q) / q_range if q_range > 0 else 0.5
        else:
            q_raw = 0.0
            q_norm = 0.5
        u = mcts.c_puct * child.prior * sqrt_parent / (1 + vc)
        children.append({
            'action': act, 'prior': child.prior, 'visits': vc,
            'q_raw': q_raw, 'q_norm': q_norm, 'u': u,
            'score': q_norm + u,
            'pol_rank': policy_ranks.get(act, -1),
        })
    children.sort(key=lambda x: -x['visits'])

    chosen = children[0]['action']
    total_visits = sum(c['visits'] for c in children)
    print(f"\n  --- {label} (sims={sims}, total_visits={total_visits}) ---")
    print(f"  q_range=[{min_q:.3f}, {max_q:.3f}] = {q_range:.3f}")
    print(f"  {'Action':<18} {'PolRk':>5} {'Prior':>6} {'N':>5} {'%':>4} "
          f"{'Q':>7} {'Q_n':>5} {'U':>5} {'Total':>6}")
    for c in children[:top_n]:
        marker = " <CHOSEN" if c['action'] == chosen else ""
        pct = 100 * c['visits'] / max(total_visits, 1)
        pol_rk = (f"#{c['pol_rank']+1}" if c['pol_rank'] >= 0 else "—")
        print(f"  {fmt_action(c['action']):<18} {pol_rk:>5} "
              f"{c['prior']:>6.3f} {c['visits']:>5} {pct:>3.0f}% "
              f"{c['q_raw']:>7.2f} {c['q_norm']:>5.3f} "
              f"{c['u']:>5.3f} {c['score']:>6.3f}{marker}")
    return chosen


def run_one_search(mcts, game):
    """Run a search, return (chosen_flat, root, min_q, max_q)."""
    action = mcts.search(game, temperature=0.0)
    if action is None:
        return None, None, None, None
    flat = (action[0][0] * 9 + action[0][1]) * 81 + \
           (action[1][0] * 9 + action[1][1])
    return flat, mcts._last_root, mcts._last_min_q, mcts._last_max_q


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--feature-value-weights', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sims-low', type=int, default=400)
    p.add_argument('--sims-high', type=int, default=1600)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--max-divergences', type=int, default=3,
                   help='Stop after this many turns with divergent picks logged')
    p.add_argument('--top-n', type=int, default=12,
                   help='How many children to show in each PUCT table')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device,
                                fp16=(args.device != 'cpu'),
                                jit_trace=True)
    fp16 = next(net.parameters()).dtype == torch.float16

    # Two MCTS instances, one per sim count
    mcts_low = MCTS(net, device, max_score=max_score,
                    num_simulations=args.sims_low, batch_size=8,
                    top_k=30, c_puct=2.5,
                    feature_weights_path=args.feature_value_weights)
    mcts_high = MCTS(net, device, max_score=max_score,
                     num_simulations=args.sims_high, batch_size=8,
                     top_k=30, c_puct=2.5,
                     feature_weights_path=args.feature_value_weights)

    g_low = ColorLinesGame(seed=args.seed); g_low.reset()
    g_high = ColorLinesGame(seed=args.seed); g_high.reset()

    print(f"Seed {args.seed}: comparing {args.sims_low} vs "
          f"{args.sims_high} sims, feature-value MCTS", flush=True)
    print(f"{'='*78}")

    turn = 0
    div_count = 0
    diverged = False
    t0 = time.time()

    while turn < args.max_turns:
        if g_low.game_over and g_high.game_over:
            break

        flat_low = flat_high = None
        if not g_low.game_over:
            flat_low, root_low, minq_low, maxq_low = run_one_search(
                mcts_low, g_low)
        if not g_high.game_over:
            flat_high, root_high, minq_high, maxq_high = run_one_search(
                mcts_high, g_high)

        # Check divergence (compare picks on whichever board is shared)
        if not diverged:
            # Boards are still in sync; compare directly
            if flat_low is not None and flat_high is not None and \
                    flat_low != flat_high:
                # First divergence — full diagnostic
                div_count += 1
                empty = int(np.sum(g_low.board == 0))
                print(f"\n  ### TURN {turn}: FIRST DIVERGENCE ###")
                print(f"  Score (both): {g_low.score}, empty cells: {empty}")
                print(f"  {args.sims_low}-sim chose: {fmt_action(flat_low)}")
                print(f"  {args.sims_high}-sim chose: {fmt_action(flat_high)}")

                # Get policy ranks for context
                policy_ranks = {}
                for rank, (act, _) in enumerate(
                        policy_top_k(net, device, g_low, k=30, fp16=fp16)):
                    policy_ranks[act] = rank

                print_puct_table(f"{args.sims_low}-sim", mcts_low,
                                 args.sims_low, root_low, minq_low,
                                 maxq_low, policy_ranks, top_n=args.top_n)
                print_puct_table(f"{args.sims_high}-sim", mcts_high,
                                 args.sims_high, root_high, minq_high,
                                 maxq_high, policy_ranks, top_n=args.top_n)

                # Highlight the comparison
                low_choice_in_high = next(
                    (c for c in [(a, ch.visit_count, ch.value_sum/max(ch.visit_count,1))
                                  for a, ch in root_high.children.items()]
                     if c[0] == flat_low), None)
                high_choice_in_low = next(
                    (c for c in [(a, ch.visit_count, ch.value_sum/max(ch.visit_count,1))
                                  for a, ch in root_low.children.items()]
                     if c[0] == flat_high), None)
                print(f"\n  Cross-reference:")
                if low_choice_in_high is not None:
                    print(f"    {args.sims_low}'s pick {fmt_action(flat_low)} in "
                          f"{args.sims_high}-sim tree: visits={low_choice_in_high[1]}, "
                          f"Q={low_choice_in_high[2]:.3f}")
                if high_choice_in_low is not None:
                    print(f"    {args.sims_high}'s pick {fmt_action(flat_high)} in "
                          f"{args.sims_low}-sim tree: visits={high_choice_in_low[1]}, "
                          f"Q={high_choice_in_low[2]:.3f}")
                diverged = True

        # Execute chosen moves on respective games
        if flat_low is not None and not g_low.game_over:
            sa = flat_low // 81; ta = flat_low % 81
            g_low.move((sa//9, sa%9), (ta//9, ta%9))
        if flat_high is not None and not g_high.game_over:
            sa = flat_high // 81; ta = flat_high % 81
            g_high.move((sa//9, sa%9), (ta//9, ta%9))

        turn += 1
        if turn % 100 == 0:
            elapsed = time.time() - t0
            print(f"  turn={turn}: low={g_low.score} ({g_low.turns}t) "
                  f"high={g_high.score} ({g_high.turns}t) ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*78}")
    print(f"Final: {args.sims_low}-sim={g_low.score} ({g_low.turns}t)  "
          f"{args.sims_high}-sim={g_high.score} ({g_high.turns}t)  "
          f"({elapsed:.0f}s)")


if __name__ == '__main__':
    main()
