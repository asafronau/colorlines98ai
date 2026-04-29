"""Diagnose a specific seed: find where and why MCTS diverges from policy.

Plays the same seed with policy-only and MCTS, finds the first turn
where they disagree, and logs the full PUCT decision table.

Usage:
    python -m alphatrain.scripts.diagnose_seed \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --seed 14 --sims 400 --device mps
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F

from game.board import ColorLinesGame, _label_empty_components
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def fmt_action(flat):
    s, t = flat // 81, flat % 81
    return f"({s//9},{s%9})->({t//9},{t%9})"


def board_stats(board):
    """Quick board health metrics."""
    empty = int(np.sum(board == 0))
    labels = _label_empty_components(board)
    components = int(labels.max())
    largest = 0
    for c in range(1, components + 1):
        size = int(np.sum(labels == c))
        if size > largest:
            largest = size
    return empty, components, largest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--sims', type=int, default=400)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--max-divergences', type=int, default=10,
                   help='Stop after this many divergences logged')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=False, jit_trace=False)
    net.eval()

    mcts = MCTS(net, device, max_score=max_score, num_simulations=args.sims,
                batch_size=8, top_k=30, c_puct=2.5)

    # Play two games in parallel: policy-only and MCTS
    game_pol = ColorLinesGame(seed=args.seed)
    game_pol.reset()
    game_mcts = ColorLinesGame(seed=args.seed)
    game_mcts.reset()

    print(f"Seed {args.seed}, sims={args.sims}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    turn = 0
    divergences = 0
    diverged = False  # once games diverge, boards differ

    while turn < args.max_turns:
        pol_over = game_pol.game_over
        mcts_over = game_mcts.game_over
        if pol_over and mcts_over:
            break

        # Policy move (on policy game)
        pol_action = None
        if not pol_over:
            obs_np = _build_obs_for_game(game_pol)
            obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            with torch.inference_mode():
                pol_logits, _ = net(obs)
            pol_np = pol_logits[0].float().cpu().numpy()
            priors = _get_legal_priors_flat(game_pol.board, pol_np, 30)
            if priors:
                pol_action = max(priors.items(), key=lambda x: x[1])[0]

        # MCTS move (on MCTS game)
        mcts_action_flat = None
        if not mcts_over:
            action = mcts.search(game_mcts, temperature=0.0,
                                 dirichlet_alpha=0.0, dirichlet_weight=0.0)
            if action is not None:
                mcts_action_flat = (action[0][0] * 9 + action[0][1]) * 81 + \
                                   (action[1][0] * 9 + action[1][1])

        # Check for divergence (only meaningful before games diverge)
        if not diverged and not pol_over and not mcts_over:
            if pol_action != mcts_action_flat and pol_action is not None \
                    and mcts_action_flat is not None:
                divergences += 1
                empty, n_comp, largest = board_stats(game_mcts.board)

                print(f"\n  DIVERGENCE #{divergences} at turn {turn}",
                      flush=True)
                print(f"  Board: empty={empty} components={n_comp} "
                      f"largest={largest} score={game_mcts.score}", flush=True)
                print(f"  Policy:  {fmt_action(pol_action)}", flush=True)
                print(f"  MCTS:    {fmt_action(mcts_action_flat)}", flush=True)

                # Log root PUCT table
                root = mcts._last_root
                min_q = mcts._last_min_q
                max_q = mcts._last_max_q
                q_range = max_q - min_q

                sqrt_parent = math.sqrt(root.visit_count)
                children = []
                for act, child in root.children.items():
                    vc = child.visit_count
                    if vc > 0:
                        q_raw = child.value_sum / vc
                        q_norm = (q_raw - min_q) / q_range \
                            if q_range > 0 else 0.5
                    else:
                        q_raw = 0.0
                        q_norm = 0.5
                    u = 2.5 * child.prior * sqrt_parent / (1 + vc)
                    score = q_norm + u
                    children.append(
                        (act, child.prior, vc, q_raw, q_norm, u, score))

                children.sort(key=lambda x: -x[2])  # sort by visits
                print(f"  q_range={q_range:.4f} min_q={min_q:.4f} "
                      f"max_q={max_q:.4f}", flush=True)
                print(f"  {'Action':<18} {'Prior':>6} {'N':>5} "
                      f"{'Q':>8} {'Q_n':>6} {'U':>6} {'Score':>7}",
                      flush=True)
                for act, prior, vc, q_raw, q_norm, u, score \
                        in children[:10]:
                    marker = ""
                    if act == pol_action:
                        marker += " <POL"
                    if act == mcts_action_flat:
                        marker += " <MCTS"
                    print(f"  {fmt_action(act):<18} {prior:>6.3f} {vc:>5} "
                          f"{q_raw:>8.2f} {q_norm:>6.3f} {u:>6.3f} "
                          f"{score:>7.3f}{marker}", flush=True)

                if divergences >= args.max_divergences:
                    print(f"\n  (stopping after {args.max_divergences} "
                          f"divergences)", flush=True)
                    break

                # After first divergence, games will have different boards
                diverged = True

        # Execute moves
        if not pol_over and pol_action is not None:
            sa = pol_action // 81
            ta = pol_action % 81
            game_pol.move((sa // 9, sa % 9), (ta // 9, ta % 9))
        elif not pol_over:
            break

        if not mcts_over and mcts_action_flat is not None:
            sa = mcts_action_flat // 81
            ta = mcts_action_flat % 81
            game_mcts.move((sa // 9, sa % 9), (ta // 9, ta % 9))
        elif not mcts_over:
            break

        turn += 1
        if turn % 500 == 0:
            elapsed = time.time() - t0
            print(f"\n  turn={turn} pol={game_pol.score} "
                  f"mcts={game_mcts.score} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"Policy: score={game_pol.score} turns={game_pol.turns}",
          flush=True)
    print(f"MCTS:   score={game_mcts.score} turns={game_mcts.turns}",
          flush=True)
    print(f"Divergences: {divergences} in {turn} turns ({elapsed:.0f}s)",
          flush=True)


if __name__ == '__main__':
    main()
