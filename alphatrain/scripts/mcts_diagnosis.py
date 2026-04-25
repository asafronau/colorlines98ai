"""Diagnose why MCTS overrides the policy.

Plays a game with MCTS. After each search, compares the policy's top move
vs MCTS's chosen move. When they disagree, logs WHY: the Q-values, visit
counts, U terms, and priors for both moves.

Also measures prior peakedness (P_max) across the game.

Usage:
    python -m alphatrain.scripts.mcts_diagnosis \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --seed 36 --sims 400 --device mps
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import argparse
import math
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=36)
    p.add_argument('--sims', type=int, default=400)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--zero-value', action='store_true',
                   help='Force value head to return 0.0 (amputation test)')
    p.add_argument('--heuristic-value', action='store_true',
                   help='Use board heuristic instead of neural value head')
    p.add_argument('--value-net', default=None,
                   help='Separate value network checkpoint')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=True, jit_trace=True)

    if args.zero_value:
        original_predict = net.predict_value
        net.predict_value = lambda val_logits, max_val=None: torch.zeros(
            val_logits.shape[0], device=val_logits.device)
        print("*** VALUE HEAD AMPUTATED (returning 0.0) ***", flush=True)

    if args.heuristic_value:
        print("*** HEURISTIC VALUE (board eval) ***", flush=True)

    # Load separate value network
    vnet = None
    if args.value_net:
        from alphatrain.model import ValueNet
        ckpt = torch.load(args.value_net, map_location='cpu', weights_only=False)
        vnet = ValueNet(in_channels=18,
                        num_blocks=ckpt['num_blocks'],
                        channels=ckpt['channels'],
                        num_value_bins=1)
        vnet.load_state_dict(ckpt['model'])
        vnet = vnet.to(device)
        vnet.requires_grad_(False)
        print(f"*** VALUE NET: {ckpt['num_blocks']}b x {ckpt['channels']}ch, "
              f"epoch={ckpt['epoch']}, acc={ckpt['accuracy']:.1f}% ***", flush=True)

    print(f"Model: max_score={max_score}, device={args.device}, "
          f"sims={args.sims}, c_puct={args.c_puct}", flush=True)

    mcts = MCTS(net, device, max_score=max_score, num_simulations=args.sims,
                batch_size=8, top_k=30, c_puct=args.c_puct,
                heuristic_value=args.heuristic_value,
                value_net=vnet)

    game = ColorLinesGame(seed=args.seed)
    game.reset()

    t0 = time.time()
    turn = 0
    disagreements = 0
    p_maxes = []
    value_preds = []

    while not game.game_over and turn < args.max_turns:
        # Get raw policy prediction
        obs_np = _build_obs_for_game(game)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        fp16 = next(net.parameters()).dtype == torch.float16
        if fp16:
            obs = obs.half()
        with torch.inference_mode():
            pol_logits, val_logits = net(obs)
            value_pred = net.predict_value(val_logits, max_val=max_score).item()
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol_np, 30)

        value_preds.append(value_pred)

        if not priors:
            break

        # Policy's top move and P_max
        policy_top = max(priors.items(), key=lambda x: x[1])
        policy_action = policy_top[0]
        p_max = policy_top[1]
        p_maxes.append(p_max)

        # Run MCTS
        result = mcts.search(game, temperature=0.0,
                             dirichlet_alpha=0.0, dirichlet_weight=0.0,
                             return_policy=True)
        if result[0] is None:
            break

        action, visit_dist = result
        mcts_action_flat = (action[0][0] * 9 + action[0][1]) * 81 + \
                           (action[1][0] * 9 + action[1][1])

        # Did MCTS disagree with policy?
        if mcts_action_flat != policy_action:
            disagreements += 1
            # Get root children stats
            root = None
            # Access root from the MCTS internals - reconstruct from visit_dist
            # Find MCTS top move visits
            mcts_top_idx = np.argmax(visit_dist)
            mcts_visits = visit_dist[mcts_top_idx]
            policy_visits = visit_dist[policy_action] if policy_action < len(visit_dist) else 0

            if turn < 20 or (turn % 100 == 0) or disagreements <= 10:
                sa = policy_action // 81
                ta = policy_action % 81
                sm = mcts_top_idx // 81
                tm = mcts_top_idx % 81
                print(f"  turn={turn} DISAGREE | "
                      f"policy=({sa//9},{sa%9})->({ta//9},{ta%9}) P={p_max:.3f} "
                      f"visits={policy_visits:.3f} | "
                      f"mcts=({sm//9},{sm%9})->({tm//9},{tm%9}) "
                      f"visits={mcts_visits:.3f} | "
                      f"val={value_pred:.1f} empty={np.sum(game.board==0)}",
                      flush=True)

        game.move(action[0], action[1])
        turn += 1

        if turn % 500 == 0:
            elapsed = time.time() - t0
            p_arr = np.array(p_maxes[-500:])
            v_arr = np.array(value_preds[-500:])
            print(f"  turn={turn} score={game.score} | "
                  f"P_max: mean={p_arr.mean():.3f} med={np.median(p_arr):.3f} "
                  f"max={p_arr.max():.3f} | "
                  f"value: mean={v_arr.mean():.1f} std={v_arr.std():.1f} | "
                  f"disagree={disagreements}/{turn} ({100*disagreements/turn:.0f}%) | "
                  f"{elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    p_arr = np.array(p_maxes)
    v_arr = np.array(value_preds)

    print(f"\n=== DIAGNOSIS: seed={args.seed} score={game.score} "
          f"turns={turn} ({elapsed:.0f}s) ===", flush=True)
    print(f"Prior peakedness (P_max):", flush=True)
    print(f"  mean={p_arr.mean():.3f} median={np.median(p_arr):.3f} "
          f"std={p_arr.std():.3f}", flush=True)
    print(f"  P_max > 0.5: {(p_arr > 0.5).sum()}/{len(p_arr)} "
          f"({100*(p_arr > 0.5).mean():.0f}%)", flush=True)
    print(f"  P_max > 0.7: {(p_arr > 0.7).sum()}/{len(p_arr)} "
          f"({100*(p_arr > 0.7).mean():.0f}%)", flush=True)
    print(f"  P_max > 0.9: {(p_arr > 0.9).sum()}/{len(p_arr)} "
          f"({100*(p_arr > 0.9).mean():.0f}%)", flush=True)
    print(f"Value head predictions:", flush=True)
    print(f"  mean={v_arr.mean():.1f} std={v_arr.std():.1f} "
          f"min={v_arr.min():.1f} max={v_arr.max():.1f}", flush=True)
    print(f"Disagreements: {disagreements}/{turn} "
          f"({100*disagreements/turn:.0f}%)", flush=True)


if __name__ == '__main__':
    main()
