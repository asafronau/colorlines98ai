"""Policy-guided stochastic root search.

For each move: generate top-K candidates from policy, then for each
candidate, roll forward with greedy policy for H turns across N
common RNG seeds. Choose by risk-aware score (mean - lambda*std).

No value head needed. Directly measures survival under real ball spawns.

Usage:
    python -m alphatrain.scripts.stochastic_search \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --seed 14 --device mps --top-k 8 --horizon 50 --samples 32
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import argparse
import numpy as np
import torch

from game.board import ColorLinesGame, _label_empty_components
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def policy_greedy_move(net, device, game, fp16=False):
    obs_np = _build_obs_for_game(game)
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    if fp16:
        obs = obs.half()
    with torch.inference_mode():
        pol_logits, _ = net(obs)
    pol_np = pol_logits[0].float().cpu().numpy()
    priors = _get_legal_priors_flat(game.board, pol_np, 30)
    if not priors:
        return None
    best = max(priors.items(), key=lambda x: x[1])[0]
    return (best // 81 // 9, best // 81 % 9, best % 81 // 9, best % 81 % 9)


def rollout_from(game, net, device, horizon, fp16=False):
    survived = 0
    while not game.game_over and survived < horizon:
        move = policy_greedy_move(net, device, game, fp16)
        if move is None:
            break
        result = game.move((move[0], move[1]), (move[2], move[3]))
        if not result['valid']:
            break
        survived += 1
    empty = int(np.sum(game.board == 0))
    return survived, empty, game.game_over


def search_move(net, device, game, top_k=8, horizon=50, samples=32,
                fp16=False, verbose=False):
    obs_np = _build_obs_for_game(game)
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    if fp16:
        obs = obs.half()
    with torch.inference_mode():
        pol_logits, _ = net(obs)
    pol_np = pol_logits[0].float().cpu().numpy()
    priors = _get_legal_priors_flat(game.board, pol_np, 30)
    if not priors:
        return None, []

    sorted_moves = sorted(priors.items(), key=lambda x: -x[1])[:top_k]
    policy_top = sorted_moves[0][0]

    # Common RNG seeds for all candidates
    base_seed = hash(game.board.tobytes()) & 0xFFFFFFFF
    rng_seeds = [base_seed + i * 7919 for i in range(samples)]

    candidates = []
    for flat, prior in sorted_moves:
        sr, sc = flat // 81 // 9, flat // 81 % 9
        tr, tc = flat % 81 // 9, flat % 81 % 9

        survivals = []
        empties = []
        deaths = 0

        for rng_seed in rng_seeds:
            from game.rng import SimpleRng
            g = game.clone(rng=SimpleRng(rng_seed))
            result = g.move((sr, sc), (tr, tc))
            if not result['valid']:
                deaths += 1
                survivals.append(0)
                empties.append(0)
                continue

            surv, empty, died = rollout_from(
                g, net, device, horizon, fp16)
            survivals.append(surv)
            empties.append(empty)
            if died:
                deaths += 1

        surv_arr = np.array(survivals)
        empty_arr = np.array(empties)

        candidates.append({
            'flat': flat,
            'move': (sr, sc, tr, tc),
            'prior': prior,
            'surv_mean': surv_arr.mean(),
            'surv_p10': np.percentile(surv_arr, 10),
            'surv_std': surv_arr.std(),
            'empty_mean': empty_arr.mean(),
            'death_rate': deaths / samples,
            'risk_score': surv_arr.mean() - 0.5 * surv_arr.std(),
        })

    candidates.sort(key=lambda x: -x['risk_score'])

    if verbose and candidates:
        print(f"  {'Move':<18} {'Prior':>6} {'Surv':>5} {'P10':>4} "
              f"{'Std':>5} {'Empty':>5} {'Death':>5} {'Risk':>6}",
              flush=True)
        for c in candidates:
            m = c['move']
            marker = ""
            if c['flat'] == policy_top:
                marker = " <POL"
            if c is candidates[0]:
                marker += " <BEST"
            print(f"  ({m[0]},{m[1]})->({m[2]},{m[3]})"
                  f"  {c['prior']:>6.3f} {c['surv_mean']:>5.1f} "
                  f"{c['surv_p10']:>4.0f} {c['surv_std']:>5.1f} "
                  f"{c['empty_mean']:>5.1f} {c['death_rate']:>5.1%} "
                  f"{c['risk_score']:>6.1f}{marker}", flush=True)

    best = candidates[0]
    return (best['move'][0], best['move'][1],
            best['move'][2], best['move'][3]), candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--horizon', type=int, default=50)
    p.add_argument('--samples', type=int, default=32)
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--verbose-every', type=int, default=200,
                   help='Log full candidate table every N turns')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(
        args.model, device, fp16=True, jit_trace=False)
    net.eval()

    game = ColorLinesGame(seed=args.seed)
    game.reset()

    print(f"Stochastic search: seed={args.seed} top_k={args.top_k} "
          f"horizon={args.horizon} samples={args.samples}", flush=True)
    print(f"{'='*70}", flush=True)

    t0 = time.time()
    turn = 0
    while not game.game_over and turn < args.max_turns:
        verbose = (turn % args.verbose_every == 0)
        if verbose:
            empty = int(np.sum(game.board == 0))
            labels = _label_empty_components(game.board)
            n_comp = int(labels.max())
            elapsed = time.time() - t0
            print(f"\n  turn={turn} score={game.score} empty={empty} "
                  f"comp={n_comp} ({elapsed:.0f}s)", flush=True)

        move, candidates = search_move(
            net, device, game,
            top_k=args.top_k, horizon=args.horizon,
            samples=args.samples, fp16=True, verbose=verbose)

        if move is None:
            break

        result = game.move((move[0], move[1]), (move[2], move[3]))
        if not result['valid']:
            break
        turn += 1

    elapsed = time.time() - t0
    print(f"\n{'='*70}", flush=True)
    print(f"Seed {args.seed}: score={game.score} turns={turn} "
          f"({elapsed:.0f}s, {elapsed/max(turn,1):.1f}s/turn)", flush=True)


if __name__ == '__main__':
    main()
