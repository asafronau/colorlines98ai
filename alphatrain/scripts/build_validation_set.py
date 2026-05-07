"""Build a K-rollout survival validation set for ValueHead calibration.

For ~N crisis-adjacent positions sampled from the V11 corpus, run
K=64-128 rollouts with the trained policy network and a different
spawn-RNG seed per rollout. Per state per horizon H, record
    P_H_hat = (count of rollouts that survive ≥ H more turns) / K

This is the gold-standard validation target: a calibrated probability
estimate, robust to single-trajectory RNG noise. ChatGPT review point:
single-trajectory `survive_H = 1` is one Bernoulli sample, not the
true probability — for tactical positions where the value head matters
most, repeated rollouts give clean labels.

Output (.pt): boards / next_pos / next_col / n_next + p_hat (N, 4)
+ rollout_K + horizons.

Usage:
    python -m alphatrain.scripts.build_validation_set \\
        --model alphatrain/data/pillar2y2_epoch_40.pt \\
        --games-dir data/selfplay_v11_s600 data/crisis_v11 \\
        --num-states 500 --rollouts-per-state 64 \\
        --output alphatrain/data/value_val_K64.pt
"""

import os
import json
import glob
import time
import random
import argparse

import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from alphatrain.mcts import _get_legal_priors_flat
from alphatrain.value_head import SURVIVAL_HORIZONS

BOARD_SIZE = 9


def mine_crisis_states(files, num_states, rng_seed=0):
    """Sample positions where the value head's signal matters most.

    Strategy: walk a random sample of games, keep positions that look
    crisis-adjacent — late-game positions, low empty-cell counts,
    crisis-replay positions. Avoid early-game positions where almost
    any move leads to long survival.
    """
    rng = random.Random(rng_seed)
    rng.shuffle(files)

    candidates = []
    for fpath in files:
        if len(candidates) >= num_states * 4:
            break
        try:
            game = json.load(open(fpath))
        except Exception:
            continue
        moves = game['moves']
        n = len(moves)
        if n < 30:
            continue
        # Sample 1-3 positions per game from the late half — that's
        # where survival becomes uncertain.
        n_pick = rng.randint(1, 3)
        for _ in range(n_pick):
            t = rng.randint(n // 2, n - 5)
            mv = moves[t]
            board = np.array(mv['board'], dtype=np.int8)
            empty = int((board == 0).sum())
            if empty < 18 or empty > 60:
                # Skip nearly-full or nearly-empty boards (rare extremes)
                continue
            nbs = mv.get('next_balls') or []
            candidates.append({
                'board': board,
                'next_balls': nbs[:3],
                'turn_in_game': t,
                'game_length': n,
                'source_file': os.path.basename(fpath),
            })

    rng.shuffle(candidates)
    return candidates[:num_states]


def _build_obs_for_game(game):
    """Build 18-channel observation tensor from a game state."""
    nb = game.next_balls
    nn = min(len(nb), 3)
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    for i in range(nn):
        pos, col = nb[i]
        nr[i] = pos[0]; nc[i] = pos[1]; ncol[i] = col
    return build_observation(game.board, nr, nc, ncol, nn)


def rollout_batch(net, device, fp16, state, K, max_horizon, base_rng_seed):
    """Run K rollouts in parallel from `state` for up to max_horizon turns.

    Returns:
        survived: (K, len(SURVIVAL_HORIZONS)) bool array — True if
            rollout k survived ≥ H more turns.
    """
    # Initialize K games with different RNG seeds
    games = []
    for k in range(K):
        g = ColorLinesGame(seed=base_rng_seed + k)
        # Reset to the target state. next_balls must be a list of
        # (pos_tuple, color) pairs, matching ColorLinesGame's format.
        nb_list = [((nb['row'], nb['col']), nb['color'])
                   for nb in state['next_balls']]
        g.reset(board=state['board'].copy(), next_balls=nb_list)
        games.append(g)

    alive = np.ones(K, dtype=bool)
    survived = np.zeros((K, len(SURVIVAL_HORIZONS)), dtype=bool)

    # Pre-allocate
    obs_batch_np = np.empty((K, 18, 9, 9), dtype=np.float32)

    horizons_set = set(SURVIVAL_HORIZONS)
    net_dtype = torch.float16 if fp16 else torch.float32

    for turn in range(max_horizon + 1):
        # Record per-horizon survival as we cross each milestone.
        # `turn` here = number of completed moves. So at turn=H, the
        # rollout has played H moves; if alive then it survived ≥ H.
        if turn in horizons_set:
            hi = SURVIVAL_HORIZONS.index(turn)
            survived[:, hi] = alive

        if turn == max_horizon:
            break  # no need to play another move
        if not alive.any():
            break

        # Build obs batch for alive games
        active_idxs = np.where(alive)[0]
        for j, k in enumerate(active_idxs):
            obs_batch_np[j] = _build_obs_for_game(games[k])
        n_active = len(active_idxs)

        # Forward
        obs_t = torch.from_numpy(
            obs_batch_np[:n_active]).to(device=device, dtype=net_dtype)
        with torch.inference_mode():
            pol_logits = net(obs_t)
        pol_np = pol_logits.float().cpu().numpy()

        # Apply moves (sequentially on CPU, cheap)
        for j, k in enumerate(active_idxs):
            g = games[k]
            priors = _get_legal_priors_flat(g.board, pol_np[j], top_k=30)
            if not priors:
                alive[k] = False
                continue
            best_action = max(priors.items(), key=lambda x: x[1])[0]
            sf = best_action // 81
            tf = best_action % 81
            r = g.move((sf // 9, sf % 9), (tf // 9, tf % 9))
            if not r['valid'] or g.game_over:
                alive[k] = False

    return survived


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--games-dir', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--num-states', type=int, default=500,
                   help='Number of crisis-adjacent states to sample')
    p.add_argument('--rollouts-per-state', type=int, default=64,
                   help='K — how many rollouts per state for label averaging')
    p.add_argument('--max-horizon', type=int, default=None,
                   help='Cap rollout length. Defaults to max(SURVIVAL_HORIZONS).')
    p.add_argument('--device', default=None,
                   help='mps / cuda / cpu. Auto-detect if not set.')
    p.add_argument('--mine-seed', type=int, default=0)
    args = p.parse_args()

    if args.max_horizon is None:
        args.max_horizon = max(SURVIVAL_HORIZONS)

    if args.device:
        device_str = args.device
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)

    # Collect game files
    files = []
    for d in args.games_dir:
        files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
    print(f"Mining {args.num_states} crisis-adjacent states from "
          f"{len(files)} games...", flush=True)
    states = mine_crisis_states(files, args.num_states, args.mine_seed)
    print(f"Mined {len(states)} states.", flush=True)

    # Load policy
    net, _ = load_model(args.model, device,
                        fp16=(device_str != 'cpu'),
                        jit_trace=True)
    fp16 = (device_str != 'cpu')

    # Run rollouts
    print(f"\nRunning {args.rollouts_per_state} rollouts per state, "
          f"max_horizon={args.max_horizon}, device={device_str}, fp16={fp16}",
          flush=True)
    p_hats = np.zeros((len(states), len(SURVIVAL_HORIZONS)),
                      dtype=np.float32)
    t0 = time.time()
    for si, state in enumerate(states):
        survived = rollout_batch(
            net, device, fp16, state,
            K=args.rollouts_per_state,
            max_horizon=args.max_horizon,
            base_rng_seed=10_000 * si + 1)
        p_hats[si] = survived.mean(axis=0)
        if (si + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (si + 1) * (len(states) - si - 1)
            print(f"  [{si+1}/{len(states)}] mean P25={p_hats[:si+1, 0].mean():.2f} "
                  f"P200={p_hats[:si+1, 3].mean():.2f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Stack the state tensors
    boards_l = [s['board'] for s in states]
    next_pos_l, next_col_l, n_next_l = [], [], []
    for s in states:
        npos = np.zeros((3, 2), dtype=np.int8)
        ncol = np.zeros(3, dtype=np.int8)
        nbs = s['next_balls']
        nn = min(len(nbs), 3)
        for i in range(nn):
            npos[i, 0] = nbs[i]['row']
            npos[i, 1] = nbs[i]['col']
            ncol[i] = nbs[i]['color']
        next_pos_l.append(npos)
        next_col_l.append(ncol)
        n_next_l.append(nn)

    data = {
        'boards': torch.tensor(np.stack(boards_l), dtype=torch.int8),
        'next_pos': torch.tensor(np.stack(next_pos_l), dtype=torch.int8),
        'next_col': torch.tensor(np.stack(next_col_l), dtype=torch.int8),
        'n_next': torch.tensor(n_next_l, dtype=torch.int8),
        'p_hat': torch.tensor(p_hats, dtype=torch.float32),
        'rollout_K': args.rollouts_per_state,
        'horizons': list(SURVIVAL_HORIZONS),
        'turn_in_game': torch.tensor(
            [s['turn_in_game'] for s in states], dtype=torch.int32),
        'game_length': torch.tensor(
            [s['game_length'] for s in states], dtype=torch.int32),
    }

    print(f"\nP_hat distribution per horizon:")
    for hi, h in enumerate(SURVIVAL_HORIZONS):
        ph = p_hats[:, hi]
        print(f"  H={h:>4}: mean={ph.mean():.3f} "
              f"P10={np.percentile(ph, 10):.2f} "
              f"P50={np.percentile(ph, 50):.2f} "
              f"P90={np.percentile(ph, 90):.2f} "
              f"  certain_dies={(ph<0.05).sum()} "
              f"certain_lives={(ph>0.95).sum()}", flush=True)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    print(f"\nSaved to {args.output} "
          f"({os.path.getsize(args.output)/1e6:.1f} MB)", flush=True)


if __name__ == '__main__':
    main()
