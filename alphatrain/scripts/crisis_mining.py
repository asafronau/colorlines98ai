"""Crisis Mining: find hard positions with policy, solve with deep search.

1. Play games with policy-only (0 sims) — dies fast, finds crisis positions
2. Recovery: rewind N turns before death, replay with deep search
3. Crisis prevention: rewind M turns before death, replay with deep search
4. Save high-quality training data for the hard positions

Usage:
    python -m alphatrain.scripts.crisis_mining \
        --model alphatrain/data/pillar2u_epoch_8.pt \
        --seed-start 100000 --seed-end 101000 \
        --recovery-turns 25 --recovery-sims 2000 \
        --prevention-turns 75 --prevention-sims 1600 \
        --continue-turns 1000 \
        --device mps --batch-size 64 \
        --save-dir data/crisis_v1
"""

import os
import time
import argparse
import json
import numpy as np
import torch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def play_policy_only(net, device, seed, fp16=False, max_turns=5000):
    """Play with greedy policy (0 sims). Returns list of snapshots."""
    game = ColorLinesGame(seed=seed)
    game.reset()

    snapshots = []
    turn = 0

    while not game.game_over and turn < max_turns:
        snapshots.append({
            'board': game.board.copy(),
            'next_balls': list(game.next_balls),
            'turn': turn,
            'score': game.score,
            'empty': int(np.sum(game.board == 0)),
        })

        obs_np = _build_obs_for_game(game)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        if fp16:
            obs = obs.half()
        with torch.inference_mode():
            pol_logits, _ = net(obs)
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol_np, 30)

        if not priors:
            break

        best_action = max(priors.items(), key=lambda x: x[1])[0]
        src_flat = best_action // 81
        tgt_flat = best_action % 81
        game.move(
            (src_flat // 9, src_flat % 9),
            (tgt_flat // 9, tgt_flat % 9))
        turn += 1

    return snapshots, game.score, turn


def replay_from_snapshot(mcts, snapshot, replay_seed, num_sims,
                         continue_turns, max_turns=5000):
    """Replay from a saved board position with MCTS search."""
    game = ColorLinesGame(seed=replay_seed)
    game.reset(
        board=snapshot['board'],
        next_balls=snapshot['next_balls'])
    game.score = snapshot['score']
    game.turns = snapshot['turn']

    mcts_replay = MCTS(
        net=mcts.net, device=mcts.device, max_score=mcts.max_score,
        num_simulations=num_sims, c_puct=mcts.c_puct,
        top_k=mcts.top_k, batch_size=mcts.batch_size,
        inference_client=mcts.inference_client)

    moves_data = []
    t0 = time.time()
    turn = 0
    capped = False
    turns_limit = min(continue_turns, max_turns - snapshot['turn'])

    while not game.game_over and turn < turns_limit:
        board_snapshot = game.board.copy().tolist()
        nb = game.next_balls
        next_balls_json = [
            {'row': int(pc[0][0]), 'col': int(pc[0][1]), 'color': int(pc[1])}
            for pc in nb
        ]

        result = mcts_replay.search(
            game, temperature=0.0,
            dirichlet_alpha=0.3, dirichlet_weight=0.25,
            return_policy=True)

        if result[0] is None:
            break

        action, policy_target = result

        top_indices = np.argsort(policy_target)[::-1][:5]
        top_moves = []
        top_scores = []
        for idx in top_indices:
            if policy_target[idx] <= 0:
                break
            flat = int(idx)
            top_moves.append({
                'sr': int(flat // 81 // 9), 'sc': int(flat // 81 % 9),
                'tr': int(flat % 81 // 9), 'tc': int(flat % 81 % 9),
            })
            top_scores.append(float(np.log(policy_target[idx] + 1e-8)))

        chosen_src, chosen_tgt = action
        moves_data.append({
            'board': board_snapshot,
            'next_balls': next_balls_json,
            'num_next': len(nb),
            'chosen_move': {
                'sr': int(chosen_src[0]), 'sc': int(chosen_src[1]),
                'tr': int(chosen_tgt[0]), 'tc': int(chosen_tgt[1]),
            },
            'top_moves': top_moves,
            'top_scores': top_scores,
        })

        move_result = game.move(action[0], action[1])
        if not move_result['valid']:
            break
        turn += 1

    if turn >= turns_limit and not game.game_over:
        capped = True

    bootstrap_value = 0.0
    if capped:
        _, bootstrap_value = mcts_replay._nn_evaluate_single(game)
        bootstrap_value = float(bootstrap_value)

    return {
        'seed': replay_seed,
        'score': game.score,
        'turns': snapshot['turn'] + turn,
        'replay_from_turn': snapshot['turn'],
        'replay_sims': num_sims,
        'moves': moves_data,
        'capped': capped,
        'bootstrap_value': bootstrap_value,
        'time': time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed-start', type=int, default=100000)
    p.add_argument('--seed-end', type=int, default=100100)
    p.add_argument('--recovery-turns', type=int, default=25,
                   help='Rewind N turns before death for recovery replay')
    p.add_argument('--recovery-sims', type=int, default=2000,
                   help='Sims for recovery replay')
    p.add_argument('--prevention-turns', type=int, default=75,
                   help='Rewind N turns before death for prevention replay')
    p.add_argument('--prevention-sims', type=int, default=1600,
                   help='Sims for prevention replay')
    p.add_argument('--continue-turns', type=int, default=1000,
                   help='Max turns to play from rewind point')
    p.add_argument('--max-turns', type=int, default=5000,
                   help='Max turns for policy-only probe game')
    p.add_argument('--device', default=None)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--save-dir', default='data/crisis_v1')
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Crisis Mining: seeds {args.seed_start}-{args.seed_end}", flush=True)
    print(f"Recovery: rw{args.recovery_turns} @ {args.recovery_sims} sims", flush=True)
    print(f"Prevention: rw{args.prevention_turns} @ {args.prevention_sims} sims", flush=True)
    print(f"Continue: {args.continue_turns} turns | Device: {device}", flush=True)

    net, max_score = load_model(args.model, device,
                                fp16=(str(device) != 'cpu'),
                                jit_trace=True)

    # Detect fp16
    fp16 = False
    try:
        fp16 = next(net.parameters()).dtype == torch.float16
    except (StopIteration, AttributeError):
        pass

    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=args.recovery_sims,
                batch_size=args.batch_size,
                top_k=30, c_puct=2.5)

    os.makedirs(args.save_dir, exist_ok=True)

    total_replays = 0
    total_states = 0
    skipped = 0
    t0 = time.time()
    n_seeds = args.seed_end - args.seed_start

    for si, seed in enumerate(range(args.seed_start, args.seed_end)):
        # Step 1: Fast policy-only game
        snapshots, pol_score, pol_turns = play_policy_only(
            net, device, seed, fp16=fp16, max_turns=args.max_turns)

        if pol_turns >= args.max_turns:
            skipped += 1
            if skipped % 50 == 0:
                print(f"  seed={seed}: policy SURVIVED (capped) — skipped "
                      f"({skipped} total)", flush=True)
            continue

        print(f"  seed={seed}: policy died turn {pol_turns} "
              f"(score={pol_score}, empty={snapshots[-1]['empty']})",
              end='', flush=True)

        # Step 2: Recovery replay (close to death)
        replays = [
            ('recovery', args.recovery_turns, args.recovery_sims),
            ('prevention', args.prevention_turns, args.prevention_sims),
        ]

        for label, rewind, sims in replays:
            rewind_idx = max(0, pol_turns - rewind)
            if rewind_idx >= len(snapshots):
                continue

            snapshot = snapshots[rewind_idx]
            replay_seed = seed * 37 + rewind  # unique per rewind point

            # Skip if already exists
            pattern = f"game_seed{seed}_{label}_score"
            existing = [f for f in os.listdir(args.save_dir) if f.startswith(pattern)]
            if existing:
                continue

            result = replay_from_snapshot(
                mcts, snapshot, replay_seed, sims,
                args.continue_turns, args.max_turns)

            fname = f"game_seed{seed}_{label}_score{result['score']}.json"
            path = os.path.join(args.save_dir, fname)
            with open(path, 'w') as f:
                json.dump(result, f)

            total_replays += 1
            total_states += len(result['moves'])

            cap = " [CAP]" if result.get('capped') else ""
            survived = result['turns'] - snapshot['turn']
            print(f" | {label}: {result['score']}pts "
                  f"({survived}t/{args.continue_turns}, "
                  f"{result['time']:.0f}s){cap}", end='', flush=True)

        print(flush=True)

        if (si + 1) % 20 == 0:
            elapsed = time.time() - t0
            done = si + 1
            eta = elapsed / done * (n_seeds - done)
            print(f"  --- [{done}/{n_seeds}] {total_replays} replays, "
                  f"{total_states:,} states, {skipped} skipped, "
                  f"ETA {eta/60:.0f}m ---", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {total_replays} replays, {total_states:,} states, "
          f"{skipped} skipped in {elapsed/60:.0f}m", flush=True)


if __name__ == '__main__':
    main()
