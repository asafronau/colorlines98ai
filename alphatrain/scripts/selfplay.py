"""Self-play data generation for AlphaZero training.

Each game records (observation, MCTS visit policy, TD value target) per move.
Games are saved as individual .pt files for easy distribution across machines.

Usage:
    # Local (M5 Max, MPS):
    python -m alphatrain.scripts.selfplay \
        --model alphatrain/data/alphatrain_td_best.pt \
        --seed-start 0 --seed-end 100 --sims 800

    # CPU machine (GCP, many cores):
    python -m alphatrain.scripts.selfplay \
        --model alphatrain/data/alphatrain_td_best.pt \
        --seed-start 500 --seed-end 1000 --sims 800 --device cpu --workers 88
"""

# Force single-threaded BLAS/OpenMP BEFORE importing numpy/torch.
# Critical for CPU multiprocessing: without this, each worker spawns
# hidden BLAS threads causing contention that destroys MCTS quality.
# On Linux (fork), env vars must be set before library initialization.
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import time
import argparse
import numpy as np
import torch

import json

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game
from alphatrain.evaluate import load_model


def save_game_json(result, save_dir):
    """Save game result as JSON compatible with build_expert_v2_tensor.py."""
    seed = result['seed']
    score = result['score']
    path = os.path.join(save_dir, f'game_seed{seed}_score{score}.json')
    data = {
        'seed': seed,
        'score': score,
        'moves': result['moves'],
    }
    if result.get('capped', False):
        data['capped'] = True
        data['bootstrap_value'] = result['bootstrap_value']
    with open(path, 'w') as f:
        json.dump(data, f)
    return path


def play_selfplay_game(mcts, seed, temperature_moves=15,
                       dirichlet_alpha=0.3, dirichlet_weight=0.25,
                       top_k_save=5, max_turns=0):
    """Play one self-play game, recording raw board data for JSON output.

    Saves in the same format as Rust expert games so build_expert_v2_tensor.py
    can process self-play and expert data identically.

    Args:
        mcts: MCTS instance
        seed: game seed
        temperature_moves: use temperature=1 for first N moves
        dirichlet_alpha: Dirichlet noise parameter
        dirichlet_weight: weight for noise at root
        top_k_save: number of top moves to save from visit counts
        max_turns: cap game at this many turns (0=no cap).
            Capped games get bootstrap_value instead of death.

    Returns:
        dict with 'seed', 'score', 'moves', 'capped', 'bootstrap_value'
    """
    game = ColorLinesGame(seed=seed)
    game.reset()

    moves_data = []
    t0 = time.time()
    turn = 0
    capped = False

    # Dynamic sims tracking
    ds_high = 0   # P_max > 0.9
    ds_mid = 0    # 0.7 < P_max <= 0.9
    ds_low = 0    # P_max <= 0.7
    ds_total_sims = 0
    ds_last_report = 0  # turn at last report

    while not game.game_over:
        if max_turns > 0 and turn >= max_turns:
            capped = True
            break
        temp = 1.0 if turn < temperature_moves else 0.0

        # Record board state BEFORE move
        board_snapshot = game.board.copy().tolist()
        nb = game.next_balls
        next_balls = []
        for pos_col in nb:
            next_balls.append({
                'row': int(pos_col[0][0]),
                'col': int(pos_col[0][1]),
                'color': int(pos_col[1]),
            })

        # MCTS search — get action + full visit count distribution
        result = mcts.search(
            game,
            temperature=temp,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_weight=dirichlet_weight,
            return_policy=True)

        if result[0] is None:
            break

        action, policy_target = result

        # Track dynamic sims stats
        if hasattr(mcts, '_last_max_prior'):
            mp = mcts._last_max_prior
            if mp > 0.5:
                ds_high += 1
            elif mp > 0.3:
                ds_mid += 1
            else:
                ds_low += 1
            ds_total_sims += mcts._last_effective_sims

        # Extract top-K moves from visit count distribution
        top_indices = np.argsort(policy_target)[::-1][:top_k_save]
        top_moves = []
        top_scores = []
        for idx in top_indices:
            if policy_target[idx] <= 0:
                break
            flat = int(idx)
            src_flat = flat // 81
            tgt_flat = flat % 81
            top_moves.append({
                'sr': int(src_flat // 9), 'sc': int(src_flat % 9),
                'tr': int(tgt_flat // 9), 'tc': int(tgt_flat % 9),
            })
            # Use log(visit_fraction + eps) as score — softmax in build script
            # recovers the original distribution
            top_scores.append(float(np.log(policy_target[idx] + 1e-8)))

        # The chosen move
        chosen_src, chosen_tgt = action
        chosen_move = {
            'sr': int(chosen_src[0]), 'sc': int(chosen_src[1]),
            'tr': int(chosen_tgt[0]), 'tc': int(chosen_tgt[1]),
        }

        moves_data.append({
            'board': board_snapshot,
            'next_balls': next_balls,
            'num_next': len(nb),
            'chosen_move': chosen_move,
            'top_moves': top_moves,
            'top_scores': top_scores,
        })

        # Execute move
        move_result = game.move(action[0], action[1])
        if not move_result['valid']:
            break

        turn += 1
        if turn % 500 == 0:
            elapsed = time.time() - t0
            ds_total = ds_high + ds_mid + ds_low
            if ds_total > 0:
                avg_sims = ds_total_sims / ds_total
                # Stats since last report
                ds_since = ds_total - ds_last_report
                print(f"    seed={seed} turn={turn} score={game.score} "
                      f"{elapsed:.0f}s | "
                      f"P>.5:{100*ds_high/ds_total:.0f}% "
                      f".3-.5:{100*ds_mid/ds_total:.0f}% "
                      f"<.3:{100*ds_low/ds_total:.0f}% "
                      f"avg_sims={avg_sims:.0f}", flush=True)
                ds_last_report = ds_total
            else:
                print(f"    seed={seed} turn={turn} score={game.score} "
                      f"{elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0

    # Bootstrap value for capped games: use the model's own value prediction
    # of the final board instead of 0 (death). This teaches the model that
    # turn 5000 boards are "still alive" with varying health levels.
    bootstrap_value = 0.0
    if capped:
        _, bootstrap_value = mcts._nn_evaluate_single(game)
        bootstrap_value = float(bootstrap_value)

    # Dynamic sims summary
    ds_total = ds_high + ds_mid + ds_low
    ds_stats = None
    if ds_total > 0:
        ds_stats = {
            'high_pct': 100 * ds_high / ds_total,
            'mid_pct': 100 * ds_mid / ds_total,
            'low_pct': 100 * ds_low / ds_total,
            'avg_sims': ds_total_sims / ds_total,
        }

    return {
        'seed': seed,
        'score': game.score,
        'turns': turn,
        'moves': moves_data,
        'capped': capped,
        'bootstrap_value': bootstrap_value,
        'time': elapsed,
        'dynamic_sims_stats': ds_stats,
    }


def _limit_threads():
    """Force single-threaded torch (env vars already set at module top)."""
    torch.set_num_threads(1)


def _server_worker(slot_id, seed_queue, result_queue,
                   obs_shm_name, pol_shm_name, val_shm_name,
                   num_workers, max_batch,
                   request_queue, response_queue,
                   num_sims, batch_size, max_score,
                   temperature_moves, dirichlet_alpha, dirichlet_weight,
                   max_turns, dynamic_sims=False):
    """Persistent worker for GPU server mode self-play."""
    torch.set_num_threads(1)

    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)

    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf,
                             request_queue, response_queue)
    mcts = MCTS(inference_client=client, max_score=max_score,
                num_simulations=num_sims, batch_size=batch_size,
                top_k=30, c_puct=2.5, dynamic_sims=dynamic_sims)

    while True:
        seed = seed_queue.get()
        if seed is None:
            break

        result = play_selfplay_game(
            mcts, seed,
            temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_weight=dirichlet_weight,
            max_turns=max_turns)

        cap_str = " [CAPPED]" if result.get('capped') else ""
        ds = result.get('dynamic_sims_stats')
        ds_str = (f" | P>.9:{ds['high_pct']:.0f}% .7-.9:{ds['mid_pct']:.0f}% "
                  f"<.7:{ds['low_pct']:.0f}% avg={ds['avg_sims']:.0f}sims"
                  if ds else "")
        print(f"  [w{slot_id}] seed={seed}: score={result['score']}, "
              f"turns={result['turns']}{cap_str}, {result['time']:.0f}s{ds_str}",
              flush=True)

        result_queue.put(result)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def _worker_play(args):
    """Worker function for CPU multiprocessing."""
    seed, model_path, device_str, num_sims, batch_size, \
        temperature_moves, dirichlet_alpha, dirichlet_weight, max_turns, \
        dynamic_sims = args

    _limit_threads()
    device = torch.device(device_str)

    net, max_score = load_model(model_path, device,
                                fp16=(device_str != 'cpu'),
                                jit_trace=True)

    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=num_sims, batch_size=batch_size,
                top_k=30, c_puct=2.5, dynamic_sims=dynamic_sims)

    result = play_selfplay_game(
        mcts, seed,
        temperature_moves=temperature_moves,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
        max_turns=max_turns)

    ds = result.get('dynamic_sims_stats')
    ds_str = (f" | P>.5:{ds['high_pct']:.0f}% .3-.5:{ds['mid_pct']:.0f}% "
              f"<.3:{ds['low_pct']:.0f}% avg={ds['avg_sims']:.0f}sims"
              if ds else "")
    print(f"  seed={seed}: score={result['score']}, "
          f"turns={result['turns']}, {result['time']:.0f}s{ds_str}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--seed-end', type=int, default=100)
    p.add_argument('--sims', type=int, default=800)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default=None,
                   help='Force device (mps/cuda/cpu). Auto-detect if not set.')
    p.add_argument('--workers', type=int, default=1,
                   help='Parallel workers (1=local MPS, >1=CPU multiprocessing)')
    p.add_argument('--value-model', default=None,
                   help='Separate ValueNet checkpoint (if None, use value head from --model)')
    p.add_argument('--deterministic', action='store_true',
                   help='Per-request GPU processing (exact scores, slower)')
    p.add_argument('--save-dir', default='data/selfplay')
    p.add_argument('--temperature-moves', type=int, default=15)
    p.add_argument('--dirichlet-alpha', type=float, default=0.3)
    p.add_argument('--dirichlet-weight', type=float, default=0.25)
    p.add_argument('--max-turns', type=int, default=0,
                   help='Cap games at this many turns (0=no cap). '
                        'Capped games use bootstrap value instead of death.')
    p.add_argument('--dynamic-sims', action='store_true',
                   help='Reduce sims for confident moves (P_max>0.9: 50 sims, '
                        'P_max>0.7: sims/4). Saves ~2-3x compute.')
    args = p.parse_args()

    if args.device:
        device_str = args.device
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'

    seeds = list(range(args.seed_start, args.seed_end))
    os.makedirs(args.save_dir, exist_ok=True)

    # Resume: skip seeds with existing game files
    completed = set()
    import re
    for f in os.listdir(args.save_dir):
        m = re.match(r'game_seed(\d+)_score\d+\.json', f)
        if m:
            completed.add(int(m.group(1)))
    if completed:
        before = len(seeds)
        seeds = [s for s in seeds if s not in completed]
        print(f"Resume: {len(completed)} games found, "
              f"skipping {before - len(seeds)} seeds", flush=True)
    n_games = len(seeds)

    if n_games == 0:
        print("All games already completed!", flush=True)
        return

    print(f"Self-play: {n_games} games (seeds {args.seed_start}-{args.seed_end-1})",
          flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Device: {device_str}, workers: {args.workers}, "
          f"sims: {args.sims}, bs: {args.batch_size}", flush=True)
    print(f"Temperature: first {args.temperature_moves} moves, "
          f"Dirichlet: alpha={args.dirichlet_alpha} weight={args.dirichlet_weight}",
          flush=True)
    print(f"Save: {args.save_dir}/", flush=True)

    t0 = time.time()
    total_states = 0
    total_score = 0

    if args.workers <= 1:
        # Local mode: single process
        if device_str == 'cpu':
            _limit_threads()
        if args.value_model:
            from alphatrain.evaluate import load_dual_model
            net, max_score = load_dual_model(
                args.model, args.value_model, torch.device(device_str),
                fp16=(device_str != 'cpu'), jit_trace=True)
        else:
            net, max_score = load_model(args.model, torch.device(device_str),
                                        fp16=(device_str != 'cpu'),
                                        jit_trace=True)
        mcts = MCTS(net, torch.device(device_str), max_score=max_score,
                     num_simulations=args.sims, batch_size=args.batch_size,
                     top_k=30, c_puct=2.5, dynamic_sims=args.dynamic_sims)

        for i, seed in enumerate(seeds):
            result = play_selfplay_game(
                mcts, seed,
                temperature_moves=args.temperature_moves,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_weight=args.dirichlet_weight,
                max_turns=args.max_turns)

            save_game_json(result, args.save_dir)

            total_states += result['turns']
            total_score += result['score']
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_games - i - 1)

            ds = result.get('dynamic_sims_stats')
            ds_str = (f" | P>.9:{ds['high_pct']:.0f}% .7-.9:{ds['mid_pct']:.0f}% "
                      f"<.7:{ds['low_pct']:.0f}% avg={ds['avg_sims']:.0f}sims"
                      if ds else "")
            print(f"  [{i+1}/{n_games}] seed={seed}: score={result['score']}, "
                  f"turns={result['turns']}, {result['time']:.0f}s "
                  f"(ETA {eta/60:.0f}m){ds_str}", flush=True)
    elif device_str == 'cpu':
        # CPU multiprocessing — env vars already set at module top
        _limit_threads()
        from multiprocessing import Pool
        worker_args = [
            (seed, args.model, 'cpu', args.sims, args.batch_size,
             args.temperature_moves, args.dirichlet_alpha,
             args.dirichlet_weight, args.max_turns, args.dynamic_sims)
            for seed in seeds
        ]

        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(
                    _worker_play, worker_args)):
                save_game_json(result, args.save_dir)

                total_states += result['turns']
                total_score += result['score']
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_games - i - 1)

                print(f"  [{i+1}/{n_games}] seed={result['seed']}: "
                      f"score={result['score']} (ETA {eta/60:.0f}m)", flush=True)

    else:
        # GPU server mode: workers>1 + MPS/CUDA
        # N CPU workers share one GPU via InferenceServer.
        # Each game runs identical MCTS (SBS=8), GPU batches across games (IBS=8*N).
        from multiprocessing import Process, Queue as MPQueue
        from alphatrain.inference_server import InferenceServer

        _limit_threads()

        # Get max_score from checkpoint
        ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
        max_score = float(ckpt.get('max_score', 30000.0))
        del ckpt

        server = InferenceServer(args.model, args.workers,
                                 device=device_str,
                                 max_batch_per_worker=args.batch_size,
                                 value_model_path=args.value_model,
                                 deterministic=args.deterministic)
        server.start()

        seed_queue = MPQueue()
        for s in seeds:
            seed_queue.put(s)
        for _ in range(args.workers):
            seed_queue.put(None)  # sentinels

        result_queue = MPQueue()

        workers = []
        for i in range(args.workers):
            p = Process(
                target=_server_worker,
                args=(i, seed_queue, result_queue,
                      server._obs_shm.name, server._pol_shm.name,
                      server._val_shm.name,
                      args.workers, args.batch_size,
                      server.request_queue, server.response_queues[i],
                      args.sims, args.batch_size, max_score,
                      args.temperature_moves, args.dirichlet_alpha,
                      args.dirichlet_weight, args.max_turns,
                      args.dynamic_sims))
            p.start()
            workers.append(p)

        try:
            for i in range(n_games):
                result = result_queue.get(timeout=7200)

                save_game_json(result, args.save_dir)

                total_states += result['turns']
                total_score += result['score']
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_games - i - 1)

                print(f"  [{i+1}/{n_games}] seed={result['seed']}: "
                      f"score={result['score']} (ETA {eta/60:.0f}m)", flush=True)
        finally:
            for p in workers:
                p.join(timeout=30)
            server.shutdown()

    elapsed = time.time() - t0
    mean_score = total_score / max(n_games, 1)
    print(f"\nDone: {n_games} games, {total_states} states, "
          f"mean score={mean_score:.0f}, {elapsed/60:.1f}m", flush=True)


if __name__ == '__main__':
    main()
