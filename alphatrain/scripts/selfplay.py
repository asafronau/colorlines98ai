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

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game
from alphatrain.evaluate import load_model


def play_selfplay_game(mcts, seed, temperature_moves=30,
                       dirichlet_alpha=0.3, dirichlet_weight=0.25,
                       gamma=0.99):
    """Play one self-play game, recording training data.

    Args:
        mcts: MCTS instance
        seed: game seed
        temperature_moves: use temperature=1 for first N moves
        dirichlet_alpha: Dirichlet noise parameter
        dirichlet_weight: weight for noise at root
        gamma: discount factor for TD returns

    Returns:
        dict with observations, policy_targets, value_targets, metadata
    """
    game = ColorLinesGame(seed=seed)
    game.reset()

    observations = []
    policy_targets = []
    scores_at_turn = []

    t0 = time.time()
    turn = 0

    while not game.game_over:
        # Temperature: explore for first N moves, greedy after
        temp = 1.0 if turn < temperature_moves else 0.0

        # Build observation before move
        obs = _build_obs_for_game(game)

        # MCTS search with policy target
        result = mcts.search(
            game,
            temperature=temp,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_weight=dirichlet_weight,
            return_policy=True)

        if result[0] is None:
            break

        action, policy_target = result

        # Record training data
        observations.append(obs)
        policy_targets.append(policy_target)
        scores_at_turn.append(game.score)

        # Execute move
        move_result = game.move(action[0], action[1])
        if not move_result['valid']:
            break

        turn += 1
        if turn % 200 == 0:
            elapsed = time.time() - t0
            print(f"    seed={seed} turn={turn} score={game.score} "
                  f"{elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    final_score = game.score

    # Compute TD value targets (gamma=0.99 discounted remaining score)
    n = len(scores_at_turn)
    scores_at_turn.append(final_score)  # sentinel for final score
    value_targets = np.zeros(n, dtype=np.float32)

    if n > 0:
        # Reward at each step = score delta
        rewards = np.zeros(n, dtype=np.float32)
        for i in range(n):
            rewards[i] = scores_at_turn[i + 1] - scores_at_turn[i]

        # Discounted return from each position
        running = 0.0
        for i in range(n - 1, -1, -1):
            running = rewards[i] + gamma * running
            value_targets[i] = running

    return {
        'observations': np.stack(observations) if observations else np.empty((0, 18, 9, 9)),
        'policy_targets': np.stack(policy_targets) if policy_targets else np.empty((0, 6561)),
        'value_targets': value_targets,
        'score': final_score,
        'turns': turn,
        'seed': seed,
        'time': elapsed,
    }


def _limit_threads():
    """Force single-threaded torch (env vars already set at module top)."""
    torch.set_num_threads(1)


def _server_worker(slot_id, seed_queue, result_queue,
                   obs_shm_name, pol_shm_name, val_shm_name,
                   num_workers, max_batch,
                   request_queue, response_queue,
                   num_sims, batch_size, max_score,
                   temperature_moves, dirichlet_alpha, dirichlet_weight, gamma):
    """Persistent worker for GPU server mode self-play.

    Pulls seeds from seed_queue, plays games using shared-memory GPU
    inference, pushes results to result_queue. Exits on None sentinel.
    """
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
                top_k=30, c_puct=2.5)

    while True:
        seed = seed_queue.get()
        if seed is None:
            break

        result = play_selfplay_game(
            mcts, seed,
            temperature_moves=temperature_moves,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_weight=dirichlet_weight,
            gamma=gamma)

        print(f"  [w{slot_id}] seed={seed}: score={result['score']}, "
              f"turns={result['turns']}, {result['time']:.0f}s", flush=True)

        result_queue.put(result)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def _worker_play(args):
    """Worker function for CPU multiprocessing."""
    seed, model_path, device_str, num_sims, batch_size, \
        temperature_moves, dirichlet_alpha, dirichlet_weight, gamma = args

    _limit_threads()
    device = torch.device(device_str)

    # Load model (each worker loads independently)
    net, max_score = load_model(model_path, device,
                                fp16=(device_str != 'cpu'),
                                jit_trace=True)

    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=num_sims, batch_size=batch_size,
                top_k=30, c_puct=2.5)

    result = play_selfplay_game(
        mcts, seed,
        temperature_moves=temperature_moves,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_weight=dirichlet_weight,
        gamma=gamma)

    print(f"  seed={seed}: score={result['score']}, "
          f"turns={result['turns']}, {result['time']:.0f}s", flush=True)
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
    p.add_argument('--save-dir', default='data/selfplay')
    p.add_argument('--temperature-moves', type=int, default=30)
    p.add_argument('--dirichlet-alpha', type=float, default=0.3)
    p.add_argument('--dirichlet-weight', type=float, default=0.25)
    p.add_argument('--gamma', type=float, default=0.99)
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
    for f in os.listdir(args.save_dir):
        if f.startswith('game_') and f.endswith('.pt'):
            try:
                completed.add(int(f[5:11]))
            except ValueError:
                pass
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
                     top_k=30, c_puct=2.5)

        for i, seed in enumerate(seeds):
            result = play_selfplay_game(
                mcts, seed,
                temperature_moves=args.temperature_moves,
                dirichlet_alpha=args.dirichlet_alpha,
                dirichlet_weight=args.dirichlet_weight,
                gamma=args.gamma)

            # Save individual game
            save_path = os.path.join(args.save_dir, f'game_{seed:06d}.pt')
            torch.save({
                'observations': torch.from_numpy(result['observations']),
                'policy_targets': torch.from_numpy(result['policy_targets']),
                'value_targets': torch.from_numpy(result['value_targets']),
                'score': result['score'],
                'turns': result['turns'],
                'seed': seed,
            }, save_path)

            total_states += result['turns']
            total_score += result['score']
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_games - i - 1)

            print(f"  [{i+1}/{n_games}] seed={seed}: score={result['score']}, "
                  f"turns={result['turns']}, {result['time']:.0f}s "
                  f"(ETA {eta/60:.0f}m)", flush=True)
    elif device_str == 'cpu':
        # CPU multiprocessing — env vars already set at module top
        _limit_threads()
        from multiprocessing import Pool
        worker_args = [
            (seed, args.model, 'cpu', args.sims, args.batch_size,
             args.temperature_moves, args.dirichlet_alpha,
             args.dirichlet_weight, args.gamma)
            for seed in seeds
        ]

        with Pool(args.workers) as pool:
            for i, result in enumerate(pool.imap_unordered(
                    _worker_play, worker_args)):
                save_path = os.path.join(
                    args.save_dir, f'game_{result["seed"]:06d}.pt')
                torch.save({
                    'observations': torch.from_numpy(result['observations']),
                    'policy_targets': torch.from_numpy(result['policy_targets']),
                    'value_targets': torch.from_numpy(result['value_targets']),
                    'score': result['score'],
                    'turns': result['turns'],
                    'seed': result['seed'],
                }, save_path)

                total_states += result['turns']
                total_score += result['score']
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_games - i - 1)

                print(f"  [{i+1}/{n_games}] saved game_{result['seed']:06d}.pt "
                      f"(ETA {eta/60:.0f}m)", flush=True)

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
                                 value_model_path=args.value_model)
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
                      args.dirichlet_weight, args.gamma))
            p.start()
            workers.append(p)

        try:
            for i in range(n_games):
                result = result_queue.get(timeout=7200)

                save_path = os.path.join(
                    args.save_dir, f'game_{result["seed"]:06d}.pt')
                torch.save({
                    'observations': torch.from_numpy(result['observations']),
                    'policy_targets': torch.from_numpy(result['policy_targets']),
                    'value_targets': torch.from_numpy(result['value_targets']),
                    'score': result['score'],
                    'turns': result['turns'],
                    'seed': result['seed'],
                }, save_path)

                total_states += result['turns']
                total_score += result['score']
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (n_games - i - 1)

                print(f"  [{i+1}/{n_games}] saved game_{result['seed']:06d}.pt "
                      f"(ETA {eta/60:.0f}m)", flush=True)
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
