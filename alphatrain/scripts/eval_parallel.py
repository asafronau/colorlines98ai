"""Parallel evaluation: Policy and MCTS players.

CPU workers handle game simulation + batched virtual loss MCTS.
One GPU process handles all NN inference via shared memory.

Usage:
    python -m alphatrain.scripts.eval_parallel \
        --model alphatrain/data/alphatrain_td_best.pt \
        --seeds 42 43 44 45 46 --games-per-seed 10 --simulations 800

    # GPU server mode (faster with multiple workers):
    python -m alphatrain.scripts.eval_parallel \
        --model alphatrain/data/alphatrain_td_best.pt \
        --seeds 42 43 44 45 46 --games-per-seed 10 --simulations 800 \
        --device mps --workers 4
"""

# Force single-threaded BLAS before imports (for CPU workers)
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
from multiprocessing import Process, Pool, Queue as MPQueue, cpu_count


# ── Policy worker (CPU with local model) ────────────────────────────

def _init_policy_worker(model_path):
    torch.set_num_threads(1)
    global _net, _device
    from alphatrain.evaluate import load_model
    _device = torch.device('cpu')
    _net, _ = load_model(model_path, _device)


def _play_policy(seed):
    from alphatrain.evaluate import make_policy_player, play_game
    player = make_policy_player(_net, _device)
    result = play_game(player, seed=seed)
    return seed, result['score'], result['turns']


# ── MCTS eval worker (persistent, shared-memory GPU inference) ──────

def _eval_mcts_worker(slot_id, seed_queue, result_queue,
                      obs_shm_name, pol_shm_name, val_shm_name,
                      num_workers, max_batch, request_queue, response_queue,
                      num_sims, c_puct, top_k, max_score):
    """Persistent worker: pull seeds, play greedy MCTS games, push results."""
    torch.set_num_threads(1)

    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE
    from alphatrain.mcts import MCTS
    from game.board import ColorLinesGame

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
                num_simulations=num_sims, c_puct=c_puct, top_k=top_k,
                batch_size=max_batch)

    while True:
        seed = seed_queue.get()
        if seed is None:
            break

        game = ColorLinesGame(seed=seed)
        game.reset()
        t0 = time.time()

        while not game.game_over:
            move = mcts.search(game)
            if move is None:
                break
            r = game.move(move[0], move[1])
            if not r['valid']:
                break
            if game.turns % 500 == 0:
                elapsed = time.time() - t0
                print(f"    [w{slot_id}] seed={seed} turn={game.turns} "
                      f"score={game.score} {elapsed:.0f}s", flush=True)

        elapsed = time.time() - t0
        ms_per_turn = elapsed / max(game.turns, 1) * 1000
        print(f"  [w{slot_id}] seed={seed}: score={game.score}, "
              f"turns={game.turns}, {elapsed:.0f}s "
              f"({ms_per_turn:.0f}ms/turn)", flush=True)

        result_queue.put((seed, game.score, game.turns))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/alphatrain_td_best.pt')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--games-per-seed', type=int, default=5)
    p.add_argument('--simulations', type=int, default=400)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default=None,
                   help='Force device (mps/cuda/cpu). Auto-detect if not set.')
    p.add_argument('--workers', type=int, default=1,
                   help='MCTS workers (1=sequential, >1=GPU server mode)')
    p.add_argument('--policy-only', action='store_true')
    p.add_argument('--mcts-only', action='store_true')
    args = p.parse_args()

    if args.device:
        device_str = args.device
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'

    seeds = args.seeds
    n_per = args.games_per_seed
    total = len(seeds) * n_per
    n_cpu = min(cpu_count(), total)

    print(f"Evaluation: {len(seeds)} seeds x {n_per} games = {total} games",
          flush=True)
    print(f"Model: {args.model}", flush=True)

    task_seeds = []
    for s in seeds:
        task_seeds.extend([s] * n_per)

    # ── Policy evaluation ──
    pol_results = []
    if not args.mcts_only:
        print(f"\n{'='*60}", flush=True)
        print(f"Policy player ({total} games, {n_cpu} CPU workers)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        with Pool(n_cpu, initializer=_init_policy_worker,
                  initargs=(args.model,)) as pool:
            pol_results = pool.map(_play_policy, task_seeds)
        print(f"Policy done: {time.time()-t0:.1f}s", flush=True)

    # ── MCTS evaluation ──
    mcts_results = []
    if not args.policy_only:
        if args.workers > 1 and device_str != 'cpu':
            mcts_results = _run_mcts_server(args, task_seeds, total, device_str)
        else:
            mcts_results = _run_mcts_local(args, task_seeds, total, device_str)

    # ── Results ──
    _print_results_table(seeds, n_per, pol_results, mcts_results,
                   show_pol=not args.mcts_only,
                   show_mcts=not args.policy_only)


def _run_mcts_local(args, task_seeds, total, device_str):
    """Run MCTS games sequentially on local device."""
    from alphatrain.evaluate import load_model
    from alphatrain.mcts import make_mcts_player
    from game.board import ColorLinesGame

    device = torch.device(device_str)

    print(f"\n{'='*60}", flush=True)
    print(f"MCTS player ({total} games, local {device}, fp16+jit, "
          f"{args.simulations} sims, bs={args.batch_size})", flush=True)
    print(f"{'='*60}", flush=True)

    net, max_score = load_model(args.model, device,
                                fp16=(device_str != 'cpu'), jit_trace=True)
    player = make_mcts_player(
        net, device, max_score=max_score,
        num_simulations=args.simulations,
        c_puct=args.c_puct, top_k=args.top_k,
        batch_size=args.batch_size)

    results = []
    t0 = time.time()
    for i, seed in enumerate(task_seeds):
        gt = time.time()
        game = ColorLinesGame(seed=seed)
        game.reset()
        while not game.game_over:
            move = player(game)
            if move is None:
                break
            r = game.move(move[0], move[1])
            if not r['valid']:
                break
        elapsed_game = time.time() - gt
        elapsed_total = time.time() - t0
        eta = elapsed_total / (i + 1) * (total - i - 1)
        ms_per_turn = elapsed_game / max(game.turns, 1) * 1000
        print(f"  [{i+1}/{total}] seed={seed}: score={game.score}, "
              f"turns={game.turns}, {elapsed_game:.0f}s "
              f"({ms_per_turn:.0f}ms/turn, ETA {eta:.0f}s)", flush=True)
        results.append((seed, game.score, game.turns))

    mcts_time = time.time() - t0
    print(f"MCTS done: {mcts_time:.0f}s ({mcts_time/total:.0f}s/game)",
          flush=True)
    return results


def _run_mcts_server(args, task_seeds, total, device_str):
    """Run MCTS games in parallel via GPU inference server with persistent workers."""
    from alphatrain.inference_server import InferenceServer

    n_workers = args.workers

    print(f"\n{'='*60}", flush=True)
    print(f"MCTS player ({total} games, {n_workers} workers + {device_str} GPU, "
          f"{args.simulations} sims, bs={args.batch_size})", flush=True)
    print(f"{'='*60}", flush=True)

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    server = InferenceServer(args.model, n_workers, device=device_str,
                             max_batch_per_worker=args.batch_size)
    server.start()

    seed_queue = MPQueue()
    for s in task_seeds:
        seed_queue.put(s)
    for _ in range(n_workers):
        seed_queue.put(None)

    result_queue = MPQueue()

    workers = []
    for i in range(n_workers):
        proc = Process(
            target=_eval_mcts_worker,
            args=(i, seed_queue, result_queue,
                  server._obs_shm.name, server._pol_shm.name,
                  server._val_shm.name,
                  n_workers, args.batch_size,
                  server.request_queue, server.response_queues[i],
                  args.simulations, args.c_puct, args.top_k, max_score))
        proc.start()
        workers.append(proc)

    results = []
    t0 = time.time()
    try:
        for i in range(total):
            seed, score, turns = result_queue.get(timeout=7200)
            results.append((seed, score, turns))

            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] seed={seed}: score={score}, "
                  f"turns={turns} (ETA {eta:.0f}s)", flush=True)
    finally:
        for proc in workers:
            proc.join(timeout=30)
        server.shutdown()

    mcts_time = time.time() - t0
    print(f"MCTS done: {mcts_time:.0f}s ({mcts_time/total:.0f}s/game)",
          flush=True)
    return results


def _print_results_table(seeds, n_per, pol_results, mcts_results,
                   show_pol=True, show_mcts=True):
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {n_per} games per seed", flush=True)
    print(f"{'='*60}\n", flush=True)

    def _by_seed(results):
        d = {}
        for seed, score, turns in results:
            d.setdefault(seed, []).append(score)
        return d

    header = f"{'Seed':>6}"
    if show_pol:
        header += f" | {'Pol Mean':>8} {'Med':>5} {'Min':>5} {'Max':>5}"
    if show_mcts:
        header += f" | {'MCTS Mean':>9} {'Med':>5} {'Min':>5} {'Max':>5}"
    if show_pol and show_mcts:
        header += f" | {'Chg':>5}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    pol_by = _by_seed(pol_results) if pol_results else {}
    mcts_by = _by_seed(mcts_results) if mcts_results else {}
    pol_all, mcts_all = [], []

    for s in seeds:
        row = f"{s:>6}"
        if show_pol:
            ps = pol_by.get(s, [0])
            pol_all.extend(ps)
            row += (f" | {np.mean(ps):>8.0f} {np.median(ps):>5.0f} "
                    f"{np.min(ps):>5} {np.max(ps):>5}")
        if show_mcts:
            ms = mcts_by.get(s, [0])
            mcts_all.extend(ms)
            row += (f" | {np.mean(ms):>9.0f} {np.median(ms):>5.0f} "
                    f"{np.min(ms):>5} {np.max(ms):>5}")
        if show_pol and show_mcts:
            pm = np.mean(pol_by.get(s, [1]))
            mm = np.mean(mcts_by.get(s, [0]))
            pct = (mm / pm - 1) * 100 if pm > 0 else 0
            row += f" | {pct:>+4.0f}%"
        print(row, flush=True)

    print("-" * len(header), flush=True)
    row = f"{'ALL':>6}"
    if show_pol:
        row += (f" | {np.mean(pol_all):>8.0f} {np.median(pol_all):>5.0f} "
                f"{np.min(pol_all):>5} {np.max(pol_all):>5}")
    if show_mcts:
        row += (f" | {np.mean(mcts_all):>9.0f} {np.median(mcts_all):>5.0f} "
                f"{np.min(mcts_all):>5} {np.max(mcts_all):>5}")
    if show_pol and show_mcts:
        pct = (np.mean(mcts_all) / np.mean(pol_all) - 1) * 100 if np.mean(pol_all) > 0 else 0
        row += f" | {pct:>+4.0f}%"
    print(row, flush=True)
    print(flush=True)


if __name__ == '__main__':
    main()
