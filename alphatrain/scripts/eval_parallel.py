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


def _eval_policy_gpu_worker(slot_id, seed_queue, result_queue,
                             obs_shm_name, pol_shm_name, val_shm_name,
                             num_workers, max_batch,
                             request_queue, response_queue):
    """Play policy-only games using GPU inference server."""
    torch.set_num_threads(1)

    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
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

    while True:
        seed = seed_queue.get()
        if seed is None:
            break

        game = ColorLinesGame(seed=seed)
        game.reset()

        while not game.game_over:
            obs_np = _build_obs_for_game(game)
            pol_np, _ = client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, 30)
            if not priors:
                break
            best_action = max(priors.items(), key=lambda x: x[1])[0]
            src_flat = best_action // 81
            tgt_flat = best_action % 81
            game.move((src_flat // 9, src_flat % 9),
                      (tgt_flat // 9, tgt_flat % 9))

        result_queue.put((seed, game.score, game.turns))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


# ── MCTS eval worker (persistent, shared-memory GPU inference) ──────

def _eval_mcts_worker(slot_id, seed_queue, result_queue,
                      obs_shm_name, pol_shm_name, val_shm_name,
                      num_workers, max_batch, request_queue, response_queue,
                      num_sims, c_puct, top_k, max_score,
                      value_net_path=None, device_str='cpu'):
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

    # Load separate value network in worker process
    vnet = None
    if value_net_path:
        from alphatrain.model import ValueNet
        device = torch.device(device_str)
        ckpt = torch.load(value_net_path, map_location='cpu', weights_only=False)
        vnet = ValueNet(in_channels=18,
                        num_blocks=ckpt['num_blocks'],
                        channels=ckpt['channels'],
                        num_value_bins=1)
        vnet.load_state_dict(ckpt['model'])
        vnet = vnet.to(device)
        vnet.requires_grad_(False)
        print(f"  [w{slot_id}] ValueNet loaded: {ckpt['num_blocks']}b x "
              f"{ckpt['channels']}ch on {device_str}", flush=True)

    mcts = MCTS(inference_client=client, max_score=max_score,
                num_simulations=num_sims, c_puct=c_puct, top_k=top_k,
                batch_size=max_batch, value_net=vnet)

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
    p.add_argument('--value-model', default=None,
                   help='Separate ValueNet checkpoint (dual-model mode)')
    p.add_argument('--deterministic', action='store_true',
                   help='Per-request GPU processing (exact scores, slower)')
    p.add_argument('--policy-only', action='store_true')
    p.add_argument('--mcts-only', action='store_true')
    p.add_argument('--value-net', default=None,
                   help='Separate ValueNet checkpoint for MCTS leaf eval')
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
        use_gpu = args.workers > 1 and device_str != 'cpu'
        if use_gpu:
            pol_results = _run_policy_server(args, task_seeds, total, device_str)
        else:
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


def _run_policy_server(args, task_seeds, total, device_str):
    """Run policy-only games using GPU inference server."""
    from alphatrain.inference_server import InferenceServer

    print(f"\n{'='*60}", flush=True)
    print(f"Policy player ({total} games, {args.workers} GPU workers)", flush=True)
    print(f"{'='*60}", flush=True)

    server = InferenceServer(args.model, args.workers,
                             device=device_str,
                             max_batch_per_worker=args.batch_size)
    server.start()

    seed_queue = MPQueue()
    for s in task_seeds:
        seed_queue.put(s)
    for _ in range(args.workers):
        seed_queue.put(None)

    result_queue = MPQueue()

    workers = []
    for i in range(args.workers):
        p = Process(
            target=_eval_policy_gpu_worker,
            args=(i, seed_queue, result_queue,
                  server._obs_shm.name, server._pol_shm.name,
                  server._val_shm.name,
                  args.workers, args.batch_size,
                  server.request_queue, server.response_queues[i]))
        p.start()
        workers.append(p)

    t0 = time.time()
    results = []
    try:
        for i in range(total):
            r = result_queue.get(timeout=3600)
            results.append(r)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"  [{i+1}/{total}] {elapsed:.0f}s (ETA {eta:.0f}s)",
                      flush=True)
    finally:
        for p in workers:
            p.join(timeout=5)
        server.shutdown()

    print(f"Policy done: {time.time()-t0:.1f}s", flush=True)
    return results


def _run_mcts_local(args, task_seeds, total, device_str):
    """Run MCTS games sequentially on local device."""
    from alphatrain.evaluate import load_model
    from alphatrain.mcts import make_mcts_player
    from game.board import ColorLinesGame

    device = torch.device(device_str)
    dual = hasattr(args, 'value_model') and args.value_model

    print(f"\n{'='*60}", flush=True)
    print(f"MCTS player ({total} games, local {device}, fp16+jit, "
          f"{args.simulations} sims, bs={args.batch_size}"
          f"{', dual-model' if dual else ''})", flush=True)
    print(f"{'='*60}", flush=True)

    if dual:
        from alphatrain.evaluate import load_dual_model
        net, max_score = load_dual_model(
            args.model, args.value_model, device,
            fp16=(device_str != 'cpu'), jit_trace=True)
    else:
        net, max_score = load_model(args.model, device,
                                    fp16=(device_str != 'cpu'), jit_trace=True)

    # Load separate value network if provided
    vnet = None
    if getattr(args, 'value_net', None):
        from alphatrain.model import ValueNet
        ckpt = torch.load(args.value_net, map_location='cpu', weights_only=False)
        vnet = ValueNet(in_channels=18,
                        num_blocks=ckpt['num_blocks'],
                        channels=ckpt['channels'],
                        num_value_bins=1)
        vnet.load_state_dict(ckpt['model'])
        vnet = vnet.to(device)
        vnet.requires_grad_(False)
        print(f"ValueNet: {ckpt['num_blocks']}b x {ckpt['channels']}ch, "
              f"acc={ckpt['accuracy']:.1f}%", flush=True)

    player = make_mcts_player(
        net, device, max_score=max_score,
        num_simulations=args.simulations,
        c_puct=args.c_puct, top_k=args.top_k,
        batch_size=args.batch_size,
        value_net=vnet)

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

    value_path = getattr(args, 'value_model', None)
    det = getattr(args, 'deterministic', False)
    server = InferenceServer(args.model, n_workers, device=device_str,
                             max_batch_per_worker=args.batch_size,
                             value_model_path=value_path,
                             deterministic=det)
    server.start()

    seed_queue = MPQueue()
    for s in task_seeds:
        seed_queue.put(s)
    for _ in range(n_workers):
        seed_queue.put(None)

    result_queue = MPQueue()

    vnet_path = getattr(args, 'value_net', None)

    workers = []
    for i in range(n_workers):
        proc = Process(
            target=_eval_mcts_worker,
            args=(i, seed_queue, result_queue,
                  server._obs_shm.name, server._pol_shm.name,
                  server._val_shm.name,
                  n_workers, args.batch_size,
                  server.request_queue, server.response_queues[i],
                  args.simulations, args.c_puct, args.top_k, max_score,
                  vnet_path, device_str))
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

    # Percentile breakdown
    for label, all_scores in [("Pol", pol_all), ("MCTS", mcts_all)]:
        if not all_scores:
            continue
        if label == "Pol" and not show_pol:
            continue
        if label == "MCTS" and not show_mcts:
            continue
        a = np.array(all_scores)
        n = len(a)
        print(f"\n  {label} percentiles ({n} games):", flush=True)
        print(f"    P1={np.percentile(a,1):.0f}  P5={np.percentile(a,5):.0f}  "
              f"P10={np.percentile(a,10):.0f}  P25={np.percentile(a,25):.0f}  "
              f"P50={np.percentile(a,50):.0f}  P75={np.percentile(a,75):.0f}  "
              f"P90={np.percentile(a,90):.0f}  P95={np.percentile(a,95):.0f}", flush=True)
        print(f"    <500: {(a<500).sum()} ({100*(a<500).mean():.1f}%)  "
              f"<1000: {(a<1000).sum()} ({100*(a<1000).mean():.1f}%)  "
              f">5000: {(a>5000).sum()} ({100*(a>5000).mean():.0f}%)  "
              f">10000: {(a>10000).sum()} ({100*(a>10000).mean():.0f}%)", flush=True)
    print(flush=True)


if __name__ == '__main__':
    main()
