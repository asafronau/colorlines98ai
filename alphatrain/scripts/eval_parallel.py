"""Parallel evaluation: policy and MCTS players on a list of seeds.

CPU workers handle game simulation + batched virtual-loss MCTS. One GPU
process handles all NN inference via shared memory.

The model is policy-only (PolicyNet); MCTS uses a feature-value evaluator
for leaf values via --feature-value-weights.

Usage:
    python -m alphatrain.scripts.eval_parallel \\
        --model alphatrain/data/pillar2x2_epoch_10.pt \\
        --feature-value-weights alphatrain/data/feature_value_weights_2x.npz \\
        --seeds $(seq 0 49) --simulations 600 \\
        --device mps --workers 16 --batch-size 8
"""

# Force single-threaded BLAS before imports (CPU workers contend otherwise).
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
from multiprocessing import Process, Queue as MPQueue


# ── Worker: policy-only player via GPU server ───────────────────────

def _policy_server_worker(slot_id, seed_queue, result_queue,
                          obs_shm_name, pol_shm_name, val_shm_name,
                          num_workers, max_batch,
                          request_queue, response_queue):
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
            best = max(priors.items(), key=lambda x: x[1])[0]
            sf = best // 81
            tf = best % 81
            game.move((sf // 9, sf % 9), (tf // 9, tf % 9))
        result_queue.put((seed, game.score, game.turns))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


# ── Worker: MCTS player via GPU server ─────────────────────────────

def _mcts_server_worker(slot_id, seed_queue, result_queue,
                        obs_shm_name, pol_shm_name, val_shm_name,
                        num_workers, max_batch, request_queue, response_queue,
                        num_sims, c_puct, top_k, max_score,
                        override_threshold, max_turns,
                        feature_weights_path, early_stop, q_weight):
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
                batch_size=max_batch,
                override_threshold=override_threshold,
                feature_weights_path=feature_weights_path,
                early_stop=early_stop, q_weight=q_weight)

    while True:
        seed = seed_queue.get()
        if seed is None:
            break
        game = ColorLinesGame(seed=seed)
        game.reset()
        t0 = time.time()
        while not game.game_over and game.turns < max_turns:
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
        ms = elapsed / max(game.turns, 1) * 1000
        print(f"  [w{slot_id}] seed={seed}: score={game.score}, "
              f"turns={game.turns}, {elapsed:.0f}s ({ms:.0f}ms/turn)",
              flush=True)
        result_queue.put((seed, game.score, game.turns))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


# ── Drivers ─────────────────────────────────────────────────────────

def _run_policy_server(args, task_seeds, total, device_str):
    from alphatrain.inference_server import InferenceServer
    n_workers = max(args.workers, 4)
    print(f"\n{'='*60}\nPolicy player ({total} games, {n_workers} workers)"
          f"\n{'='*60}", flush=True)
    server = InferenceServer(args.model, n_workers, device=device_str,
                             max_batch_per_worker=args.batch_size,
                             use_compile=args.compile)
    server.start()

    seed_queue = MPQueue()
    for s in task_seeds: seed_queue.put(s)
    for _ in range(n_workers): seed_queue.put(None)
    result_queue = MPQueue()
    procs = []
    for i in range(n_workers):
        p = Process(target=_policy_server_worker,
                    args=(i, seed_queue, result_queue,
                          server._obs_shm.name, server._pol_shm.name,
                          server._val_shm.name, n_workers, args.batch_size,
                          server.request_queue, server.response_queues[i]))
        p.start()
        procs.append(p)

    t0 = time.time()
    results = []
    try:
        for i in range(total):
            results.append(result_queue.get(timeout=3600))
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (total - i - 1)
                print(f"  [{i+1}/{total}] {elapsed:.0f}s (ETA {eta:.0f}s)",
                      flush=True)
    finally:
        for p in procs:
            p.join(timeout=5)
        server.shutdown()
    print(f"Policy done: {time.time()-t0:.1f}s", flush=True)
    return results


def _run_mcts_server(args, task_seeds, total, device_str):
    from alphatrain.inference_server import InferenceServer
    n_workers = args.workers
    print(f"\n{'='*60}\nMCTS player ({total} games, {n_workers} workers + "
          f"{device_str} GPU, {args.simulations} sims, bs={args.batch_size})"
          f"\n{'='*60}", flush=True)

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    server = InferenceServer(args.model, n_workers, device=device_str,
                             max_batch_per_worker=args.batch_size,
                             use_compile=args.compile)
    server.start()

    seed_queue = MPQueue()
    for s in task_seeds: seed_queue.put(s)
    for _ in range(n_workers): seed_queue.put(None)
    result_queue = MPQueue()
    procs = []
    for i in range(n_workers):
        p = Process(target=_mcts_server_worker,
                    args=(i, seed_queue, result_queue,
                          server._obs_shm.name, server._pol_shm.name,
                          server._val_shm.name, n_workers, args.batch_size,
                          server.request_queue, server.response_queues[i],
                          args.simulations, args.c_puct, args.top_k, max_score,
                          args.override_threshold, args.max_turns,
                          args.feature_value_weights, args.early_stop,
                          args.q_weight))
        p.start()
        procs.append(p)

    t0 = time.time()
    results = []
    try:
        for i in range(total):
            seed, score, turns = result_queue.get(timeout=7200)
            results.append((seed, score, turns))
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] seed={seed}: score={score}, "
                  f"turns={turns} (ETA {eta:.0f}s)", flush=True)
    finally:
        for p in procs:
            p.join(timeout=30)
        server.shutdown()
    wall = time.time() - t0
    print(f"MCTS done: {wall:.0f}s ({wall/total:.0f}s/game)", flush=True)
    return results


def _run_mcts_local(args, task_seeds, total, device_str):
    """Single-process fallback (used when --workers 1)."""
    from alphatrain.evaluate import load_model
    from alphatrain.mcts import make_mcts_player
    from game.board import ColorLinesGame

    device = torch.device(device_str)
    print(f"\n{'='*60}\nMCTS player ({total} games, local {device}, "
          f"{args.simulations} sims, bs={args.batch_size})\n{'='*60}",
          flush=True)
    # JIT-traced models can't expose backbone_features (the value head
    # path needs the un-traced model). Skip JIT when --value-head-path.
    use_jit = args.value_head_path is None
    net, max_score = load_model(args.model, device,
                                fp16=(device_str != 'cpu'), jit_trace=use_jit)
    player = make_mcts_player(
        net, device, max_score=max_score,
        num_simulations=args.simulations,
        c_puct=args.c_puct, top_k=args.top_k, batch_size=args.batch_size,
        override_threshold=args.override_threshold,
        feature_weights_path=args.feature_value_weights,
        early_stop=args.early_stop,
        q_weight=args.q_weight,
        value_head_path=args.value_head_path)

    results = []
    t0 = time.time()
    for i, seed in enumerate(task_seeds):
        gt = time.time()
        game = ColorLinesGame(seed=seed)
        game.reset()
        while not game.game_over and game.turns < args.max_turns:
            move = player(game)
            if move is None:
                break
            r = game.move(move[0], move[1])
            if not r['valid']:
                break
        gw = time.time() - gt
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (total - i - 1)
        ms = gw / max(game.turns, 1) * 1000
        print(f"  [{i+1}/{total}] seed={seed}: score={game.score}, "
              f"turns={game.turns}, {gw:.0f}s ({ms:.0f}ms/turn, "
              f"ETA {eta:.0f}s)", flush=True)
        results.append((seed, game.score, game.turns))
    print(f"MCTS done: {time.time()-t0:.0f}s", flush=True)
    return results


# ── Output formatting ──────────────────────────────────────────────

def _print_results_table(seeds, pol_results, mcts_results,
                         show_pol=True, show_mcts=True):
    print(f"\n{'='*60}\nResults: {len(seeds)} seeds\n{'='*60}\n",
          flush=True)

    def by_seed(rs):
        d = {}
        for seed, score, _ in rs:
            d[seed] = score  # one game per seed
        return d

    header = f"{'Seed':>6}"
    if show_pol:
        header += f" | {'Pol':>6}"
    if show_mcts:
        header += f" | {'MCTS':>6}"
    if show_pol and show_mcts:
        header += f" | {'Chg':>5}"
    print(header, flush=True)
    print('-' * len(header), flush=True)

    pol_by = by_seed(pol_results) if pol_results else {}
    mcts_by = by_seed(mcts_results) if mcts_results else {}
    pol_all, mcts_all = [], []
    for s in seeds:
        row = f"{s:>6}"
        if show_pol:
            ps = pol_by.get(s, 0)
            pol_all.append(ps)
            row += f" | {ps:>6}"
        if show_mcts:
            ms = mcts_by.get(s, 0)
            mcts_all.append(ms)
            row += f" | {ms:>6}"
        if show_pol and show_mcts:
            pm = pol_by.get(s, 0)
            mm = mcts_by.get(s, 0)
            pct = (mm / pm - 1) * 100 if pm > 0 else 0
            row += f" | {pct:>+4.0f}%"
        print(row, flush=True)
    print('-' * len(header), flush=True)
    row = f"{'MEAN':>6}"
    if show_pol:
        row += f" | {np.mean(pol_all):>6.0f}"
    if show_mcts:
        row += f" | {np.mean(mcts_all):>6.0f}"
    if show_pol and show_mcts:
        pct = (np.mean(mcts_all) / np.mean(pol_all) - 1) * 100 \
            if np.mean(pol_all) > 0 else 0
        row += f" | {pct:>+4.0f}%"
    print(row, flush=True)

    for label, all_scores in [("Pol", pol_all), ("MCTS", mcts_all)]:
        if not all_scores:
            continue
        if (label == "Pol" and not show_pol) or \
                (label == "MCTS" and not show_mcts):
            continue
        a = np.array(all_scores)
        n = len(a)
        print(f"\n  {label} percentiles ({n} games):", flush=True)
        print(f"    P1={np.percentile(a,1):.0f}  P5={np.percentile(a,5):.0f}  "
              f"P10={np.percentile(a,10):.0f}  P25={np.percentile(a,25):.0f}  "
              f"P50={np.percentile(a,50):.0f}  P75={np.percentile(a,75):.0f}  "
              f"P90={np.percentile(a,90):.0f}  P95={np.percentile(a,95):.0f}",
              flush=True)
        print(f"    <500: {(a<500).sum()} ({100*(a<500).mean():.1f}%)  "
              f"<1000: {(a<1000).sum()} ({100*(a<1000).mean():.1f}%)  "
              f">5000: {(a>5000).sum()} ({100*(a>5000).mean():.0f}%)  "
              f">10000: {(a>10000).sum()} ({100*(a>10000).mean():.0f}%)",
              flush=True)
    print(flush=True)


# ── Main ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[42, 43, 44, 45, 46])
    p.add_argument('--simulations', type=int, default=400)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--q-weight', type=float, default=1.0,
                   help='PUCT score = q_weight * q_norm + U. Default 1.0 '
                        '(legacy behavior). 0.5 was the empirical sweet '
                        'spot for distilled-policy generations 2Y/2Y2 '
                        '(HISTORY lessons 127-128). 0.0 = pure-prior '
                        'search (diagnostic for whether the leaf '
                        'evaluator is contributing or just adding noise).')
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=8,
                   help='MCTS batch size. >8 destroys quality '
                        '(HISTORY lesson 115).')
    p.add_argument('--override-threshold', type=float, default=0.0,
                   help='Skip MCTS pick if visit count is too close to '
                        "policy's pick (0.2 = 20%%). 0 = always trust MCTS.")
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--device', default=None,
                   help='mps/cuda/cpu. Auto-detect if not set.')
    p.add_argument('--workers', type=int, default=1,
                   help='MCTS workers (1=local single-process, >1=GPU server)')
    p.add_argument('--policy-only', action='store_true',
                   help='Run only the policy player (no MCTS)')
    p.add_argument('--mcts-only', action='store_true',
                   help='Run only the MCTS player (no policy baseline)')
    p.add_argument('--feature-value-weights', default=None,
                   help='Path to feature_value_weights.npz. Required when '
                        'running MCTS — the model has no NN value head.')
    p.add_argument('--value-head-path', default=None,
                   help='Path to a trained ValueHead checkpoint (Phase 3 '
                        'NN value head). Mutually exclusive with '
                        '--feature-value-weights. LOCAL-MODE ONLY for now '
                        '(server mode would need to wire the head over '
                        'backbone features — TODO).')
    p.add_argument('--early-stop', action='store_true',
                   help='Exit MCTS early when greedy root child is locked '
                        'in. Eval-only — preserves pick, not visit dist.')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile(reduce-overhead) in the GPU server. '
                        'CUDA only; ignored elsewhere.')
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
    total = len(seeds)

    # Fail-fast: MCTS needs the feature evaluator. PolicyNet has no NN
    # value head; without feature weights, search runs blind on zeros.
    if not args.policy_only and not args.feature_value_weights:
        raise SystemExit(
            "MCTS requires --feature-value-weights (policy model has no "
            "NN value head). Pass --policy-only to skip MCTS.")

    print(f"Evaluation: {total} games (one per seed)", flush=True)
    print(f"Model: {args.model}", flush=True)

    task_seeds = list(seeds)

    # ── Policy phase ──
    pol_results = []
    if not args.mcts_only and device_str != 'cpu':
        pol_results = _run_policy_server(args, task_seeds, total, device_str)

    # ── MCTS phase ──
    mcts_results = []
    if not args.policy_only:
        if args.workers > 1 and device_str != 'cpu':
            mcts_results = _run_mcts_server(args, task_seeds, total, device_str)
        else:
            mcts_results = _run_mcts_local(args, task_seeds, total, device_str)

    _print_results_table(seeds, pol_results, mcts_results,
                         show_pol=not args.mcts_only,
                         show_mcts=not args.policy_only)


if __name__ == '__main__':
    main()
