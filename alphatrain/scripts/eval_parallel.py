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
import torch.nn as nn
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
                      value_net_path=None, device_str='cpu',
                      terminal_value=None, override_threshold=0.0,
                      max_turns=5000):
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
        vnet = vnet.to(device).half()
        vnet.requires_grad_(False)
        print(f"  [w{slot_id}] ValueNet loaded: {ckpt['num_blocks']}b x "
              f"{ckpt['channels']}ch on {device_str} [fp16]", flush=True)

    mcts = MCTS(inference_client=client, max_score=max_score,
                num_simulations=num_sims, c_puct=c_puct, top_k=top_k,
                batch_size=max_batch, value_net=vnet,
                terminal_value=terminal_value,
                override_threshold=override_threshold)

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
    p.add_argument('--override-threshold', type=float, default=0.0,
                   help='Only override policy if MCTS has >X%% more visits '
                        '(0.2 = 20%%). 0 = always use MCTS top move.')
    p.add_argument('--max-turns', type=int, default=5000,
                   help='Cap games at this many turns')
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
    p.add_argument('--ranking-head', default=None,
                   help='Ranking head checkpoint (trained on frozen backbone)')
    p.add_argument('--value-mode', default=None,
                   help='Value mode: zero, hash, iid:<std>, or None (model head). '
                        'E.g. --value-mode hash or --value-mode iid:14')
    p.add_argument('--terminal-value', type=float, default=None,
                   help='Force terminal (game-over) value. '
                        'E.g. --terminal-value 0 for normalized terminals.')
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
        if device_str != 'cpu':
            # Always use GPU server for policy eval (fp16, matches MCTS precision)
            pol_workers = max(args.workers, 4)
            pol_results = _run_policy_server(
                args, task_seeds, total, device_str,
                n_workers_override=pol_workers)
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
        use_server = args.workers > 1 and device_str != 'cpu'
        if use_server:
            mcts_results = _run_mcts_server(args, task_seeds, total, device_str)
        else:
            mcts_results = _run_mcts_local(args, task_seeds, total, device_str)

    # ── Results ──
    _print_results_table(seeds, n_per, pol_results, mcts_results,
                   show_pol=not args.mcts_only,
                   show_mcts=not args.policy_only)


def _run_policy_server(args, task_seeds, total, device_str,
                       n_workers_override=None):
    """Run policy-only games using GPU inference server."""
    from alphatrain.inference_server import InferenceServer

    n_workers = n_workers_override or args.workers

    print(f"\n{'='*60}", flush=True)
    print(f"Policy player ({total} games, {n_workers} GPU workers)", flush=True)
    print(f"{'='*60}", flush=True)

    server = InferenceServer(args.model, n_workers,
                             device=device_str,
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
        p = Process(
            target=_eval_policy_gpu_worker,
            args=(i, seed_queue, result_queue,
                  server._obs_shm.name, server._pol_shm.name,
                  server._val_shm.name,
                  n_workers, args.batch_size,
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

    # When using ranking head, we need raw backbone access (no JIT)
    use_jit = not getattr(args, 'ranking_head', None)
    if dual:
        from alphatrain.evaluate import load_dual_model
        net, max_score = load_dual_model(
            args.model, args.value_model, device,
            fp16=(device_str != 'cpu'), jit_trace=use_jit)
    else:
        net, max_score = load_model(args.model, device,
                                    fp16=(device_str != 'cpu'),
                                    jit_trace=use_jit)

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
        vnet = vnet.to(device).half()
        vnet.requires_grad_(False)
        print(f"ValueNet: {ckpt['num_blocks']}b x {ckpt['channels']}ch, "
              f"acc={ckpt['accuracy']:.1f}% [fp16]", flush=True)

    # Load ranking head if provided (overrides value_net)
    if getattr(args, 'ranking_head', None):
        from alphatrain.scripts.train_ranking_head import (
            RankingHead, SpatialRankingHead)
        ckpt = torch.load(args.ranking_head, map_location='cpu',
                          weights_only=False)
        if ckpt.get('spatial', False):
            rhead = SpatialRankingHead(
                channels=256,
                value_channels=ckpt.get('value_channels', 32),
                value_hidden=ckpt.get('value_hidden', 256))
        else:
            rhead = RankingHead(in_features=ckpt.get('in_features', 256),
                                hidden=ckpt.get('hidden', 0))
        rhead.load_state_dict(ckpt['head'])
        if device_str != 'cpu':
            rhead = rhead.to(device).half()
        else:
            rhead = rhead.to(device)
        rhead.requires_grad_(False)
        print(f"RankingHead: hidden={ckpt.get('hidden', 0)}, "
              f"val_acc={100*ckpt.get('val_acc', 0):.1f}%", flush=True)

        # Wrap as a "value_net" that MCTS can use:
        # backbone features → global avg pool → ranking head → scalar
        import torch.nn.functional as _F

        is_spatial = ckpt.get('spatial', False)

        class RankingWrapper(nn.Module):
            def __init__(self, backbone, head, spatial):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.spatial = spatial
            def forward(self, x):
                # Zero fake next_balls channels to match training data
                x = x.clone()
                x[:, 8:12] = 0
                with torch.inference_mode():
                    out = self.backbone.stem(x)
                    out = self.backbone.blocks(out)
                    out = _F.relu(self.backbone.backbone_bn(out))
                if not self.spatial:
                    out = out.mean(dim=(2, 3))
                return self.head(out)  # (batch, 1)

        # Build wrapper using the policy net's frozen backbone
        vnet = RankingWrapper(net, rhead, is_spatial)
        vnet = vnet.to(device)
        print(f"  Wrapped as value_net on backbone", flush=True)

    # Value mode for local MCTS (zero, hash, iid:<std>)
    vmode = getattr(args, 'value_mode', None)
    if vmode == 'zero':
        print("*** VALUE MODE: zero ***", flush=True)
        net.predict_value = lambda val_logits, max_val=None: \
            torch.zeros(val_logits.shape[0], device=val_logits.device)
    elif vmode and vmode.startswith('iid:'):
        std = float(vmode.split(':')[1])
        print(f"*** VALUE MODE: iid N(100, {std}) ***", flush=True)
        net.predict_value = lambda val_logits, max_val=None, _s=std: \
            torch.randn(val_logits.shape[0], device=val_logits.device) * _s + 100
    elif vmode == 'hash':
        import hashlib
        print("*** VALUE MODE: deterministic hash ***", flush=True)
        def _hash_value(val_logits, max_val=None):
            # Not used in local mode — hash is in the server
            return torch.zeros(val_logits.shape[0], device=val_logits.device) + 100
        net.predict_value = _hash_value
    elif vmode:
        print(f"*** VALUE MODE: {vmode} (unknown, using default) ***", flush=True)

    # Terminal value: explicit flag > auto (0.0 for synthetic/value_net) > None
    has_custom_value = vmode or vnet or getattr(args, 'ranking_head', None)
    tv = args.terminal_value if args.terminal_value is not None \
        else (0.0 if has_custom_value else None)

    player = make_mcts_player(
        net, device, max_score=max_score,
        num_simulations=args.simulations,
        c_puct=args.c_puct, top_k=args.top_k,
        batch_size=args.batch_size,
        value_net=vnet, terminal_value=tv,
        override_threshold=getattr(args, 'override_threshold', 0.0))

    results = []
    t0 = time.time()
    for i, seed in enumerate(task_seeds):
        gt = time.time()
        game = ColorLinesGame(seed=seed)
        game.reset()
        max_t = getattr(args, 'max_turns', 5000)
        while not game.game_over and game.turns < max_t:
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

    vnet_path = getattr(args, 'value_net', None) or \
                getattr(args, 'value_model', None)
    rhead_path = getattr(args, 'ranking_head', None)
    det = getattr(args, 'deterministic', False)
    vmode = getattr(args, 'value_mode', None)
    server = InferenceServer(args.model, n_workers, device=device_str,
                             max_batch_per_worker=args.batch_size,
                             value_model_path=vnet_path,
                             deterministic=det,
                             value_mode=vmode,
                             ranking_head_path=rhead_path)
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
                  vnet_path, device_str,
                  args.terminal_value if args.terminal_value is not None
                  else (0.0 if (vmode or vnet_path or rhead_path) else None),
                  getattr(args, 'override_threshold', 0.0),
                  getattr(args, 'max_turns', 5000)))
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
