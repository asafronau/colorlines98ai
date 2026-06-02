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


def _largest_empty_component(board):
    """Size of the largest 4-connected empty region (for --record-game)."""
    visited = np.zeros_like(board, dtype=bool)
    best = 0
    for r0 in range(9):
        for c0 in range(9):
            if board[r0, c0] != 0 or visited[r0, c0]:
                continue
            sz = 0
            stack = [(r0, c0)]
            visited[r0, c0] = True
            while stack:
                r, c = stack.pop()
                sz += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr, nc] and board[nr, nc] == 0):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if sz > best:
                best = sz
    return int(best)


def _count_empty_components(board):
    """Number of 4-connected empty regions (for --record-game)."""
    visited = np.zeros_like(board, dtype=bool)
    n = 0
    for r0 in range(9):
        for c0 in range(9):
            if board[r0, c0] != 0 or visited[r0, c0]:
                continue
            n += 1
            stack = [(r0, c0)]
            visited[r0, c0] = True
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr, nc] and board[nr, nc] == 0):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return int(n)


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

    # Optional trajectory dump — instrumented per-turn metrics for failure
    # analysis. Activated via env var POLICY_SAVE_TRAJ_DIR. Avoids touching
    # the worker signature so eval_parallel still launches normally.
    import os, json
    save_dir = os.environ.get('POLICY_SAVE_TRAJ_DIR', '')
    record_path = os.environ.get('POLICY_RECORD_GAME', '')
    max_turns = int(os.environ.get('POLICY_MAX_TURNS', '1000000'))
    while True:
        seed = seed_queue.get()
        if seed is None:
            break
        game = ColorLinesGame(seed=seed)
        game.reset()
        traj = [] if save_dir else None
        rec = [] if record_path else None  # play_gui --replay format
        prev_score = 0
        while not game.game_over and game.turns < max_turns:
            obs_np = _build_obs_for_game(game)
            pol_np, _ = client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, 30)
            if not priors:
                break
            # Snapshot pre-move metrics + full state (for Phase-B counterfactuals).
            if traj is not None:
                vals_arr = np.fromiter(priors.values(), dtype=np.float64)
                vals_arr.sort()
                top1 = float(vals_arr[-1]) if vals_arr.size else 0.0
                top2 = float(vals_arr[-2]) if vals_arr.size > 1 else 0.0
                # Largest empty component via BFS over empty cells
                board_arr = game.board
                visited = np.zeros_like(board_arr, dtype=bool)
                lec = 0
                n_components = 0
                stack = []
                for r0 in range(9):
                    for c0 in range(9):
                        if board_arr[r0, c0] != 0 or visited[r0, c0]:
                            continue
                        n_components += 1
                        sz = 0
                        stack.append((r0, c0))
                        visited[r0, c0] = True
                        while stack:
                            r, c = stack.pop()
                            sz += 1
                            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                nr, nc = r + dr, c + dc
                                if (0 <= nr < 9 and 0 <= nc < 9
                                        and not visited[nr, nc]
                                        and board_arr[nr, nc] == 0):
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                        if sz > lec:
                            lec = sz
                traj.append({
                    'turn': int(game.turns),
                    'score': int(game.score),
                    'empties': int((board_arr == 0).sum()),
                    'lec': int(lec),
                    'n_components': int(n_components),
                    'n_legal_top30': len(priors),
                    'top1_p': float(top1),
                    'top1_top2_gap': float(top1 - top2),
                    'board': board_arr.astype(np.int8).tolist(),
                    'next_balls': [[[int(p[0]), int(p[1])], int(c)]
                                    for p, c in game.next_balls],
                })
            best = max(priors.items(), key=lambda x: x[1])[0]
            sf = best // 81
            tf = best % 81
            chosen = ((sf // 9, sf % 9), (tf // 9, tf % 9))

            # Capture a play_gui --replay frame BEFORE executing the move.
            if rec is not None:
                board_arr = game.board
                top_sorted = sorted(priors.items(), key=lambda x: -x[1])[:10]
                rec_frame = {
                    'turn': int(game.turns),
                    'score_before': int(game.score),
                    'board': board_arr.astype(np.int8).tolist(),
                    'next_balls': [[[int(p[0]), int(p[1])], int(c)]
                                    for p, c in game.next_balls],
                    'empties': int((board_arr == 0).sum()),
                    'lec': _largest_empty_component(board_arr),
                    'n_components': _count_empty_components(board_arr),
                    'chosen_move': [[int(chosen[0][0]), int(chosen[0][1])],
                                    [int(chosen[1][0]), int(chosen[1][1])]],
                    'top_k': [
                        {'move': [[int((m // 81) // 9), int((m // 81) % 9)],
                                  [int((m % 81) // 9), int((m % 81) % 9)]],
                         'prob': float(pr)}
                        for m, pr in top_sorted
                    ],
                }

            res = game.move(*chosen)
            if traj is not None and res.get('cleared', 0) > 0:
                # Annotate the last entry with what the move cleared
                traj[-1]['cleared'] = int(res['cleared'])
                traj[-1]['score_delta'] = int(game.score - prev_score)
            if rec is not None:
                rec_frame['score'] = int(game.score)
                rec_frame['result'] = {
                    'valid': bool(res.get('valid', True)),
                    'cleared': int(res.get('cleared', 0)),
                    'score': int(game.score - rec_frame['score_before']),
                }
                rec.append(rec_frame)
            prev_score = game.score
        if traj is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir,
                                    f'traj_seed{seed}.json'), 'w') as f:
                json.dump({
                    'seed': seed,
                    'final_score': int(game.score),
                    'final_turn': int(game.turns),
                    'game_over': bool(game.game_over),
                    'metrics': traj,
                }, f)
        if rec is not None:
            with open(record_path, 'w') as f:
                json.dump({
                    'seed': seed,
                    'model': '(eval_parallel policy server)',
                    'final_score': int(game.score),
                    'final_turns': int(game.turns),
                    'died': bool(game.game_over),
                    'frames': rec,
                }, f)
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
                        feature_weights_path, early_stop, q_weight,
                        value_head_path):
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
                early_stop=early_stop, q_weight=q_weight,
                value_head_path=value_head_path)

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
                             use_compile=args.compile,
                             fp16=not args.fp32)
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
                             use_compile=args.compile,
                             value_head_path=args.value_head_path)
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
                          args.q_weight, args.value_head_path))
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
                         show_pol=True, show_mcts=True, show_scores=False):
    print(f"\n{'='*60}\nResults: {len(seeds)} seeds\n{'='*60}\n",
          flush=True)

    def by_seed(rs):
        d = {}
        for seed, score, turns in rs:
            d[seed] = (score, turns)  # one game per seed
        return d

    pol_by = by_seed(pol_results) if pol_results else {}
    mcts_by = by_seed(mcts_results) if mcts_results else {}
    pol_all = [pol_by.get(s, (0, 0))[0] for s in seeds] if show_pol else []
    mcts_all = [mcts_by.get(s, (0, 0))[0] for s in seeds] if show_mcts else []

    if show_scores:
        header = f"{'Seed':>6}"
        if show_pol:
            header += f" | {'Pol':>6}"
        if show_mcts:
            header += f" | {'MCTS':>6}"
        if show_pol and show_mcts:
            header += f" | {'Chg':>5}"
        print(header, flush=True)
        print('-' * len(header), flush=True)
        for s in seeds:
            row = f"{s:>6}"
            if show_pol:
                row += f" | {pol_by.get(s, (0, 0))[0]:>6}"
            if show_mcts:
                row += f" | {mcts_by.get(s, (0, 0))[0]:>6}"
            if show_pol and show_mcts:
                pm = pol_by.get(s, (0, 0))[0]
                mm = mcts_by.get(s, (0, 0))[0]
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

    for label, by, all_scores in [("Pol", pol_by, pol_all),
                                    ("MCTS", mcts_by, mcts_all)]:
        if not all_scores:
            continue
        a = np.array(all_scores)
        n = len(a)
        imin = int(np.argmin(a))
        imax = int(np.argmax(a))
        min_seed, max_seed = seeds[imin], seeds[imax]
        min_turns = by.get(min_seed, (0, 0))[1]
        max_turns = by.get(max_seed, (0, 0))[1]
        print(f"\n  {label} stats ({n} games):", flush=True)
        print(f"    min={a.min():.0f} (seed {min_seed} @ {min_turns}t)  "
              f"max={a.max():.0f} (seed {max_seed} @ {max_turns}t)  "
              f"mean={a.mean():.0f}", flush=True)
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
    p.add_argument('--max-turns', type=int, default=1_000_000,
                   help='Turn cap. Default 1,000,000 = effectively no cap '
                        '(games play to natural death, per project policy; no '
                        'game reaches this — max observed ~70k turns). Lower '
                        'it only to bound wall-clock on a quick smoke.')
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
                        '--feature-value-weights. Works in both local '
                        '(--workers 1) and server (--workers >1) modes; '
                        'in server mode the GPU loop runs the head fused '
                        'with the policy net and ships scalar V per leaf.')
    p.add_argument('--early-stop', action='store_true',
                   help='Exit MCTS early when greedy root child is locked '
                        'in. Eval-only — preserves pick, not visit dist.')
    p.add_argument('--compile', action='store_true',
                   help='torch.compile(reduce-overhead) in the GPU server. '
                        'CUDA only; ignored elsewhere.')
    p.add_argument('--show-scores', action='store_true',
                   help='Print per-seed score table. Default off; summary '
                        'stats (mean/min/max/percentiles/thresholds) always '
                        'print.')
    p.add_argument('--fp32', action='store_true',
                   help='Run the policy inference server in fp32 instead of '
                        'fp16. ~2x slower on MPS but DETERMINISTIC across '
                        'device and batch size — a single-seed re-run '
                        'reproduces the exact game from a multi-seed scan '
                        '(fp16 diverges via batch-dependent kernel choice). '
                        'Use with --record-game to autopsy a specific seed.')
    p.add_argument('--record-game', type=str, default=None,
                   help='Record the full per-turn trajectory (board, '
                        'next_balls, top-K, LEC, n_components, chosen move) '
                        'to this JSON path, in play_gui --replay format. '
                        'Intended for a SINGLE --seeds value. Pair with '
                        '--fp32 for an exactly-reproducible recording.')
    args = p.parse_args()

    # Cap policy games at --max-turns (the worker had no cap → strong games
    # ran to thousands of turns). Threaded via env to avoid touching the
    # worker signature. A game hitting the cap is NOT a natural death.
    os.environ['POLICY_MAX_TURNS'] = str(args.max_turns)

    # Recording is wired into the policy worker via env var (avoids touching
    # the worker process signature). Only meaningful with the policy player.
    if args.record_game:
        os.environ['POLICY_RECORD_GAME'] = args.record_game
        if args.mcts_only:
            raise SystemExit("--record-game needs the policy player; "
                             "remove --mcts-only.")
        if len(args.seeds) != 1:
            print(f"  WARNING: --record-game with {len(args.seeds)} seeds; "
                  f"each seed overwrites the same file. Pass one seed.",
                  flush=True)

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

    # Fail-fast: MCTS needs a leaf-value source. PolicyNet has no NN
    # value head built in; provide either the linear feature evaluator
    # (--feature-value-weights) or the trained NN ValueHead
    # (--value-head-path). Without one, search runs blind on zeros.
    if not args.policy_only and not (args.feature_value_weights
                                      or args.value_head_path):
        raise SystemExit(
            "MCTS requires --feature-value-weights or --value-head-path. "
            "Pass --policy-only to skip MCTS.")
    if args.feature_value_weights and args.value_head_path:
        raise SystemExit(
            "--feature-value-weights and --value-head-path are mutually "
            "exclusive. Pick one Q source.")

    print(f"Evaluation: {total} games (one per seed)", flush=True)
    print(f"Model: {args.model}", flush=True)

    task_seeds = list(seeds)

    # The policy player runs only through the GPU inference server — there is no
    # CPU policy path. Asking for it on cpu used to SILENTLY return 0-score
    # games (every seed 0@0t); crash loudly instead of producing garbage.
    if not args.mcts_only and device_str == 'cpu':
        raise SystemExit(
            "Policy eval requires a GPU device (the inference server has no CPU "
            "path; --device cpu silently yields 0-score games). Use "
            "--device mps or --device cuda. (For CPU, run MCTS-only.)")

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
                         show_mcts=not args.policy_only,
                         show_scores=args.show_scores)


if __name__ == '__main__':
    main()
