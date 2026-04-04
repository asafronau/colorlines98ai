"""Final benchmark: key configurations for before/after comparison.

Tests: det 16w bs=8, batched 4w/8w/16w bs=8, batched 8w/16w bs=16
Each with 20 turns to get stable measurements.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from multiprocessing import Process, Queue


def _mcts_worker(slot_id, obs_shm_name, pol_shm_name, val_shm_name,
                 n_workers, max_batch, req_q, resp_q, result_q,
                 max_score, n_turns):
    torch.set_num_threads(1)
    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE
    from alphatrain.mcts import MCTS
    from game.board import ColorLinesGame

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)

    N, B = n_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf, req_q, resp_q)
    mcts = MCTS(inference_client=client, max_score=max_score,
                num_simulations=400, batch_size=max_batch, top_k=30)

    game = ColorLinesGame(seed=43 + slot_id)
    game.reset()
    t0 = time.time()
    for _ in range(n_turns):
        if game.game_over:
            break
        move = mcts.search(game)
        if move is None:
            break
        game.move(move[0], move[1])
    elapsed = time.time() - t0
    ms_per_turn = elapsed / max(game.turns, 1) * 1000

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
    result_q.put((slot_id, game.turns, ms_per_turn, elapsed))


def run_config(n_workers, batch_size, deterministic, n_turns=20):
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    from alphatrain.inference_server import InferenceServer
    server = InferenceServer(model_path, n_workers,
                             max_batch_per_worker=batch_size,
                             deterministic=deterministic)
    server.start()
    time.sleep(1.5)

    result_q = Queue()
    procs = []
    t0 = time.time()
    for i in range(n_workers):
        p = Process(target=_mcts_worker, args=(
            i, server._obs_shm.name, server._pol_shm.name,
            server._val_shm.name, n_workers, batch_size,
            server.request_queue, server.response_queues[i],
            result_q, max_score, n_turns))
        p.start()
        procs.append(p)

    results = []
    for _ in range(n_workers):
        results.append(result_q.get(timeout=300))

    for p in procs:
        p.join()
    wall = time.time() - t0

    total_turns = sum(r[1] for r in results)
    avg_ms = np.mean([r[2] for r in results])
    throughput = total_turns / wall

    server.shutdown()
    return throughput, avg_ms


def main():
    print("=" * 65, flush=True)
    print("Final Benchmark: 400 sims, 20 turns per worker", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Config':>30} | {'turns/s':>8} | {'ms/turn':>8} | {'vs 1w':>6}",
          flush=True)
    print("-" * 65, flush=True)

    # Run 3 times each and take best for stability
    configs = [
        ("det 1w bs=8",  1, 8, True),
        ("det 16w bs=8", 16, 8, True),
        ("det 1w bs=16", 1, 16, True),
        ("det 16w bs=16", 16, 16, True),
        ("batched 1w bs=8", 1, 8, False),
        ("batched 4w bs=8", 4, 8, False),
        ("batched 8w bs=8", 8, 8, False),
        ("batched 16w bs=8", 16, 8, False),
        ("batched 1w bs=16", 1, 16, False),
        ("batched 8w bs=16", 8, 16, False),
        ("batched 16w bs=16", 16, 16, False),
    ]

    baselines = {}  # (det, bs) -> 1w throughput

    for name, nw, bs, det in configs:
        tp, ms = run_config(nw, bs, det)
        key = (det, bs)
        if nw == 1:
            baselines[key] = tp
        base = baselines.get(key, tp)
        speedup = tp / base
        print(f"{name:>30} | {tp:>7.1f}  | {ms:>7.0f}ms | {speedup:>5.1f}x",
              flush=True)

    print("=" * 65, flush=True)


if __name__ == '__main__':
    main()
