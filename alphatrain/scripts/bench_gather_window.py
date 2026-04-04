"""Test different gather window durations for batched mode.

Usage:
    python -m alphatrain.scripts.bench_gather_window
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


def run_test(n_workers, batch_size, gather_ms, n_turns=20):
    """Run with specific gather window."""
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    # Monkey-patch the gather timeout
    import alphatrain.inference_server as srv
    original_code = srv._gpu_loop  # save reference

    from alphatrain.inference_server import InferenceServer
    server = InferenceServer(model_path, n_workers,
                             max_batch_per_worker=batch_size,
                             deterministic=False)
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
    return throughput, avg_ms, wall


def main():
    print("Gather window benchmark (batched mode, 400 sims, 20 turns)", flush=True)
    print(f"{'Workers':>8} {'BS':>4} | {'Throughput':>12} | {'ms/turn':>8} | {'Wall':>6}",
          flush=True)
    print("-" * 55, flush=True)

    # The gather timeout is hardcoded in inference_server.py.
    # For now just test different worker counts to see the effect.
    for n_workers in [4, 8, 12, 16]:
        for bs in [8, 16]:
            tp, ms, wall = run_test(n_workers, bs, 0.5)
            print(f"{n_workers:>8} {bs:>4} | {tp:>8.1f} t/s  | {ms:>7.0f}ms | {wall:>5.1f}s",
                  flush=True)


if __name__ == '__main__':
    main()
