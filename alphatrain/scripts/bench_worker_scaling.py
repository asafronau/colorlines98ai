"""Find optimal worker count for GPU inference server.

Tests 1, 2, 4, 8, 18 workers on the same seed, measures ms/turn and wall-clock.

Usage:
    python -m alphatrain.scripts.bench_worker_scaling
"""

import time
import numpy as np
import torch
from multiprocessing import Process, Queue, cpu_count
from alphatrain.inference_server import InferenceServer


def _worker(seed, slot_id, obs_shm_name, pol_shm_name, val_shm_name,
            num_workers, max_batch, request_queue, response_queue,
            num_sims, max_score, result_queue):
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
                num_simulations=num_sims, top_k=30, batch_size=max_batch)

    game = ColorLinesGame(seed=seed)
    game.reset()
    t0 = time.time()
    # Play exactly 50 turns (for consistent comparison)
    for _ in range(50):
        if game.game_over:
            break
        move = mcts.search(game)
        if move is None:
            break
        game.move(move[0], move[1])
    elapsed = time.time() - t0
    turns = game.turns
    ms_per_turn = elapsed / max(turns, 1) * 1000

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
    result_queue.put((turns, ms_per_turn, elapsed))


def main():
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    batch_size = 32
    num_sims = 400

    print(f"Worker scaling benchmark (seed=43, 50 turns, {num_sims} sims, bs={batch_size})\n",
          flush=True)
    print(f"{'Workers':>8} | {'ms/turn':>8} | {'Wall-clock':>10} | {'Throughput':>12}", flush=True)
    print("-" * 50, flush=True)

    for n_workers in [1, 2, 4, 8, 18]:
        server = InferenceServer(model_path, n_workers, max_batch_per_worker=batch_size)
        server.start()
        time.sleep(1)

        t0 = time.time()
        processes = []
        result_queues = []
        for i in range(n_workers):
            rq = Queue()
            result_queues.append(rq)
            proc = Process(target=_worker, args=(
                43, i, server._obs_shm.name, server._pol_shm.name,
                server._val_shm.name, n_workers, batch_size,
                server.request_queue, server.response_queues[i],
                num_sims, max_score, rq))
            proc.start()
            processes.append(proc)

        results = []
        for proc, rq in zip(processes, result_queues):
            results.append(rq.get(timeout=300))
            proc.join()
        wall = time.time() - t0

        avg_ms = np.mean([r[1] for r in results])
        total_turns = sum(r[0] for r in results)
        throughput = total_turns / wall

        print(f"{n_workers:>8} | {avg_ms:>7.0f}ms | {wall:>9.1f}s | "
              f"{throughput:>8.1f} turns/s", flush=True)

        server.shutdown()

    print(flush=True)


if __name__ == '__main__':
    main()
