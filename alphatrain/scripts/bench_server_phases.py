"""Instrument the GPU server loop to measure time in each phase.

Uses a patched version of _gpu_loop with timing counters.
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory
from queue import Empty


def _instrumented_gpu_loop(model_path, device_str, num_workers, max_batch,
                           obs_shm_name, pol_shm_name, val_shm_name,
                           request_queue, response_queues, value_model_path,
                           deterministic):
    """GPU loop with per-phase timing."""
    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)

    from alphatrain.inference_server import OBS_SHAPE, POL_SIZE

    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    device = torch.device(device_str)
    from alphatrain.evaluate import load_model
    net, max_score = load_model(model_path, device)
    net = net.half()
    dummy = torch.randn(1, 18, 9, 9, device=device).half()
    net_traced = torch.jit.trace(net, dummy)

    max_total = num_workers * max_batch
    gpu_obs = torch.empty(max_total, 18, 9, 9, device=device, dtype=torch.float16)
    obs_staging = np.empty((max_total,) + OBS_SHAPE, dtype=np.float32)

    GPU_BATCH_CAP = 128
    GATHER_TIMEOUT = 0.0005 if num_workers > 1 else 0.0
    GATHER_MIN_EVALS = min(32, num_workers * max_batch)

    print(f"Instrumented GPU loop ready ({device_str})", flush=True)

    t_gather = 0
    t_copy_in = 0
    t_forward = 0
    t_copy_out = 0
    t_signal = 0
    total_evals = 0
    total_fwd = 0
    batch_sizes = []

    _queue_get = request_queue.get
    _queue_get_nowait = request_queue.get_nowait

    while True:
        t0 = time.perf_counter()
        item = _queue_get()
        if item is None:
            break

        pending = [item]
        pending_evals = item[1]

        # Gather phase
        if GATHER_TIMEOUT > 0 and pending_evals < GATHER_MIN_EVALS:
            deadline = time.monotonic() + GATHER_TIMEOUT
            while pending_evals < GPU_BATCH_CAP:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = request_queue.get(timeout=remaining)
                    if item is None:
                        request_queue.put(None)
                        break
                    pending.append(item)
                    pending_evals += item[1]
                except Empty:
                    break

        while pending_evals < GPU_BATCH_CAP:
            try:
                item = _queue_get_nowait()
                if item is None:
                    request_queue.put(None)
                    break
                pending.append(item)
                pending_evals += item[1]
            except Empty:
                break

        t1 = time.perf_counter()
        t_gather += t1 - t0

        # Copy-in phase
        total_count = 0
        for slot_id, count in pending:
            np.copyto(obs_staging[total_count:total_count + count],
                      obs_buf[slot_id, :count])
            total_count += count
        gpu_obs[:total_count] = torch.from_numpy(obs_staging[:total_count]).half()
        t2 = time.perf_counter()
        t_copy_in += t2 - t1

        # Forward phase
        with torch.inference_mode():
            pol_logits, val_logits = net_traced(gpu_obs[:total_count])
            values = net.predict_value(val_logits, max_val=max_score)
        t3 = time.perf_counter()
        t_forward += t3 - t2

        # Copy-out phase
        pol_cpu = pol_logits.float().cpu()
        val_cpu = values.float().cpu()
        pol_np = pol_cpu.numpy()
        val_np = val_cpu.numpy()
        offset = 0
        for slot_id, count in pending:
            np.copyto(pol_buf[slot_id, :count], pol_np[offset:offset + count])
            np.copyto(val_buf[slot_id, :count], val_np[offset:offset + count])
            offset += count
        t4 = time.perf_counter()
        t_copy_out += t4 - t3

        # Signal phase
        for slot_id, count in pending:
            response_queues[slot_id].put(1)
        t5 = time.perf_counter()
        t_signal += t5 - t4

        total_evals += total_count
        total_fwd += 1
        batch_sizes.append(total_count)

        if total_fwd % 100 == 0:
            elapsed = time.time()
            print(f"  [GPU] {total_evals} evals, {total_fwd} fwd, "
                  f"avg_bs={np.mean(batch_sizes[-100:]):.0f}", flush=True)

    total_time = t_gather + t_copy_in + t_forward + t_copy_out + t_signal
    print(f"\n=== GPU Server Phase Breakdown ({total_fwd} forwards, "
          f"{total_evals} evals) ===", flush=True)
    print(f"  Gather:   {t_gather*1000:.0f}ms ({t_gather/total_time*100:.1f}%) "
          f"-- waiting for requests", flush=True)
    print(f"  Copy-in:  {t_copy_in*1000:.0f}ms ({t_copy_in/total_time*100:.1f}%) "
          f"-- SHM -> GPU", flush=True)
    print(f"  Forward:  {t_forward*1000:.0f}ms ({t_forward/total_time*100:.1f}%) "
          f"-- GPU compute", flush=True)
    print(f"  Copy-out: {t_copy_out*1000:.0f}ms ({t_copy_out/total_time*100:.1f}%) "
          f"-- GPU -> SHM", flush=True)
    print(f"  Signal:   {t_signal*1000:.0f}ms ({t_signal/total_time*100:.1f}%) "
          f"-- queue.put", flush=True)
    print(f"  Total:    {total_time*1000:.0f}ms", flush=True)
    print(f"  Avg bs:   {np.mean(batch_sizes):.0f} "
          f"(min={np.min(batch_sizes)}, max={np.max(batch_sizes)})", flush=True)
    print(f"  Throughput: {total_evals/total_time:.0f} evals/s", flush=True)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


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

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
    result_q.put((slot_id, game.turns, elapsed))


def main():
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    from alphatrain.inference_server import InferenceServer, OBS_SIZE, POL_SIZE

    n_workers = 16
    bs = 8
    n_turns = 15

    # Create server manually with instrumented loop
    server = InferenceServer.__new__(InferenceServer)
    server.model_path = model_path
    server.value_model_path = None
    server.deterministic = False
    server.num_workers = n_workers
    server.max_batch = bs
    server.device_str = 'mps'

    N, B = n_workers, bs
    server._obs_shm = SharedMemory(create=True, size=max(1, N * B * OBS_SIZE * 4))
    server._pol_shm = SharedMemory(create=True, size=max(1, N * B * POL_SIZE * 4))
    server._val_shm = SharedMemory(create=True, size=max(1, N * B * 4))

    server.obs_buf = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=server._obs_shm.buf)
    server.pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=server._pol_shm.buf)
    server.val_buf = np.ndarray((N, B), dtype=np.float32, buffer=server._val_shm.buf)

    server.request_queue = Queue()
    server.response_queues = [Queue() for _ in range(n_workers)]

    server._process = Process(
        target=_instrumented_gpu_loop,
        args=(model_path, 'mps', n_workers, bs,
              server._obs_shm.name, server._pol_shm.name, server._val_shm.name,
              server.request_queue, server.response_queues, None, False),
        daemon=True)
    server._process.start()
    time.sleep(1.5)

    result_q = Queue()
    procs = []
    t0 = time.time()
    for i in range(n_workers):
        p = Process(target=_mcts_worker, args=(
            i, server._obs_shm.name, server._pol_shm.name,
            server._val_shm.name, n_workers, bs,
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
    print(f"\nResult: {total_turns} turns in {wall:.1f}s = "
          f"{total_turns/wall:.1f} turns/s", flush=True)

    server.shutdown()


if __name__ == '__main__':
    main()
