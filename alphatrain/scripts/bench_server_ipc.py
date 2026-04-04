"""Profile inference server IPC overhead in isolation.

Measures:
1. Queue.put() + Queue.get() round-trip time
2. Shared memory copy time (obs write, result read)
3. torch.from_numpy().half() conversion time
4. GPU forward pass time at various batch sizes
5. Result copy-back time (float().cpu().numpy())
6. End-to-end server latency per eval

Usage:
    python -m alphatrain.scripts.bench_server_ipc
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from multiprocessing import Queue, Process
from queue import Empty


def bench_queue_roundtrip():
    """Measure Queue.put + Queue.get latency."""
    q = Queue()
    n = 5000

    # Warmup
    for _ in range(100):
        q.put((0, 8))
        q.get()

    t0 = time.perf_counter()
    for _ in range(n):
        q.put((0, 8))
        q.get()
    elapsed = time.perf_counter() - t0
    print(f"  Queue put+get (tuple):    {elapsed/n*1e6:.1f} us/roundtrip", flush=True)

    # Compare with int signal
    t0 = time.perf_counter()
    for _ in range(n):
        q.put(1)
        q.get()
    elapsed = time.perf_counter() - t0
    print(f"  Queue put+get (int):      {elapsed/n*1e6:.1f} us/roundtrip", flush=True)


def bench_shm_copy():
    """Measure shared-memory array copy overhead."""
    from multiprocessing.shared_memory import SharedMemory

    B = 16
    obs_size = B * 18 * 9 * 9 * 4  # float32
    pol_size = B * 6561 * 4
    val_size = B * 4

    shm_obs = SharedMemory(create=True, size=obs_size)
    shm_pol = SharedMemory(create=True, size=pol_size)
    shm_val = SharedMemory(create=True, size=val_size)

    obs_buf = np.ndarray((B, 18, 9, 9), dtype=np.float32, buffer=shm_obs.buf)
    pol_buf = np.ndarray((B, 6561), dtype=np.float32, buffer=shm_pol.buf)
    val_buf = np.ndarray((B,), dtype=np.float32, buffer=shm_val.buf)

    obs_np = np.random.randn(B, 18, 9, 9).astype(np.float32)
    pol_np = np.random.randn(B, 6561).astype(np.float32)
    val_np = np.random.randn(B).astype(np.float32)

    n = 2000

    # Worker side: write obs to shared memory
    t0 = time.perf_counter()
    for _ in range(n):
        obs_buf[:8] = obs_np[:8]
    elapsed = time.perf_counter() - t0
    print(f"  SHM obs write (bs=8):     {elapsed/n*1e6:.1f} us", flush=True)

    t0 = time.perf_counter()
    for _ in range(n):
        obs_buf[:16] = obs_np[:16]
    elapsed = time.perf_counter() - t0
    print(f"  SHM obs write (bs=16):    {elapsed/n*1e6:.1f} us", flush=True)

    # GPU side: read obs from shared memory -> GPU tensor
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    gpu_obs = torch.empty(B, 18, 9, 9, device=device, dtype=torch.float16)

    # Method 1: from_numpy().half() (current)
    t0 = time.perf_counter()
    for _ in range(n):
        gpu_obs[:8] = torch.from_numpy(obs_buf[:8]).half()
    elapsed = time.perf_counter() - t0
    print(f"  from_numpy+half (bs=8):   {elapsed/n*1e6:.1f} us", flush=True)

    # Method 2: from_numpy().to(device, dtype)
    t0 = time.perf_counter()
    for _ in range(n):
        gpu_obs[:8] = torch.from_numpy(obs_buf[:8]).to(device=device, dtype=torch.float16)
    elapsed = time.perf_counter() - t0
    print(f"  from_numpy+to (bs=8):     {elapsed/n*1e6:.1f} us", flush=True)

    # Method 3: Pre-allocated CPU tensor + pin (for CUDA)
    cpu_obs = torch.empty(B, 18, 9, 9, dtype=torch.float32)
    t0 = time.perf_counter()
    for _ in range(n):
        np.copyto(cpu_obs.numpy()[:8], obs_buf[:8])
        gpu_obs[:8] = cpu_obs[:8].half().to(device)
    elapsed = time.perf_counter() - t0
    print(f"  copyto+half+to (bs=8):    {elapsed/n*1e6:.1f} us", flush=True)

    # GPU side: write results back
    pol_out = torch.randn(B, 6561, device=device, dtype=torch.float16)
    val_out = torch.randn(B, device=device, dtype=torch.float16)

    # Method 1: float().cpu().numpy() (current)
    t0 = time.perf_counter()
    for _ in range(n):
        pol_buf[:8] = pol_out[:8].float().cpu().numpy()
        val_buf[:8] = val_out[:8].float().cpu().numpy()
    elapsed = time.perf_counter() - t0
    print(f"  pol+val copyback (bs=8):  {elapsed/n*1e6:.1f} us", flush=True)

    # Method 2: Pre-allocated CPU tensor
    pol_cpu = torch.empty(B, 6561, dtype=torch.float32)
    val_cpu = torch.empty(B, dtype=torch.float32)
    t0 = time.perf_counter()
    for _ in range(n):
        torch.float32
        pol_cpu[:8].copy_(pol_out[:8])
        np.copyto(pol_buf[:8], pol_cpu[:8].numpy())
        val_cpu[:8].copy_(val_out[:8])
        np.copyto(val_buf[:8], val_cpu[:8].numpy())
    elapsed = time.perf_counter() - t0
    print(f"  preallocated copyback(8): {elapsed/n*1e6:.1f} us", flush=True)

    # Cleanup
    shm_obs.close(); shm_obs.unlink()
    shm_pol.close(); shm_pol.unlink()
    shm_val.close(); shm_val.unlink()


def bench_gpu_forward():
    """Measure raw GPU forward pass time."""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("  No GPU available, skipping", flush=True)
        return

    from alphatrain.evaluate import load_model
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}", flush=True)
        return

    net, max_score = load_model(model_path, device, fp16=True, jit_trace=True)

    # Warmup
    dummy = torch.randn(8, 18, 9, 9, device=device, dtype=torch.float16)
    for _ in range(10):
        with torch.inference_mode():
            net(dummy)

    n = 500
    for bs in [1, 4, 8, 16, 32, 64, 128]:
        x = torch.randn(bs, 18, 9, 9, device=device, dtype=torch.float16)

        t0 = time.perf_counter()
        for _ in range(n):
            with torch.inference_mode():
                pol, val = net(x)
                _ = net.predict_value(val, max_val=max_score)
        elapsed = time.perf_counter() - t0
        evals_s = bs * n / elapsed
        print(f"  GPU forward bs={bs:>3}: {elapsed/n*1e3:.2f} ms/batch, "
              f"{evals_s:.0f} evals/s, {elapsed/n/bs*1e6:.0f} us/eval", flush=True)


OBS_SHAPE_BENCH = (18, 9, 9)
POL_SIZE_BENCH = 6561
N_REQ_BENCH = 200

def _e2e_worker(slot_id, obs_shm_name, pol_shm_name, val_shm_name,
                n_workers, max_batch, req_q, resp_q, result_q):
    torch.set_num_threads(1)
    from multiprocessing.shared_memory import SharedMemory

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)

    N, B = n_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE_BENCH, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE_BENCH), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    obs_buf[slot_id, :8] = np.random.randn(8, 18, 9, 9).astype(np.float32)

    # Warmup
    for _ in range(5):
        req_q.put((slot_id, 8))
        resp_q.get()

    t0 = time.perf_counter()
    for _ in range(N_REQ_BENCH):
        req_q.put((slot_id, 8))
        resp_q.get()
    elapsed = time.perf_counter() - t0

    obs_shm.close()
    pol_shm.close()
    val_shm.close()
    result_q.put((slot_id, elapsed, N_REQ_BENCH * 8))


def bench_e2e_server():
    """Measure end-to-end throughput with actual server."""
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}", flush=True)
        return

    from alphatrain.inference_server import InferenceServer
    from multiprocessing import Queue as MPQ

    for det in [True, False]:
        mode = "deterministic" if det else "batched"
        for n_workers in [1, 4, 8, 16]:
            server = InferenceServer(model_path, n_workers,
                                     max_batch_per_worker=8,
                                     deterministic=det)
            server.start()
            time.sleep(1.5)

            result_q = MPQ()
            procs = []
            for i in range(n_workers):
                p = Process(target=_e2e_worker, args=(
                    i, server._obs_shm.name, server._pol_shm.name,
                    server._val_shm.name, n_workers, 8,
                    server.request_queue, server.response_queues[i],
                    result_q))
                p.start()
                procs.append(p)

            total_evals = 0
            max_elapsed = 0
            for _ in range(n_workers):
                sid, elapsed, evals = result_q.get(timeout=60)
                total_evals += evals
                max_elapsed = max(max_elapsed, elapsed)

            for p in procs:
                p.join()
            server.shutdown()

            throughput = total_evals / max_elapsed
            print(f"  {mode:>13} {n_workers:>2}w: {throughput:.0f} evals/s "
                  f"({max_elapsed:.1f}s for {total_evals} evals)", flush=True)


def main():
    print("=== Queue round-trip ===", flush=True)
    bench_queue_roundtrip()

    print("\n=== Shared memory copy ===", flush=True)
    bench_shm_copy()

    print("\n=== GPU forward pass ===", flush=True)
    bench_gpu_forward()

    print("\n=== End-to-end server throughput ===", flush=True)
    bench_e2e_server()


if __name__ == '__main__':
    main()
