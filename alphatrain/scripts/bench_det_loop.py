"""Microbenchmark the deterministic loop iterations on MPS.

Measures each step of the per-request processing:
1. obs copy to GPU
2. net forward
3. predict_value
4. result copy to SHM
5. queue signal

Usage:
    python -m alphatrain.scripts.bench_det_loop
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory


def main():
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    if not os.path.exists(model_path):
        print("Model not found", flush=True)
        return

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        print("No MPS device", flush=True)
        return

    from alphatrain.evaluate import load_model
    net, max_score = load_model(model_path, device)
    net = net.half()
    dummy = torch.randn(1, 18, 9, 9, device=device).half()
    net_traced = torch.jit.trace(net, dummy)

    B = 16
    N = 1
    POL_SIZE = 6561
    gpu_obs = torch.empty(B, 18, 9, 9, device=device, dtype=torch.float16)

    shm_obs = SharedMemory(create=True, size=N * B * 18 * 9 * 9 * 4)
    shm_pol = SharedMemory(create=True, size=N * B * POL_SIZE * 4)
    shm_val = SharedMemory(create=True, size=N * B * 4)

    obs_buf = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=shm_obs.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=shm_pol.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=shm_val.buf)

    obs_buf[0] = np.random.randn(B, 18, 9, 9).astype(np.float32)

    n = 500
    for bs in [8, 16]:
        # Warmup
        for _ in range(20):
            gpu_obs[:bs] = torch.from_numpy(obs_buf[0, :bs]).half()
            with torch.inference_mode():
                pol, val = net_traced(gpu_obs[:bs])
                v = net.predict_value(val, max_val=max_score)
            pol_buf[0, :bs] = pol.float().cpu().numpy()
            val_buf[0, :bs] = v.float().cpu().numpy()

        # Measure each step
        t_copy_in = 0
        t_forward = 0
        t_predict = 0
        t_copy_out_pol = 0
        t_copy_out_val = 0
        t_total = 0

        t0_total = time.perf_counter()
        for _ in range(n):
            t0 = time.perf_counter()
            gpu_obs[:bs] = torch.from_numpy(obs_buf[0, :bs]).half()
            t1 = time.perf_counter()
            t_copy_in += t1 - t0

            with torch.inference_mode():
                t0 = time.perf_counter()
                pol, val = net_traced(gpu_obs[:bs])
                t1 = time.perf_counter()
                t_forward += t1 - t0

                t0 = time.perf_counter()
                v = net.predict_value(val, max_val=max_score)
                t1 = time.perf_counter()
                t_predict += t1 - t0

            t0 = time.perf_counter()
            pol_buf[0, :bs] = pol.float().cpu().numpy()
            t1 = time.perf_counter()
            t_copy_out_pol += t1 - t0

            t0 = time.perf_counter()
            val_buf[0, :bs] = v.float().cpu().numpy()
            t1 = time.perf_counter()
            t_copy_out_val += t1 - t0

        t_total = time.perf_counter() - t0_total

        print(f"\nbs={bs}, {n} iterations:", flush=True)
        print(f"  copy_in:      {t_copy_in/n*1e6:>6.0f} us  ({t_copy_in/t_total*100:5.1f}%)",
              flush=True)
        print(f"  forward:      {t_forward/n*1e6:>6.0f} us  ({t_forward/t_total*100:5.1f}%)",
              flush=True)
        print(f"  predict_val:  {t_predict/n*1e6:>6.0f} us  ({t_predict/t_total*100:5.1f}%)",
              flush=True)
        print(f"  copy_out_pol: {t_copy_out_pol/n*1e6:>6.0f} us  ({t_copy_out_pol/t_total*100:5.1f}%)",
              flush=True)
        print(f"  copy_out_val: {t_copy_out_val/n*1e6:>6.0f} us  ({t_copy_out_val/t_total*100:5.1f}%)",
              flush=True)
        other = t_total - t_copy_in - t_forward - t_predict - t_copy_out_pol - t_copy_out_val
        print(f"  other/pyloop: {other/n*1e6:>6.0f} us  ({other/t_total*100:5.1f}%)",
              flush=True)
        print(f"  TOTAL:        {t_total/n*1e6:>6.0f} us  ({t_total/n*1e3:.2f} ms)", flush=True)
        print(f"  evals/s:      {bs*n/t_total:.0f}", flush=True)

    shm_obs.close(); shm_obs.unlink()
    shm_pol.close(); shm_pol.unlink()
    shm_val.close(); shm_val.unlink()


if __name__ == '__main__':
    main()
