"""Test fp16 policy transfer with forward pass in pipeline."""

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
        print("No MPS", flush=True)
        return

    from alphatrain.evaluate import load_model
    net, max_score = load_model(model_path, device)
    net = net.half()
    dummy = torch.randn(1, 18, 9, 9, device=device).half()
    net_traced = torch.jit.trace(net, dummy)

    POL_SIZE = 6561
    n = 500

    for bs in [8, 16]:
        gpu_obs = torch.randn(bs, 18, 9, 9, device=device, dtype=torch.float16)

        shm32 = SharedMemory(create=True, size=bs * POL_SIZE * 4)
        pol32 = np.ndarray((bs, POL_SIZE), dtype=np.float32, buffer=shm32.buf)

        shm16 = SharedMemory(create=True, size=bs * POL_SIZE * 2)
        pol16 = np.ndarray((bs, POL_SIZE), dtype=np.float16, buffer=shm16.buf)

        val32 = np.empty(bs, dtype=np.float32)

        # Warmup
        for _ in range(20):
            with torch.inference_mode():
                pol, val = net_traced(gpu_obs)
                v = net.predict_value(val, max_val=max_score)
            pol32[:] = pol.float().cpu().numpy()
            val32[:] = v.float().cpu().numpy()

        # Method 1: Current (fp32 transfer)
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.inference_mode():
                pol, val = net_traced(gpu_obs)
                v = net.predict_value(val, max_val=max_score)
            pol32[:] = pol.float().cpu().numpy()
            val32[:] = v.float().cpu().numpy()
        t_fp32 = time.perf_counter() - t0
        print(f"  bs={bs} forward + fp32 transfer:  {t_fp32/n*1e6:.0f} us "
              f"({bs*n/t_fp32:.0f} evals/s)", flush=True)

        # Method 2: fp16 transfer (skip .float() conversion)
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.inference_mode():
                pol, val = net_traced(gpu_obs)
                v = net.predict_value(val, max_val=max_score)
            pol16[:] = pol.cpu().numpy()
            val32[:] = v.float().cpu().numpy()
        t_fp16 = time.perf_counter() - t0
        print(f"  bs={bs} forward + fp16 transfer:  {t_fp16/n*1e6:.0f} us "
              f"({bs*n/t_fp16:.0f} evals/s) {t_fp32/t_fp16:.2f}x", flush=True)

        # Method 3: fp16 transfer + cast to fp32 in worker
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.inference_mode():
                pol, val = net_traced(gpu_obs)
                v = net.predict_value(val, max_val=max_score)
            pol16[:] = pol.cpu().numpy()
            val32[:] = v.float().cpu().numpy()
            # Simulate worker reading fp16 -> fp32
            _ = pol16.astype(np.float32)
        t_fp16_cast = time.perf_counter() - t0
        print(f"  bs={bs} forward + fp16 + cast:    {t_fp16_cast/n*1e6:.0f} us "
              f"({bs*n/t_fp16_cast:.0f} evals/s) {t_fp32/t_fp16_cast:.2f}x", flush=True)

        print(flush=True)

        shm32.close(); shm32.unlink()
        shm16.close(); shm16.unlink()


if __name__ == '__main__':
    main()
