"""Test fp16 vs fp32 policy transfer overhead."""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory


def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        print("No MPS", flush=True)
        return

    POL_SIZE = 6561
    n = 1000

    for bs in [8, 16]:
        pol_gpu = torch.randn(bs, POL_SIZE, device=device, dtype=torch.float16)

        # Float32 shm
        shm32 = SharedMemory(create=True, size=bs * POL_SIZE * 4)
        pol32 = np.ndarray((bs, POL_SIZE), dtype=np.float32, buffer=shm32.buf)

        # Float16 shm
        shm16 = SharedMemory(create=True, size=bs * POL_SIZE * 2)
        pol16 = np.ndarray((bs, POL_SIZE), dtype=np.float16, buffer=shm16.buf)

        # Warmup
        for _ in range(20):
            pol32[:] = pol_gpu.float().cpu().numpy()
            pol16[:] = pol_gpu.cpu().numpy()

        # Method 1: float().cpu().numpy() -> fp32 SHM (current)
        t0 = time.perf_counter()
        for _ in range(n):
            pol32[:] = pol_gpu.float().cpu().numpy()
        t_fp32 = time.perf_counter() - t0
        print(f"  bs={bs} float().cpu().numpy() -> fp32 SHM: {t_fp32/n*1e6:.0f} us",
              flush=True)

        # Method 2: cpu().numpy() -> fp16 SHM (half bandwidth)
        t0 = time.perf_counter()
        for _ in range(n):
            pol16[:] = pol_gpu.cpu().numpy()
        t_fp16 = time.perf_counter() - t0
        print(f"  bs={bs} cpu().numpy() -> fp16 SHM:        {t_fp16/n*1e6:.0f} us "
              f"({t_fp32/t_fp16:.1f}x faster)", flush=True)

        # Method 3: Direct half -> fp32 via np.copyto with casting
        t0 = time.perf_counter()
        for _ in range(n):
            np_16 = pol_gpu.cpu().numpy()
            np.copyto(pol32, np_16, casting='same_kind')
        t_cast = time.perf_counter() - t0
        print(f"  bs={bs} cpu().numpy() -> cast fp32 SHM:   {t_cast/n*1e6:.0f} us",
              flush=True)

        # Method 4: Just the .cpu() step alone
        t0 = time.perf_counter()
        for _ in range(n):
            _ = pol_gpu.cpu()
        t_cpu_only = time.perf_counter() - t0
        print(f"  bs={bs} .cpu() alone:                     {t_cpu_only/n*1e6:.0f} us",
              flush=True)

        # Method 5: .float() on GPU, then .cpu()
        t0 = time.perf_counter()
        for _ in range(n):
            _ = pol_gpu.float().cpu()
        t_float_cpu = time.perf_counter() - t0
        print(f"  bs={bs} .float().cpu():                   {t_float_cpu/n*1e6:.0f} us",
              flush=True)

        print(flush=True)

        shm32.close(); shm32.unlink()
        shm16.close(); shm16.unlink()


if __name__ == '__main__':
    main()
