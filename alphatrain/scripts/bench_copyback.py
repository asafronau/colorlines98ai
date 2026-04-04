"""Microbenchmark: compare result copy-back strategies on MPS.

Measures:
1. Current: pol_logits.float().cpu().numpy() (allocates new tensors)
2. Pre-alloc: pol_cpu_buf[:n].copy_(pol_logits); np.copyto(dst, src)
3. Direct: pol_buf[...] = pol_logits.float().cpu().numpy() (assign to shm)
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory


def main():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    POL_SIZE = 6561
    n = 2000

    for bs in [8, 16, 64, 128]:
        # Simulate GPU output
        pol_gpu = torch.randn(bs, POL_SIZE, device=device, dtype=torch.float16)
        val_gpu = torch.randn(bs, device=device, dtype=torch.float16)

        # Shared memory destination (simulating pol_buf)
        shm = SharedMemory(create=True, size=bs * POL_SIZE * 4 + bs * 4)
        pol_dst = np.ndarray((bs, POL_SIZE), dtype=np.float32, buffer=shm.buf)
        val_dst = np.ndarray((bs,), dtype=np.float32,
                             buffer=shm.buf[bs * POL_SIZE * 4:])

        # Pre-allocated CPU buffers
        pol_cpu = torch.empty(bs, POL_SIZE, dtype=torch.float32)
        val_cpu = torch.empty(bs, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            pol_dst[:] = pol_gpu.float().cpu().numpy()
            val_dst[:] = val_gpu.float().cpu().numpy()

        # Method 1: .float().cpu().numpy() -> assign to SHM
        t0 = time.perf_counter()
        for _ in range(n):
            pol_dst[:] = pol_gpu.float().cpu().numpy()
            val_dst[:] = val_gpu.float().cpu().numpy()
        t1 = time.perf_counter() - t0
        print(f"  bs={bs:>3} float().cpu().numpy():       {t1/n*1e6:.1f} us", flush=True)

        # Method 2: copy_ to pre-alloc + np.copyto
        t0 = time.perf_counter()
        for _ in range(n):
            pol_cpu.copy_(pol_gpu)
            np.copyto(pol_dst, pol_cpu.numpy())
            val_cpu.copy_(val_gpu)
            np.copyto(val_dst, val_cpu.numpy())
        t2 = time.perf_counter() - t0
        print(f"  bs={bs:>3} copy_ + np.copyto:          {t2/n*1e6:.1f} us", flush=True)

        # Method 3: Just .float().cpu() to preallocated
        t0 = time.perf_counter()
        for _ in range(n):
            torch.float32  # noop reference
            p = pol_gpu.float().cpu()
            np.copyto(pol_dst, p.numpy())
            v = val_gpu.float().cpu()
            np.copyto(val_dst, v.numpy())
        t3 = time.perf_counter() - t0
        print(f"  bs={bs:>3} float().cpu() + np.copyto:  {t3/n*1e6:.1f} us", flush=True)

        # Method 4: Use float() on GPU, then cpu().numpy()
        t0 = time.perf_counter()
        for _ in range(n):
            pol_f = pol_gpu.float()
            pol_c = pol_f.cpu()
            np.copyto(pol_dst, pol_c.numpy())
            val_f = val_gpu.float()
            val_c = val_f.cpu()
            np.copyto(val_dst, val_c.numpy())
        t4 = time.perf_counter() - t0
        print(f"  bs={bs:>3} float() gpu + cpu() + copyto: {t4/n*1e6:.1f} us", flush=True)

        print(flush=True)

        shm.close()
        shm.unlink()


if __name__ == '__main__':
    main()
