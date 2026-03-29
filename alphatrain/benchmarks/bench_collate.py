"""Benchmark dataset collate function speed.

Measures observation building throughput at different batch sizes.
Target: >10K samples/s on MPS, >40K on CUDA.
"""

import time
import argparse
import torch
from alphatrain.dataset import TensorDatasetGPU
from torch.utils.data import DataLoader


def bench_collate(tensor_file, device_name, batch_sizes=(256, 1024, 4096)):
    ds = TensorDatasetGPU(tensor_file, augment=True, device=device_name)

    # Warmup JIT
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0,
                         collate_fn=ds.collate)
    batch = next(iter(loader))
    print(f"Warmup done. obs={batch[0].shape}")
    if device_name == 'mps':
        torch.mps.synchronize()

    for bs in batch_sizes:
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                             collate_fn=ds.collate)
        it = iter(loader)
        next(it)  # warmup
        if device_name == 'mps':
            torch.mps.synchronize()
        elif device_name == 'cuda':
            torch.cuda.synchronize()

        N = 10
        t0 = time.perf_counter()
        for _ in range(N):
            next(it)
        if device_name == 'mps':
            torch.mps.synchronize()
        elif device_name == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        ms = (t1 - t0) / N * 1000
        sps = bs / (ms / 1000)
        print(f"  batch={bs}: {ms:.0f}ms/batch = {sps:.0f} samples/s")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', default='alphatrain/data/alphatrain_pairwise.pt')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    print(f"=== Collate Benchmark ({args.device}) ===")
    bench_collate(args.tensor_file, args.device)
