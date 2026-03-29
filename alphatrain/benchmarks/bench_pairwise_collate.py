"""Benchmark pairwise collate throughput.

Compares standard collate vs pairwise collate (3x obs building).
Must stay above 5K samples/s on MPS, 15K on CUDA for viable training.

Usage:
    python -m alphatrain.benchmarks.bench_pairwise_collate
"""

import os
import time
import argparse
import torch
from torch.utils.data import DataLoader


def bench(dataset, collate_fn, batch_size, device_name, label, n_batches=10):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_fn)
    it = iter(loader)
    # Warmup
    next(it)
    if device_name == 'mps':
        torch.mps.synchronize()
    elif device_name == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_batches):
        next(it)
    if device_name == 'mps':
        torch.mps.synchronize()
    elif device_name == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) / n_batches * 1000
    sps = batch_size / (ms / 1000)
    print(f"  {label}: {ms:.0f}ms/batch = {sps:.0f} samples/s")
    return sps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', default='alphatrain/data/alphatrain_pairwise.pt')
    p.add_argument('--device', default=None)
    p.add_argument('--batch-size', type=int, default=4096)
    args = p.parse_args()

    if not os.path.exists(args.tensor_file):
        print(f"ERROR: {args.tensor_file} not found")
        return

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = 'mps'
        elif torch.cuda.is_available():
            args.device = 'cuda'
        else:
            args.device = 'cpu'

    from alphatrain.dataset import TensorDatasetGPU
    ds = TensorDatasetGPU(args.tensor_file, augment=True, device=args.device)

    print(f"=== Pairwise Collate Benchmark ({args.device}, "
          f"batch={args.batch_size}) ===")

    std_sps = bench(ds, ds.collate, args.batch_size, args.device, "Standard collate")

    if ds.has_pairs:
        pair_sps = bench(ds, ds.collate_pairwise, args.batch_size, args.device,
                         "Pairwise collate")
        ratio = std_sps / max(pair_sps, 1)
        print(f"\n  Pairwise overhead: {ratio:.1f}x slower than standard")
        print(f"  Pairwise throughput: {pair_sps:.0f} s/s "
              f"({'OK' if pair_sps > 5000 else 'TOO SLOW'})")
    else:
        print("  No pairwise data in tensor file")


if __name__ == '__main__':
    main()
