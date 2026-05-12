"""Precompute 18-channel observations for an existing tensor file.

Eliminates the per-batch obs-building cost during training (the 20-iter
component-labeling loop in dataset.collate is the dominant overhead on
H100). Adds an `obs_precomputed` (fp16) field to the tensor; dataset.py
uses it directly when present and skips obs construction.

Tensor size grows by ~14GB (9.77M × 18 × 81 × 2 bytes fp16). Gzipped
upload is ~5-7GB.

Usage:
    python -m alphatrain.scripts.precompute_obs \\
        --input alphatrain/data/v12_pillar2z.pt \\
        --output alphatrain/data/v12_pillar2z_obs.pt \\
        --device mps --batch-size 8192
"""

import os
import time
import argparse
import torch

from alphatrain.dataset import TensorDatasetGPU


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True,
                   help='Input tensor file (e.g., v12_pillar2z.pt)')
    p.add_argument('--output', required=True,
                   help='Output tensor file (with obs_precomputed added)')
    p.add_argument('--device', default=None,
                   help='Device for obs build (auto-detect mps/cuda/cpu)')
    p.add_argument('--batch-size', type=int, default=8192,
                   help='Batch size for obs construction')
    args = p.parse_args()

    if args.device:
        device_str = args.device
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    print(f"Device: {device_str}", flush=True)

    # Load via the same dataset class so obs-build code is reused exactly.
    ds = TensorDatasetGPU(args.input, augment=False, device=device_str)
    n = ds.boards.shape[0]
    print(f"States: {n:,}", flush=True)

    print(f"Computing obs in batches of {args.batch_size}...", flush=True)
    obs_all = torch.empty((n, 18, 9, 9), dtype=torch.float16,
                          device='cpu', pin_memory=False)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            idx = torch.arange(start, end, device=device, dtype=torch.long)
            boards = ds.boards[idx]
            obs_batch = ds._build_obs_core(
                boards,
                next_pos=ds.next_pos[idx],
                next_col=ds.next_col[idx],
                n_next=ds.n_next[idx],
            )
            obs_all[start:end] = obs_batch.to(torch.float16).cpu()
            if (start // args.batch_size) % 20 == 0:
                elapsed = time.time() - t0
                pct = 100 * end / n
                eta = elapsed / max(end, 1) * (n - end)
                print(f"  {end:>9,}/{n:,} ({pct:.1f}%) "
                      f"elapsed={elapsed:.0f}s ETA={eta:.0f}s",
                      flush=True)

    print(f"Done building obs in {time.time()-t0:.0f}s", flush=True)
    print(f"obs_precomputed shape: {tuple(obs_all.shape)}, dtype: {obs_all.dtype}",
          flush=True)
    print(f"obs_precomputed size: {obs_all.numel() * obs_all.element_size() / 1e9:.1f}GB",
          flush=True)

    # Reload original tensor (raw bytes) to merge, avoiding re-saving fields
    # via the dataset abstraction (which strips some metadata).
    data = torch.load(args.input, weights_only=False)
    data['obs_precomputed'] = obs_all
    data['obs_dtype'] = 'float16'

    print(f"Writing {args.output}...", flush=True)
    t0 = time.time()
    torch.save(data, args.output)
    print(f"Saved in {time.time()-t0:.0f}s "
          f"({os.path.getsize(args.output)/1e9:.1f}GB)", flush=True)


if __name__ == '__main__':
    main()
