"""Profile collate to identify remaining bottlenecks.

Usage:
    python -m alphatrain.scripts.profile_collate
"""

import os
import time
import torch


def main():
    path = 'data/alphatrain_pairwise.pt'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    from alphatrain.dataset import TensorDatasetGPU
    ds = TensorDatasetGPU(path, augment=True, device=device)

    B = 4096
    indices = list(range(B))

    # Warmup
    ds.collate(indices)
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()

    def sync():
        if device == 'mps':
            torch.mps.synchronize()
        elif device == 'cuda':
            torch.cuda.synchronize()

    # Profile standard collate components
    N = 5
    print(f"=== Profiling collate ({device}, B={B}, N={N} iters) ===\n")

    # Full standard collate
    sync()
    t0 = time.perf_counter()
    for _ in range(N):
        ds.collate(indices)
        sync()
    full_ms = (time.perf_counter() - t0) / N * 1000
    print(f"Full standard collate: {full_ms:.0f}ms")

    # Just obs building (the hot path)
    base_idx = torch.tensor(indices, dtype=torch.long, device=ds.device) // ds.augment_factor
    boards = ds.boards[base_idx]
    sync()
    t0 = time.perf_counter()
    for _ in range(N):
        ds._build_obs_core(boards,
                           next_pos=ds.next_pos[base_idx],
                           next_col=ds.next_col[base_idx],
                           n_next=ds.n_next[base_idx])
        sync()
    obs_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  _build_obs_core (with next_balls): {obs_ms:.0f}ms")

    # Just afterstate obs (no next_balls)
    sync()
    t0 = time.perf_counter()
    for _ in range(N):
        ds._build_obs_boards_only(boards)
        sync()
    after_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  _build_obs_boards_only: {after_ms:.0f}ms")

    # Just line potentials
    obs = torch.zeros(B, 18, 9, 9, device=ds.device)
    sync()
    t0 = time.perf_counter()
    for _ in range(N):
        ds._build_line_potentials_gpu(boards, obs, B)
        sync()
    lp_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  _build_line_potentials_gpu: {lp_ms:.0f}ms")

    # Just component area (channels 0-12 minus line potentials)
    # Approximate by obs_core - line_potentials - colors
    sync()
    t0 = time.perf_counter()
    for _ in range(N):
        obs2 = torch.zeros(B, 18, 9, 9, device=ds.device)
        for c in range(1, 8):
            obs2[:, c - 1] = (boards == c).float()
        obs2[:, 7] = (boards == 0).float()
        sync()
    color_ms = (time.perf_counter() - t0) / N * 1000
    print(f"  Color channels (0-7): {color_ms:.0f}ms")

    comp_ms = after_ms - lp_ms - color_ms
    print(f"  Component area (ch 12, estimated): {max(0, comp_ms):.0f}ms")

    # Full pairwise collate
    sync()
    t0 = time.perf_counter()
    for _ in range(N):
        ds.collate_pairwise(indices)
        sync()
    pair_ms = (time.perf_counter() - t0) / N * 1000
    print(f"\nFull pairwise collate: {pair_ms:.0f}ms")
    print(f"  Overhead: {pair_ms - full_ms:.0f}ms ({pair_ms/full_ms:.1f}x)")

    print(f"\nBreakdown:")
    print(f"  Colors (ch 0-7):     {color_ms:.0f}ms ({color_ms/after_ms*100:.0f}%)")
    print(f"  Component (ch 12):   {max(0,comp_ms):.0f}ms ({max(0,comp_ms)/after_ms*100:.0f}%)")
    print(f"  Line potentials:     {lp_ms:.0f}ms ({lp_ms/after_ms*100:.0f}%)")


if __name__ == '__main__':
    main()
