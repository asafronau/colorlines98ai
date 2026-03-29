"""Verify GPU line potentials match CPU Numba implementation.

Usage:
    python -m alphatrain.scripts.verify_gpu_line_potentials
"""

import os
import numpy as np
import torch
from alphatrain.observation import build_line_potentials_batch


def main():
    path = 'alphatrain/data/alphatrain_pairwise.pt'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'

    from alphatrain.dataset import TensorDatasetGPU
    ds = TensorDatasetGPU(path, augment=False, device=device)

    # Take 100 random boards
    N = 100
    idx = torch.randint(0, ds.boards.shape[0], (N,))
    boards = ds.boards[idx]

    # GPU path
    obs_gpu = torch.zeros(N, 18, 9, 9, device=ds.device)
    ds._build_line_potentials_gpu(boards, obs_gpu, N)
    gpu_lp = obs_gpu[:, 13:18].cpu().numpy()

    # CPU path (reference)
    boards_np = boards.cpu().numpy().astype(np.int64)
    obs_cpu = np.zeros((N, 18, 9, 9), dtype=np.float32)
    build_line_potentials_batch(boards_np, obs_cpu)
    cpu_lp = obs_cpu[:, 13:18]

    # Compare
    max_diff = np.abs(gpu_lp - cpu_lp).max()
    mean_diff = np.abs(gpu_lp - cpu_lp).mean()
    n_mismatch = (np.abs(gpu_lp - cpu_lp) > 1e-5).sum()

    print(f"Compared {N} boards, channels 13-17:")
    print(f"  max_diff: {max_diff:.6f}")
    print(f"  mean_diff: {mean_diff:.6f}")
    print(f"  mismatches (>1e-5): {n_mismatch} / {N * 5 * 9 * 9}")

    if max_diff < 1e-4:
        print("  PASS: GPU matches CPU")
    else:
        print("  FAIL: GPU does NOT match CPU")
        # Show first mismatch
        for i in range(N):
            for ch in range(5):
                diff = np.abs(gpu_lp[i, ch] - cpu_lp[i, ch])
                if diff.max() > 1e-4:
                    r, c = np.unravel_index(diff.argmax(), (9, 9))
                    print(f"  board {i}, ch {13+ch}, ({r},{c}): "
                          f"gpu={gpu_lp[i,ch,r,c]:.4f} cpu={cpu_lp[i,ch,r,c]:.4f} "
                          f"board_val={boards_np[i,r,c]}")
                    return


if __name__ == '__main__':
    main()
