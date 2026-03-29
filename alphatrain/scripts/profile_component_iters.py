"""Check how many label propagation iterations are actually needed.

Usage:
    python -m alphatrain.scripts.profile_component_iters
"""

import os
import torch
import numpy as np


def main():
    path = 'data/alphatrain_pairwise.pt'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    device = 'cpu'  # need to check convergence, CPU is fine
    from alphatrain.dataset import TensorDatasetGPU
    ds = TensorDatasetGPU(path, augment=False, device=device)

    # Sample 1000 boards
    N = 1000
    idx = torch.randint(0, ds.boards.shape[0], (N,))
    boards = ds.boards[idx]
    empty = (boards == 0).float()
    flat_empty = empty.reshape(N, 81)

    labels = torch.arange(1, 82).unsqueeze(0).expand(N, 81).float()
    labels = labels * flat_empty
    neighbors = ds._neighbors

    for it in range(30):
        old = labels.clone()
        for ni in range(4):
            nb_idx = neighbors[:, ni]
            valid = nb_idx >= 0
            if not valid.any():
                continue
            vc = torch.where(valid)[0]
            nb_labels = labels[:, nb_idx[vc]]
            self_labels = labels[:, vc]
            both = (self_labels > 0) & (nb_labels > 0)
            new_min = torch.where(both & (nb_labels < self_labels),
                                   nb_labels, self_labels)
            labels[:, vc] = new_min
        changed = (labels != old).any(dim=1).sum().item()
        if changed == 0:
            print(f"Converged at iteration {it}")
            break
        print(f"Iter {it}: {changed}/{N} boards still changing")


if __name__ == '__main__':
    main()
