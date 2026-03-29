"""Verify afterstate observations now include next_balls channels.

Usage:
    python -m alphatrain.scripts.verify_afterstate_nextballs
"""

import os
import torch


def main():
    path = 'alphatrain/data/alphatrain_pairwise.pt'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    from alphatrain.dataset import TensorDatasetGPU
    ds = TensorDatasetGPU(path, augment=True, device=device)

    indices = list(range(32))
    obs, policy, val, good_obs, bad_obs, margin = ds.collate_pairwise(indices)

    # Check channels 8-11 (next_balls) are populated
    next_ball_energy = good_obs[:, 8:12].abs().sum().item()
    pre_move_energy = obs[:, 8:12].abs().sum().item()

    print(f"Pre-move obs channels 8-11 energy: {pre_move_energy:.1f}")
    print(f"Afterstate obs channels 8-11 energy: {next_ball_energy:.1f}")

    if next_ball_energy > 0:
        print("PASS: Afterstate observations include next_balls")
    else:
        print("FAIL: Afterstate observations have zero next_balls")

    # Verify good and bad have same next_balls (same parent)
    good_nb = good_obs[:, 8:12]
    bad_nb = bad_obs[:, 8:12]
    same = torch.allclose(good_nb, bad_nb)
    print(f"Good/bad share same next_balls: {same}")


if __name__ == '__main__':
    main()
