"""Verify TD tensor file loads correctly and has sensible value targets.

Usage:
    python -m alphatrain.scripts.verify_td_data
"""

import torch
import numpy as np


def main():
    print("Loading TD tensor file...", flush=True)
    d = torch.load('data/alphatrain_td.pt', weights_only=True)

    print(f"Samples: {d['boards'].shape[0]:,}")
    print(f"value_mode: {d.get('value_mode', 'unknown')}")
    print(f"gamma: {d.get('gamma', 'unknown')}")

    # Decode two-hot value targets back to scalar
    bins = torch.linspace(0, d['max_score'], d['num_value_bins'])
    val_scalars = (d['val_targets'] * bins).sum(dim=-1).numpy()

    print(f"\nValue target stats:")
    print(f"  mean: {val_scalars.mean():.0f}")
    print(f"  std:  {val_scalars.std():.0f}")
    print(f"  min:  {val_scalars.min():.0f}")
    print(f"  max:  {val_scalars.max():.0f}")
    print(f"  median: {np.median(val_scalars):.0f}")

    # Check distribution
    pct = np.percentile(val_scalars, [10, 25, 50, 75, 90])
    print(f"  p10={pct[0]:.0f} p25={pct[1]:.0f} p50={pct[2]:.0f} "
          f"p75={pct[3]:.0f} p90={pct[4]:.0f}")

    # Verify it's NOT all the same value (the old bug)
    n_unique = len(np.unique(np.round(val_scalars, 0)))
    print(f"  unique values: {n_unique:,}")

    # Compare to old data if available
    import os
    old_path = 'data/alphatrain_v1.pt'
    if os.path.exists(old_path):
        print("\nLoading old tensor file for comparison...", flush=True)
        d_old = torch.load(old_path, weights_only=True)
        old_scalars = (d_old['val_targets'] * bins).sum(dim=-1).numpy()
        old_unique = len(np.unique(np.round(old_scalars, 0)))
        print(f"Old value targets: mean={old_scalars.mean():.0f}, "
              f"std={old_scalars.std():.0f}, unique={old_unique:,}")
        print(f"\nImprovement: {n_unique:,} unique values vs {old_unique:,} "
              f"({n_unique/max(old_unique,1):.0f}x more diverse)")
    else:
        print(f"\n(Old tensor file not found at {old_path}, skipping comparison)")


if __name__ == '__main__':
    main()
