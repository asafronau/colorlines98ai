"""Merge individual self-play game files into a single training tensor file.

Usage:
    python -m alphatrain.scripts.build_selfplay_tensors \
        --games-dir data/selfplay \
        --output alphatrain/data/selfplay_iter1.pt
"""

import os
import argparse
import glob
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/selfplay')
    p.add_argument('--output', default='alphatrain/data/selfplay_iter1.pt')
    p.add_argument('--max-score', type=float, default=500.0)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.games_dir, 'game_*.pt')))
    print(f"Found {len(files)} game files in {args.games_dir}", flush=True)

    all_obs = []
    all_pol = []
    all_val = []
    scores = []
    skipped = 0

    for i, f in enumerate(files):
        g = torch.load(f, weights_only=False)
        if g['turns'] == 0:
            skipped += 1
            continue
        all_obs.append(g['observations'])
        all_pol.append(g['policy_targets'])
        all_val.append(g['value_targets'])
        scores.append(g['score'])

        if (i + 1) % 100 == 0:
            n_states = sum(t.shape[0] for t in all_obs)
            print(f"  {i+1}/{len(files)} games, {n_states:,} states", flush=True)

    if skipped:
        print(f"Skipped {skipped} empty games", flush=True)

    observations = torch.cat(all_obs)
    policy_targets = torch.cat(all_pol)
    value_targets = torch.cat(all_val)
    scores = np.array(scores)

    print(f"\nMerged: {len(scores)} games, {observations.shape[0]:,} states",
          flush=True)
    print(f"Observations: {observations.shape} ({observations.dtype})", flush=True)
    print(f"Policy targets: {policy_targets.shape}", flush=True)
    print(f"Value targets: {value_targets.shape} "
          f"(mean={value_targets.mean():.1f}, max={value_targets.max():.1f})",
          flush=True)
    print(f"Game scores: mean={scores.mean():.0f}, median={np.median(scores):.0f}, "
          f"min={scores.min()}, max={scores.max()}", flush=True)

    data = {
        'observations': observations,
        'policy_targets': policy_targets,
        'value_targets': value_targets,
        'max_score': args.max_score,
        'format': 'selfplay',
        'n_games': len(scores),
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
