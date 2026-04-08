"""Analyze the distribution of TD returns to understand value head contrast."""

import torch
import time

print("Loading tensor...", flush=True)
t0 = time.time()
data = torch.load('alphatrain/data/expert_v2_pairwise.pt', weights_only=True, map_location='cpu')
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

val_targets = data['val_targets']
bins = torch.linspace(0, float(data['max_score']), int(data['num_value_bins']))
scalars = (val_targets * bins).sum(dim=-1)

N = len(scalars)
print(f"\n{N:,} positions total")

# Count by value ranges
for threshold in [0, 10, 25, 50, 100, 150, 175, 190, 200, 210, 220, 250]:
    n = (scalars <= threshold).sum().item()
    print(f"  TD return <= {threshold:>4}: {n:>10,} ({100*n/N:>6.2f}%)")

print(f"\n  Mean: {scalars.mean():.1f}")
print(f"  Std:  {scalars.std():.1f}")

# The "death zone" — how concentrated is the distribution?
tight = ((scalars >= 190) & (scalars <= 240)).sum().item()
print(f"\n  In [190, 240]: {tight:,} ({100*tight/N:.1f}%) — the 'everything looks the same' zone")

# Value head needs to distinguish: what fraction of positions are truly different?
low = (scalars < 150).sum().item()
high = (scalars > 250).sum().item()
mid = N - low - high
print(f"\n  Low (<150):  {low:,} ({100*low/N:.2f}%)")
print(f"  Mid (150-250): {mid:,} ({100*mid/N:.1f}%)")
print(f"  High (>250): {high:,} ({100*high/N:.2f}%)")
