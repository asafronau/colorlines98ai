"""Re-encode value targets with a new max_score for categorical head.

Decodes existing two-hot targets (max_score=2000) to scalar TD returns,
prints distribution statistics, then re-encodes with new max_score.

Usage:
    python -m alphatrain.scripts.reencode_value_targets [--new-max-score 500]
"""

import argparse
import numpy as np
import torch
import os
import time


def decode_twohot(targets, max_score, num_bins):
    """Decode two-hot targets to scalar values."""
    bins = torch.linspace(0, max_score, num_bins, device=targets.device)
    return (targets * bins).sum(dim=-1)


def encode_twohot(scalars, max_score, num_bins):
    """Encode scalar values as two-hot categorical targets."""
    bins = torch.linspace(0, max_score, num_bins, device=scalars.device)
    scalars = scalars.clamp(0, max_score)
    # Find bin indices
    pos = scalars / max_score * (num_bins - 1)
    low = pos.long().clamp(0, num_bins - 2)
    frac = (pos - low.float()).clamp(0, 1)
    targets = torch.zeros(len(scalars), num_bins, device=scalars.device)
    targets.scatter_(1, low.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    targets.scatter_(1, (low + 1).unsqueeze(1), frac.unsqueeze(1))
    return targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='alphatrain/data/expert_v2_pairwise.pt')
    parser.add_argument('--output', default=None,
                        help='Output path (default: input with _reencoded suffix)')
    parser.add_argument('--new-max-score', type=float, default=500.0)
    parser.add_argument('--stats-only', action='store_true',
                        help='Only print statistics, do not re-encode')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_ms{int(args.new_max_score)}{ext}"

    print(f"Loading {args.input}...", flush=True)
    t0 = time.time()
    data = torch.load(args.input, weights_only=True, map_location='cpu')
    print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

    old_max = float(data['max_score'])
    num_bins = int(data['num_value_bins'])
    val_targets = data['val_targets']  # (N, 64) two-hot
    N = val_targets.shape[0]

    print(f"\nCurrent: {N:,} targets, {num_bins} bins, max_score={old_max}", flush=True)

    # Decode to scalars
    print("Decoding two-hot to scalars...", flush=True)
    scalars = decode_twohot(val_targets, old_max, num_bins)

    # Distribution statistics
    print(f"\n{'='*60}", flush=True)
    print(f"TD Return Distribution ({N:,} values)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Min:    {scalars.min().item():.1f}", flush=True)
    print(f"  P10:    {torch.quantile(scalars.float(), 0.10).item():.1f}", flush=True)
    print(f"  P25:    {torch.quantile(scalars.float(), 0.25).item():.1f}", flush=True)
    print(f"  Median: {torch.quantile(scalars.float(), 0.50).item():.1f}", flush=True)
    print(f"  Mean:   {scalars.float().mean().item():.1f}", flush=True)
    print(f"  P75:    {torch.quantile(scalars.float(), 0.75).item():.1f}", flush=True)
    print(f"  P90:    {torch.quantile(scalars.float(), 0.90).item():.1f}", flush=True)
    print(f"  P95:    {torch.quantile(scalars.float(), 0.95).item():.1f}", flush=True)
    print(f"  P99:    {torch.quantile(scalars.float(), 0.99).item():.1f}", flush=True)
    print(f"  P99.9:  {torch.quantile(scalars.float(), 0.999).item():.1f}", flush=True)
    print(f"  Max:    {scalars.max().item():.1f}", flush=True)

    new_max = args.new_max_score
    n_clipped = (scalars > new_max).sum().item()
    pct_clipped = 100.0 * n_clipped / N
    print(f"\n  Values > {new_max}: {n_clipped:,} ({pct_clipped:.2f}%)", flush=True)

    if args.stats_only:
        return

    # Re-encode
    print(f"\nRe-encoding with max_score={new_max}, {num_bins} bins...", flush=True)
    print(f"  Bin width: {new_max/(num_bins-1):.1f} (was {old_max/(num_bins-1):.1f})", flush=True)
    new_targets = encode_twohot(scalars, new_max, num_bins)

    # Verify round-trip
    recovered = decode_twohot(new_targets, new_max, num_bins)
    err = (recovered - scalars.clamp(0, new_max)).abs()
    print(f"  Round-trip error: mean={err.mean().item():.3f}, max={err.max().item():.3f}",
          flush=True)

    # Save
    data['val_targets'] = new_targets
    data['max_score'] = new_max
    print(f"\nSaving to {args.output}...", flush=True)
    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved ({size_mb:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
