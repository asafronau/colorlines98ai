"""Build mixed training tensor: 50% heuristic expert + 50% sharpened self-play.

Converts both data sources to a unified format (pre-computed obs, dense policy,
scalar values) and merges into one tensor file.

Usage:
    python -m alphatrain.scripts.build_mixed_tensors \
        --expert alphatrain/data/alphatrain_pairwise.pt \
        --selfplay alphatrain/data/selfplay_iter1.pt \
        --output alphatrain/data/mixed_iter1.pt \
        --policy-temperature 0.1
"""

import os
import argparse
import time
import numpy as np
import torch

from alphatrain.observation import build_observation
from alphatrain.dataset import score_to_twohot

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4


def _build_obs_from_boards(boards, next_pos, next_col, n_next):
    """Build observations from raw board data (CPU, numba JIT)."""
    n = boards.shape[0]
    obs = np.zeros((n, 18, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for i in range(n):
        board = boards[i].numpy().astype(np.int8)
        nr = np.zeros(3, dtype=np.intp)
        nc = np.zeros(3, dtype=np.intp)
        ncol = np.zeros(3, dtype=np.intp)
        nn = int(n_next[i])
        for j in range(nn):
            nr[j] = int(next_pos[i, j, 0])
            nc[j] = int(next_pos[i, j, 1])
            ncol[j] = int(next_col[i, j])
        obs[i] = build_observation(board, nr, nc, ncol, nn)

        if (i + 1) % 50000 == 0:
            print(f"    {i+1}/{n} observations built", flush=True)

    return torch.from_numpy(obs)


def _sparse_to_dense_policy(pol_indices, pol_values, pol_nnz=None):
    """Convert sparse policy to dense (N, 6561)."""
    n = pol_indices.shape[0]
    policy = torch.zeros(n, NUM_MOVES, dtype=torch.float32)
    for i in range(n):
        k = int(pol_nnz[i]) if pol_nnz is not None else pol_indices.shape[1]
        if k > 0:
            policy[i].scatter_(0, pol_indices[i, :k].long(), pol_values[i, :k])
    return policy


def _decode_twohot_values(val_targets, max_score, num_bins):
    """Decode two-hot categorical values to scalar."""
    bins = torch.linspace(0, max_score, num_bins)
    return (val_targets * bins).sum(dim=-1)


def _sharpen_policy(policy, temperature):
    """Apply temperature sharpening to policy distributions."""
    if temperature >= 1.0:
        return policy
    # pol^(1/T), then renormalize
    inv_t = 1.0 / temperature
    sharpened = policy ** inv_t
    sums = sharpened.sum(dim=-1, keepdim=True)
    sums = sums.clamp(min=1e-8)
    return sharpened / sums


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--expert', default='alphatrain/data/alphatrain_pairwise.pt',
                   help='Heuristic expert tensor file')
    p.add_argument('--selfplay', default='alphatrain/data/selfplay_iter1.pt',
                   help='Self-play tensor file')
    p.add_argument('--output', default='alphatrain/data/mixed_iter1.pt')
    p.add_argument('--policy-temperature', type=float, default=0.1,
                   help='Temperature for sharpening self-play policy (0.1=sharp)')
    p.add_argument('--max-expert', type=int, default=0,
                   help='Max expert states to use (0=match selfplay count)')
    p.add_argument('--max-score', type=float, default=500.0)
    args = p.parse_args()

    # ── Load self-play data ──
    print("Loading self-play data...", flush=True)
    sp = torch.load(args.selfplay, weights_only=False)
    sp_obs = sp['observations']
    sp_pol = sp['policy_targets']
    sp_val = sp['value_targets']
    n_sp = sp_obs.shape[0]
    print(f"  Self-play: {n_sp:,} states", flush=True)

    # Sharpen self-play policy targets
    if args.policy_temperature < 1.0:
        print(f"  Sharpening policy (T={args.policy_temperature})...", flush=True)
        sp_pol = _sharpen_policy(sp_pol, args.policy_temperature)
        # Verify sharpness
        entropy = -(sp_pol * (sp_pol + 1e-10).log()).sum(dim=-1).mean()
        print(f"  Post-sharpening entropy: {entropy:.2f} "
              f"(was ~3.8 at T=1.0)", flush=True)

    # ── Load and convert expert data ──
    print("\nLoading expert data...", flush=True)
    ex = torch.load(args.expert, weights_only=True)
    n_ex_total = ex['boards'].shape[0]
    max_score = float(ex['max_score'])
    num_bins = int(ex['num_value_bins'])

    # Subsample expert data to match self-play count
    n_ex = args.max_expert if args.max_expert > 0 else n_sp
    n_ex = min(n_ex, n_ex_total)
    print(f"  Expert: {n_ex_total:,} total, using {n_ex:,} (subsampled)", flush=True)

    # Random subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(n_ex_total, size=n_ex, replace=False)
    indices.sort()
    idx = torch.from_numpy(indices)

    boards = ex['boards'][idx]
    next_pos = ex['next_pos'][idx]
    next_col = ex['next_col'][idx]
    n_next = ex['n_next'][idx]

    # Build observations from boards (CPU, numba)
    print("  Building observations from boards...", flush=True)
    t0 = time.time()
    ex_obs = _build_obs_from_boards(boards, next_pos, next_col, n_next)
    print(f"  Built in {time.time()-t0:.0f}s", flush=True)

    # Convert sparse policy to dense
    print("  Converting sparse policy to dense...", flush=True)
    ex_pol = _sparse_to_dense_policy(
        ex['pol_indices'][idx], ex['pol_values'][idx],
        ex.get('pol_nnz', None)
        if 'pol_nnz' not in ex else ex['pol_nnz'][idx])

    # Decode two-hot values to scalar
    ex_val = _decode_twohot_values(ex['val_targets'][idx], max_score, num_bins)
    print(f"  Expert value stats: mean={ex_val.mean():.1f}, "
          f"max={ex_val.max():.1f}", flush=True)

    # ── Merge ──
    print(f"\nMerging: {n_ex:,} expert + {n_sp:,} self-play = "
          f"{n_ex + n_sp:,} total", flush=True)

    observations = torch.cat([ex_obs, sp_obs])
    policy_targets = torch.cat([ex_pol, sp_pol])
    value_targets = torch.cat([ex_val, sp_val])

    print(f"Observations: {observations.shape}", flush=True)
    print(f"Policy targets: {policy_targets.shape}", flush=True)
    print(f"Value targets: {value_targets.shape} "
          f"(mean={value_targets.mean():.1f})", flush=True)

    data = {
        'observations': observations,
        'policy_targets': policy_targets,
        'value_targets': value_targets,
        'max_score': args.max_score,
        'format': 'selfplay',
        'n_expert': n_ex,
        'n_selfplay': n_sp,
        'policy_temperature': args.policy_temperature,
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(data, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved to {args.output} ({size_mb:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
