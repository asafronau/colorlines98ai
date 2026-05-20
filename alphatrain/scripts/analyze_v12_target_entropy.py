"""Pre-training sanity: what does V12 target entropy look like, and how
does it change under different target_temperature sharpening levels?

Loads the V12 sparse policy targets (pol_indices, pol_values), reconstructs
the per-state target distribution, then computes its entropy under several
temperatures.

Used to choose a sane sharpening ladder before committing 6-8h of training.
Specifically: detect whether T=0.25 collapses targets toward one-hot
(bad — replaces soft over-distillation with hard noise-distillation).

Output: per-temperature mean/median entropy + concentration metrics.
The training-time loss applies:

    sharp = target ** (1/T)
    sharp = sharp / sharp.sum()

so this script replicates the same op.

Usage:
    python -m alphatrain.scripts.analyze_v12_target_entropy \\
        --tensor-file alphatrain/data/v12_pillar2z.pt \\
        --n-samples 5000 \\
        --temperatures 1.0 0.75 0.5 0.25 0.1
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor-file', required=True)
    p.add_argument('--n-samples', type=int, default=5000)
    p.add_argument('--temperatures', type=float, nargs='+',
                   default=[1.0, 0.75, 0.5, 0.25, 0.1])
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"Loading {args.tensor_file}...", flush=True)
    t0 = time.time()
    data = torch.load(args.tensor_file, weights_only=False)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    pol_indices = data['pol_indices']
    pol_values = data['pol_values']
    pol_nnz = data.get('pol_nnz', None)
    if isinstance(pol_indices, torch.Tensor):
        pol_indices = pol_indices.numpy()
    if isinstance(pol_values, torch.Tensor):
        pol_values = pol_values.numpy()
    if pol_nnz is not None and isinstance(pol_nnz, torch.Tensor):
        pol_nnz = pol_nnz.numpy()

    N = pol_indices.shape[0]
    K = pol_indices.shape[1]
    print(f"  {N:,} states, top-{K} sparse targets", flush=True)

    sel = rng.choice(N, size=min(args.n_samples, N), replace=False)
    print(f"  sampling {len(sel)} states", flush=True)

    def _entropy(probs):
        """Entropy of a single 1D probability distribution (nan-safe)."""
        valid = probs > 0
        return -float(np.sum(probs[valid] * np.log(probs[valid])))

    def _stats_for_t(T):
        ent = np.zeros(len(sel), dtype=np.float32)
        top1 = np.zeros(len(sel), dtype=np.float32)
        top1_top2 = np.zeros(len(sel), dtype=np.float32)
        nnz_eff = np.zeros(len(sel), dtype=np.int32)
        for i_out, i in enumerate(sel):
            k = (int(pol_nnz[i]) if pol_nnz is not None
                  else int((pol_values[i] > 0).sum()))
            if k < 1:
                continue
            v = pol_values[i, :k].astype(np.float64)
            # Skip empty / degenerate
            v_sum = v.sum()
            if v_sum <= 0:
                continue
            v_norm = v / v_sum
            if T == 1.0:
                sharp = v_norm
            else:
                sharp = v_norm ** (1.0 / T)
                ss = sharp.sum()
                if ss > 0:
                    sharp = sharp / ss
                else:
                    continue
            ent[i_out] = _entropy(sharp)
            top1[i_out] = float(sharp.max())
            sharp_sorted = np.sort(sharp)[::-1]
            if len(sharp_sorted) >= 2:
                top1_top2[i_out] = float(sharp_sorted[0] - sharp_sorted[1])
            else:
                top1_top2[i_out] = float(sharp_sorted[0])
            # effective nnz = how many components > 1e-3
            nnz_eff[i_out] = int(np.sum(sharp > 1e-3))
        return {
            'T': T,
            'entropy_mean': float(ent.mean()),
            'entropy_median': float(np.median(ent)),
            'top1_mean': float(top1.mean()),
            'top1_median': float(np.median(top1)),
            'top1_p10': float(np.percentile(top1, 10)),
            'top1_p90': float(np.percentile(top1, 90)),
            'top1_top2_mean': float(top1_top2.mean()),
            'top1_top2_median': float(np.median(top1_top2)),
            'nnz_eff_mean': float(nnz_eff.mean()),
            'nnz_eff_p10': float(np.percentile(nnz_eff, 10)),
            'nnz_eff_p90': float(np.percentile(nnz_eff, 90)),
            'frac_near_onehot': float((top1 > 0.95).mean()),
        }

    print(f"\nComputing target entropy under each temperature...\n", flush=True)
    print(f"{'T':>5} {'entropy':>9} {'top1 mean':>10} "
          f"{'top1 P50':>9} {'top1 P90':>9} "
          f"{'gap mean':>9} {'eff nnz':>9} {'~1-hot %':>10}")
    print('-' * 72)
    results = []
    for T in args.temperatures:
        st = _stats_for_t(T)
        results.append(st)
        print(f"{st['T']:>5.2f} {st['entropy_mean']:>9.3f} "
              f"{st['top1_mean']:>10.3f} {st['top1_median']:>9.3f} "
              f"{st['top1_p90']:>9.3f} "
              f"{st['top1_top2_mean']:>9.3f} {st['nnz_eff_mean']:>9.1f} "
              f"{100*st['frac_near_onehot']:>9.1f}%")

    print(f"\nInterpretation:")
    print(f"  entropy_mean drops as T → 0 (sharper)")
    print(f"  top1_mean rises as T → 0")
    print(f"  ~1-hot % > ~30 means T is likely too aggressive (hard-label "
          f"regime, brittle to MCTS visit noise)")
    print(f"  eff nnz ≈ 1 at T=0.1 confirms one-hot collapse")


if __name__ == '__main__':
    main()
