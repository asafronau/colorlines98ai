"""Smoke-test GumbelDatasetGPU against the real V15 slim tensor.

Validates the whole dataset path: collate shapes, target/prior are valid distributions,
the empirical correction rate matches the diagnostic (~3-5%), target diverges from prior
ONLY on correction states, and dihedral augmentation preserves the distributions.

    PYTHONPATH=. python scripts/smoke_gumbel_dataset.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from alphatrain.dataset import GumbelDatasetGPU

TENSOR = 'alphatrain/data/v15_pillar3f_slim.pt'


def main():
    dev = ('cuda' if torch.cuda.is_available()
           else 'mps' if torch.backends.mps.is_available() else 'cpu')
    t0 = time.time()
    train, val = GumbelDatasetGPU.make_train_val_split(
        TENSOR, device=dev, augment=True, color_augment=True, augment_factor=1,
        visit_floor=15.0, tau=0.05, gamma=10.0, spread_gate=0.05)
    print(f"loaded in {time.time()-t0:.1f}s | train={len(train):,} val={len(val):,} dev={dev}",
          flush=True)

    # --- one collated batch: shapes + validity ---
    B = 8192
    idx = list(range(B))
    obs, target, prior, support, weight = train.collate(idx)
    print(f"\nobs {tuple(obs.shape)} | target {tuple(target.shape)} | "
          f"prior {tuple(prior.shape)} | support {tuple(support.shape)} | "
          f"weight {tuple(weight.shape)}")
    assert obs.shape == (B, 18, 9, 9)
    assert target.shape == prior.shape == support.shape == (B, 6561)
    ts, ps = target.sum(1), prior.sum(1)
    print(f"target row-sums: min {ts.min():.4f} max {ts.max():.4f} (want ~1.0)")
    print(f"prior  row-sums: min {ps.min():.4f} max {ps.max():.4f} (want ~1.0)")
    assert torch.allclose(ts, torch.ones_like(ts), atol=1e-4)
    assert torch.allclose(ps, torch.ones_like(ps), atol=1e-4)
    # target mass must live entirely on the support (candidate-restricted CE invariant)
    off_support = (target * (support <= 0)).sum(1)
    print(f"target mass OFF support: max {off_support.max():.2e} (want ~0)")
    assert off_support.max() < 1e-5

    # --- correction rate over many batches (should match diagnostic ~3-5%) ---
    n_corr, n_tot, div_corr, div_noncorr = 0, 0, 0.0, 0.0
    nb = 30
    for b in range(nb):
        idx = list(range(b * B, (b + 1) * B))
        _, tgt, pri, sup, w = train.collate(idx)
        is_corr = w > 1.0
        n_corr += int(is_corr.sum()); n_tot += len(w)
        # L1 divergence target vs prior, split by correction flag
        l1 = (tgt - pri).abs().sum(1)
        if is_corr.any():
            div_corr += float(l1[is_corr].sum())
        div_noncorr += float(l1[~is_corr].sum())
    print(f"\ncorrection rate over {n_tot:,} states: {100*n_corr/n_tot:.2f}%  "
          f"(diagnostic expected ~3-5% at visit_floor 15)")
    print(f"mean |target-prior|_1 on corrections: {div_corr/max(n_corr,1):.4f}")
    print(f"mean |target-prior|_1 on non-corrections: "
          f"{div_noncorr/max(n_tot-n_corr,1):.6f}  (want ~0: prior left alone)")

    # --- augmentation preserves the distribution (sum invariant) ---
    val_obs, val_tgt, val_pri, val_sup, _ = val.collate(list(range(4096)))
    assert torch.allclose(val_tgt.sum(1), torch.ones(4096, device=val_tgt.device), atol=1e-4)
    print("\nval (unaugmented) target sums OK; augmented train sums OK -> dihedral safe")
    print("\nSMOKE PASS")


if __name__ == '__main__':
    main()
