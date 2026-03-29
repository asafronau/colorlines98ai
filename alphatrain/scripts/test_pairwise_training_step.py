"""Smoke test: one pairwise training step on real data.

Verifies the full pipeline works end-to-end:
load data → collate_pairwise → forward pass → ranking loss → backward.

Usage:
    python -m alphatrain.scripts.test_pairwise_training_step
"""

import os
import torch
import torch.nn as nn
from alphatrain.model import AlphaTrainNet
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.train import cross_entropy_soft


def main():
    path = 'data/alphatrain_pairwise.pt'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'

    print(f"Loading dataset to {device}...", flush=True)
    ds = TensorDatasetGPU(path, augment=True, device=device)
    assert ds.has_pairs, "Dataset must have pairwise data"

    print("Building model...", flush=True)
    net = AlphaTrainNet(num_blocks=2, channels=64).to(device)  # small for speed
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    max_score = ds.max_score

    # One pairwise collate
    print("Collating batch...", flush=True)
    indices = list(range(32))
    obs, pol_tgt, val_tgt, good_obs, bad_obs, margin = ds.collate_pairwise(indices)
    print(f"  obs: {obs.shape}, good_obs: {good_obs.shape}, margin: {margin.shape}")

    # Forward pass
    print("Forward pass...", flush=True)
    net.train()
    pol_logits, val_logits = net(obs)
    pol_loss = cross_entropy_soft(pol_logits, pol_tgt)
    val_loss = cross_entropy_soft(val_logits, val_tgt)

    _, good_val = net(good_obs)
    _, bad_val = net(bad_obs)
    v_good = net.predict_value(good_val, max_val=max_score)
    v_bad = net.predict_value(bad_val, max_val=max_score)

    print(f"  V(good): {v_good[:4].tolist()}")
    print(f"  V(bad):  {v_bad[:4].tolist()}")
    print(f"  Margin:  {margin[:4].tolist()}")

    # Ranking loss
    target = torch.ones_like(v_good)
    margin_loss_fn = nn.MarginRankingLoss(margin=0.0, reduction='none')
    rank_loss_raw = margin_loss_fn(v_good, v_bad, target)
    margin_norm = margin / (margin.mean() + 1e-8)
    rank_loss = (rank_loss_raw * margin_norm).mean()

    total_loss = pol_loss + val_loss + rank_loss
    print(f"  pol_loss={pol_loss.item():.4f}, val_loss={val_loss.item():.4f}, "
          f"rank_loss={rank_loss.item():.4f}")
    print(f"  total_loss={total_loss.item():.4f}")

    # Backward
    print("Backward pass...", flush=True)
    optimizer.zero_grad()
    total_loss.backward()

    grad_norm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    n_with_grad = sum(1 for p in net.parameters() if p.grad is not None)
    print(f"  {n_with_grad} params with gradients, total grad norm={grad_norm:.4f}")

    # Step
    optimizer.step()
    print("\nSUCCESS: Full pairwise training step completed.", flush=True)


if __name__ == '__main__':
    main()
