"""Smoke test: one training step with scalar value head (bins=1).

Usage:
    python -m alphatrain.scripts.test_scalar_value_step
"""

import os
import torch
import torch.nn.functional as F
from alphatrain.model import AlphaTrainNet
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.train import cross_entropy_soft


def main():
    path = 'alphatrain/data/alphatrain_pairwise.pt'
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    print("Loading dataset...", flush=True)
    ds = TensorDatasetGPU(path, augment=True, device=device)

    # Scalar value head
    print("Building scalar value model (bins=1)...", flush=True)
    net = AlphaTrainNet(num_blocks=2, channels=64, num_value_bins=1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)

    print(f"  value_fc2 shape: {net.value_fc2.weight.shape}")
    assert net.num_value_bins == 1

    # Collate
    indices = list(range(32))
    obs, pol_tgt, val_tgt, good_obs, bad_obs, margin = ds.collate_pairwise(indices)

    # Forward
    net.train()
    pol_logits, val_logits = net(obs)
    print(f"  val_logits shape: {val_logits.shape}")  # should be (32, 1)

    pol_loss = cross_entropy_soft(pol_logits, pol_tgt)

    # No val_CE for scalar head — only ranking
    pair_obs = torch.cat([good_obs, bad_obs], dim=0)
    _, pair_val = net(pair_obs)
    good_val, bad_val = pair_val.chunk(2, dim=0)
    v_good = net.predict_value(good_val)
    v_bad = net.predict_value(bad_val)

    print(f"  V(good): {v_good[:4].tolist()}")
    print(f"  V(bad):  {v_bad[:4].tolist()}")
    print(f"  Margin:  {margin[:4].tolist()}")

    margin_scaled = margin * (5.0 / (margin.mean() + 1e-8))
    rank_loss = F.relu(margin_scaled - (v_good - v_bad)).mean()
    loss = pol_loss + rank_loss

    print(f"  pol_loss={pol_loss.item():.4f}, rank_loss={rank_loss.item():.4f}")
    print(f"  total={loss.item():.4f} (no val_CE)")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    print(f"  grad_norm={grad_norm:.4f}")

    optimizer.step()
    print("\nSUCCESS: Scalar value training step completed.", flush=True)


if __name__ == '__main__':
    main()
