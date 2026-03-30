"""Smoke test: one training step with scalar value head (bins=1) + anchor loss.

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
    max_score = ds.max_score

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

    # Anchor loss: MSE between sigmoid-clamped prediction and TD target
    v_pred = net.predict_value(val_logits, max_val=max_score)
    anchor_bins = torch.linspace(0, max_score, ds.num_value_bins, device=device)
    true_scalar = (val_tgt * anchor_bins).sum(dim=-1)
    anchor_loss = F.mse_loss(v_pred, true_scalar)
    print(f"  V(pred) range: [{v_pred.min():.1f}, {v_pred.max():.1f}]")
    print(f"  TD target range: [{true_scalar.min():.1f}, {true_scalar.max():.1f}]")
    print(f"  anchor_loss={anchor_loss.item():.4f}")

    # Ranking loss
    pair_obs = torch.cat([good_obs, bad_obs], dim=0)
    _, pair_val = net(pair_obs)
    good_val, bad_val = pair_val.chunk(2, dim=0)
    v_good = net.predict_value(good_val, max_val=max_score)
    v_bad = net.predict_value(bad_val, max_val=max_score)

    print(f"  V(good): {v_good[:4].tolist()}")
    print(f"  V(bad):  {v_bad[:4].tolist()}")
    print(f"  Margin:  {margin[:4].tolist()}")

    margin_scaled = margin * (5.0 / (margin.mean() + 1e-8))
    rank_loss = F.relu(margin_scaled - (v_good - v_bad)).mean()

    # Combined: pol + rank + 0.1 * anchor
    anchor_weight = 0.1
    loss = pol_loss + rank_loss + anchor_weight * anchor_loss

    print(f"  pol_loss={pol_loss.item():.4f}, rank_loss={rank_loss.item():.4f}, "
          f"anchor_loss={anchor_loss.item():.4f}")
    print(f"  total={loss.item():.4f} (pol + rank + {anchor_weight}*anchor)")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    print(f"  grad_norm={grad_norm:.4f}")

    optimizer.step()
    print("\nSUCCESS: Scalar value + anchor loss training step completed.", flush=True)


if __name__ == '__main__':
    main()
