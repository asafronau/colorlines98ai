"""Train a stationary-risk aux head on frozen pillar3b backbone.

Phase 1 step of the floor-lift experiment. Trains a small MLP on top of
pillar3b's penultimate feature map to predict 6 forward-100-turn outcomes:
  min_empty_H, min_lec_H, empty_delta_H, lec_delta_H, score_rate_H, clear_rate_H

Decision gate at the end:
  - Compute AUC on binary derivatives: min_lec_H<10, min_empty_H<25
  - If AUC > 0.75 -> signal is real and learnable -> proceed to Phase 2
  - If AUC < 0.65 -> backbone features don't carry stationary risk
  - If 0.65-0.75 -> marginal, investigate per-bucket calibration

Usage:
    python scripts/train_stationary_risk_head.py \\
        --backbone alphatrain/data/pillar3b_epoch_20.pt \\
        --data alphatrain/data/stationary_risk_v1.pt \\
        --out alphatrain/data/stationary_risk_head_v1.pt \\
        --epochs 5 --batch-size 4096
"""
from __future__ import annotations
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation


LABEL_NAMES = ['min_empty_H', 'min_lec_H', 'empty_delta_H',
                'lec_delta_H', 'score_rate_H', 'clear_rate_H',
                'lec_under_10_frac_H', 'lec_under_15_frac_H',
                'lec_shortfall_15_H']
N_LABELS = 9


def build_obs_batch(boards_b, npos_b, ncol_b, nn_b):
    """Build (B, 18, 9, 9) obs from a batch of state tensors."""
    B = boards_b.shape[0]
    out = np.empty((B, 18, 9, 9), dtype=np.float32)
    for i in range(B):
        nr = np.zeros(3, dtype=np.intp)
        nc = np.zeros(3, dtype=np.intp)
        nco = np.zeros(3, dtype=np.intp)
        n = int(nn_b[i])
        for j in range(min(n, 3)):
            nr[j] = npos_b[i, j, 0].item()
            nc[j] = npos_b[i, j, 1].item()
            nco[j] = ncol_b[i, j].item()
        out[i] = build_observation(boards_b[i].numpy(), nr, nc, nco, n)
    return out


class StationaryRiskHeadGAP(nn.Module):
    """GAP head: backbone -> (B, C, 9, 9) -> GAP -> (B, C) -> MLP."""
    def __init__(self, in_channels=256, hidden=128, n_outputs=N_LABELS):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_outputs)

    def forward(self, feats_2d):
        x = feats_2d.mean(dim=(2, 3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class StationaryRiskHeadSpatial(nn.Module):
    """Spatial head: preserves 9x9 structure via 1x1 conv -> flatten -> MLP.
    Per ChatGPT 2026-05-23: LEC depends on graph topology, which compresses
    badly under GAP. Spatial projection keeps per-cell features.
    """
    def __init__(self, in_channels=256, proj_channels=32, hidden=256,
                  n_outputs=N_LABELS):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, proj_channels, 1)
        self.proj_bn = nn.BatchNorm2d(proj_channels)
        self.fc1 = nn.Linear(proj_channels * 81, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_outputs)

    def forward(self, feats_2d):
        x = F.relu(self.proj_bn(self.proj(feats_2d)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_head(head_type, in_channels, hidden, n_outputs):
    if head_type == 'gap':
        return StationaryRiskHeadGAP(in_channels, hidden, n_outputs)
    elif head_type == 'spatial':
        return StationaryRiskHeadSpatial(in_channels, 32, hidden, n_outputs)
    else:
        raise ValueError(f"unknown head_type {head_type}")


def compute_auc(y_true, y_score):
    """Simple ROC AUC. y_true binary, y_score continuous."""
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',
                         default='alphatrain/data/pillar3b_epoch_20.pt')
    parser.add_argument('--data',
                         default='alphatrain/data/stationary_risk_v1.pt')
    parser.add_argument('--out',
                         default='alphatrain/data/stationary_risk_head_v1.pt')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-frac', type=float, default=0.1,
                         help='Game-level split (windows from same seed '
                              'go to same split, no leakage).')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--head-type', default='gap', choices=['gap', 'spatial'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    net, _ = load_model(args.backbone, device,
                         fp16=(device.type != 'cpu'))
    net_dtype = next(net.parameters()).dtype
    for p in net.parameters():
        p.requires_grad_(False)
    net.train(False)

    with torch.no_grad():
        dummy = torch.zeros(1, 18, 9, 9, device=device, dtype=net_dtype)
        feats = net.backbone_features(dummy)
    feat_channels = feats.shape[1]
    print(f"Backbone feature channels: {feat_channels}", flush=True)

    print(f"Loading {args.data}...", flush=True)
    data = torch.load(args.data, weights_only=False)
    N = data['boards'].shape[0]
    labels = data['labels'].float()
    meta = data['meta']
    seeds = meta[:, 1].numpy()
    print(f"  N={N:,} windows, label shape {labels.shape}")
    print(f"  corpus_names: {data['corpus_names']}")

    corpus_seed_keys = (meta[:, 0].numpy().astype(np.int64) * 10_000_000
                         + seeds.astype(np.int64))
    uniq_keys = np.unique(corpus_seed_keys)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(uniq_keys)
    n_val_games = max(1, int(len(uniq_keys) * args.val_frac))
    val_keys = set(uniq_keys[:n_val_games].tolist())
    train_mask = np.array([k not in val_keys for k in corpus_seed_keys])
    val_mask = ~train_mask
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    print(f"  Train games: {len(uniq_keys) - n_val_games}, "
          f"val games: {n_val_games}")
    print(f"  Train windows: {len(train_idx):,}, "
          f"val windows: {len(val_idx):,}")

    train_labels = labels[train_idx]
    label_mean = train_labels.mean(dim=0)
    label_std = train_labels.std(dim=0).clamp_min(1e-6)
    print(f"\nLabel normalization (train set):")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<18s}: mu={label_mean[i]:>8.3f}  sigma={label_std[i]:>8.3f}")
    norm_labels = (labels - label_mean) / label_std

    head = make_head(args.head_type, feat_channels,
                      args.hidden, N_LABELS).to(device)
    print(f"Head type: {args.head_type}", flush=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    print(f"Head params: {sum(p.numel() for p in head.parameters()):,}",
          flush=True)

    boards = data['boards']
    next_pos = data['next_pos']
    next_col = data['next_col']
    n_next = data['n_next']

    def score_batch(indices):
        """Build obs + backbone features + head outputs in normalized space."""
        obs_np = build_obs_batch(boards[indices], next_pos[indices],
                                  next_col[indices], n_next[indices])
        obs_t = torch.from_numpy(obs_np).to(device=device, dtype=net_dtype)
        with torch.no_grad():
            feats = net.backbone_features(obs_t)
        feats_f = feats.float()
        return head(feats_f)

    best_val_mse = float('inf')
    for epoch in range(args.epochs):
        et = time.time()
        perm = np.random.permutation(train_idx)
        head.train(True)
        total_loss = 0.0
        n_batches = 0
        for batch_start in range(0, len(perm), args.batch_size):
            batch_idx = perm[batch_start:batch_start + args.batch_size]
            pred = score_batch(batch_idx)
            target = norm_labels[batch_idx].to(device)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
            if n_batches % 5 == 0:
                print(f"  ep{epoch+1} batch {n_batches} loss={loss.item():.4f}",
                      flush=True)
        train_loss = total_loss / max(1, n_batches)

        head.train(False)
        val_preds = []
        val_targets = []
        for batch_start in range(0, len(val_idx), args.batch_size):
            batch = val_idx[batch_start:batch_start + args.batch_size]
            with torch.no_grad():
                p = score_batch(batch)
            val_preds.append(p.cpu().float().numpy())
            val_targets.append(norm_labels[batch].numpy())
        val_pred = np.concatenate(val_preds, axis=0)
        val_tgt = np.concatenate(val_targets, axis=0)
        val_mse = ((val_pred - val_tgt) ** 2).mean()

        val_pred_denorm = val_pred * label_std.numpy() + label_mean.numpy()
        val_labels_actual = labels[val_idx].numpy()

        aucs = {}
        # cols are positions in LABEL_NAMES
        risk_specs = [
            ('min_lec<10', 1, np.less, 10, True),
            ('min_lec<15', 1, np.less, 15, True),
            ('min_empty<30', 0, np.less, 30, True),
            ('min_empty<25', 0, np.less, 25, True),
            ('lec_delta<=-10', 3, np.less_equal, -10, True),
            ('lec_under_10_frac>0.2', 6, np.greater, 0.2, False),
            ('lec_under_15_frac>0.5', 7, np.greater, 0.5, False),
            ('lec_shortfall>3', 8, np.greater, 3.0, False),
        ]
        for thr_name, col, op, thr, low_is_risky in risk_specs:
            y_true = op(val_labels_actual[:, col], thr).astype(int)
            # If "less" criteria, low predicted value = high risk -> flip.
            # If "greater" (fractions/shortfall): high predicted = high risk.
            y_score = (-val_pred_denorm[:, col] if low_is_risky
                        else val_pred_denorm[:, col])
            aucs[thr_name] = compute_auc(y_true, y_score)

        elapsed = time.time() - et
        auc_str = '  '.join(f"{k}:{v:.3f}" for k, v in aucs.items())
        print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}  "
              f"val_mse={val_mse:.4f}  [{elapsed:.0f}s]")
        print(f"  AUCs: {auc_str}", flush=True)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                'head_state': head.state_dict(),
                'head_type': args.head_type,
                'label_mean': label_mean.numpy(),
                'label_std': label_std.numpy(),
                'label_names': LABEL_NAMES,
                'feat_channels': feat_channels,
                'hidden': args.hidden,
                'epoch': epoch + 1,
                'val_mse': val_mse,
                'aucs': aucs,
            }, args.out)
            print(f"  ** Saved best to {args.out} **", flush=True)

    print(f"\n=== Decision gate ===")
    print(f"  AUC > 0.75 -> Phase 1 confirmed -> proceed to Phase 2")
    print(f"  AUC < 0.65 -> backbone features don't carry stationary risk")
    print(f"  0.65 <= AUC <= 0.75 -> marginal; investigate further")
    best = torch.load(args.out, weights_only=False)
    print(f"  Best AUCs: {best['aucs']}")


if __name__ == '__main__':
    main()
