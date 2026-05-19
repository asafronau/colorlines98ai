"""Train ValueHead on top of a frozen PolicyNet backbone.

Phase 3 step 4. Reads the multi-horizon survival tensor produced by
build_value_targets.py, builds the 18-channel observation on the GPU
each batch, runs the frozen pillar2y2 backbone, and trains a small
ValueHead with masked BCE per horizon.

Validation uses the K=N rollout soft-label set produced by
build_validation_set.py — gold-standard P_H probability targets,
not single-trajectory noisy labels.

Usage:
    python -m alphatrain.scripts.train_value_head \\
        --backbone alphatrain/data/pillar2y2_epoch_40.pt \\
        --train-data alphatrain/data/value_targets_v11.pt \\
        --val-data alphatrain/data/value_val_K64.pt \\
        --epochs 5 --batch-size 4096 --lr 1e-3 \\
        --out alphatrain/data/value_head_v11.pt
"""

import os
import time
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphatrain.evaluate import load_model
from alphatrain.value_head import (
    ValueHead, SURVIVAL_HORIZONS, NUM_HORIZONS,
    DEFAULT_HORIZON_WEIGHTS, save as save_value_head,
)


def _maybe_build_observation_batch(boards_b, npos_b, ncol_b, nn_b, device):
    """Build (B, 18, 9, 9) float32 observation batch on the device.

    Falls back to per-sample build_observation if a batched JIT helper
    isn't available — boards/next_balls handling lives in observation.py.
    """
    try:
        from alphatrain.observation import build_observation_batch as _b
        # Batched JIT path
        return _b(boards_b.numpy(), npos_b.numpy(),
                  ncol_b.numpy(), nn_b.numpy())
    except Exception:
        from alphatrain.observation import build_observation
        # Per-sample fallback
        out = np.empty((boards_b.shape[0], 18, 9, 9), dtype=np.float32)
        for i in range(boards_b.shape[0]):
            nr = np.zeros(3, dtype=np.intp)
            nc = np.zeros(3, dtype=np.intp)
            nco = np.zeros(3, dtype=np.intp)
            nn_i = int(nn_b[i].item())
            for j in range(min(nn_i, 3)):
                nr[j] = npos_b[i, j, 0].item()
                nc[j] = npos_b[i, j, 1].item()
                nco[j] = ncol_b[i, j].item()
            out[i] = build_observation(
                boards_b[i].numpy(), nr, nc, nco, nn_i)
        return out


def _shuffle_indices(n, rng):
    perm = np.arange(n)
    rng.shuffle(perm)
    return perm


def _bce_per_horizon(logits, labels, masks):
    """Masked BCE-with-logits, per horizon and across horizons.

    Args:
        logits: (B, H) float
        labels: (B, H) {0, 1}
        masks:  (B, H) {0, 1} — 1 = use this label in loss

    Returns:
        scalar loss (mean over usable (sample, horizon) pairs),
        per-horizon loss array (H,).
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction='none')   # (B, H)
    masked = bce * masks.float()
    # Per-horizon: sum / count of mask
    denom_h = masks.float().sum(dim=0).clamp_min(1.0)
    per_horizon = masked.sum(dim=0) / denom_h
    # Overall scalar
    total_denom = masks.float().sum().clamp_min(1.0)
    overall = masked.sum() / total_denom
    return overall, per_horizon


def _mse_per_horizon(preds, targets):
    """MSE loss per horizon and across horizons (no masking — every target valid).

    Args:
        preds: (B, H) float — raw regression outputs (no sigmoid)
        targets: (B, H) float — continuous [0, 1] targets

    Returns:
        scalar loss, per-horizon loss array.
    """
    sq = (preds - targets).pow(2)            # (B, H)
    per_horizon = sq.mean(dim=0)             # (H,)
    overall = sq.mean()
    return overall, per_horizon


def _calibration_metrics(probs, p_hat, n_bins=10):
    """Rough calibration: bucket predicted probs, compare to mean p_hat
    in each bucket. Returns max abs gap across buckets."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    gaps = []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() < 5:
            continue
        gap = abs(probs[sel].mean() - p_hat[sel].mean())
        gaps.append(gap)
    return float(max(gaps)) if gaps else 0.0


def _eval_on_val_set(net, head, val_data, device, fp16):
    """Compute calibration / ranking metrics on the K-rollout val set.

    Returns dict with per-horizon Pearson r between predicted prob and
    P_hat, mean abs error, max calibration gap.
    """
    boards = val_data['boards']
    npos = val_data['next_pos']
    ncol = val_data['next_col']
    nn = val_data['n_next']
    p_hat = val_data['p_hat'].numpy()  # (N, H)
    N = boards.shape[0]

    net_dtype = torch.float16 if fp16 else torch.float32
    all_probs = np.zeros_like(p_hat)
    BATCH = 512
    head.train(False)
    for i in range(0, N, BATCH):
        bs = slice(i, min(i + BATCH, N))
        obs_np = _maybe_build_observation_batch(
            boards[bs], npos[bs], ncol[bs], nn[bs], device)
        obs_t = torch.from_numpy(obs_np).to(device=device, dtype=net_dtype)
        with torch.inference_mode():
            feats = net.backbone_features(obs_t)
            logits = head(feats.float())
            probs = torch.sigmoid(logits)
        all_probs[bs] = probs.cpu().numpy()

    metrics = {}
    for hi, h in enumerate(SURVIVAL_HORIZONS):
        pp = all_probs[:, hi]
        ph = p_hat[:, hi]
        # Pearson r — avoid divide-by-zero on degenerate sets
        if pp.std() < 1e-6 or ph.std() < 1e-6:
            r = 0.0
        else:
            r = float(np.corrcoef(pp, ph)[0, 1])
        mae = float(np.abs(pp - ph).mean())
        cal_gap = _calibration_metrics(pp, ph)
        metrics[f'H{h}_r'] = r
        metrics[f'H{h}_mae'] = mae
        metrics[f'H{h}_cal_gap'] = cal_gap
    return metrics, all_probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True,
                   help='Path to PolicyNet checkpoint (frozen during training)')
    p.add_argument('--train-data', required=True,
                   help='Path to value_targets_v11.pt from build_value_targets.py')
    p.add_argument('--val-data', required=False, default=None,
                   help='Path to K-rollout val set from build_validation_set.py')
    p.add_argument('--out', required=True,
                   help='Path to write the trained ValueHead checkpoint')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--hidden', type=int, default=32,
                   help='ValueHead hidden width')
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit-states', type=int, default=0,
                   help='Cap training set size (0 = use all). Useful for '
                        'fast iteration during development.')
    args = p.parse_args()

    if args.device:
        device_str = args.device
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'
    device = torch.device(device_str)
    fp16 = (device_str != 'cpu')

    # ── Load backbone (FROZEN) ──
    print(f"Loading backbone {args.backbone} on {device}...", flush=True)
    net, _ = load_model(args.backbone, device,
                        fp16=fp16, jit_trace=False)
    for p_ in net.parameters():
        p_.requires_grad_(False)
    net.train(False)
    backbone_channels = net.channels

    # ── Load training data (peek for target_type before building head) ──
    print(f"\nLoading training data {args.train_data}...", flush=True)
    train_data = torch.load(args.train_data, weights_only=False)
    target_type = train_data.get('target_type', 'survival')
    print(f"  target_type={target_type}", flush=True)

    boards = train_data['boards']
    npos = train_data['next_pos']
    ncol = train_data['next_col']
    nn_arr = train_data['n_next']
    is_train = train_data['is_train'].numpy()

    if target_type == 'density':
        targets_field = train_data['density_targets']
        masks_field = None
        horizons = tuple(train_data['horizons'])
        num_outputs = len(horizons)
    elif target_type == 'survival':
        targets_field = train_data['survive_labels']
        masks_field = train_data['survive_masks']
        horizons = SURVIVAL_HORIZONS
        num_outputs = NUM_HORIZONS
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    # ── Build value head (sized by target type) ──
    head = ValueHead(in_channels=backbone_channels, hidden=args.hidden,
                     num_outputs=num_outputs)
    head = head.to(device)
    n_head_params = sum(p.numel() for p in head.parameters())
    print(f"ValueHead: {n_head_params:,} params, hidden={args.hidden}, "
          f"num_outputs={num_outputs}, horizons={horizons}", flush=True)

    # Train/val split (val here is just for loss tracking — calibration
    # comes from the separate K-rollout val set).
    train_idxs = np.nonzero(is_train)[0]
    inner_val_idxs = np.nonzero(~is_train)[0]
    if args.limit_states > 0:
        train_idxs = train_idxs[:args.limit_states]
        inner_val_idxs = inner_val_idxs[:max(args.limit_states // 10, 1000)]
    print(f"Train: {len(train_idxs):,}  Inner-val: {len(inner_val_idxs):,}",
          flush=True)

    # ── Load K-rollout val set (calibration ground truth) ──
    # Skipped for 'density' mode (calibration metrics built for survival probs)
    val_data = None
    if target_type == 'survival' and args.val_data:
        print(f"Loading K-rollout val set {args.val_data}...", flush=True)
        val_data = torch.load(args.val_data, weights_only=False)
        print(f"  {val_data['boards'].shape[0]} states × K={val_data['rollout_K']} "
              f"rollouts, horizons={val_data['horizons']}", flush=True)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── Training loop ──
    rng = np.random.default_rng(args.seed)
    print(f"\n=== Training {args.epochs} epochs (bs={args.batch_size}, "
          f"lr={args.lr:.1e}) ===", flush=True)
    t0 = time.time()
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        head.train(True)
        perm = _shuffle_indices(len(train_idxs), rng)
        train_idxs_ep = train_idxs[perm]
        n_batches = math.ceil(len(train_idxs_ep) / args.batch_size)

        running_loss = 0.0
        running_per_h = np.zeros(num_outputs)
        running_n = 0
        for bi in range(n_batches):
            slc = train_idxs_ep[bi * args.batch_size:(bi + 1) * args.batch_size]
            obs_np = _maybe_build_observation_batch(
                boards[slc], npos[slc], ncol[slc], nn_arr[slc], device)
            obs_t = torch.from_numpy(obs_np).to(device=device,
                                                 dtype=torch.float16 if fp16 else torch.float32)
            with torch.inference_mode():
                feats = net.backbone_features(obs_t)

            feats = feats.float().detach()  # detach: backbone is frozen
            logits = head(feats)

            if target_type == 'density':
                targets_t = targets_field[slc].to(device).float()
                loss, per_h = _mse_per_horizon(logits, targets_t)
            else:
                labels_t = targets_field[slc].to(device).long()
                masks_t = masks_field[slc].to(device).float()
                loss, per_h = _bce_per_horizon(logits, labels_t, masks_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(slc)
            running_per_h += per_h.detach().cpu().numpy() * len(slc)
            running_n += len(slc)

            if (bi + 1) % 100 == 0 or bi == n_batches - 1:
                avg = running_loss / running_n
                avg_h = running_per_h / running_n
                elapsed = time.time() - t0
                rate = running_n / max(elapsed, 1e-3)
                h_str = ' '.join(f"{v:.3f}" for v in avg_h)
                print(f"  [{bi+1}/{n_batches}] loss={avg:.4f}  "
                      f"per-H=[{h_str}]  "
                      f"{rate:.0f} samples/s ({elapsed:.0f}s)", flush=True)

        # Inner val (single-trajectory, noisy)
        head.train(False)
        with torch.inference_mode():
            iv_loss = 0.0
            iv_per_h = np.zeros(num_outputs)
            iv_n = 0
            for bi in range(0, len(inner_val_idxs), args.batch_size):
                slc = inner_val_idxs[bi:bi + args.batch_size]
                obs_np = _maybe_build_observation_batch(
                    boards[slc], npos[slc], ncol[slc], nn_arr[slc], device)
                obs_t = torch.from_numpy(obs_np).to(
                    device=device,
                    dtype=torch.float16 if fp16 else torch.float32)
                feats = net.backbone_features(obs_t).float()
                logits = head(feats)
                if target_type == 'density':
                    targets_t = targets_field[slc].to(device).float()
                    l, p_h = _mse_per_horizon(logits, targets_t)
                else:
                    labels_t = targets_field[slc].to(device).long()
                    masks_t = masks_field[slc].to(device).float()
                    l, p_h = _bce_per_horizon(logits, labels_t, masks_t)
                iv_loss += l.item() * len(slc)
                iv_per_h += p_h.cpu().numpy() * len(slc)
                iv_n += len(slc)
            iv_loss /= max(iv_n, 1)
            iv_per_h /= max(iv_n, 1)

        # K-rollout calibration metrics (survival mode only)
        cal_metrics = {}
        if target_type == 'survival' and val_data is not None:
            cal_metrics, _ = _eval_on_val_set(
                net, head, val_data, device, fp16)
        print(f"\nEpoch {epoch+1}/{args.epochs}: "
              f"train_loss={running_loss/running_n:.4f}  "
              f"inner_val_loss={iv_loss:.4f}", flush=True)
        iv_h_str = ' '.join(f"{v:.4f}" for v in iv_per_h)
        print(f"  per-H val MSE/BCE: [{iv_h_str}]", flush=True)
        if cal_metrics:
            for hi, h in enumerate(horizons):
                print(f"  H={h}: r={cal_metrics[f'H{h}_r']:.3f}  "
                      f"mae={cal_metrics[f'H{h}_mae']:.3f}  "
                      f"cal_gap={cal_metrics[f'H{h}_cal_gap']:.3f}", flush=True)

        if iv_loss < best_val_loss:
            best_val_loss = iv_loss
            save_value_head(
                head, args.out, backbone_path=args.backbone,
                train_args=vars(args), horizons=horizons,
                target_type=target_type,
                val_metrics={
                    'inner_val_loss': iv_loss,
                    'calibration': cal_metrics,
                    'epoch': epoch + 1,
                })
            print(f"  ** New best, saved to {args.out} **", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f}m. Best inner-val loss: "
          f"{best_val_loss:.4f}", flush=True)


if __name__ == '__main__':
    main()
