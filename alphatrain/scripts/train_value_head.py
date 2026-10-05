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
    ValueHead, SpatialValueHead, SURVIVAL_HORIZONS, NUM_HORIZONS,
    DEFAULT_HORIZON_WEIGHTS, save as save_value_head, save_spatial,
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


def _bce_per_horizon(logits, labels, masks, weights=None):
    """Masked BCE-with-logits, per horizon and across horizons.

    Args:
        logits: (B, H) float
        labels: (B, H) {0, 1}
        masks:  (B, H) {0, 1} — 1 = use this label in loss
        weights: optional (B,) importance weights for unbiased subsampling

    Returns:
        scalar loss (mean over usable (sample, horizon) pairs),
        per-horizon loss array (H,).
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction='none')   # (B, H)
    w = masks.float() if weights is None else masks.float() * weights[:, None].float()
    masked = bce * w
    # Per-horizon: sum / count of weighted mask
    denom_h = w.sum(dim=0).clamp_min(1.0)
    per_horizon = masked.sum(dim=0) / denom_h
    # Overall scalar
    total_denom = w.sum().clamp_min(1.0)
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


def _eval_on_val_set(net, head, val_data, device, fp16, cat_obs=False):
    """Compute calibration / ranking metrics on the K-rollout val set.

    Returns dict with per-horizon Pearson r between predicted prob and
    P_hat, mean abs error, max calibration gap.
    """
    boards = val_data['boards']
    npos = val_data['next_pos']
    ncol = val_data['next_col']
    nn_arr = val_data['n_next']
    p_hat = val_data['p_hat'].numpy()  # (N, H)
    N = boards.shape[0]

    net_dtype = next(net.parameters()).dtype
    all_probs = np.zeros_like(p_hat)
    BATCH = 512
    head.train(False)
    for i in range(0, N, BATCH):
        bs = slice(i, min(i + BATCH, N))
        obs_np = _maybe_build_observation_batch(
            boards[bs], npos[bs], ncol[bs], nn_arr[bs], device)
        obs_t = torch.from_numpy(obs_np).to(device=device, dtype=net_dtype)
        with torch.inference_mode():
            feats = net.backbone_features(obs_t).float()
            if cat_obs:
                feats = torch.cat([feats, obs_t.float()], dim=1)
            logits = head(feats)
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


def _build_gpu_obs(ds_helper, boards_b, npos_b, ncol_b, nn_b, device, augment=False,
                   obs_inv_luts=None, line_inv_luts=None):
    """Build (B, 18, 9, 9) observations on GPU with optional D4 x S7 augmentation."""
    B = boards_b.shape[0]
    boards_d = boards_b.to(device)
    npos_d = npos_b.to(device)
    ncol_d = ncol_b.to(device)
    nn_d = nn_b.to(device)
    if augment:
        perm_1_7 = torch.argsort(torch.rand(B, 7, device=device), dim=1) + 1
        perm = torch.zeros(B, 8, dtype=torch.long, device=device)
        perm[:, 1:8] = perm_1_7
        boards_d = torch.gather(perm, 1, boards_d.long().view(B, -1)).view(B, 9, 9).to(torch.int8)
        ncol_d = torch.gather(perm, 1, ncol_d.long()).to(torch.int8)
    obs = ds_helper._build_obs_core(boards_d, next_pos=npos_d, next_col=ncol_d, n_next=nn_d)
    if augment and obs_inv_luts is not None:
        from alphatrain.dataset import _transform_observation
        transforms = torch.randint(0, 8, (B,), device=device, dtype=torch.long)
        for t in range(1, 8):
            mask = transforms == t
            if mask.any():
                obs[mask] = _transform_observation(obs[mask], obs_inv_luts[t], line_inv_luts[t])
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True,
                   help='Path to PolicyNet checkpoint (frozen by default during training)')
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
    p.add_argument('--arch', choices=['gap', 'spatial'], default='gap',
                   help='gap: 1x1 conv + global average pool + linear (~3k params); spatial: SpatialValueHead '
                        '(1x1 conv + two 3x3 residual blocks + mean/max pool + MLP), which keeps the board geometry')
    p.add_argument('--spatial-mid', type=int, default=64, help='SpatialValueHead width')
    p.add_argument('--cat-obs', action='store_true',
                   help='Concatenate the 18-channel observation to backbone_features (in_channels = C + 18)')
    p.add_argument('--augment', action='store_true',
                   help='Apply random D4 dihedral and S7 color permutation augmentation on GPU during training')
    p.add_argument('--gpu-obs', action='store_true',
                   help='Build observations on GPU (automatically enabled when --augment is set)')
    p.add_argument('--neg-subsample', type=int, default=1,
                   help='Subsample all-survive [1,1,1,1] training states by 1/M each epoch with importance weight M '
                        '(1 = keep all states; >1 = unbiased speedup focusing compute on death/slide trajectories)')
    p.add_argument('--unfreeze-blocks', type=int, default=0,
                   help='Number of final backbone residual blocks to unfreeze (0 = frozen backbone, -1 = full backbone)')
    p.add_argument('--backbone-lr', type=float, default=1e-4,
                   help='Learning rate for unfrozen backbone parameters')
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
    unfreeze = (args.unfreeze_blocks != 0)
    use_gpu_obs = args.gpu_obs or args.augment or unfreeze

    # ── Load backbone ──
    print(f"Loading backbone {args.backbone} on {device}...", flush=True)
    # Keep master weights in fp32 when unfreezing so AdamW updates do not underflow in fp16
    net, _ = load_model(args.backbone, device,
                        fp16=(fp16 and not unfreeze), jit_trace=False)
    if unfreeze:
        from alphatrain.inference_cpp.export_ts import fp16_safe_batchnorm
        fp16_safe_batchnorm(net)
    for p_ in net.parameters():
        p_.requires_grad_(False)
    net.train(False)
    if unfreeze:
        if args.unfreeze_blocks < 0:
            for p_ in net.parameters():
                p_.requires_grad_(True)
            # Freeze unused policy head parameters
            for name, p_ in net.named_parameters():
                if any(name.startswith(pf) for pf in ('policy_', 'pair_', 'line_', 'adj_', 'src_', 'dst_')):
                    p_.requires_grad_(False)
        else:
            n_blk = len(net.blocks)
            k_unfreeze = min(args.unfreeze_blocks, n_blk)
            for blk in net.blocks[n_blk - k_unfreeze:]:
                for p_ in blk.parameters():
                    p_.requires_grad_(True)
            if hasattr(net, 'backbone_bn'):
                for p_ in net.backbone_bn.parameters():
                    p_.requires_grad_(True)
        n_unfrozen = sum(p_.numel() for p_ in net.parameters() if p_.requires_grad)
        print(f"  Unfroze {n_unfrozen:,} backbone parameters (unfreeze_blocks={args.unfreeze_blocks}, "
              f"backbone_lr={args.backbone_lr:.1e})", flush=True)
    backbone_channels = net.channels + (18 if args.cat_obs else 0)

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
    if args.arch == 'spatial':
        head = SpatialValueHead(in_channels=backbone_channels, mid_channels=args.spatial_mid,
                                num_outputs=num_outputs)
    else:
        head = ValueHead(in_channels=backbone_channels, hidden=args.hidden,
                         num_outputs=num_outputs)
    head = head.to(device)
    n_head_params = sum(p.numel() for p in head.parameters())
    print(f"{type(head).__name__}: {n_head_params:,} params, in_channels={backbone_channels}, "
          f"hidden={args.hidden}, num_outputs={num_outputs}, horizons={horizons}", flush=True)

    # Train/val split (val here is just for loss tracking — calibration
    # comes from the separate K-rollout val set).
    train_idxs = np.nonzero(is_train)[0]
    inner_val_idxs = np.nonzero(~is_train)[0]
    if args.limit_states > 0:
        train_idxs = train_idxs[:args.limit_states]
        inner_val_idxs = inner_val_idxs[:max(args.limit_states // 10, 1000)]
    print(f"Train: {len(train_idxs):,}  Inner-val: {len(inner_val_idxs):,}",
          flush=True)

    # Pre-split train_idxs for importance-weighted negative subsampling if requested
    pos_train_idxs = None
    neg_train_idxs = None
    if args.neg_subsample > 1 and target_type == 'survival':
        train_labels_np = targets_field[train_idxs].numpy()
        has_death = (train_labels_np.min(axis=1) == 0)
        pos_train_idxs = train_idxs[has_death]
        neg_train_idxs = train_idxs[~has_death]
        print(f"  Importance-weighted subsampling (M={args.neg_subsample}): "
              f"keeping 100% of {len(pos_train_idxs):,} death-window states (w=1.0) + "
              f"1/{args.neg_subsample} of {len(neg_train_idxs):,} calm states (w={args.neg_subsample}.0) per epoch",
              flush=True)

    ds_helper = None
    obs_inv_luts = None
    line_inv_luts = None
    if use_gpu_obs:
        from alphatrain.dataset import TensorDatasetGPU, _OBS_INV_LUTS, _LINE_INV_LUTS
        ds_helper = TensorDatasetGPU.__new__(TensorDatasetGPU)
        ds_helper.device = device
        obs_inv_luts = torch.tensor(np.stack(_OBS_INV_LUTS), dtype=torch.long, device=device)
        line_inv_luts = torch.tensor(np.stack(_LINE_INV_LUTS), dtype=torch.long, device=device)

    # ── Load K-rollout val set (calibration ground truth) ──
    val_data = None
    if target_type == 'survival' and args.val_data:
        print(f"Loading K-rollout val set {args.val_data}...", flush=True)
        val_data = torch.load(args.val_data, weights_only=False)
        print(f"  {val_data['boards'].shape[0]} states × K={val_data['rollout_K']} "
              f"rollouts, horizons={val_data['horizons']}", flush=True)

    # ── Optimizer ──
    param_groups = [{'params': list(head.parameters()), 'lr': args.lr}]
    if unfreeze:
        backbone_params = [p_ for p_ in net.parameters() if p_.requires_grad]
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Precompute fullness balls on inner_val_idxs for within-bin H=200 AUC tracking
    iv_balls = (boards[inner_val_idxs] > 0).sum(dim=(1, 2)).numpy() if target_type == 'survival' else None

    # ── Training loop ──
    rng = np.random.default_rng(args.seed)
    print(f"\n=== Training {args.epochs} epochs (bs={args.batch_size}, "
          f"lr={args.lr:.1e}, augment={args.augment}, unfreeze={args.unfreeze_blocks}) ===", flush=True)
    t0 = time.time()
    best_val_loss = float('inf')
    best_bin_auc = -1.0

    for epoch in range(args.epochs):
        head.train(True)
        net.train(False)  # keep backbone BatchNorm running stats frozen
        if pos_train_idxs is not None:
            n_neg_draw = max(1, len(neg_train_idxs) // args.neg_subsample)
            neg_draw = rng.choice(neg_train_idxs, size=n_neg_draw, replace=False)
            ep_idxs = np.concatenate([pos_train_idxs, neg_draw])
            ep_weights = np.concatenate([
                np.ones(len(pos_train_idxs), dtype=np.float32),
                np.full(len(neg_draw), float(args.neg_subsample), dtype=np.float32),
            ])
            perm = _shuffle_indices(len(ep_idxs), rng)
            train_idxs_ep = ep_idxs[perm]
            train_weights_ep = ep_weights[perm]
        else:
            perm = _shuffle_indices(len(train_idxs), rng)
            train_idxs_ep = train_idxs[perm]
            train_weights_ep = None
        n_batches = math.ceil(len(train_idxs_ep) / args.batch_size)

        running_loss = 0.0
        running_per_h = np.zeros(num_outputs)
        running_n = 0
        for bi in range(n_batches):
            slc = train_idxs_ep[bi * args.batch_size:(bi + 1) * args.batch_size]
            w_t = (torch.from_numpy(train_weights_ep[bi * args.batch_size:(bi + 1) * args.batch_size]).to(device)
                   if train_weights_ep is not None else None)
            if use_gpu_obs:
                obs_t = _build_gpu_obs(
                    ds_helper, boards[slc], npos[slc], ncol[slc], nn_arr[slc],
                    device, augment=args.augment, obs_inv_luts=obs_inv_luts, line_inv_luts=line_inv_luts)
                if fp16 and not unfreeze:
                    obs_t = obs_t.half()
            else:
                obs_np = _maybe_build_observation_batch(
                    boards[slc], npos[slc], ncol[slc], nn_arr[slc], device)
                obs_t = torch.from_numpy(obs_np).to(device=device,
                                                     dtype=torch.float16 if fp16 else torch.float32)
            if unfreeze:
                if args.unfreeze_blocks > 0:
                    n_blk = len(net.blocks)
                    k_un = min(args.unfreeze_blocks, n_blk)
                    with torch.no_grad():
                        out = net.stem(obs_t)
                        for blk in net.blocks[:n_blk - k_un]:
                            out = blk(out)
                    out = out.detach()
                    for blk in net.blocks[n_blk - k_un:]:
                        out = blk(out)
                    feats = F.relu(net.backbone_bn(out))
                else:
                    feats = net.backbone_features(obs_t)
                if args.cat_obs:
                    feats = torch.cat([feats, obs_t.float()], dim=1)
                logits = head(feats)
            else:
                with torch.no_grad():
                    feats = net.backbone_features(obs_t)
                feats = feats.float().detach()
                if args.cat_obs:
                    feats = torch.cat([feats, obs_t.float().detach()], dim=1)
                logits = head(feats)

            if target_type == 'density':
                targets_t = targets_field[slc].to(device).float()
                loss, per_h = _mse_per_horizon(logits, targets_t)
            else:
                labels_t = targets_field[slc].to(device).long()
                masks_t = masks_field[slc].to(device).float()
                loss, per_h = _bce_per_horizon(logits, labels_t, masks_t, weights=w_t)

            optimizer.zero_grad()
            loss.backward()
            if unfreeze:
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            batch_w = float(w_t.sum().item()) if w_t is not None else float(len(slc))
            running_loss += loss.item() * batch_w
            running_per_h += per_h.detach().cpu().numpy() * batch_w
            running_n += batch_w

            if (bi + 1) % 100 == 0 or bi == n_batches - 1:
                avg = running_loss / running_n
                avg_h = running_per_h / running_n
                elapsed = time.time() - t0
                rate = (bi + 1) * args.batch_size / max(elapsed, 1e-3)
                h_str = ' '.join(f"{v:.3f}" for v in avg_h)
                print(f"  [{bi+1}/{n_batches}] loss={avg:.4f}  "
                      f"per-H=[{h_str}]  "
                      f"{rate:.0f} samples/s ({elapsed:.0f}s)", flush=True)

        # Inner val (single-trajectory, noisy)
        head.train(False)
        net.train(False)
        iv_probs_200 = np.zeros(len(inner_val_idxs), dtype=np.float32) if target_type == 'survival' else None
        with torch.inference_mode():
            iv_loss = 0.0
            iv_per_h = np.zeros(num_outputs)
            iv_n = 0
            for bi in range(0, len(inner_val_idxs), args.batch_size):
                slc = inner_val_idxs[bi:bi + args.batch_size]
                if use_gpu_obs:
                    obs_t = _build_gpu_obs(
                        ds_helper, boards[slc], npos[slc], ncol[slc], nn_arr[slc],
                        device, augment=False)
                    if fp16 and not unfreeze:
                        obs_t = obs_t.half()
                else:
                    obs_np = _maybe_build_observation_batch(
                        boards[slc], npos[slc], ncol[slc], nn_arr[slc], device)
                    obs_t = torch.from_numpy(obs_np).to(
                        device=device,
                        dtype=torch.float16 if fp16 else torch.float32)
                feats = net.backbone_features(obs_t).float()
                if args.cat_obs:
                    feats = torch.cat([feats, obs_t.float()], dim=1)
                logits = head(feats)
                if iv_probs_200 is not None:
                    iv_probs_200[bi:bi + len(slc)] = torch.sigmoid(logits[:, -1]).cpu().numpy()
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

        # Within-fullness-bin H=200 ROC AUC on inner_val
        mean_bin_auc = float('nan')
        bin_aucs = []
        if iv_probs_200 is not None:
            from alphatrain.scripts.value_sees_slide import BINS, roc_auc
            iv_labels_200 = targets_field[inner_val_idxs, -1].numpy()
            iv_masks_200 = masks_field[inner_val_idxs, -1].numpy()
            for lo, hi in BINS:
                valid = (iv_balls >= lo) & (iv_balls < hi) & (iv_masks_200 > 0)
                y_die = (iv_labels_200[valid] == 0).astype(np.int32)
                s_die = 1.0 - iv_probs_200[valid]
                bin_aucs.append(roc_auc(y_die, s_die))
            if not all(np.isnan(bin_aucs)):
                mean_bin_auc = float(np.nanmean(bin_aucs))

        # K-rollout calibration metrics (survival mode only)
        cal_metrics = {}
        if target_type == 'survival' and val_data is not None:
            cal_metrics, _ = _eval_on_val_set(
                net, head, val_data, device, fp16, cat_obs=args.cat_obs)
        print(f"\nEpoch {epoch+1}/{args.epochs}: "
              f"train_loss={running_loss/running_n:.4f}  "
              f"inner_val_loss={iv_loss:.4f}  "
              f"mean_H200_bin_AUC={mean_bin_auc:.4f}", flush=True)
        iv_h_str = ' '.join(f"{v:.4f}" for v in iv_per_h)
        print(f"  per-H val MSE/BCE: [{iv_h_str}]", flush=True)
        if bin_aucs:
            b_str = ' / '.join(f"{a:.3f}" for a in bin_aucs)
            print(f"  H=200 bin AUCs [36..62]: {b_str}", flush=True)
        if cal_metrics:
            for hi, h in enumerate(horizons):
                print(f"  H={h}: r={cal_metrics[f'H{h}_r']:.3f}  "
                      f"mae={cal_metrics[f'H{h}_mae']:.3f}  "
                      f"cal_gap={cal_metrics[f'H{h}_cal_gap']:.3f}", flush=True)

        improved = (mean_bin_auc > best_bin_auc) if not np.isnan(mean_bin_auc) else (iv_loss < best_val_loss)
        if iv_loss < best_val_loss:
            best_val_loss = iv_loss
        if not np.isnan(mean_bin_auc) and mean_bin_auc > best_bin_auc:
            best_bin_auc = mean_bin_auc
        if improved:
            metrics = {
                'inner_val_loss': iv_loss,
                'mean_bin_auc_H200': mean_bin_auc,
                'bin_aucs_H200': bin_aucs,
                'calibration': cal_metrics,
                'epoch': epoch + 1,
            }
            if args.arch == 'spatial':
                save_spatial(head, args.out, backbone_path=args.backbone, train_args=vars(args),
                             val_metrics=metrics, target_type=target_type, horizons=horizons)
            else:
                save_value_head(
                    head, args.out, backbone_path=args.backbone,
                    train_args=vars(args), horizons=horizons,
                    target_type=target_type, val_metrics=metrics)
            if unfreeze or args.cat_obs:
                ck = torch.load(args.out, map_location='cpu', weights_only=False)
                ck['cat_obs'] = bool(args.cat_obs)
                if unfreeze:
                    ck['backbone_state_dict'] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                torch.save(ck, args.out)
            print(f"  ** New best (mean_H200_bin_AUC={mean_bin_auc:.4f}, iv_loss={iv_loss:.4f}), saved to {args.out} **", flush=True)

    print(f"\nDone in {(time.time()-t0)/60:.1f}m. Best inner-val loss: "
          f"{best_val_loss:.4f}  Best mean H=200 bin AUC: {best_bin_auc:.4f}", flush=True)


if __name__ == '__main__':
    main()
