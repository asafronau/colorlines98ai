"""Train a SpatialValueHead via margin-weighted pairwise (BPR) ranking loss.

Input: pairwise tensor from build_pairwise_dataset.py
  (anchor, win_afterstate, lose_afterstate, margin, metric)

Loss per pair:  margin * -log_sigmoid(V_w - V_l)
  - V_w, V_l are the head's scalar outputs on win/lose afterstates
  - margin scales each pair's contribution by ChatGPT recommendation

Validation split BY anchor_board hash, not by pair — prevents sibling leakage
(pairs from the same anchor never appear in both train and val).

Usage:
    python -m alphatrain.scripts.train_ranking_head \\
        --backbone alphatrain/data/pillar2z_epoch_19.pt \\
        --pairwise alphatrain/data/pairwise_v12.pt \\
        --epochs 15 --batch-size 512 --lr 1e-3 \\
        --device mps \\
        --out alphatrain/data/value_head_spatial.pt
"""

import os
import time
import math
import hashlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation
from alphatrain.value_head import SpatialValueHead, save_spatial, load_spatial


def board_hash_to_bucket(board_int8, n_buckets=10):
    """Deterministic hash-based bucketing for train/val split by anchor board."""
    bs = board_int8.tobytes()
    h = hashlib.md5(bs).digest()
    return int.from_bytes(h[:4], 'little') % n_buckets


def build_obs_array(boards, next_pos, next_col, n_next):
    """Loop-build 18-channel obs from (board, next_balls) state.

    Slow Python loop but ~3000 states is trivial. Used at startup to
    pre-build all observations; the train loop is GPU-bound.
    """
    N = len(boards)
    out = np.zeros((N, 18, 9, 9), dtype=np.float32)
    for i in range(N):
        b = np.asarray(boards[i], dtype=np.int8)
        nn_i = int(n_next[i])
        nr = np.asarray(next_pos[i, :, 0], dtype=np.intp)
        nc = np.asarray(next_pos[i, :, 1], dtype=np.intp)
        ncol = np.asarray(next_col[i], dtype=np.intp)
        out[i] = build_observation(b, nr, nc, ncol, nn_i)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True)
    p.add_argument('--pairwise', required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--device', default='mps')
    p.add_argument('--val-bucket', type=int, default=0,
                   help='Bucket index for val (anchors hash%%10 == this).')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--mid-channels', type=int, default=64,
                   help='SpatialValueHead intermediate channels (capacity knob).')
    p.add_argument('--unweighted', action='store_true',
                   help='Use unweighted BPR (ignore per-pair margins). The default '
                        'margin-weighted form has mixed units (cap_rate 0-1 vs turns 50+ '
                        'vs score 100s) and effectively ignores cap_rate pairs.')
    p.add_argument('--metric-filter', default=None,
                   help='Comma-separated list of metric names to keep '
                        '(e.g. "cap_rate" or "cap_rate,turns"). Drops other pairs.')
    p.add_argument('--select-by', choices=['val_loss', 'val_acc'],
                   default='val_loss',
                   help='Best-checkpoint criterion. val_acc selects the highest '
                        'val accuracy seen (matters for MCTS use); val_loss is the '
                        'classical choice but for BPR can disagree with val_acc.')
    p.add_argument('--calibrate-terminal', action='store_true',
                   help='After training, shift the head bias so that p01 of train '
                        "outputs lands at +0.1. Terminal V=0 (used by MCTS) is then "
                        'below every live state, fixing the death-bias bug.')
    p.add_argument('--terminal-margin', type=float, default=0.1,
                   help='Target value for p01 of live-state outputs after offset.')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')

    # ── Backbone (frozen) ──
    print(f"Loading backbone {args.backbone}...", flush=True)
    net, _ = load_model(args.backbone, device, fp16=fp16, jit_trace=False)
    for p_ in net.parameters():
        p_.requires_grad = False
    backbone_channels = net.channels

    # ── Pairwise data ──
    print(f"Loading pairwise data {args.pairwise}...", flush=True)
    data = torch.load(args.pairwise, weights_only=False)
    N = data['anchor_boards'].shape[0]
    print(f"  pairs: {N:,}", flush=True)

    # Optional metric filter (e.g., keep only cap_rate pairs).
    if args.metric_filter:
        keep = set(s.strip() for s in args.metric_filter.split(','))
        metric_names = data['metric_names']
        keep_mask = np.array([m in keep for m in metric_names])
        before = N
        for k in list(data.keys()):
            v = data[k]
            if isinstance(v, torch.Tensor) and v.shape and v.shape[0] == N:
                data[k] = v[torch.from_numpy(keep_mask)]
            elif isinstance(v, list) and len(v) == N:
                data[k] = [v[i] for i in range(N) if keep_mask[i]]
        N = int(keep_mask.sum())
        print(f"  metric filter {keep}: {N:,} of {before:,} pairs kept",
              flush=True)

    # Train/val split by anchor_board hash
    anchor_bs = data['anchor_boards'].numpy()
    buckets = np.array([board_hash_to_bucket(anchor_bs[i]) for i in range(N)])
    val_mask = (buckets == args.val_bucket)
    train_idxs = np.nonzero(~val_mask)[0]
    val_idxs = np.nonzero(val_mask)[0]
    print(f"  train: {len(train_idxs):,}  val: {len(val_idxs):,}  "
          f"(split by anchor_board hash, val_bucket={args.val_bucket}/10)",
          flush=True)

    # ── Pre-build observations (one-time CPU cost, then GPU iteration) ──
    print(f"Building observations...", flush=True)
    t0 = time.time()
    win_obs = build_obs_array(
        data['win_boards'].numpy(),
        data['win_next_pos'].numpy(),
        data['win_next_col'].numpy(),
        data['win_n_next'].numpy())
    lose_obs = build_obs_array(
        data['lose_boards'].numpy(),
        data['lose_next_pos'].numpy(),
        data['lose_next_col'].numpy(),
        data['lose_n_next'].numpy())
    margins = data['margins'].numpy()
    print(f"  built in {time.time()-t0:.0f}s", flush=True)

    # To GPU as fp16 (matches backbone dtype)
    win_obs_t = torch.from_numpy(win_obs).to(device,
        dtype=torch.float16 if fp16 else torch.float32)
    lose_obs_t = torch.from_numpy(lose_obs).to(device,
        dtype=torch.float16 if fp16 else torch.float32)
    margins_t = torch.from_numpy(margins).to(device, dtype=torch.float32)

    # ── Head + optimizer ──
    head = SpatialValueHead(in_channels=backbone_channels,
                             mid_channels=args.mid_channels,
                             num_outputs=1).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"SpatialValueHead: {n_params:,} params, "
          f"mid_channels={args.mid_channels}", flush=True)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)

    def forward_pair(idxs):
        """Compute V_w, V_l, margins for a batch of pair indices."""
        idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=device)
        # no_grad (not inference_mode) — inference_mode tensors can't be
        # used downstream by autograd-tracked ops (the head's forward).
        with torch.no_grad():
            wf = net.backbone_features(win_obs_t.index_select(0, idxs_t)).float()
            lf = net.backbone_features(lose_obs_t.index_select(0, idxs_t)).float()
        V_w = head(wf).squeeze(-1)
        V_l = head(lf).squeeze(-1)
        m = margins_t.index_select(0, idxs_t)
        return V_w, V_l, m

    def compute_loss(V_w, V_l, m):
        """BPR. Either unweighted (-log_sigmoid mean) or margin-weighted."""
        diff = V_w - V_l
        per_pair = -F.logsigmoid(diff)
        if args.unweighted:
            loss = per_pair.mean()
        else:
            w = m / (m.mean().clamp_min(1e-6))
            loss = (per_pair * w).mean()
        accuracy = (diff > 0).float().mean()
        return loss, accuracy

    print(f"\n=== Training {args.epochs} epochs "
          f"(bs={args.batch_size}, lr={args.lr:.1e}, "
          f"select_by={args.select_by}) ===", flush=True)
    best_val_loss = float('inf')
    best_val_acc = -1.0
    best_epoch = 0
    t_start = time.time()
    rng = np.random.default_rng(args.seed)

    for epoch in range(args.epochs):
        head.train(True)
        perm = rng.permutation(train_idxs)
        n_batches = math.ceil(len(perm) / args.batch_size)
        running_loss = 0.0
        running_acc = 0.0
        running_n = 0
        for bi in range(n_batches):
            batch_idxs = perm[bi * args.batch_size:(bi + 1) * args.batch_size]
            V_w, V_l, m = forward_pair(batch_idxs)
            loss, acc = compute_loss(V_w, V_l, m)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(batch_idxs)
            running_acc += acc.item() * len(batch_idxs)
            running_n += len(batch_idxs)

        # Val
        head.train(False)
        with torch.inference_mode():
            v_loss_sum = 0.0
            v_acc_sum = 0.0
            v_n = 0
            for bi in range(0, len(val_idxs), args.batch_size):
                batch_idxs = val_idxs[bi:bi + args.batch_size]
                V_w, V_l, m = forward_pair(batch_idxs)
                vl, va = compute_loss(V_w, V_l, m)
                v_loss_sum += vl.item() * len(batch_idxs)
                v_acc_sum += va.item() * len(batch_idxs)
                v_n += len(batch_idxs)
            val_loss = v_loss_sum / max(v_n, 1)
            val_acc = v_acc_sum / max(v_n, 1)

        tl = running_loss / max(running_n, 1)
        ta = running_acc / max(running_n, 1)
        elapsed = time.time() - t_start
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"train loss={tl:.4f} acc={ta:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}  "
              f"[{elapsed:.0f}s]", flush=True)

        is_best = (val_acc > best_val_acc) if args.select_by == 'val_acc' \
            else (val_loss < best_val_loss)
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch + 1
            save_spatial(head, args.out, backbone_path=args.backbone,
                          train_args=vars(args),
                          val_metrics={
                              'epoch': epoch + 1,
                              'val_loss': val_loss,
                              'val_accuracy': val_acc,
                              'train_loss': tl,
                              'train_accuracy': ta,
                          })
            print(f"  ** New best, saved to {args.out} **", flush=True)

    print(f"\nTraining done in {(time.time()-t_start)/60:.1f}m. "
          f"Best @ epoch {best_epoch}: val_loss={best_val_loss:.4f}  "
          f"val_acc={best_val_acc:.3f}", flush=True)

    if args.calibrate_terminal:
        print(f"\n=== Terminal V calibration ===", flush=True)
        # Reload the best checkpoint (parameters at end of loop may not be best).
        best_head, _ = load_spatial(args.out, device=device)
        best_head.train(False)
        # Compute head outputs over ALL train pairs (V_w and V_l combined).
        outs = []
        with torch.inference_mode():
            for bi in range(0, len(train_idxs), args.batch_size):
                idxs = train_idxs[bi:bi + args.batch_size]
                idxs_t = torch.as_tensor(idxs, dtype=torch.long, device=device)
                wf = net.backbone_features(
                    win_obs_t.index_select(0, idxs_t)).float()
                lf = net.backbone_features(
                    lose_obs_t.index_select(0, idxs_t)).float()
                outs.append(best_head(wf).squeeze(-1).cpu().numpy())
                outs.append(best_head(lf).squeeze(-1).cpu().numpy())
        outs = np.concatenate(outs)
        p01 = float(np.percentile(outs, 1.0))
        p50 = float(np.percentile(outs, 50.0))
        p99 = float(np.percentile(outs, 99.0))
        offset = -p01 + args.terminal_margin
        print(f"  pre-offset:  p01={p01:.3f}  p50={p50:.3f}  p99={p99:.3f}",
              flush=True)
        print(f"  offset (added to fc2.bias): {offset:+.3f}", flush=True)
        print(f"  post-offset: p01={p01+offset:+.3f}  p50={p50+offset:+.3f}  "
              f"p99={p99+offset:+.3f}", flush=True)
        print(f"  → terminal V=0 is now {args.terminal_margin:.2f} below p01.",
              flush=True)
        with torch.no_grad():
            best_head.fc2.bias.add_(offset)
        save_spatial(best_head, args.out, backbone_path=args.backbone,
                      train_args=vars(args),
                      val_metrics={
                          'epoch': best_epoch,
                          'val_loss': best_val_loss,
                          'val_accuracy': best_val_acc,
                          'calibration': {
                              'pre_p01': p01, 'pre_p50': p50, 'pre_p99': p99,
                              'offset': offset,
                              'terminal_margin': args.terminal_margin,
                          },
                      })
        print(f"  Saved calibrated head to {args.out}", flush=True)


if __name__ == '__main__':
    main()
