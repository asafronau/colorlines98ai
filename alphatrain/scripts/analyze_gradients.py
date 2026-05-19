"""Measure gradient norms and alignment between V12 and oracle losses.

Cheap diagnostic — no training, just forward+backward+capture+zero. Answers:

  - Is the oracle gradient too weak vs V12 to steer the model?
        metric: ||λ · g_oracle|| / ||g_v12||
        rule of thumb: <0.05 = oracle is noise; 0.10-0.30 = effective; >0.5 = oracle dominates

  - Is oracle fighting V12 (gradient conflict)?
        metric: cosine(g_v12, g_oracle)
        positive = aligned; ~0 = orthogonal; negative = conflict

  - Where in the network is the disagreement?
        per-layer breakdown (stem / each block / policy head)

Run on A_ep12 (clean control), C_ep7 (oracle peak), C_ep12 (oracle faded) at
λ ∈ {0.05, 0.10} on the same batches. Compare:

  - Does C_ep12 show smaller relative oracle gradient than C_ep7? (Oracle
    gradient dies off as weights settle.)
  - Does cosine flip sign between A and C? (C may "agree" with oracle
    direction because it's been pushed there.)
  - Is oracle gradient concentrated in policy head, or also touching backbone?

Usage:
    python -m alphatrain.scripts.analyze_gradients \\
        --checkpoint alphatrain/data/c_smoke_epoch_7.pt \\
        --v12-tensor alphatrain/data/v12_pillar2z.pt \\
        --oracle-tensor alphatrain/data/phase1_oracle_path_b.pt \\
        --lambda 0.05 --n-batches 5 --device mps
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet
from alphatrain.train_path_b import (
    OracleDataset,
    distillation_loss,
    oracle_loss,
)


def _layer_group(name: str) -> str:
    """Map a parameter name to a coarse group for reporting.

    The model has no `policy_head.*` prefix — the policy head consists of
    policy_conv1, policy_bn, policy_conv2, after a final backbone_bn.
    """
    if name.startswith('stem.'):
        return 'stem'
    if name.startswith('blocks.'):
        block_idx = name.split('.')[1]
        return f'block_{block_idx}'
    if name.startswith('backbone_bn.'):
        return 'backbone_bn'
    if (name.startswith('policy_conv')
            or name.startswith('policy_bn.')):
        return 'policy_head'
    if name.startswith('value_'):
        return 'value_head'
    return 'other'


def _per_group_stats(g_v12, g_oracle, lambda_):
    """Aggregate per-group ||g_v12||, ||g_oracle||, cosine(g_v12, g_oracle).

    Treats all params in a group as one big flattened vector.
    """
    groups = defaultdict(lambda: {'v12_sqs': 0.0, 'oracle_sqs': 0.0,
                                    'dot': 0.0, 'n_params': 0})
    for name, gv in g_v12.items():
        go = g_oracle.get(name)
        if go is None:
            continue
        grp = _layer_group(name)
        gv_f = gv.detach().flatten().cpu().to(torch.float64)
        go_f = go.detach().flatten().cpu().to(torch.float64)
        groups[grp]['v12_sqs'] += float((gv_f * gv_f).sum().item())
        groups[grp]['oracle_sqs'] += float((go_f * go_f).sum().item())
        groups[grp]['dot'] += float((gv_f * go_f).sum().item())
        groups[grp]['n_params'] += gv.numel()

    rows = []
    total = {'v12_sqs': 0.0, 'oracle_sqs': 0.0, 'dot': 0.0}
    for grp, st in groups.items():
        n_v12 = st['v12_sqs'] ** 0.5
        n_oracle = st['oracle_sqs'] ** 0.5
        cos = (st['dot'] / (n_v12 * n_oracle)
                if n_v12 > 0 and n_oracle > 0 else 0.0)
        ratio = lambda_ * n_oracle / max(n_v12, 1e-30)
        rows.append((grp, n_v12, n_oracle, ratio, cos, st['n_params']))
        total['v12_sqs'] += st['v12_sqs']
        total['oracle_sqs'] += st['oracle_sqs']
        total['dot'] += st['dot']
    # Global
    n_v12 = total['v12_sqs'] ** 0.5
    n_oracle = total['oracle_sqs'] ** 0.5
    cos = (total['dot'] / (n_v12 * n_oracle)
            if n_v12 > 0 and n_oracle > 0 else 0.0)
    ratio = lambda_ * n_oracle / max(n_v12, 1e-30)
    rows.append(('GLOBAL', n_v12, n_oracle, ratio, cos, sum(g.numel() for g in g_v12.values())))
    return rows


def _capture_grad(model, loss, retain_graph=False):
    """Backward `loss` and return {name: grad.clone()}; zero grads after."""
    loss.backward(retain_graph=retain_graph)
    grads = {name: p.grad.detach().clone()
              for name, p in model.named_parameters()
              if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


def _snapshot_bn_stats(model):
    """Capture BN running_mean/running_var (and num_batches_tracked)."""
    snap = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            snap[name] = (mod.running_mean.detach().clone(),
                           mod.running_var.detach().clone(),
                           mod.num_batches_tracked.detach().clone())
    return snap


def _restore_bn_stats(model, snap):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d) and name in snap:
            rm, rv, nbt = snap[name]
            mod.running_mean.copy_(rm)
            mod.running_var.copy_(rv)
            mod.num_batches_tracked.copy_(nbt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--v12-tensor', required=True)
    p.add_argument('--oracle-tensor', required=True)
    p.add_argument('--lambda', dest='lambda_', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=10.0)
    p.add_argument('--noise-floor', type=float, default=0.05)
    p.add_argument('--scale', type=float, default=0.20)
    p.add_argument('--v12-batch-size', type=int, default=4096)
    p.add_argument('--oracle-batch-size', type=int, default=4096)
    p.add_argument('--n-batches', type=int, default=5,
                   help='Number of batches to average over.')
    p.add_argument('--num-blocks', type=int, default=10)
    p.add_argument('--channels', type=int, default=256)
    p.add_argument('--device', default=None)
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)

    # ── Load model ──
    print(f"\nLoading {args.checkpoint}...", flush=True)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model = AlphaTrainNet(num_blocks=args.num_blocks,
                          channels=args.channels).to(device)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                 if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)
    # Train(True) so BN behaves the same as during training. We don't update
    # any parameters; just want the same forward semantics.
    model.train(True)
    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"  epoch={epoch}  val_loss={val_loss:.4f}", flush=True)

    # ── Load V12 + oracle ──
    print(f"\nLoading V12 {args.v12_tensor}...", flush=True)
    train_set, _ = TensorDatasetGPU.make_train_val_split(
        args.v12_tensor,
        val_split=0.05,
        augment=False,
        color_augment=False,
        augment_factor=1,
        device=str(device),
        seed=args.seed,
    )
    v12_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.v12_batch_size, shuffle=True,
        num_workers=0, collate_fn=train_set.collate)
    v12_iter = iter(v12_loader)

    print(f"Loading oracle {args.oracle_tensor}...", flush=True)
    ods = OracleDataset(args.oracle_tensor, device=device)
    train_idx, _ = ods.split(val_frac=0.05, seed=args.seed)
    sample_gen = torch.Generator(device=device.type).manual_seed(args.seed)

    # ── Accumulate per-batch readings ──
    print(f"\nMeasuring gradients on {args.n_batches} batches "
          f"(λ={args.lambda_}, β={args.beta})...", flush=True)
    print(f"  (single concat-forward matching training; "
          f"BN snapshot/restore around each batch)", flush=True)
    all_rows = defaultdict(list)
    for b in range(args.n_batches):
        try:
            obs_v, pol_v = next(v12_iter)
        except StopIteration:
            v12_iter = iter(v12_loader)
            obs_v, pol_v = next(v12_iter)
        obs_v = obs_v.to(device, non_blocking=True)
        pol_v = pol_v.to(device, non_blocking=True)

        o_obs, o_act, o_cap, o_nm, o_dc = ods.sample(
            train_idx, args.oracle_batch_size, sample_gen)

        # ── Snapshot BN running stats BEFORE forward; restore AFTER backward
        # to keep the model bit-identical across diagnostic batches ──
        bn_snap = _snapshot_bn_stats(model)

        # ── Single concat-forward, matching train_path_b.py training ──
        model.zero_grad(set_to_none=True)
        B_v = obs_v.shape[0]
        obs_cat = torch.cat([obs_v, o_obs], dim=0)
        logits_cat = model(obs_cat)
        if isinstance(logits_cat, tuple):
            logits_cat = logits_cat[0]

        logits_v = logits_cat[:B_v]
        logits_o = logits_cat[B_v:].float()
        L_v12 = distillation_loss(logits_v, pol_v)
        L_oracle, _ = oracle_loss(
            logits_o, o_act, o_cap, o_nm, o_dc,
            beta=args.beta, noise_floor=args.noise_floor, scale=args.scale)

        # Two backward passes from the SAME forward graph
        g_v12 = _capture_grad(model, L_v12, retain_graph=True)
        g_oracle = _capture_grad(model, L_oracle)

        # Restore BN to where it was before this batch
        _restore_bn_stats(model, bn_snap)

        # ── Per-layer stats ──
        rows = _per_group_stats(g_v12, g_oracle, args.lambda_)
        for grp, n_v12, n_oracle, ratio, cos, n_params in rows:
            all_rows[grp].append((n_v12, n_oracle, ratio, cos, n_params))

        # Print scalars for sanity
        gl = next(r for r in rows if r[0] == 'GLOBAL')
        print(f"  batch {b+1}: L_v12={L_v12.item():.4f}  "
              f"L_oracle={L_oracle.item():.4f}  "
              f"||g_v12||={gl[1]:.3e}  ||g_oracle||={gl[2]:.3e}  "
              f"λ·ratio={gl[3]:.4f}  cos={gl[4]:+.4f}", flush=True)

    # ── Report averages ──
    print(f"\n=== Per-group averages (over {args.n_batches} batches) ===")
    print(f"{'group':<12} {'||g_v12||':>10} {'||g_oracle||':>12} "
          f"{'λ·ratio':>9} {'cos':>9} {'n_params':>10}")
    print('-' * 72)
    order = ['stem'] + [f'block_{i}' for i in range(args.num_blocks)] + \
            ['policy_head', 'value_head', 'other', 'GLOBAL']
    for grp in order:
        rows = all_rows.get(grp)
        if not rows:
            continue
        arr = np.array(rows)  # (n_batches, 5)
        m = arr.mean(axis=0)
        sep = '=' if grp == 'GLOBAL' else ' '
        print(f"{grp:<12} {m[0]:>10.3e} {m[1]:>12.3e} {m[2]:>9.4f} "
              f"{m[3]:>+9.4f} {int(m[4]):>10}{sep}")

    print(f"\nInterpretation key:")
    print(f"  λ·ratio = ||λ · g_oracle|| / ||g_v12||")
    print(f"     < 0.05: oracle is noise vs V12")
    print(f"     0.05-0.30: oracle is effective influence")
    print(f"     > 0.5: oracle dominates V12")
    print(f"  cos = cosine(g_v12, g_oracle)")
    print(f"     +0.5+: aligned (oracle reinforces V12)")
    print(f"     ~0:     orthogonal")
    print(f"     <-0.2: conflict (oracle fights V12)")


if __name__ == '__main__':
    main()
