"""Audit information lost by the deployed survival-head scalar.

The production fused module computes ``sum(w * sigmoid(logit_H))`` in the
model dtype.  On healthy boards every short-horizon probability can round to
one in FP16, leaving MCTS with almost no within-tree Q range even when the raw
head logits still contain ordering information.  This audit reports native
FP16 saturation and compares several *diagnostic* scalarizations against
held-out survival labels.  It does not promote a scalarization; that still
requires a fixed-state search sweep and a 5k gameplay distribution.

Run under caffeinate, for example::

    caffeinate -i -s ../.venv/bin/python -m \
      alphatrain.scripts.audit_value_scalar \
      --backbone alphatrain/data/small128_vh3.pt \
      --head alphatrain/data/value_head_vh3.pt \
      --data alphatrain/data/value_targets_vh3.pt --device mps
"""

from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
from scipy.stats import rankdata

from alphatrain import value_head as vh
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model


def auc(scores, labels):
    """Tie-correct binary ROC AUC, or NaN for a one-class slice."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = rankdata(scores, method='average')
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return (f'min={values.min():.5g} P10={np.percentile(values, 10):.5g} '
            f'P50={np.median(values):.5g} '
            f'P90={np.percentile(values, 90):.5g} '
            f'max={values.max():.5g} std={values.std():.5g}')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--data', required=True,
                   help='Survival-target tensor with boards and labels.')
    p.add_argument('--split', choices=['val', 'train', 'all'], default='val')
    p.add_argument('--n-samples', type=int, default=50_000)
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true',
                   help='Diagnostic override; deployed FP16 is the default.')
    args = p.parse_args()

    raw = torch.load(args.data, map_location='cpu', weights_only=False)
    required = ('boards', 'next_pos', 'next_col', 'n_next',
                'survive_labels', 'survive_masks')
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f'{args.data} missing {missing}')
    eligible = np.arange(len(raw['boards']))
    if args.split != 'all':
        if 'is_train' not in raw:
            raise ValueError(f'{args.split} split requested but no is_train')
        wanted = args.split == 'train'
        eligible = np.flatnonzero(raw['is_train'].numpy() == wanted)
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(
        eligible, min(args.n_samples, len(eligible)), replace=False))
    labels = raw['survive_labels'][selected].numpy()
    masks = raw['survive_masks'][selected].numpy().astype(bool)
    horizons = list(raw.get('horizons', vh.SURVIVAL_HORIZONS))

    # TensorDatasetGPU is the golden GPU observation builder.  Give it a tiny
    # dummy policy schema so this audit exercises exactly the training/search
    # observation path without copying the full value tensor to the device.
    n = len(selected)
    sampled = {
        'boards': raw['boards'][selected],
        'next_pos': raw['next_pos'][selected],
        'next_col': raw['next_col'][selected],
        'n_next': raw['n_next'][selected],
        'pol_indices': torch.zeros((n, 1), dtype=torch.int64),
        'pol_values': torch.ones((n, 1), dtype=torch.float32),
        'pol_nnz': torch.ones(n, dtype=torch.int64),
        'max_score': 0.0,
    }
    handle = tempfile.NamedTemporaryFile(
        prefix='value_scalar_audit_', suffix='.pt', dir='/tmp', delete=False)
    tmp_path = handle.name
    handle.close()
    torch.save(sampled, tmp_path)
    del sampled, raw

    device = torch.device(args.device)
    fp16 = not args.fp32 and device.type != 'cpu'
    net, _ = load_model(args.backbone, device, fp16=fp16)
    head, metadata, head_type = vh.load_any(args.head, device)
    if head_type != 'value_head' or metadata.get('target_type') != 'survival':
        raise ValueError('audit requires a multi-horizon survival head')
    dtype = torch.float16 if fp16 else torch.float32
    head = head.to(dtype)
    head.train(False)
    net.train(False)
    ds = TensorDatasetGPU(tmp_path, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    weights = torch.tensor(vh.DEFAULT_HORIZON_WEIGHTS, device=device)

    chunks = {name: [] for name in (
        'logits', 'prob_native', 'prob_float', 'current_native',
        'current_float', 'logit_weighted', 'h100_logit', 'min_logit')}
    try:
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            obs = ds._build_obs_core(
                ds.boards[start:end], next_pos=ds.next_pos[start:end],
                next_col=ds.next_col[start:end], n_next=ds.n_next[start:end])
            _, feats = net.forward_with_features(obs.to(dtype))
            logits_native = head(feats)
            prob_native = torch.sigmoid(logits_native)
            current_native = (
                prob_native * weights.to(dtype)).sum(-1)
            logits = logits_native.float()
            prob_float = torch.sigmoid(logits)
            chunks['logits'].append(logits.cpu())
            chunks['prob_native'].append(prob_native.float().cpu())
            chunks['prob_float'].append(prob_float.cpu())
            chunks['current_native'].append(current_native.float().cpu())
            chunks['current_float'].append(
                (prob_float * weights).sum(-1).cpu())
            chunks['logit_weighted'].append((logits * weights).sum(-1).cpu())
            h100 = min(2, logits.shape[1] - 1)
            chunks['h100_logit'].append(logits[:, h100].cpu())
            chunks['min_logit'].append(logits.min(-1).values.cpu())
            if end % (args.batch_size * 10) == 0 or end == n:
                print(f'  {end:,}/{n:,}', flush=True)
    finally:
        os.unlink(tmp_path)

    out = {key: torch.cat(parts).numpy() for key, parts in chunks.items()}
    print(f'\n{n:,} {args.split} rows; device={device} dtype={dtype}; '
          f'horizons={horizons}', flush=True)
    for hi, horizon in enumerate(horizons):
        native = out['prob_native'][:, hi]
        print(f'  H{horizon} logit: {summary(out["logits"][:, hi])}',
              flush=True)
        print(f'       sigmoid({dtype}): {summary(native)}; '
              f'exact0={100 * (native == 0).mean():.2f}% '
              f'exact1={100 * (native == 1).mean():.2f}%', flush=True)

    scalar_names = ('current_native', 'current_float', 'logit_weighted',
                    'h100_logit', 'min_logit')
    print('\n[scalar spread]', flush=True)
    for name in scalar_names:
        values = out[name]
        unique = len(np.unique(values))
        _, counts = np.unique(values, return_counts=True)
        print(f'  {name:16s} {summary(values)}; unique={unique:,}; '
              f'mode_share={100 * counts.max() / len(values):.2f}%',
              flush=True)

    print('\n[AUC versus held-out single-trajectory survival labels]',
          flush=True)
    print('  scalar          ' + ' '.join(f'H{h:>4}' for h in horizons),
          flush=True)
    for name in scalar_names:
        values = []
        for hi in range(len(horizons)):
            use = masks[:, hi]
            values.append(auc(out[name][use], labels[use, hi]))
        print(f'  {name:16s} ' + ' '.join(f'{v:5.3f}' for v in values),
              flush=True)


if __name__ == '__main__':
    main()
