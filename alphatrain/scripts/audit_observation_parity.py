"""Measure the historical GPU component-feature error on real states.

Older GPU observation builders stopped connected-component propagation after
20 rounds.  Obstacles can make a winding empty component much longer, while
Python and C++ inference use an exact flood fill.  This audit reports how often
the old channel 12 differed and whether that changed the frozen policy's legal
argmax.  It does not mutate the corpus.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.dataset import NUM_CHANNELS, TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.fleet_gpu import gpu_legal_mask


def legacy_component_heatmap(boards):
    """Reproduce the former 20-round dataset implementation exactly."""
    batch = len(boards)
    empty = boards == 0
    labels = torch.arange(
        1, 82, device=boards.device, dtype=torch.long).reshape(
            1, 9, 9).expand(batch, 9, 9).clone()
    labels *= empty.long()
    for _ in range(20):
        old = labels
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = TensorDatasetGPU._shift(labels, dr, dc)
            both = (labels > 0) & (nb > 0)
            labels = torch.where(both & (nb < labels), nb, labels)
        if (labels == old).all():
            break

    offsets = torch.arange(batch, device=boards.device).reshape(
        batch, 1, 1) * 82
    flat_labels = (labels + offsets) * (labels > 0).long()
    flat = flat_labels.reshape(-1)
    counts = torch.zeros(int(flat.max().item()) + 1, device=boards.device)
    valid = flat > 0
    counts.scatter_add_(0, flat[valid], torch.ones(valid.sum(),
                                                   device=boards.device))
    out = torch.zeros_like(flat, dtype=torch.float32)
    out[valid] = counts[flat[valid]] / 81.0
    return out.reshape(batch, 9, 9)


def _extend(store, key, tensor):
    store[key].extend(tensor.detach().float().cpu().numpy().tolist())


def _summary(values):
    values = np.asarray(values)
    return (f'mean={values.mean():.6f} P50={np.median(values):.6f} '
            f'P90={np.percentile(values, 90):.6f} '
            f'P99={np.percentile(values, 99):.6f}')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--n-samples', type=int, default=20_000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--rng-skip-population', type=int, default=0,
                   help='Advance RNG by one same-sized sample from this '
                        'population (for reproducing a later corpus cap).')
    p.add_argument('--split', choices=('all', 'train', 'val'), default='all',
                   help='Optionally restrict to the stored game-group split.')
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true')
    args = p.parse_args()

    device = torch.device(args.device)
    ds = TensorDatasetGPU(args.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    ds.return_flywheel = True
    if args.split != 'all':
        if ds.split is None:
            raise ValueError(f'{args.tensor} has no stored split')
        wanted = 0 if args.split == 'train' else 1
        ds.base_indices = (ds.split == wanted).nonzero(as_tuple=True)[0]
    rng = np.random.default_rng(args.seed)
    if args.rng_skip_population:
        rng.choice(args.rng_skip_population,
                   size=min(args.n_samples, args.rng_skip_population),
                   replace=False)
    selected = np.sort(rng.choice(
        len(ds.base_indices), size=min(args.n_samples, len(ds.base_indices)),
        replace=False)).astype(np.int64)
    names = ((ds.metadata or {}).get('source_names')
             or [str(i) for i in range(int(ds.source_id.max()) + 1)])
    model, _ = load_model(args.base, device, fp16=not args.fp32)
    dtype = next(model.parameters()).dtype

    metrics = defaultdict(list)
    source_all = []
    for start in range(0, len(selected), args.batch_size):
        ix_cpu = torch.from_numpy(selected[start:start + args.batch_size])
        obs_exact, _, _, behavior, hard_target, source = ds.collate(ix_cpu)
        actual = ds.base_indices[ix_cpu.to(device)]
        boards = ds.boards[actual]
        legacy_heat = legacy_component_heatmap(boards)
        exact_heat = obs_exact[:, 12]
        delta = (legacy_heat - exact_heat).abs()
        mismatch = delta.flatten(1).max(1).values > 0
        obs_legacy = obs_exact.clone()
        obs_legacy[:, 12] = legacy_heat

        exact_logits = model(obs_exact.to(dtype))
        legacy_logits = model(obs_legacy.to(dtype))
        if isinstance(exact_logits, tuple):
            exact_logits = exact_logits[0]
        if isinstance(legacy_logits, tuple):
            legacy_logits = legacy_logits[0]
        exact_logits = exact_logits.float()
        legacy_logits = legacy_logits.float()
        exact_logp = F.log_softmax(exact_logits, -1)
        legacy_logp = F.log_softmax(legacy_logits, -1)
        kl = (exact_logp.exp() * (exact_logp - legacy_logp)).sum(1)
        legal = gpu_legal_mask(boards)
        exact_action = exact_logits.masked_fill(~legal, float('-inf')).argmax(1)
        legacy_action = legacy_logits.masked_fill(
            ~legal, float('-inf')).argmax(1)
        legal_exact = exact_logits.masked_fill(~legal, float('-inf'))
        behavior = behavior.long()
        behavior_logit = exact_logits.gather(
            1, behavior.unsqueeze(1)).squeeze(1)
        behavior_gap = legal_exact.max(1).values - behavior_logit
        behavior_rank = (legal_exact > behavior_logit.unsqueeze(1)).sum(1) + 1

        _extend(metrics, 'mismatch', mismatch)
        _extend(metrics, 'changed_cells', (delta > 0).sum((1, 2)))
        _extend(metrics, 'max_channel_error', delta.flatten(1).max(1).values)
        _extend(metrics, 'full_kl', kl)
        _extend(metrics, 'legal_action_change', exact_action != legacy_action)
        _extend(metrics, 'behavior_is_legal', legal.gather(
            1, behavior.long().unsqueeze(1)).squeeze(1))
        _extend(metrics, 'hard_target_is_legal', legal.gather(
            1, hard_target.long().unsqueeze(1)).squeeze(1))
        _extend(metrics, 'exact_behavior_agree', exact_action == behavior)
        _extend(metrics, 'legacy_behavior_agree', legacy_action == behavior)
        _extend(metrics, 'hard_target_is_behavior', hard_target == behavior)
        _extend(metrics, 'behavior_logit_gap', behavior_gap)
        _extend(metrics, 'behavior_rank', behavior_rank)
        source_all.extend(source.cpu().numpy().tolist())

    source_all = np.asarray(source_all)

    def report(label, mask):
        mismatch = np.asarray(metrics['mismatch'])[mask].astype(bool)
        action = np.asarray(metrics['legal_action_change'])[mask].astype(bool)
        print(f'\n[{label}] n={int(mask.sum()):,}', flush=True)
        print(f'  legacy channel-12 mismatch: {100*mismatch.mean():.3f}%',
              flush=True)
        for key in ('changed_cells', 'max_channel_error', 'full_kl'):
            print(f'  {key:22s} {_summary(np.asarray(metrics[key])[mask])}',
                  flush=True)
        print(f'  legal argmax changed:      {100*action.mean():.3f}% all; '
              f'{100*action[mismatch].mean() if mismatch.any() else 0:.3f}% '
              f'of mismatched states', flush=True)
        exact = np.asarray(metrics['exact_behavior_agree'])[mask]
        legacy = np.asarray(metrics['legacy_behavior_agree'])[mask]
        print(f'  behavior agreement: exact={100*exact.mean():.3f}% '
              f'legacy={100*legacy.mean():.3f}%', flush=True)
        target_same = np.asarray(metrics['hard_target_is_behavior'])[mask]
        print(f'  hard target is behavior:   {100*target_same.mean():.3f}%',
              flush=True)
        behavior_legal = np.asarray(metrics['behavior_is_legal'])[mask]
        target_legal = np.asarray(metrics['hard_target_is_legal'])[mask]
        print(f'  recorded legality: behavior={100*behavior_legal.mean():.3f}% '
              f'hard_target={100*target_legal.mean():.3f}%', flush=True)
        gaps = np.asarray(metrics['behavior_logit_gap'])[mask]
        ranks = np.asarray(metrics['behavior_rank'])[mask]
        disagree = ~exact.astype(bool)
        print(f'  recorded-move gap/rank:  {_summary(gaps)}; '
              f'top3={100*(ranks <= 3).mean():.2f}%', flush=True)
        if disagree.any():
            dg = gaps[disagree]
            print(f'  mismatch-only gap:       {_summary(dg)}; '
                  f'gap<=.01 {100*(dg <= .01).mean():.2f}% '
                  f'gap<=.05 {100*(dg <= .05).mean():.2f}%', flush=True)

    all_mask = np.ones(len(source_all), dtype=bool)
    print(f'{args.tensor}; dtype={dtype}; split={args.split}; '
          f'sample={len(source_all):,}',
          flush=True)
    report('all', all_mask)
    for sid, name in enumerate(names):
        mask = source_all == sid
        if mask.any():
            report(name, mask)


if __name__ == '__main__':
    main()
