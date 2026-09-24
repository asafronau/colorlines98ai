"""Compare scratch widths on the same hard-CE corpus states.

This is a fixed-state representation/optimization diagnostic, not a gameplay
evaluation.  It answers whether a wider checkpoint's wall-clock advantage is
localized to dangerous states or merely reflects better imitation everywhere.
All policy statistics use the exact legal action set; the training-space NLL
over all 6561 logits is reported separately.

Run under caffeinate, for example::

    caffeinate -i -s python -m alphatrain.scripts.audit_scratch_width \
      --tensor alphatrain/data/r2_bulk.pt \
      --models alphatrain/data/scratch128_ckpts_epoch_16.pt \
               alphatrain/data/scratch192_ckpts_epoch_7.pt \
      --device mps
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.fleet_gpu import gpu_legal_mask


def _report(label, mask, metrics):
    n = int(mask.sum())
    if n == 0:
        return
    print(
        f'  {label:13s} n={n:6,d}  '
        f'match={100 * metrics["match"][mask].mean():6.2f}%  '
        f'top3={100 * metrics["top3"][mask].mean():6.2f}%  '
        f'NLLfull={metrics["full_nll"][mask].mean():6.3f}  '
        f'NLLlegal={metrics["legal_nll"][mask].mean():6.3f}  '
        f'p(label)={metrics["target_p"][mask].mean():6.3f}  '
        f'p(own)={metrics["own_p"][mask].mean():6.3f}',
        flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--n-samples', type=int, default=50_000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true',
                   help='Diagnostic override; FP16 is the default protocol.')
    args = p.parse_args()

    device = torch.device(args.device)
    ds = TensorDatasetGPU(args.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    n_total = len(ds.base_indices)
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(
        n_total, size=min(args.n_samples, n_total), replace=False))
    ds.base_indices = torch.from_numpy(selected).to(device)
    n = len(selected)
    danger = (ds.disagree_mask[ds.base_indices].float().cpu().numpy()
              if ds.disagree_mask is not None else None)
    print(f'auditing {n:,}/{n_total:,} fixed corpus rows on {device}; '
          f'precision={"fp32" if args.fp32 else "fp16"}', flush=True)

    all_metrics = {}
    for model_path in args.models:
        model, _ = load_model(model_path, device, fp16=not args.fp32)
        dtype = next(model.parameters()).dtype
        chunks = {k: [] for k in (
            'action', 'target', 'match', 'top3', 'full_nll', 'legal_nll',
            'target_p', 'own_p', 'full_argmax_legal')}

        for start in range(0, n, args.batch_size):
            pos = torch.arange(start, min(start + args.batch_size, n))
            obs, policy = ds.collate(pos)
            target = policy.argmax(1).long()
            logits = model(obs.to(dtype))
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.float()
            full_logp = F.log_softmax(logits, dim=1)
            actual = ds.base_indices[pos.to(device)]
            legal = gpu_legal_mask(ds.boards[actual])
            legal_logits = logits.masked_fill(~legal, float('-inf'))
            legal_logz = torch.logsumexp(legal_logits, dim=1)
            target_logit = logits.gather(1, target[:, None]).squeeze(1)
            action = legal_logits.argmax(1)
            own_logit = legal_logits.max(1).values
            rank = (legal_logits > target_logit[:, None]).sum(1) + 1
            full_action = logits.argmax(1)

            values = {
                'action': action,
                'target': target,
                'match': action == target,
                'top3': rank <= 3,
                'full_nll': -full_logp.gather(
                    1, target[:, None]).squeeze(1),
                'legal_nll': legal_logz - target_logit,
                'target_p': (target_logit - legal_logz).exp(),
                'own_p': (own_logit - legal_logz).exp(),
                'full_argmax_legal': legal.gather(
                    1, full_action[:, None]).squeeze(1),
            }
            for key, value in values.items():
                chunks[key].append(value.detach().cpu())
            done = min(start + args.batch_size, n)
            if done % (args.batch_size * 20) == 0 or done == n:
                print(f'  {os.path.basename(model_path)}: {done:,}/{n:,}',
                      flush=True)

        metrics = {key: torch.cat(parts).numpy()
                   for key, parts in chunks.items()}
        all_metrics[os.path.basename(model_path)] = metrics
        print(f'\n[{os.path.basename(model_path)}]', flush=True)
        _report('all', np.ones(n, dtype=bool), metrics)
        print(f'  full argmax legal: '
              f'{100 * metrics["full_argmax_legal"].mean():.2f}%',
              flush=True)
        if danger is not None:
            for label, lo, hi in (
                    ('danger <.10', -np.inf, .10),
                    ('danger .10-.30', .10, .30),
                    ('danger .30-.60', .30, .60),
                    ('danger >=.60', .60, np.inf)):
                _report(label, (danger >= lo) & (danger < hi), metrics)
        del model
        if device.type == 'mps':
            torch.mps.empty_cache()

    names = list(all_metrics)
    if len(names) > 1:
        print('\n[fixed-state pairwise diagnostics]', flush=True)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = all_metrics[names[i]], all_metrics[names[j]]
                disagree = a['action'] != b['action']
                nd = int(disagree.sum())
                if nd:
                    a_right = (a['action'] == a['target']) & disagree
                    b_right = (b['action'] == b['target']) & disagree
                    print(
                        f'  {names[i]} vs {names[j]}: '
                        f'agree={100 * (1 - nd / n):.2f}%; '
                        f'on {nd:,} disagreements label won by '
                        f'A={100 * a_right.sum() / nd:.2f}% '
                        f'B={100 * b_right.sum() / nd:.2f}%', flush=True)


if __name__ == '__main__':
    main()
