"""Audit the functional size of a flywheel policy update before training.

The bounded trainer anchors the model's *full* 6561-way softmax, while play
renormalizes logits over legal moves.  If the base assigns appreciable mass to
illegal actions, a nominal ``eta`` is a larger dose after legal conditioning.
This script measures that effect in the same FP16 inference protocol used by
evaluation and reports it separately for every current-lineage source.

Run real audits under caffeinate, for example::

    caffeinate -i -s python -m alphatrain.scripts.audit_flywheel_targets \
      --targets alphatrain/data/flywheel_vh3_targets.pt \
      --base alphatrain/data/small128_vh3.pt --device mps
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.fleet_gpu import gpu_legal_mask


def _pct(values):
    values = np.asarray(values)
    return (f'mean={values.mean():.4f} P10={np.percentile(values, 10):.4f} '
            f'P50={np.percentile(values, 50):.4f} '
            f'P90={np.percentile(values, 90):.4f}')


def _extend(store, name, value):
    store[name].extend(value.detach().float().cpu().numpy().tolist())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--n-samples', type=int, default=20_000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true',
                   help='Diagnostic override; FP16 is the default protocol.')
    p.add_argument('--eta', type=float, nargs='+',
                   default=[0.02, 0.05, 0.1, 0.2])
    p.add_argument('--soft-alpha', type=float, nargs='+',
                   default=[0.0, 0.5, 1.0])
    args = p.parse_args()

    device = torch.device(args.device)
    ds = TensorDatasetGPU(args.targets, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    ds.return_flywheel = True
    if ds.target_weight is None or ds.source_id is None:
        raise ValueError('target tensor is missing flywheel fields')
    eligible = (ds.target_weight > 0).nonzero(as_tuple=True)[0].cpu().numpy()
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(
        eligible, size=min(args.n_samples, len(eligible)), replace=False))
    source_names = ((ds.metadata or {}).get('source_names')
                    or [str(i) for i in range(int(ds.source_id.max()) + 1)])

    model, _ = load_model(args.base, device, fp16=not args.fp32)
    dtype = next(model.parameters()).dtype
    print(f'auditing {len(selected):,}/{len(eligible):,} weighted target rows; '
          f'device={device} dtype={dtype}', flush=True)

    base_metrics = defaultdict(list)
    dose_metrics = {
        (alpha, eta): defaultdict(list)
        for alpha in args.soft_alpha for eta in args.eta
    }
    source_all = []

    for start in range(0, len(selected), args.batch_size):
        ix = torch.from_numpy(selected[start:start + args.batch_size])
        obs, target, weight, behavior, teacher, source = ds.collate(ix)
        logits = model(obs.to(dtype))
        if isinstance(logits, tuple):
            logits = logits[0]
        logits = logits.float()
        logp = F.log_softmax(logits, dim=-1)
        full_p = logp.exp()
        boards = ds.boards[ix.to(device)]
        legal = gpu_legal_mask(boards).to(full_p.dtype)
        legal_p = full_p * legal
        legal_mass = legal_p.sum(1).clamp_min(1e-30)
        base_legal = legal_p / legal_mass.unsqueeze(1)
        masked_logits = logits.masked_fill(legal == 0, float('-inf'))
        base_action = masked_logits.argmax(1)
        # ``teacher`` is the clean search winner when behavior/label searches
        # were separated, and the historical visit winner otherwise. Dense
        # argmax picks the lowest action id on tied visit counts, whereas the
        # generator keeps its stable candidate ordering.
        search_action = teacher.long()
        search_top_share = target.max(1).values
        teacher_share = target.gather(
            1, search_action.unsqueeze(1)).squeeze(1)
        behavior_share = target.gather(
            1, behavior.long().unsqueeze(1)).squeeze(1)
        visit_tie = (target == search_top_share.unsqueeze(1)).sum(1) > 1
        target_legal_mass = (target * legal).sum(1)
        n_legal = legal.sum(1)

        _extend(base_metrics, 'legal_mass', legal_mass)
        _extend(base_metrics, 'legal_top1', base_legal.max(1).values)
        _extend(base_metrics, 'search_top_share', search_top_share)
        _extend(base_metrics, 'target_legal_mass', target_legal_mass)
        _extend(base_metrics, 'n_legal', n_legal)
        _extend(base_metrics, 'base_search_agree', base_action == search_action)
        _extend(base_metrics, 'behavior_teacher_agree',
                behavior.long() == search_action)
        _extend(base_metrics, 'teacher_is_visit_max',
                teacher_share == search_top_share)
        _extend(base_metrics, 'behavior_is_visit_max',
                behavior_share == search_top_share)
        _extend(base_metrics, 'visit_top_tie', visit_tie)
        _extend(base_metrics, 'base_search_ce', -(target * logp).sum(1))
        source_all.extend(source.cpu().numpy().tolist())

        hard = torch.zeros_like(target)
        hard.scatter_(1, teacher.long().unsqueeze(1), 1.0)
        for alpha in args.soft_alpha:
            search_mix = alpha * target + (1.0 - alpha) * hard
            for eta in args.eta:
                effective_eta = eta * weight.float()
                q_num = (full_p
                         + effective_eta.unsqueeze(1) * search_mix)
                q = q_num / (1.0 + effective_eta).unsqueeze(1)
                q_legal_num = (legal_p
                               + effective_eta.unsqueeze(1) * search_mix)
                q_legal = q_legal_num / (
                    legal_mass + effective_eta).unsqueeze(1)
                q_action = q_legal.argmax(1)
                full_kl = (full_p * (logp - q.clamp_min(1e-30).log())).sum(1)
                legal_kl = torch.where(
                    base_legal > 0,
                    base_legal * (base_legal.clamp_min(1e-30).log()
                                  - q_legal.clamp_min(1e-30).log()),
                    torch.zeros_like(base_legal)).sum(1)
                effective_legal_dose = effective_eta / (
                    legal_mass + effective_eta)
                metrics = dose_metrics[(alpha, eta)]
                _extend(metrics, 'full_kl', full_kl)
                _extend(metrics, 'legal_kl', legal_kl)
                _extend(metrics, 'legal_dose', effective_legal_dose)
                _extend(metrics, 'flip', q_action != base_action)
                _extend(metrics, 'search_adopt', q_action == search_action)
                _extend(metrics, 'base_disagree',
                        base_action != search_action)

        done = min(start + args.batch_size, len(selected))
        if done % (args.batch_size * 10) == 0 or done == len(selected):
            print(f'  {done:,}/{len(selected):,}', flush=True)

    source_all = np.asarray(source_all)

    def report_base(label, mask):
        print(f'\n[{label}] n={int(mask.sum()):,}', flush=True)
        for key in ('legal_mass', 'legal_top1', 'search_top_share',
                    'target_legal_mass', 'n_legal', 'base_search_ce'):
            print(f'  {key:22s} {_pct(np.asarray(base_metrics[key])[mask])}',
                  flush=True)
        for key in ('base_search_agree', 'behavior_teacher_agree',
                    'teacher_is_visit_max', 'behavior_is_visit_max',
                    'visit_top_tie'):
            values = np.asarray(base_metrics[key])[mask]
            print(f'  {key:22s} {100 * values.mean():.2f}%', flush=True)

    all_mask = np.ones(len(source_all), dtype=bool)
    report_base('all flywheel targets', all_mask)
    for sid, name in enumerate(source_names):
        mask = source_all == sid
        if mask.any():
            report_base(name, mask)

    print('\n[analytic same-state bounded targets: all sources]', flush=True)
    edit_fraction = np.asarray(
        dose_metrics[(args.soft_alpha[0], args.eta[0])]['base_disagree'])
    print(f'  actor/search edits: {100 * edit_fraction.mean():.2f}%',
          flush=True)
    print('  alpha  eta | legal dose P50/P90 | legal flips | edit adoption | '
          'KLfull mean/P90 | KLlegal mean/P90', flush=True)
    for alpha in args.soft_alpha:
        for eta in args.eta:
            metrics = dose_metrics[(alpha, eta)]
            ld = np.asarray(metrics['legal_dose'])
            flip = np.asarray(metrics['flip'])
            adopt = np.asarray(metrics['search_adopt'])
            edit = np.asarray(metrics['base_disagree']).astype(bool)
            edit_adopt = adopt[edit].mean() if edit.any() else float('nan')
            fk = np.asarray(metrics['full_kl'])
            lk = np.asarray(metrics['legal_kl'])
            print(f'  {alpha:5.2f} {eta:4.2f} | '
                  f'{np.median(ld):6.2%}/{np.percentile(ld, 90):6.2%} | '
                  f'{100*flip.mean():10.2f}% | {100*edit_adopt:13.2f}% | '
                  f'{fk.mean():.5f}/{np.percentile(fk, 90):.5f} | '
                  f'{lk.mean():.5f}/{np.percentile(lk, 90):.5f}', flush=True)


if __name__ == '__main__':
    main()
