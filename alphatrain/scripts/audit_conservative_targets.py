"""Audit bounded policy-improvement targets before training them.

For selected corpus rows, compare the search/played target with both the
frozen base policy's full output distribution and its *exact all-legal*
conditional distribution.  On disagreement rows, consider the same-state
conservative objective

    CE(p_base, p_student) + eta * CE(one_hot(search), p_student).

In unconstrained probability space its optimum is exactly

    p_target = (p_base + eta * one_hot(search)) / (1 + eta).

This gives ``eta`` an interpretable meaning: the search action becomes the new
argmax iff eta exceeds its probability gap to the base action.  Unlike hard CE,
agreement rows do not self-sharpen, and every edit has a known KL/mass budget.

Example:
    python -m alphatrain.scripts.audit_conservative_targets \
        --tensor alphatrain/data/r2_frontier.pt \
        --strata alphatrain/data/r2_frontier_strata.npz \
        --base alphatrain/data/small128_vh3.pt --channels 128 \
        --n-samples 30000 --min-visit-share 0.36 --device cpu
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.mcts import _get_all_legal_priors_flat


def percentiles(x):
    if not len(x):
        return 'n/a'
    return ' '.join(
        f'P{q}={np.percentile(x, q):.4f}' for q in (10, 50, 90))


def report_group(name, rows, eta_values):
    if not rows:
        print(f'\n[{name}] n=0', flush=True)
        return
    # The trainer's KL is over the full output distribution, so these are the
    # probabilities relevant to the analytic optimum and flip threshold.
    p_t = np.asarray([r['p_teacher_full'] for r in rows])
    p_b = np.asarray([r['p_base_full'] for r in rows])
    gap = p_b - p_t
    p_t_legal = np.asarray([r['p_teacher_legal'] for r in rows])
    p_b_legal = np.asarray([r['p_base_legal'] for r in rows])
    legal_mass = np.asarray([r['legal_mass'] for r in rows])
    share = np.asarray([r['visit_share'] for r in rows])
    ranks = np.asarray([r['rank'] for r in rows])
    print(f'\n[{name}] n={len(rows):,}', flush=True)
    print(f'  search visit top-share: {percentiles(share)}', flush=True)
    print(f'  total legal mass:       {percentiles(legal_mass)}', flush=True)
    print(f'  legal-cond p(search):   {percentiles(p_t_legal)}', flush=True)
    print(f'  legal-cond p(base max): {percentiles(p_b_legal)}', flush=True)
    print(f'  full p(search):         {percentiles(p_t)}', flush=True)
    print(f'  full p(base legal max): {percentiles(p_b)}', flush=True)
    print(f'  full probability gap:   {percentiles(gap)}', flush=True)
    print(f'  base rank(search):      P50={np.median(ranks):.0f} '
          f'P90={np.percentile(ranks, 90):.0f}', flush=True)
    for eta in eta_values:
        # Exact optimum of CE(base)+eta*CE(search), restricted to this state.
        q_t = (p_t + eta) / (1.0 + eta)
        flips = eta > gap
        # KL(p_base || q_target).  Every non-teacher action is divided by
        # 1+eta; handle p_teacher=0 by continuity.
        t_term = np.where(
            p_t > 0, p_t * np.log(p_t / np.maximum(q_t, 1e-300)), 0.0)
        kl = t_term + (1.0 - p_t) * np.log1p(eta)
        delta = eta / (1.0 + eta)
        print(f'  eta={eta:g} (mass dose={delta:.1%}): '
              f'analytic flips={100*flips.mean():5.1f}%  '
              f'q(search) P50={np.median(q_t):.3f}  '
              f'KL(base||target) P50={np.median(kl):.4f} '
              f'P90={np.percentile(kl, 90):.4f}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--strata', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--n-samples', type=int, default=30_000)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='cpu')
    p.add_argument('--min-visit-share', type=float, default=0.36)
    p.add_argument('--eta', type=float, nargs='+',
                   default=[0.05, 0.1, 0.2, 0.3, 0.5])
    args = p.parse_args()

    dev = torch.device(args.device)
    data = torch.load(args.tensor, map_location='cpu', weights_only=False)
    strata = np.load(args.strata, allow_pickle=False)['strata'].astype(str)
    n = int(data['boards'].shape[0])
    if len(strata) != n:
        raise ValueError(f'strata rows {len(strata):,} != tensor rows {n:,}')

    rng = np.random.default_rng(args.seed)
    idx_np = np.sort(rng.choice(
        n, size=min(args.n_samples, n), replace=False)).astype(np.int64)
    idx = torch.from_numpy(idx_np).to(dev)
    teacher = data['pol_indices'][
        torch.arange(n), data['pol_values'].argmax(1)].numpy()
    visit_share = data['pol_values'].max(1).values.numpy()

    ds = TensorDatasetGPU(args.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=args.device)
    net, _ = load_model(args.base, dev, fp16=False)
    dtype = next(net.parameters()).dtype
    rows = []
    print(f'auditing {len(idx_np):,}/{n:,} rows on {dev}...', flush=True)
    for start in range(0, len(idx_np), args.batch_size):
        b = idx[start:start + args.batch_size]
        obs = ds._build_obs_core(
            ds.boards[b], next_pos=ds.next_pos[b],
            next_col=ds.next_col[b], n_next=ds.n_next[b])
        with torch.inference_mode():
            out = net(obs.to(dtype))
            logits_t = (out[0] if isinstance(out, tuple) else out).float()
            full_probs = logits_t.softmax(dim=-1).cpu().numpy()
            logits = logits_t.cpu().numpy()
        boards = ds.boards[b].cpu().numpy().astype(np.int8)
        for j, corpus_i in enumerate(idx_np[start:start + len(b)]):
            priors = _get_all_legal_priors_flat(boards[j], logits[j])
            if not priors:
                continue
            ordered = sorted(priors.items(), key=lambda kv: -kv[1])
            base_move, base_prob_legal = ordered[0]
            target_move = int(teacher[corpus_i])
            legal_idx = np.fromiter(priors, dtype=np.int64)
            legal_mass = float(full_probs[j, legal_idx].sum())
            rank = next((r for r, (a, _) in enumerate(ordered, 1)
                         if a == target_move), len(ordered) + 1)
            rows.append({
                'stratum': str(strata[corpus_i]),
                'visit_share': float(visit_share[corpus_i]),
                'legal_mass': legal_mass,
                'p_teacher_legal': float(priors.get(target_move, 0.0)),
                'p_base_legal': float(base_prob_legal),
                'p_teacher_full': float(full_probs[j, target_move]),
                'p_base_full': float(full_probs[j, base_move]),
                'rank': rank,
                'disagree': target_move != base_move,
            })
        done = min(start + args.batch_size, len(idx_np))
        if done % (args.batch_size * 10) == 0 or done == len(idx_np):
            print(f'  {done:,}/{len(idx_np):,}', flush=True)

    crisis = [r for r in rows if r['stratum'].startswith('f_')]
    broad = [r for r in rows if r['stratum'].startswith('b_')]
    print(f'\nrows={len(rows):,}: crisis={len(crisis):,}, broad={len(broad):,}',
          flush=True)
    for name, group in (('crisis', crisis), ('broad', broad)):
        disagree = [r for r in group if r['disagree']]
        print(f'{name} disagreement: {len(disagree):,}/{len(group):,} '
              f'({100*len(disagree)/max(len(group), 1):.1f}%)', flush=True)
        report_group(f'{name} disagreements (all)', disagree, args.eta)
        confident = [r for r in disagree
                     if r['visit_share'] >= args.min_visit_share]
        report_group(
            f'{name} disagreements (visit share >= {args.min_visit_share:g})',
            confident, args.eta)


if __name__ == '__main__':
    main()
