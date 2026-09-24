"""Read-only, seeded checks of the September project-resumption hypotheses.

Run from the repo root. Samples are diagnostic, not gameplay evaluations.
The CPU flood fill is independent of the tensor component implementation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from alphatrain.scripts.fleet_gpu import gpu_label_components
from game.board import _label_empty_components


def summary(x):
    x = np.asarray(x, dtype=float)
    return dict(mean=float(x.mean()), p10=float(np.percentile(x, 10)),
                p50=float(np.median(x)), p90=float(np.percentile(x, 90)))


def canonical_labels(board):
    labels = _label_empty_components(board).reshape(-1)
    out = np.zeros(81, dtype=np.int64)
    for label in np.unique(labels):
        if label:
            cells = np.flatnonzero(labels == label)
            out[cells] = cells.min() + 1
    return out.reshape(9, 9)


def audit_tensor(path, samples, seed):
    d = torch.load(path, weights_only=True, mmap=True, map_location='cpu')
    n = len(d['boards'])
    rng = np.random.default_rng(seed)
    ix = np.sort(rng.choice(n, min(n, samples), replace=False))
    selected = {k: v[ix].numpy() for k, v in d.items()
                if isinstance(v, torch.Tensor) and len(v) == n}
    out = {'path': str(path), 'rows': n, 'sample_size': len(ix),
           'seed': seed, 'policy_width': d['pol_values'].shape[1],
           'metadata': d.get('metadata', {}), 'sources': {}}
    names = d['metadata']['source_names']
    for sid, name in enumerate(names):
        mask = selected['source_id'] == sid
        if not mask.any():
            continue
        v = selected['cand_visit'][mask].astype(float)
        p = selected['cand_prior'][mask].astype(float)
        support = np.arange(p.shape[1])[None, :] < selected['pol_nnz'][mask, None]
        if not np.isfinite(p[support]).all():
            raise ValueError('nonfinite prior inside recorded support')
        # Schema 4 stores LOG priors, and omits zero-visit candidates. Compare
        # distributions on that common support; report the retained prior mass.
        p = np.where(support, np.exp(p), 0.0)
        prior_mass = p.sum(1)
        v /= v.sum(1, keepdims=True)
        p /= p.sum(1, keepdims=True)
        kl = (v * (np.log(np.maximum(v, 1e-30)) -
                   np.log(np.maximum(p, 1e-30)))).sum(1)
        edits = (selected['base_move'][mask] != selected['teacher_move'][mask])
        decisiveness = v.max(1) ** 3
        s = {
            'sample_rows': int(mask.sum()),
            'root_value': summary(selected['root_value'][mask]),
            'tree_q_range': summary(selected['q_max'][mask] - selected['q_min'][mask]),
            'prior_top_share': summary(p.max(1)),
            'retained_prior_mass': summary(prior_mass),
            'visit_top_share': summary(v.max(1)),
            'top5_visit_mass': summary(np.sort(v, axis=1)[:, -5:].sum(1)),
            'search_prior_kl': summary(kl),
            'decisiveness_kl_correlation': float(np.corrcoef(decisiveness, kl)[0, 1]),
            'edit_fraction': float(edits.mean()),
            'edit_decisiveness_weight_fraction': float(decisiveness[edits].sum() / decisiveness.sum()),
            'edit_kl_weight_fraction': float(kl[edits].sum() / max(kl.sum(), 1e-30)),
            'capped_game_row_fraction': float(selected['game_capped'][mask].mean()),
        }
        out['sources'][name] = s
        print(name, json.dumps(s), flush=True)

    # All groups, not just the sampled rows: train/validation overlap.
    split = d['split'].numpy()
    groups = d['group_seed'].numpy()
    out['split'] = {
        'train_rows': int((split == 0).sum()),
        'val_rows': int((split == 1).sum()),
        'overlapping_group_seeds': int(len(np.intersect1d(
            np.unique(groups[split == 0]), np.unique(groups[split == 1])))),
    }
    failures = []
    for start in range(0, len(ix), 512):
        boards = selected['boards'][start:start + 512]
        got = gpu_label_components(torch.from_numpy(boards)).numpy()
        for j, board in enumerate(boards):
            expected = canonical_labels(board)
            if not np.array_equal(got[j], expected):
                failures.append({'tensor_row': int(ix[start + j]),
                                 'board': board.tolist(),
                                 'actual': got[j].tolist(),
                                 'expected': expected.tolist()})
        if start % 5120 == 0:
            print(f'component audit {start + len(boards)}/{len(ix)}; '
                  f'failures={len(failures)}', flush=True)
    out['component_failures'] = failures
    out['component_failure_fraction'] = len(failures) / len(ix)
    return out


def audit_checkpoint(path):
    d = torch.load(path, weights_only=False, map_location='cpu')
    state = d.get('model_state_dict', d.get('model', d))
    bad = {}
    for k, v in state.items():
        if k.endswith('running_var'):
            mask = ~torch.isfinite(v.half())
            if mask.any():
                bad[k] = {'indices': mask.nonzero().flatten().tolist(),
                          'fp32_values': v[mask].tolist()}
    return {'path': str(path), 'bn_fp16_nonfinite': bad}


def audit_eval(base_path, candidate_path):
    a = np.genfromtxt(base_path, delimiter=',', names=True)
    b = np.genfromtxt(candidate_path, delimiter=',', names=True)
    out = {'base': str(base_path), 'candidate': str(candidate_path),
           'n_base': len(a), 'n_candidate': len(b),
           'mean_difference': float(b['score'].mean() - a['score'].mean()),
           'base_score': summary(a['score']), 'candidate_score': summary(b['score'])}
    # Equal IDs are paired only for this covariance diagnostic. Matching IDs
    # alone do not establish matching engine/initialization protocols.
    common, ia, ib = np.intersect1d(a['seed'], b['seed'], return_indices=True)
    if len(common) > 1:
        aa, bb = a['score'][ia], b['score'][ib]
        out['matching_seed_score_correlation'] = float(np.corrcoef(aa, bb)[0, 1])
        out['matching_seed_count'] = len(common)
    if 'capped' in a.dtype.names and 'capped' in b.dtype.names:
        pa, pb = float(a['capped'].mean()), float(b['capped'].mean())
        se = (pa * (1-pa) / len(a) + pb * (1-pb) / len(b)) ** .5
        interior = 0 < pa < 1 and 0 < pb < 1
        out.update(base_cap_rate=pa, candidate_cap_rate=pb,
                   cap_difference=pb-pa,
                   cap_difference_normal_ci=([pb-pa-1.96*se, pb-pa+1.96*se] if interior else None),
                   approximate_80pct_power_mde=(2.802*se if interior else None))
        if 'turns' in a.dtype.names and 'turns' in b.dtype.names:
            out['base_censoring_turns'] = np.unique(a['turns'][a['capped'] == 1]).tolist()
            out['candidate_censoring_turns'] = np.unique(b['turns'][b['capped'] == 1]).tolist()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--eval-pair', nargs=2, action='append', default=[])
    p.add_argument('--samples', type=int, default=20000)
    p.add_argument('--seed', type=int, default=20260905)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.samples <= 0:
        p.error('--samples must be positive')
    torch.set_num_threads(2)
    out = {'tensor': audit_tensor(args.tensor, args.samples, args.seed)}
    if args.checkpoint:
        out['checkpoint'] = audit_checkpoint(args.checkpoint)
    out['evals'] = [audit_eval(*pair) for pair in args.eval_pair]
    for entry in out['evals']:
        print(json.dumps(entry), flush=True)
    Path(args.output).write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()
