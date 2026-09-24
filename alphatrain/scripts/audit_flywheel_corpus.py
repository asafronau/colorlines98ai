"""Report source, target, and split semantics of a flywheel tensor."""

from __future__ import annotations

import argparse

import numpy as np
import torch


def pct(x):
    return ' '.join(f'P{q}={np.percentile(x, q):.3f}'
                    for q in (10, 50, 90)) if len(x) else 'n/a'


def game_row_stats(game_id):
    _, rows = np.unique(game_id, return_counts=True)
    return (f'games={len(rows):,}; rows/game '
            f'P10={np.percentile(rows, 10):.0f} '
            f'P50={np.percentile(rows, 50):.0f} '
            f'P90={np.percentile(rows, 90):.0f} '
            f'P99={np.percentile(rows, 99):.0f} max={rows.max():,}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('tensor')
    args = p.parse_args()
    d = torch.load(args.tensor, map_location='cpu', weights_only=False)
    md = d.get('metadata', {})
    n = len(d['boards'])
    names = md.get('source_names', [])
    src = d['source_id'].numpy()
    weight = d['target_weight'].numpy()
    split = d['split'].numpy()
    behavior = d['behavior_move'].numpy()
    teacher = d.get('teacher_move')
    teacher = teacher.numpy() if teacher is not None else None
    base_move = d.get('base_move')
    base_move = base_move.numpy() if base_move is not None else None
    game_id = d['game_id'].numpy()
    group_seed = d.get('group_seed')
    group_seed = group_seed.numpy() if group_seed is not None else game_id
    nnz = d['pol_nnz'].numpy()
    pi = d['pol_indices'].numpy()
    pv = d['pol_values'].numpy()

    print(f'{args.tensor}: {n:,} rows; kind={md.get("kind", "?")} '
          f'iteration={md.get("iteration", "?")}')
    print(f'base={md.get("base_checkpoint", "?")}')
    print(f'label protocol={md.get("label_protocol", "?")}')
    print(f'replay games={len(np.unique(game_id)):,}; '
          f'source-seed groups={len(np.unique(group_seed)):,}; '
          f'validation rows={int((split == 1).sum()):,} '
          f'({100*(split == 1).mean():.2f}%)')
    print(game_row_stats(game_id))
    values, counts = np.unique(weight, return_counts=True)
    print('target weights: ' + ', '.join(
        f'{float(v):g}:{int(c):,}' for v, c in zip(values, counts)))

    valid = (nnz > 0) & (weight > 0)
    if valid.any():
        top_col = pv.argmax(1)
        top_action = pi[np.arange(n), top_col]
        top_share = pv.max(1)
        # Cast before the clamp: 1e-30 underflows to zero in float16 and made
        # NumPy evaluate log(0) even on entries later masked by np.where.
        pv_float = pv.astype(np.float64)
        entropy = -(np.where(
            pv_float > 0,
            pv_float * np.log(np.maximum(pv_float, 1e-30)), 0).sum(1))
        print(f'weighted targets={int(valid.sum()):,}; nnz {pct(nnz[valid])}; '
              f'top-share {pct(top_share[valid])}; entropy {pct(entropy[valid])}')
        print(f'behavior == raw-visit argmax: '
              f'{100*(behavior[valid] == top_action[valid]).mean():.2f}%')
        if teacher is not None and base_move is not None:
            recorded = valid & (teacher >= 0) & (base_move >= 0)
            if recorded.any():
                print('teacher == recorded actor-prior argmax: '
                      f'{100*(teacher[recorded] == base_move[recorded]).mean():.2f}% '
                      f'({int(recorded.sum()):,} full-record rows)')

    for sid in np.unique(src):
        mask = src == sid
        name = names[int(sid)] if int(sid) < len(names) else str(int(sid))
        usable = mask & (weight > 0)
        print(f'  {name}: rows={int(mask.sum()):,}; '
              f'target-weight>0={int(usable.sum()):,}; '
              f'val={100*(split[mask] == 1).mean():.2f}%; '
              f'{game_row_stats(game_id[mask])}')


if __name__ == '__main__':
    main()
