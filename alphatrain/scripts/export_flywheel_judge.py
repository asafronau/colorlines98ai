"""Export held-out flywheel edits for the native common-random-number judge.

The output is a CLRJ binary consumed by ``inference_cpp/build/rollout_judge``
and a row-aligned ``*_rows.npz`` sidecar.  Sampling is explicitly stratified:
successful exploit trajectories, failed exploit broad/tail states, explore,
crisis prevention, and crisis recovery.  This prevents the abundant broad
states from hiding a useful (or harmful) crisis class in one aggregate mean.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct

import numpy as np


REQUIRED_ARRAYS = (
    'boards', 'next_pos', 'next_col', 'n_next', 'teacher_move', 'base_move',
    'source_id', 'trajectory_kind', 'split', 'turn', 'game_turns',
    'game_capped', 'group_seed', 'target_weight',
)


def load_arrays(state_dir: Path):
    arrays = {}
    for name in REQUIRED_ARRAYS:
        path = state_dir / f'{name}.npy'
        if not path.exists():
            raise FileNotFoundError(path)
        arrays[name] = np.load(path, mmap_mode='r')
    n = len(arrays['boards'])
    for name, value in arrays.items():
        if len(value) != n:
            raise ValueError(f'{name} has {len(value):,} rows, expected {n:,}')
    return arrays


def stratum_masks(arrays, eligible):
    source = np.asarray(arrays['source_id'])
    trajectory = np.asarray(arrays['trajectory_kind'])
    capped = np.asarray(arrays['game_capped']).astype(bool)
    remaining = (np.asarray(arrays['game_turns'], dtype=np.int64)
                 - np.asarray(arrays['turn'], dtype=np.int64))
    exploit = source == 0
    failed = ~capped
    return {
        'exploit_success': eligible & exploit & capped,
        'exploit_failed_broad': eligible & exploit & failed & (remaining > 100),
        'exploit_failed_tail20': eligible & exploit & failed & (remaining <= 20),
        'explore': eligible & (source == 1),
        'prevention': eligible & (trajectory == 2),
        'recovery': eligible & (trajectory == 3),
    }


def sample_strata(masks, per_stratum, seed):
    rng = np.random.default_rng(seed)
    selected = []
    names = []
    inventory = {}
    for name, mask in masks.items():
        available = np.flatnonzero(mask)
        take = min(per_stratum, len(available))
        chosen = (rng.choice(available, size=take, replace=False)
                  if take else np.empty(0, dtype=np.int64))
        chosen.sort()
        selected.append(chosen)
        names.extend([name] * take)
        inventory[name] = {'available': int(len(available)), 'selected': take}
    if not selected or not any(len(x) for x in selected):
        raise ValueError('no eligible judge rows')
    return np.concatenate(selected), np.asarray(names), inventory


def atomic_write_bin(path, rows, arrays, top_share):
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as handle:
        handle.write(b'CLRJ')
        handle.write(struct.pack('<i', len(rows)))
        for row in rows:
            handle.write(np.asarray(
                arrays['boards'][row], dtype=np.int8).reshape(81).tobytes())
            handle.write(struct.pack('<i', int(arrays['n_next'][row])))
            for i in range(3):
                handle.write(struct.pack(
                    '<iii', int(arrays['next_pos'][row, i, 0]),
                    int(arrays['next_pos'][row, i, 1]),
                    int(arrays['next_col'][row, i])))
            handle.write(struct.pack(
                '<iif', int(arrays['teacher_move'][row]),
                int(arrays['base_move'][row]), float(top_share[row])))
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--base-policy-sidecar', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--per-stratum', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20260818)
    parser.add_argument('--strata', nargs='+',
                        help=('Optional subset to emit. Sampling is still run '
                              'for every stratum first, so the same seed '
                              'selects the same rows as an earlier full export.'))
    parser.add_argument('--include-train', action='store_true',
                        help='Sample all rows; default uses held-out groups only.')
    args = parser.parse_args()
    if args.per_stratum <= 0:
        raise ValueError('--per-stratum must be positive')

    state_dir = Path(args.state_dir)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(state_dir)
    n = len(arrays['boards'])

    sidecar = np.load(args.base_policy_sidecar, allow_pickle=False)
    for key in ('recorded_disagree', 'recorded_base_valid', 'target_top_share'):
        if key not in sidecar:
            raise ValueError(f'base-policy sidecar has no {key!r}')
        if len(sidecar[key]) != n:
            raise ValueError(f'{key} row count does not match corpus')
    edit = sidecar['recorded_disagree'].astype(bool)
    valid = sidecar['recorded_base_valid'].astype(bool)
    eligible = edit & valid
    eligible &= np.asarray(arrays['teacher_move']) >= 0
    eligible &= np.asarray(arrays['base_move']) >= 0
    eligible &= (np.asarray(arrays['teacher_move'])
                 != np.asarray(arrays['base_move']))
    if not args.include_train:
        eligible &= np.asarray(arrays['split']) == 1

    masks = stratum_masks(arrays, eligible)
    rows, strata, inventory = sample_strata(
        masks, args.per_stratum, args.seed)
    if args.strata:
        unknown = sorted(set(args.strata) - set(masks))
        if unknown:
            raise ValueError(f'unknown --strata: {unknown}')
        keep = np.isin(strata, args.strata)
        rows, strata = rows[keep], strata[keep]
        if not len(rows):
            raise ValueError('--strata selected no rows')
    top_share = sidecar['target_top_share']
    atomic_write_bin(output, rows, arrays, top_share)

    metadata = {
        'schema_version': 1,
        'state_dir': str(state_dir),
        'base_policy_sidecar': args.base_policy_sidecar,
        'seed': args.seed,
        'held_out_only': not args.include_train,
        'per_stratum': args.per_stratum,
        'emitted_strata': (args.strata if args.strata else list(masks)),
        'inventory': inventory,
    }
    sidecar_path = output.with_name(output.stem + '_rows.npz')
    tmp_sidecar = sidecar_path.with_name(sidecar_path.name + '.tmp.npz')
    np.savez(
        tmp_sidecar,
        rows=rows,
        stratum=strata,
        source_id=np.asarray(arrays['source_id'][rows]),
        trajectory_kind=np.asarray(arrays['trajectory_kind'][rows]),
        group_seed=np.asarray(arrays['group_seed'][rows]),
        turn=np.asarray(arrays['turn'][rows]),
        game_turns=np.asarray(arrays['game_turns'][rows]),
        game_capped=np.asarray(arrays['game_capped'][rows]),
        target_weight=np.asarray(arrays['target_weight'][rows]),
        teacher_move=np.asarray(arrays['teacher_move'][rows]),
        base_move=np.asarray(arrays['base_move'][rows]),
        top_share=np.asarray(top_share[rows]),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(tmp_sidecar, sidecar_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f'wrote {output}: {len(rows):,} held-out edit states')
    print(f'wrote {sidecar_path}')


if __name__ == '__main__':
    main()
