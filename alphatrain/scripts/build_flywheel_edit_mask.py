"""Build an immutable, provenance-checked subset of flywheel search edits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state-dir', required=True)
    parser.add_argument('--base-policy-sidecar', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--trajectory-kinds', type=int, nargs='+')
    parser.add_argument('--source-ids', type=int, nargs='+')
    parser.add_argument('--min-top-share', type=float, default=0.0)
    args = parser.parse_args()
    if not args.trajectory_kinds and not args.source_ids:
        raise ValueError('select --trajectory-kinds and/or --source-ids')
    if not 0 <= args.min_top_share <= 1:
        raise ValueError('--min-top-share must be in [0,1]')

    state_dir = Path(args.state_dir)
    trajectory = np.load(state_dir / 'trajectory_kind.npy', mmap_mode='r')
    source = np.load(state_dir / 'source_id.npy', mmap_mode='r')
    sidecar = np.load(args.base_policy_sidecar, allow_pickle=False)
    required = ('recorded_disagree', 'recorded_base_valid',
                'target_top_share')
    for key in required:
        if key not in sidecar:
            raise ValueError(f'base-policy sidecar has no {key!r}')
        if len(sidecar[key]) != len(trajectory):
            raise ValueError(f'{key} row count does not match state directory')

    selected = np.ones(len(trajectory), dtype=bool)
    if args.trajectory_kinds:
        selected &= np.isin(trajectory, args.trajectory_kinds)
    if args.source_ids:
        selected &= np.isin(source, args.source_ids)
    selected &= sidecar['target_top_share'] >= args.min_top_share
    edit = (sidecar['recorded_disagree'].astype(bool)
            & sidecar['recorded_base_valid'].astype(bool) & selected)

    metadata = (json.loads(str(sidecar['metadata']))
                if 'metadata' in sidecar else {})
    metadata['edit_mask'] = {
        'schema_version': 1,
        'base_policy_sidecar': args.base_policy_sidecar,
        'state_dir': str(state_dir),
        'trajectory_kinds': args.trajectory_kinds,
        'source_ids': args.source_ids,
        'min_top_share': args.min_top_share,
        'selected_edit_rows': int(edit.sum()),
    }
    counts_by_trajectory = {
        str(int(kind)): int((edit & (trajectory == kind)).sum())
        for kind in np.unique(trajectory[edit])
    }
    counts_by_source = {
        str(int(sid)): int((edit & (source == sid)).sum())
        for sid in np.unique(source[edit])
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.tmp.npz')
    np.savez(
        tmp,
        recorded_disagree=edit.astype(np.int8),
        disagree=edit.astype(np.int8),
        target_top_share=np.asarray(sidecar['target_top_share']),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(tmp, output)
    report = {
        'output': str(output),
        'rows': int(len(edit)),
        'selected_edit_rows': int(edit.sum()),
        'counts_by_trajectory_kind': counts_by_trajectory,
        'counts_by_source_id': counts_by_source,
        'selection': metadata['edit_mask'],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
