"""Build a corpus-matched mask of teacher edits shared by two policy bases.

For an architecture absorption control, ``teacher != reference`` alone is not
enough: the candidate may already play that teacher move.  Conversely,
``teacher != candidate`` includes ordinary reference-policy imitation rows.
The intersection selects states on which the same frozen teacher action is a
genuine correction to both bases.  All remaining states can still contribute
base-KL preservation; they simply receive no hard-teacher dose.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def edit_partition(reference_edit, candidate_edit):
    reference_edit = np.asarray(reference_edit, dtype=bool)
    candidate_edit = np.asarray(candidate_edit, dtype=bool)
    if reference_edit.shape != candidate_edit.shape:
        raise ValueError('sidecar masks have different shapes')
    return {
        'shared': reference_edit & candidate_edit,
        'reference_only': reference_edit & ~candidate_edit,
        'candidate_only': ~reference_edit & candidate_edit,
        'neither': ~reference_edit & ~candidate_edit,
    }


def _metadata(sidecar):
    if 'metadata' not in sidecar:
        return {}
    return json.loads(str(sidecar['metadata']))


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-sidecar', required=True)
    p.add_argument('--candidate-sidecar', required=True)
    p.add_argument('--reference-key', default='recorded_disagree')
    p.add_argument('--candidate-key', default='disagree')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    reference = np.load(args.reference_sidecar, allow_pickle=False)
    candidate = np.load(args.candidate_sidecar, allow_pickle=False)
    if args.reference_key not in reference:
        raise ValueError(
            f'reference sidecar has no {args.reference_key!r}')
    if args.candidate_key not in candidate:
        raise ValueError(
            f'candidate sidecar has no {args.candidate_key!r}')
    rmd, cmd = _metadata(reference), _metadata(candidate)
    rinv = rmd.get('tensor_inventory_sha256')
    cinv = cmd.get('tensor_inventory_sha256')
    if rinv and cinv and rinv != cinv:
        raise ValueError('sidecars describe different tensor inventories')
    if ('target_argmax' in reference and 'target_argmax' in candidate
            and not np.array_equal(reference['target_argmax'],
                                   candidate['target_argmax'])):
        raise ValueError('sidecars do not contain the same teacher actions')

    parts = edit_partition(reference[args.reference_key],
                           candidate[args.candidate_key])
    n = len(parts['shared'])
    counts = {name: int(mask.sum()) for name, mask in parts.items()}
    metadata = {
        'schema_version': 1,
        'kind': 'shared_teacher_edit_mask',
        'rows': n,
        'tensor_inventory_sha256': rinv or cinv,
        'reference_sidecar': args.reference_sidecar,
        'candidate_sidecar': args.candidate_sidecar,
        'reference_key': args.reference_key,
        'candidate_key': args.candidate_key,
        'reference_base_sha256': rmd.get('base_sha256'),
        'candidate_base_sha256': cmd.get('base_sha256'),
        'partition_counts': counts,
        'semantics': ('teacher action differs from both the reference actor '
                      'and candidate deployment legal argmax'),
    }
    top_share = reference.get('target_top_share')
    if top_share is None:
        top_share = np.zeros(n, dtype=np.float16)
    _atomic_savez(
        args.output,
        # Existing trainer/auditor keys are retained deliberately.  Both are
        # the composite mask, not a claim about either single base.
        disagree=parts['shared'],
        recorded_disagree=parts['shared'],
        target_top_share=top_share,
        protocol=np.array(metadata['semantics']),
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        'output': args.output,
        'rows': n,
        'partition_counts': counts,
        'partition_fractions': {
            name: count / n for name, count in counts.items()
        },
    }, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
