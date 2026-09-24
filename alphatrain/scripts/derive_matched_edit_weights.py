"""Match candidate agreement/edit loss mass to a reference flywheel arm.

Different base policies can disagree with one frozen teacher corpus at very
different rates. Reusing one disagreement multiplier would then confound an
architecture comparison with a different correction/retention dose. This
utility derives candidate agreement and disagreement weights that reproduce
the reference arm's two exact weighted masses, including per-row target and
per-source weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def derive_weights(base_weight, reference_edit, candidate_edit,
                   reference_agree_weight, reference_disagree_weight):
    """Return candidate weights and exact mass diagnostics."""
    base_weight = np.asarray(base_weight, dtype=np.float64)
    reference_edit = np.asarray(reference_edit, dtype=bool)
    candidate_edit = np.asarray(candidate_edit, dtype=bool)
    if not (base_weight.shape == reference_edit.shape
            == candidate_edit.shape):
        raise ValueError('weight and edit masks must have identical shapes')
    if np.any(base_weight < 0):
        raise ValueError('base weights must be nonnegative')

    reference_agree_mass = float(
        base_weight[~reference_edit].sum() * reference_agree_weight)
    reference_edit_mass = float(
        base_weight[reference_edit].sum() * reference_disagree_weight)
    candidate_agree_base = float(base_weight[~candidate_edit].sum())
    candidate_edit_base = float(base_weight[candidate_edit].sum())
    if candidate_agree_base <= 0 or candidate_edit_base <= 0:
        raise ValueError('candidate needs positive-weight agree and edit rows')

    agree_weight = reference_agree_mass / candidate_agree_base
    disagree_weight = reference_edit_mass / candidate_edit_base
    total = reference_agree_mass + reference_edit_mass
    return {
        'rows': int(len(base_weight)),
        'reference_edit_rows': int(reference_edit.sum()),
        'reference_edit_fraction': float(reference_edit.mean()),
        'candidate_edit_rows': int(candidate_edit.sum()),
        'candidate_edit_fraction': float(candidate_edit.mean()),
        'reference_effective_agree_mass': reference_agree_mass,
        'reference_effective_edit_mass': reference_edit_mass,
        'reference_edit_dose_fraction': reference_edit_mass / total,
        'candidate_agree_weight': agree_weight,
        'candidate_disagree_weight': disagree_weight,
        'candidate_effective_agree_mass': (
            candidate_agree_base * agree_weight),
        'candidate_effective_edit_mass': (
            candidate_edit_base * disagree_weight),
    }


def atomic_json(payload, path):
    tmp = str(path) + '.tmp'
    with open(tmp, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state-dir', required=True,
                   help='Resumable corpus-build directory containing '
                        'source_id.npy and target_weight.npy.')
    p.add_argument('--reference-sidecar', required=True)
    p.add_argument('--candidate-sidecar', required=True)
    p.add_argument('--reference-key', default='recorded_disagree')
    p.add_argument('--candidate-key', default='disagree')
    p.add_argument('--source-weights', type=float, nargs='+', required=True)
    p.add_argument('--reference-agree-weight', type=float, default=1.0)
    p.add_argument('--reference-disagree-weight', type=float, required=True)
    p.add_argument('--output')
    args = p.parse_args()

    state_dir = Path(args.state_dir)
    source = np.load(state_dir / 'source_id.npy', mmap_mode='r')
    target_weight = np.load(state_dir / 'target_weight.npy', mmap_mode='r')
    if len(source) != len(target_weight):
        raise ValueError('source_id and target_weight row counts differ')
    source_weights = np.asarray(args.source_weights, dtype=np.float64)
    if np.any(source < 0) or np.any(source >= len(source_weights)):
        raise ValueError('source_id is outside --source-weights')
    base_weight = (np.asarray(target_weight, dtype=np.float64)
                   * source_weights[np.asarray(source, dtype=np.int64)])

    reference = np.load(args.reference_sidecar, allow_pickle=False)
    candidate = np.load(args.candidate_sidecar, allow_pickle=False)
    if args.reference_key not in reference:
        raise ValueError(f'reference sidecar has no {args.reference_key!r}')
    if args.candidate_key not in candidate:
        raise ValueError(f'candidate sidecar has no {args.candidate_key!r}')
    result = derive_weights(
        base_weight, reference[args.reference_key],
        candidate[args.candidate_key], args.reference_agree_weight,
        args.reference_disagree_weight)
    result.update({
        'schema_version': 1,
        'state_dir': args.state_dir,
        'reference_sidecar': args.reference_sidecar,
        'candidate_sidecar': args.candidate_sidecar,
        'reference_key': args.reference_key,
        'candidate_key': args.candidate_key,
        'source_weights': list(args.source_weights),
        'reference_agree_weight': args.reference_agree_weight,
        'reference_disagree_weight': args.reference_disagree_weight,
    })
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        atomic_json(result, args.output)


if __name__ == '__main__':
    main()
