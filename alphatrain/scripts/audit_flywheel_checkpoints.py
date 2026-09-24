"""Functional audit of flywheel checkpoints on fixed game-group splits.

Reports frozen-base KL/retention and hard-teacher adoption.  This is a cheap
checkpoint selector before gameplay, not promotion evidence.  Run under
``caffeinate -i -s``; FP16 is the default inference protocol.
"""

from __future__ import annotations

import argparse
import copy
import json
import math

import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.train_flywheel import (
    TensorBatchLoader, _atomic_json_save, _sha256, audit, cap_dataset,
    make_datasets,
)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--targets', required=True)
    p.add_argument('--anchors', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--base-policy-sidecar',
                   help='Optional add_fulllegal_mask output; enables exact '
                        'agreement/disagreement strata.')
    p.add_argument(
        '--sidecar-disagree-key', default='disagree',
        choices=('disagree', 'recorded_disagree'),
        help='Edit stratum to read from --base-policy-sidecar.')
    p.add_argument(
        '--strata-sidecar',
        help=('Optional corpus-matched sidecar defining reported edit strata. '
              '--base-policy-sidecar still validates --base provenance.'))
    p.add_argument(
        '--strata-key', choices=('disagree', 'recorded_disagree'),
        help='Mask in --strata-sidecar.')
    p.add_argument('--min-target-share', type=float, default=0.30)
    p.add_argument('--bounded-eta', type=float,
                   help=('Report the teacher-argmax adoption of the exact '
                         'unconstrained bounded optimum at this eta.'))
    p.add_argument('--bounded-soft-alpha', type=float, default=0.0,
                   help='Search target mixture: 1=visits, 0=hard teacher.')
    p.add_argument('--bounded-source-weights', type=float, nargs='+',
                   help='Source multipliers used by the bounded run.')
    p.add_argument('--bounded-agree-weight', type=float, default=1.0,
                   help='Agree-row multiplier used by the bounded run.')
    p.add_argument('--bounded-disagree-weight', type=float, default=1.0,
                   help='Edit-row multiplier used by the bounded run.')
    p.add_argument('--rows-per-source', type=int, default=50_000)
    p.add_argument('--split', choices=('validation', 'train'),
                   default='validation',
                   help=('Game-group split to audit. Validation is the '
                         'default; train diagnoses fit versus generalization.'))
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=20260808)
    p.add_argument('--device', default='mps')
    p.add_argument('--fp32', action='store_true',
                   help='Diagnostic override; FP16 is the default protocol.')
    p.add_argument('--output',
                   help='Optional atomic JSON output for persistent gates.')
    args = p.parse_args()
    if args.bounded_eta is not None and args.bounded_eta < 0:
        raise ValueError('--bounded-eta must be nonnegative')
    if not 0 <= args.bounded_soft_alpha <= 1:
        raise ValueError('--bounded-soft-alpha must be in [0,1]')
    if args.strata_sidecar and not args.base_policy_sidecar:
        raise ValueError('--strata-sidecar requires --base-policy-sidecar')
    if args.strata_key and not args.strata_sidecar:
        raise ValueError('--strata-key requires --strata-sidecar')

    device = torch.device(args.device)
    audit_train = args.split == 'train'
    target = make_datasets(
        args.targets, device, augment=False, color_augment=False,
        augment_factor=1, train=audit_train)
    anchor = make_datasets(
        args.anchors, device, augment=False, color_augment=False,
        augment_factor=1, train=audit_train)
    rng = np.random.default_rng(args.seed)
    cap_dataset(anchor, args.rows_per_source, rng)
    target_views = {'target/all': target}
    names = (target.metadata or {}).get('source_names', [])
    source = target.source_id[target.base_indices]
    for sid, name in enumerate(names):
        view = copy.copy(target)
        view.base_indices = target.base_indices[source == sid]
        cap_dataset(view, args.rows_per_source, rng)
        target_views[f'target/source/{name}'] = view
    if args.base_policy_sidecar:
        side = np.load(args.base_policy_sidecar, allow_pickle=False)
        if 'metadata' in side:
            metadata = json.loads(str(side['metadata']))
            recorded_hash = metadata.get('base_sha256')
            if recorded_hash and recorded_hash != _sha256(args.base):
                raise ValueError(
                    'sidecar base checkpoint hash does not match '
                    f'{args.base!r}')
            if not recorded_hash and metadata.get('base') != args.base:
                raise ValueError(
                    f'sidecar base {metadata.get("base")!r} != {args.base!r}; '
                    'legacy sidecar has no content hash')
        strata = side
        strata_key = args.sidecar_disagree_key
        if args.strata_sidecar:
            strata = np.load(args.strata_sidecar, allow_pickle=False)
            strata_key = args.strata_key or args.sidecar_disagree_key
            if 'metadata' in strata:
                strata_metadata = json.loads(str(strata['metadata']))
                expected_inventory = (target.metadata or {}).get(
                    'input_inventory_sha256')
                recorded_inventory = strata_metadata.get(
                    'tensor_inventory_sha256')
                if (expected_inventory and recorded_inventory
                        and expected_inventory != recorded_inventory):
                    raise ValueError('strata sidecar corpus inventory does '
                                     'not match target tensor')
        if strata_key not in strata:
            raise ValueError(
                f'strata sidecar has no {strata_key!r} field')
        sidecar_disagree = strata[strata_key]
        if len(sidecar_disagree) != len(target.boards):
            raise ValueError('strata sidecar row count mismatch')
        full_disagree = torch.from_numpy(
            sidecar_disagree.astype(bool)).to(device)
        full_share = torch.from_numpy(
            strata['target_top_share'].astype(np.float32)).to(device)
        # Carry the immutable optimization stratum into every copied view so
        # train_flywheel.audit reports adoption/optimum for the mined mask,
        # rather than recomputing all online base/teacher disagreements.
        for view in target_views.values():
            view.flywheel_disagree = full_disagree
        for label, mask in (
                ('agree', ~full_disagree),
                ('disagree', full_disagree),
                (f'disagree_share_ge_{args.min_target_share:g}',
                 full_disagree & (full_share >= args.min_target_share))):
            view = copy.copy(target)
            view.base_indices = target.base_indices[mask[target.base_indices]]
            cap_dataset(view, args.rows_per_source, rng)
            target_views[f'target/{label}'] = view
    # Cap the aggregate view last; the stratum views above retain their own
    # independent fixed samples while sharing the immutable tensor backing.
    cap_dataset(target, args.rows_per_source, rng)
    loaders = {
        name: TensorBatchLoader(view, args.batch_size, shuffle=False)
        for name, view in target_views.items()
    }
    loaders['anchor'] = TensorBatchLoader(
        anchor, args.batch_size, shuffle=False)
    max_batches = math.ceil(args.rows_per_source / args.batch_size)
    amp_dtype = torch.float32 if args.fp32 else torch.float16
    base, _ = load_model(args.base, device, fp16=not args.fp32)
    base.requires_grad_(False)

    print(f'fixed {args.split} rows: ' + ', '.join(
        f'{name}={len(loader.dataset):,}' for name, loader in loaders.items())
        + f'; precision={amp_dtype}', flush=True)
    model_reports = {}
    for path in args.models:
        student, _ = load_model(path, device, fp16=not args.fp32)
        student.requires_grad_(False)
        metrics = audit(
            student, base, loaders, device,
            max_batches=max_batches, amp_dtype=amp_dtype,
            bounded_eta=args.bounded_eta,
            bounded_source_weights=args.bounded_source_weights,
            bounded_agree_weight=args.bounded_agree_weight,
            bounded_disagree_weight=args.bounded_disagree_weight,
            bounded_soft_alpha=args.bounded_soft_alpha)
        model_reports[path] = metrics
        print(f'\n[{path}]', flush=True)
        for name, v in metrics.items():
            print(
                f'  {name:6s} n={v["n"]:,} '
                f'retain={v["retain"]:.4f} '
                f'KLfull={v["mean_kl"]:.6f}/{v["p90_kl"]:.6f} '
                f'KLlegal={v["mean_legal_kl"]:.6f}/'
                f'{v["p90_legal_kl"]:.6f} '
                f'teacher={v["teacher_before"]:.4f}->'
                f'{v["teacher_after"]:.4f}/{v["teacher_n"]} '
                f'edit_adopt={v["edit_adopt"]:.4f}/{v["edit_n"]} '
                f'agree_preserve={v["agree_preserve"]:.4f}'
                + (f' ideal@eta={v["bounded_optimum_edit_adopt"]:.4f} '
                   f'gap50/90={v["edit_probability_gap_p50"]:.4f}/'
                   f'{v["edit_probability_gap_p90"]:.4f} '
                   f'teacherP={v["edit_teacher_probability_before"]:.4f}->'
                   f'{v["edit_teacher_probability_after"]:.4f} '
                   f'qKLstudent(edit)={v["bounded_optimum_residual_kl"]:.5f} '
                   f'qKLstudent/base(all)='
                   f'{v["bounded_optimum_all_residual_kl"]:.5f}/'
                   f'{v["bounded_optimum_all_base_kl"]:.5f} '
                   f'reverseBaseKLq='
                   f'{v["bounded_optimum_all_base_to_target_kl"]:.5f}'
                   if 'bounded_optimum_edit_adopt' in v else ''),
                flush=True)
        del student
        if device.type == 'mps':
            torch.mps.empty_cache()
    if args.output:
        _atomic_json_save({
            'schema_version': 1,
            'targets': args.targets,
            'anchors': args.anchors,
            'base': args.base,
            'base_policy_sidecar': args.base_policy_sidecar,
            'sidecar_disagree_key': args.sidecar_disagree_key,
            'strata_sidecar': args.strata_sidecar,
            'strata_key': args.strata_key,
            'split': args.split,
            'rows_per_source': args.rows_per_source,
            'batch_size': args.batch_size,
            'seed': args.seed,
            'device': args.device,
            'precision': 'fp32' if args.fp32 else 'fp16',
            'bounded_eta': args.bounded_eta,
            'bounded_soft_alpha': args.bounded_soft_alpha,
            'bounded_source_weights': args.bounded_source_weights,
            'bounded_agree_weight': args.bounded_agree_weight,
            'bounded_disagree_weight': args.bounded_disagree_weight,
            'models': model_reports,
        }, args.output)
        print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
