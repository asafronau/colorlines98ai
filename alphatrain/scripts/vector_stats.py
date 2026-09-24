"""Norms and cosines of checkpoint displacement vectors vs a base — decide
merge doses BEFORE building candidates (round-2 plan: beta norm-scaled).

    python -m alphatrain.scripts.vector_stats \
        --base alphatrain/data/small128_vh3.pt \
        --others alphatrain/data/r2bulk_dg_ckpts_epoch_1.pt ... \
        --ref checkpoints/h4z/iter5_hall4g0_ckpts_epoch_12.pt \
        --ref-base alphatrain/data/small128_vh2.pt
"""
import argparse
import os

import torch
import torch.nn.functional as F

from alphatrain.scripts.interp_checkpoints import sd_of


def parameter_groups(base):
    """Separate learned weights, BN affine parameters, and BN buffers.

    Full-state vector norms/cosines can be dominated by BatchNorm running
    means/variances even though those buffers are not optimizer parameters.
    Reporting only the concatenated state therefore answers the wrong
    geometric question for task-vector merges.
    """
    bn_prefixes = {
        k.rsplit('.', 1)[0]
        for k in base
        if k.endswith('.running_mean') or k.endswith('.running_var')
    }
    groups = {'weights': [], 'bn_affine': [], 'bn_buffers': [], 'all_float': []}
    for k in sorted(base):
        if not base[k].dtype.is_floating_point:
            continue
        groups['all_float'].append(k)
        prefix, leaf = k.rsplit('.', 1)
        if leaf in ('running_mean', 'running_var'):
            groups['bn_buffers'].append(k)
        elif prefix in bn_prefixes and leaf in ('weight', 'bias'):
            groups['bn_affine'].append(k)
        else:
            groups['weights'].append(k)
    return groups


def flat_delta(sd, base, keys):
    return torch.cat([(sd[k].float() - base[k].float()).flatten()
                      for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--others', nargs='+', required=True)
    ap.add_argument('--ref')
    ap.add_argument('--ref-base')
    a = ap.parse_args()
    base = sd_of(a.base)
    groups = parameter_groups(base)
    states = {}
    basenames = [os.path.basename(p) for p in a.others]
    for i, p in enumerate(a.others):
        # Step checkpoints from different experiment directories commonly have
        # the same basename.  Never silently overwrite one comparison arm.
        label = basenames[i]
        if basenames.count(label) > 1:
            label = os.path.join(os.path.basename(os.path.dirname(p)), label)
        if label in states:
            label = f'{i}:{p}'
        states[label] = (sd_of(p), base)
    if a.ref:
        rb = sd_of(a.ref_base or a.base)
        states['REF:' + os.path.basename(a.ref)] = (sd_of(a.ref), rb)
    names = list(states)
    print(f'base = {a.base}')
    for group in ('weights', 'bn_affine', 'bn_buffers', 'all_float'):
        # A reference may have a distinct base checkpoint but the same model
        # schema, so reuse the key classification by name.
        deltas = {
            n: flat_delta(sd, b, groups[group])
            for n, (sd, b) in states.items()
        }
        print(f'\n[{group}] elements={len(deltas[names[0]]):,}')
        for n in names:
            print(f'  ||{n}|| = {deltas[n].norm():.4f}')
        print('  cosines:')
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                c = F.cosine_similarity(
                    deltas[names[i]], deltas[names[j]], dim=0)
                print(f'    cos({names[i]}, {names[j]}) = {c:.4f}')


if __name__ == '__main__':
    main()
