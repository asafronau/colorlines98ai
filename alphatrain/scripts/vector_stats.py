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

from alphatrain.scripts.interp_checkpoints import sd_of


def flat_delta(sd, base):
    return torch.cat([(sd[k].float() - base[k].float()).flatten()
                      for k in sorted(base) if base[k].dtype.is_floating_point])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--others', nargs='+', required=True)
    ap.add_argument('--ref')
    ap.add_argument('--ref-base')
    a = ap.parse_args()
    base = sd_of(a.base)
    deltas = {}
    for p in a.others:
        deltas[os.path.basename(p)] = flat_delta(sd_of(p), base)
    if a.ref:
        rb = sd_of(a.ref_base or a.base)
        deltas['REF:' + os.path.basename(a.ref)] = flat_delta(sd_of(a.ref), rb)
    names = list(deltas)
    print(f'base = {a.base}')
    for n in names:
        print(f'  ||{n}|| = {deltas[n].norm():.2f}')
    print('cosines:')
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = torch.nn.functional.cosine_similarity(
                deltas[names[i]], deltas[names[j]], dim=0)
            print(f'  cos({names[i]}, {names[j]}) = {c:.3f}')


if __name__ == '__main__':
    main()
