"""Full-state interpolation theta = (1-a)*base + a*other — ALL params and BN
buffers (for checkpoints trained with live BN, where merge_checkpoints'
frozen-BN assertion doesn't apply). Strips torch.compile prefixes.

    python -m alphatrain.scripts.interp_checkpoints \
        --base alphatrain/data/small128_vh2.pt \
        --other alphatrain/data/iter5_hall4g0_ckpts_epoch_12.pt \
        --alpha 0.25 --out checkpoints/h4z/al025.pt
"""
import argparse

import torch


def sd_of(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--other', required=True)
    ap.add_argument('--alpha', type=float, required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    b, o = sd_of(a.base), sd_of(a.other)
    assert set(b) == set(o), 'key mismatch'
    out = {}
    for k in b:
        if b[k].dtype.is_floating_point:
            out[k] = ((1 - a.alpha) * b[k].float()
                      + a.alpha * o[k].float()).to(b[k].dtype)
        else:  # num_batches_tracked etc.
            out[k] = o[k].clone() if a.alpha >= 0.5 else b[k].clone()
    torch.save({'model': out, 'policy_only': True,
                'interp': {'base': a.base, 'other': a.other, 'alpha': a.alpha}},
               a.out)
    print(f'{a.out}: (1-{a.alpha})*base + {a.alpha}*other (full state incl. BN)')


if __name__ == '__main__':
    main()
