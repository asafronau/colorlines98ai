"""Average K task vectors into one checkpoint: theta = base + mean_k(delta_k).

    python -m alphatrain.scripts.avg_vectors \
        --base alphatrain/data/small128_vh2.pt \
        --checkpoints checkpoints/r1vh2/s0/ft_epoch_30.pt ... \
        --out checkpoints/r1vh2/ft_avg.pt
"""
import argparse

import torch


def sd_of(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    base = sd_of(a.base)
    reps = [sd_of(c) for c in a.checkpoints]
    out = {}
    for k in base:
        if 'running_' in k or 'num_batches' in k:
            out[k] = base[k].clone()  # BN stats stay the base's (frozen-BN ft)
        else:
            deltas = torch.stack([r[k].float() - base[k].float() for r in reps])
            out[k] = (base[k].float() + deltas.mean(0)).to(base[k].dtype)
    torch.save({'model': out, 'policy_only': True,
                'avg_of': a.checkpoints, 'base': a.base}, a.out)
    print(f'{a.out}: base + mean of {len(reps)} vectors')


if __name__ == '__main__':
    main()
