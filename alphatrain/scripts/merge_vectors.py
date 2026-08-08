"""Multi-vector merge: theta = base + sum_i alpha_i * (other_i - base),
full state incl. BN buffers (float buffers combined by the same formula;
num_batches_tracked taken from base). Generalizes interp_checkpoints to
N vectors for the round-2 bulk+frontier merges.

    python -m alphatrain.scripts.merge_vectors \
        --base alphatrain/data/small128_vh3.pt \
        --add 0.5 alphatrain/data/r2bulk_dg_ckpts_epoch_1.pt \
        --add 0.1 alphatrain/data/r2frontier_ckpts_epoch_12.pt \
        --out checkpoints/r2m/dg1a05_f12b01.pt
"""
import argparse

import torch

from alphatrain.scripts.interp_checkpoints import sd_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--add', nargs=2, action='append', required=True,
                    metavar=('ALPHA', 'CKPT'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    base = sd_of(a.base)
    out = {k: v.float().clone() if v.dtype.is_floating_point else v.clone()
           for k, v in base.items()}
    for alpha_s, path in a.add:
        alpha = float(alpha_s)
        sd = sd_of(path)
        assert set(sd) == set(base), f'key mismatch: {path}'
        for k in base:
            if base[k].dtype.is_floating_point:
                out[k] += alpha * (sd[k].float() - base[k].float())
    out = {k: v.to(base[k].dtype) for k, v in out.items()}
    torch.save({'model': out, 'policy_only': True,
                'merge': {'base': a.base, 'add': a.add}}, a.out)
    print(f'{a.out}: base + ' + ' + '.join(f'{al}*({p} - base)'
                                           for al, p in a.add))


if __name__ == '__main__':
    main()
