"""Mix a new (small) policy-slim tensor with a REHEARSAL sample from a larger
one (the base's own training corpus) — the mC 'replay > regularization' pattern.

    PYTHONPATH=. python -m alphatrain.scripts.mix_tensors \
        --main alphatrain/data/gate3_crisis.pt \
        --rehearsal alphatrain/data/distill_pillar3k.pt \
        --rehearsal-ratio 3.0 --output alphatrain/data/gate3_mix.pt
"""
import argparse, sys
import numpy as np
import torch

sys.path.insert(0, '.')

KEYS = ['boards', 'next_pos', 'next_col', 'n_next',
        'pol_indices', 'pol_values', 'pol_nnz']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--main', required=True)
    p.add_argument('--rehearsal', required=True)
    p.add_argument('--rehearsal-ratio', type=float, default=3.0,
                   help='rehearsal rows = ratio * main rows')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', required=True)
    a = p.parse_args()

    main_t = torch.load(a.main, map_location='cpu', weights_only=False)
    reh = torch.load(a.rehearsal, map_location='cpu', weights_only=False)
    n_main = main_t['boards'].shape[0]
    n_reh = min(int(n_main * a.rehearsal_ratio), reh['boards'].shape[0])
    rng = np.random.default_rng(a.seed)
    idx = torch.from_numpy(
        rng.choice(reh['boards'].shape[0], n_reh, replace=False)).long()

    out = {}
    for k in KEYS:
        km, kr = main_t[k], reh[k][idx]
        # pol_indices K may differ (both are top-5 in current pipelines)
        if km.shape[1:] != kr.shape[1:]:
            raise SystemExit(f'shape mismatch on {k}: {km.shape} vs {kr.shape}')
        out[k] = torch.cat([km, kr], dim=0)
    # carry scalar metadata from the main tensor
    for k in ('max_score', 'num_channels', 'value_mode', 'gamma', 'num_value_bins'):
        if k in main_t:
            out[k] = main_t[k]
    out['mix_info'] = {'main': a.main, 'n_main': n_main,
                       'rehearsal': a.rehearsal, 'n_rehearsal': n_reh,
                       'seed': a.seed}
    torch.save(out, a.output)
    print(f'{a.output}: {n_main:,} main + {n_reh:,} rehearsal '
          f'= {out["boards"].shape[0]:,} states '
          f'({100*n_main/out["boards"].shape[0]:.0f}% new signal)')


if __name__ == '__main__':
    main()
