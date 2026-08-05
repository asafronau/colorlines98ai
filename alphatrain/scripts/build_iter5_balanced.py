"""Build the FAITHFUL 70/30 Scale-A corpus (review #7): all crisis rows +
sampled selfplay rows from the existing iter5.pt (rows are ordered
[crisis, selfplay] by build_expert_v2_tensor's --games-dir order; boundary
verified: 1,453,708 + 4,721,555 = 6,175,263).

    python -m alphatrain.scripts.build_iter5_balanced
"""
import numpy as np
import torch

N_CRISIS = 1_453_708
N_SELF = 4_721_555
TARGET_SELF = 623_018  # -> 2,076,726 total @ 70/30


def main():
    d = torch.load('alphatrain/data/iter5.pt', map_location='cpu',
                   weights_only=False)
    n = d['boards'].shape[0]
    assert n == N_CRISIS + N_SELF, f'boundary mismatch: {n}'
    rng = np.random.default_rng(0)
    sp = np.sort(rng.choice(np.arange(N_CRISIS, n), TARGET_SELF, replace=False))
    idx = torch.from_numpy(np.concatenate([np.arange(N_CRISIS), sp]))
    out = {}
    for k, v in d.items():
        out[k] = v[idx] if isinstance(v, torch.Tensor) and v.shape[:1] == (n,) \
            else v
    out['balance_info'] = {'crisis': N_CRISIS, 'selfplay': TARGET_SELF,
                           'source': 'iter5.pt', 'seed': 0}
    torch.save(out, 'alphatrain/data/iter5_bal.pt')
    print(f'iter5_bal.pt: {N_CRISIS:,} crisis + {TARGET_SELF:,} selfplay '
          f'= {N_CRISIS + TARGET_SELF:,} states '
          f'({100 * N_CRISIS / (N_CRISIS + TARGET_SELF):.1f}% crisis)')


if __name__ == '__main__':
    main()
