"""Add a FULL-LEGAL fp16 disagree_mask to a corpus (review #5 protocol).

mask[i] = 1 where the corpus target argmax != the base policy's fp16 legal
argmax over ALL legal moves (deployment protocol — add_disagree_mask.py is
support-restricted and fp-agnostic; this replaces it for iteration-4).

    python -m alphatrain.scripts.add_fulllegal_mask \
        --tensor alphatrain/data/vh2c_crisis.pt \
        --base alphatrain/data/small128_vh1.pt
"""
import argparse

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)

    d = torch.load(a.tensor, map_location='cpu', weights_only=False)
    n = d['boards'].shape[0]
    tgt = d['pol_indices'][torch.arange(n), d['pol_values'].argmax(1)].numpy()
    ds = TensorDatasetGPU(a.tensor, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)
    net, _ = load_model(a.base, dev, fp16=True)
    varg = np.full(n, -1, dtype=np.int64)
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                 next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
        with torch.no_grad():
            lg = net(obs.to(torch.float16)).float().cpu().numpy()
        bd = ds.boards[s:e].cpu().numpy().astype(np.int8)
        for i in range(e - s):
            k, fi, _ = _legal_priors_jit(bd[i], lg[i], 1)
            if k:
                varg[s + i] = int(fi[0])
        if (s // 2048) % 10 == 0:
            print(f'  {e:,}/{n:,}', flush=True)
    mask = torch.from_numpy((tgt != varg).astype(np.int8))
    d['disagree_mask'] = mask
    d['disagree_mask_protocol'] = 'full-legal fp16 vs ' + a.base
    torch.save(d, a.tensor)
    print(f'{a.tensor}: disagree_mask set on {int(mask.sum()):,}/{n:,} rows '
          f'({100 * mask.float().mean():.1f}%)')


if __name__ == '__main__':
    main()
