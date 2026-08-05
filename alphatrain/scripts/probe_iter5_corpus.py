"""Corpus sanity probe for iter5.pt (review-#7 prep): compare target geometry
against v14_rev3.pt (the corpus this exact recipe WON with, on the 256ch line).

Per corpus (10k sample): top-share distribution, target argmax agreement with
the training base's fp16 legal argmax, dw3 weight mass on disagreements, and
what T=0.7 sharpening does to top-1 target mass.

    python -m alphatrain.scripts.probe_iter5_corpus
"""
import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model

PAIRS = [
    ('iter5(vh2)', 'alphatrain/data/iter5.pt', 'alphatrain/data/small128_vh2.pt'),
    ('v14_rev3(pillar3f)', 'alphatrain/data/v14_rev3.pt',
     'alphatrain/data/pillar3f.pt'),
]


def main():
    dev = torch.device('mps')
    rng = np.random.default_rng(0)
    print(f'{"corpus":20s} {"ts_mean":>8s} {"ts<0.1":>7s} {"agree%":>7s} '
          f'{"dw3_dis%":>9s} {"sharp_top1":>10s} {"nnz_med":>8s}')
    for name, path, base in PAIRS:
        try:
            d = torch.load(path, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            print(f'{name:20s} MISSING {path}')
            continue
        n = d['boards'].shape[0]
        pick = np.sort(rng.choice(n, 10000, replace=False))
        pv = d['pol_values'][pick].float()
        s = pv.sum(1)
        ok = (s > 0).numpy()
        p = pv / s.clamp(min=1e-8).unsqueeze(1)
        ts = p.max(1).values.numpy()
        nnz = (pv > 0).sum(1).numpy()
        sharp = p.pow(1 / 0.7)
        sharp = sharp / sharp.sum(1, keepdim=True).clamp(min=1e-8)
        st1 = sharp.max(1).values.numpy()
        try:
            net, _ = load_model(base, dev, fp16=True)
        except FileNotFoundError:
            print(f'{name:20s} ts={ts[ok].mean():.3f} (base model missing)')
            continue
        tmp = path + '.probe.tmp'
        torch.save({'boards': d['boards'][pick], 'next_pos': d['next_pos'][pick],
                    'next_col': d['next_col'][pick], 'n_next': d['n_next'][pick],
                    'pol_indices': torch.zeros((10000, 5), dtype=torch.int64),
                    'pol_values': torch.zeros((10000, 5), dtype=torch.float32),
                    'max_score': 0.0}, tmp)
        ds = TensorDatasetGPU(tmp, augment=False, color_augment=False,
                              augment_factor=1, device='mps')
        tgt = d['pol_indices'][pick][torch.arange(10000),
                                     d['pol_values'][pick].argmax(1)].numpy()
        varg = np.full(10000, -1, dtype=np.int64)
        for st in range(0, 10000, 2048):
            e = min(st + 2048, 10000)
            obs = ds._build_obs_core(ds.boards[st:e], next_pos=ds.next_pos[st:e],
                                     next_col=ds.next_col[st:e],
                                     n_next=ds.n_next[st:e])
            with torch.no_grad():
                lg = net(obs.to(torch.float16)).float().cpu().numpy()
            bd = ds.boards[st:e].cpu().numpy().astype(np.int8)
            for i in range(e - st):
                k, fi, _ = _legal_priors_jit(bd[i], lg[i], 1)
                if k:
                    varg[st + i] = int(fi[0])
        import os
        os.remove(tmp)
        del net
        m = ok & (varg >= 0)
        agree = (tgt[m] == varg[m])
        w = ts[m] ** 3
        dis_mass = 100 * w[~agree].sum() / w.sum()
        print(f'{name:20s} {ts[m].mean():8.3f} '
              f'{100 * (ts[m] < 0.1).mean():6.1f}% {100 * agree.mean():6.1f}% '
              f'{dis_mass:8.1f}% {st1[m].mean():10.3f} '
              f'{np.median(nnz[m]):8.0f}')


if __name__ == '__main__':
    main()
