"""Verify review #5's quantitative claims before adopting its plan.

  1. Corpus peakedness: top-share (max/sum of pol_values) distributions.
  2. dw=3 gradient allocation: share of top_share^3 weight on rows where the
     corpus target argmax != vh1's legal argmax (10k samples, fp32).
  3. Rehearsal capture on vh2c_mix: top_share^3 mass on main vs rehearsal
     partitions (main-first ordering, n_main = 85,928).

    python -m alphatrain.scripts.diag_review5_verify
"""
import os

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model

CORPORA = [
    ('gate3', 'alphatrain/data/gate3_crisis.pt'),
    ('shallow', 'alphatrain/data/vh2try_crisis.pt'),
    ('deep', 'alphatrain/data/vh2c_crisis.pt'),
    ('v14_rev3(256ch)', 'alphatrain/data/v14_rev3.pt'),
]


def top_share(d):
    pv = d['pol_values'].float()
    s = pv.sum(1)
    ok = s > 0
    return (pv.max(1).values[ok] / s[ok]).numpy(), ok


def main():
    dev = torch.device('mps')
    net, _ = load_model('alphatrain/data/small128_vh1.pt', dev, fp16=False)
    rng = np.random.default_rng(0)

    print(f'{"corpus":18s} {"N":>10s} {"ts_mean":>8s} {"ts_med":>7s} '
          f'{">0.40":>6s} | {"dis%":>6s} {"dw3_mass_on_dis%":>16s}')
    for name, path in CORPORA:
        if not os.path.exists(path):
            print(f'{name:18s} MISSING {path}')
            continue
        d = torch.load(path, map_location='cpu', weights_only=False)
        ts, ok = top_share(d)
        line = (f'{name:18s} {len(ts):10,} {ts.mean():8.3f} '
                f'{np.median(ts):7.3f} {100 * (ts > 0.40).mean():5.1f}%')
        if name != 'v14_rev3(256ch)':  # vh1-argmax comparison is 128ch-only
            n = d['boards'].shape[0]
            pick = np.sort(rng.choice(np.where(ok.numpy())[0],
                                      min(10000, int(ok.sum())), replace=False))
            tmp = path + '.v.tmp'
            torch.save({'boards': d['boards'][pick],
                        'next_pos': d['next_pos'][pick],
                        'next_col': d['next_col'][pick],
                        'n_next': d['n_next'][pick],
                        'pol_indices': d['pol_indices'][pick],
                        'pol_values': d['pol_values'][pick],
                        'max_score': 0.0}, tmp)
            ds = TensorDatasetGPU(tmp, augment=False, color_augment=False,
                                  augment_factor=1, device='mps')
            tgt = d['pol_indices'][pick][
                torch.arange(len(pick)), d['pol_values'][pick].argmax(1)].numpy()
            varg = np.full(len(pick), -1, dtype=np.int64)
            for s in range(0, len(pick), 2048):
                e = min(s + 2048, len(pick))
                obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                         next_col=ds.next_col[s:e],
                                         n_next=ds.n_next[s:e])
                with torch.no_grad():
                    lg = net(obs).float().cpu().numpy()
                bd = ds.boards[s:e].cpu().numpy().astype(np.int8)
                for i in range(e - s):
                    k, fi, _ = _legal_priors_jit(bd[i], lg[i], 1)
                    if k:
                        varg[s + i] = int(fi[0])
            os.remove(tmp)
            tss, _ = top_share({'pol_values': d['pol_values'][pick]})
            dis = tgt != varg
            w = tss ** 3
            line += (f' | {100 * dis.mean():5.1f}% '
                     f'{100 * w[dis].sum() / w.sum():15.1f}%')
        print(line, flush=True)

    mix = torch.load('alphatrain/data/vh2c_mix.pt', map_location='cpu',
                     weights_only=False)
    n_main = mix['mix_info']['n_main']
    ts_all, ok = top_share(mix)
    idx = np.where(ok.numpy())[0]
    w = ts_all ** 3
    main_mask = idx < n_main
    print(f'\nvh2c_mix dw=3 weight capture: main(deep) rows '
          f'{100 * w[main_mask].sum() / w.sum():.1f}%  vs rehearsal '
          f'{100 * w[~main_mask].sum() / w.sum():.1f}%  '
          f'(rows: {n_main:,} vs {mix["boards"].shape[0] - n_main:,})')


if __name__ == '__main__':
    main()
