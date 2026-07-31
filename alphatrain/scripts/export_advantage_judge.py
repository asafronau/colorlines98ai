"""Export ALL full-legal disagreement rows of a corpus for per-row advantage
judging (review #5 fallback: advantage-filtered policy improvement).

Writes the CLRJ judge bin with teacher_move = corpus target argmax and
base_move = vh1's fp16 legal argmax, plus a row-index sidecar npz.

    python -m alphatrain.scripts.export_advantage_judge \
        --tensor alphatrain/data/vh2c_crisis.pt \
        --out alphatrain/inference_cpp/data/adv_judge_states.bin
"""
import argparse
import struct

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default='alphatrain/data/vh2c_crisis.pt')
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--out',
                   default='alphatrain/inference_cpp/data/adv_judge_states.bin')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)

    d = torch.load(a.tensor, map_location='cpu', weights_only=False)
    n = d['boards'].shape[0]
    assert 'disagree_mask' in d, 'run add_fulllegal_mask first'
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

    rows = np.where((varg >= 0) & (tgt != varg))[0]
    boards = d['boards'].numpy().reshape(n, 81)
    np_pos = d['next_pos'].numpy()
    ncol = d['next_col'].numpy()
    nn = d['n_next'].numpy()
    ts = (d['pol_values'].max(1).values
          / d['pol_values'].sum(1).clamp(min=1e-6)).numpy()
    with open(a.out, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(rows)))
        for i in rows:
            f.write(boards[i].astype(np.int8).tobytes())
            f.write(struct.pack('<i', int(nn[i])))
            for t in range(3):
                f.write(struct.pack('<iii', int(np_pos[i, t, 0]),
                                    int(np_pos[i, t, 1]), int(ncol[i, t])))
            f.write(struct.pack('<iif', int(tgt[i]), int(varg[i]),
                                float(ts[i])))
    np.savez(a.out.replace('.bin', '_rows.npz'),
             rows=rows, tgt=tgt[rows], vh1=varg[rows], top_share=ts[rows])
    print(f'{a.out}: {len(rows):,} disagreement rows '
          f'({100 * len(rows) / n:.1f}%) + rows sidecar')


if __name__ == '__main__':
    main()
