"""Danger-score a corpus with the FRESH vh3 survival head: per row,
danger = 1 - P(survive >= 100 turns), stored as a CONTINUOUS float in
disagree_mask so --disagree-gamma g trains with w = 1 + g*danger.
(Amplify critical MOMENTS wherever they occur — owner doctrine.)

    python -m alphatrain.scripts.danger_score_corpus \
        --tensor alphatrain/data/r2_bulk.pt
"""
import argparse

import numpy as np
import torch

import alphatrain.value_head as vh
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tensor', required=True)
    ap.add_argument('--backbone', default='alphatrain/data/small128_vh3.pt')
    ap.add_argument('--head', default='alphatrain/data/value_head_vh3.pt')
    ap.add_argument('--horizon-idx', type=int, default=2,
                    help='0..3 = horizons 25/50/100/200; default H100')
    a = ap.parse_args()
    dev = torch.device('mps')
    net, _ = load_model(a.backbone, dev, fp16=True)
    head, _, _ = vh.load_any(a.head, dev)
    head.train(False)
    head.half()

    ds = TensorDatasetGPU(a.tensor, augment=False, color_augment=False,
                          augment_factor=1, device='mps')
    n = ds.boards.shape[0]
    danger = np.zeros(n, dtype=np.float32)
    for s in range(0, n, 4096):
        e = min(s + 4096, n)
        obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                 next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
        with torch.no_grad():
            _, feats = net.forward_with_features(obs.to(torch.float16))
            p = torch.sigmoid(head(feats)).float().cpu().numpy()
        danger[s:e] = 1.0 - p[:, a.horizon_idx]
        if (s // 4096) % 200 == 0:
            print(f'  {e:,}/{n:,}', flush=True)

    d = torch.load(a.tensor, map_location='cpu', weights_only=False)
    d['disagree_mask'] = torch.from_numpy(danger)
    d['disagree_mask_protocol'] = f'danger=1-P(survive H100), {a.head}'
    torch.save(d, a.tensor)
    t = torch.load(a.tensor, weights_only=True)
    q = np.percentile(danger, [50, 90, 99])
    print(f'{a.tensor}: danger stored (median {q[0]:.3f}, P90 {q[1]:.3f}, '
          f'P99 {q[2]:.3f}); weights_only OK ({t["boards"].shape[0]:,})')


if __name__ == '__main__':
    main()
