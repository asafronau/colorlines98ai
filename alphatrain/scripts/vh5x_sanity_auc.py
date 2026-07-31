"""Sanity gate: per-horizon AUC of the 5x-data head vs the falsified fresh
head, on the SAME disjoint-seed validation rows (Bernoulli labels, masked).

    python -m alphatrain.scripts.vh5x_sanity_auc
"""
import argparse

import numpy as np
import torch

import alphatrain.value_head as vh
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--heads', nargs='+',
                   default=['alphatrain/data/value_head_small128_vh1.pt',
                            'alphatrain/data/value_head_vh5x.pt'])
    p.add_argument('--data', default='alphatrain/data/value_targets_vh5x.pt')
    p.add_argument('--n', type=int, default=100000)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(0)

    d = torch.load(a.data, map_location='cpu', weights_only=False)
    val = np.where(~d['is_train'].numpy())[0]
    pick = np.sort(rng.choice(val, min(a.n, len(val)), replace=False))
    labels = d['survive_labels'][pick].numpy()
    masks = d['survive_masks'][pick].numpy()

    tmp = a.data + '.auc.tmp'
    n = len(pick)
    torch.save({'boards': d['boards'][pick], 'next_pos': d['next_pos'][pick],
                'next_col': d['next_col'][pick], 'n_next': d['n_next'][pick],
                'pol_indices': torch.zeros((n, 5), dtype=torch.int64),
                'pol_values': torch.zeros((n, 5), dtype=torch.float32),
                'max_score': 0.0}, tmp)
    ds = TensorDatasetGPU(tmp, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)
    net, _ = load_model(a.backbone, dev, fp16=True)

    print(f'{n:,} disjoint-seed val rows; horizons {list(d["horizons"])}')
    for hp in a.heads:
        head, _, _ = vh.load_any(hp, dev)
        head.train(False)
        head.half()
        preds = np.zeros((n, labels.shape[1]), dtype=np.float32)
        for s in range(0, n, 4096):
            e = min(s + 4096, n)
            obs = ds._build_obs_core(
                ds.boards[s:e], next_pos=ds.next_pos[s:e],
                next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
            with torch.no_grad():
                _, feats = net.forward_with_features(obs.to(torch.float16))
                preds[s:e] = torch.sigmoid(head(feats)).float().cpu().numpy()
        aucs = []
        for hi in range(labels.shape[1]):
            m = masks[:, hi] == 1
            aucs.append(auc(preds[m, hi], labels[m, hi]))
        print(f'{hp.split("/")[-1]:34s} AUC per-H: '
              + '  '.join(f'{x:.4f}' for x in aucs))
    import os
    os.remove(tmp)


if __name__ == '__main__':
    main()
