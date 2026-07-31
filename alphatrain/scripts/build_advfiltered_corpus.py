"""Build the advantage-filtered corpus (review #5 fallback): only rows whose
per-row judged uplift clears the genuine bar, weighted by measured advantage.

    python -m alphatrain.scripts.build_advfiltered_corpus \
        --output alphatrain/data/advfilt.pt
"""
import argparse
import csv

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default='alphatrain/data/vh2c_crisis.pt')
    p.add_argument('--rows',
                   default='alphatrain/inference_cpp/data/adv_judge_states_rows.npz')
    p.add_argument('--results',
                   default='alphatrain/inference_cpp/data/adv_judge_results.csv')
    p.add_argument('--min-uplift', type=float, default=0.08)
    p.add_argument('--output', default='alphatrain/data/advfilt.pt')
    a = p.parse_args()

    side = np.load(a.rows)
    with open(a.results) as f:
        res = list(csv.DictReader(f))
    assert len(res) == len(side['rows'])
    up = np.array([float(r['base_died']) - float(r['teacher_died'])
                   for r in res])
    keep = up >= a.min_uplift
    rows = side['rows'][keep]
    d = torch.load(a.tensor, map_location='cpu', weights_only=False)
    idx = torch.from_numpy(rows)
    out = {
        'boards': d['boards'][idx],
        'next_pos': d['next_pos'][idx],
        'next_col': d['next_col'][idx],
        'n_next': d['n_next'][idx],
        'tgt_idx': d['pol_indices'][idx],
        'tgt_prob': d['pol_values'][idx],
        'vh1_move': torch.from_numpy(side['vh1'][keep].astype(np.int64)),
        'weight': torch.from_numpy(up[keep].astype(np.float32)),
        'seed': torch.from_numpy(rows.astype(np.int64)),  # pseudo-seed = row
        '_stats': {'n_seeds': len(rows),
                   'min_margin': f'judged uplift >= {a.min_uplift}'},
    }
    ok = out['tgt_idx'][
        torch.arange(len(rows)), out['tgt_prob'].argmax(1)] != out['vh1_move']
    assert ok.all()
    torch.save(out, a.output)
    print(f'{a.output}: {len(rows):,} judged-positive rows '
          f'(mean uplift {up[keep].mean():.3f}, of {len(up):,} judged)')


if __name__ == '__main__':
    main()
