"""Independent-rejudge analysis (review #6): shrinkage + selection noise.

Compares each row's ORIGINAL selection estimate (stored as corpus weight)
against the FRESH independent estimate (new seeds, R=256).

    python -m alphatrain.scripts.rejudge_analysis \
        --corpus alphatrain/data/advfilt2.pt \
        --results alphatrain/inference_cpp/data/rejudge_r2_results.csv
"""
import argparse
import csv

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--results', required=True)
    a = p.parse_args()
    c = torch.load(a.corpus, map_location='cpu', weights_only=False)
    orig = c['weight'].numpy()
    with open(a.results) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(orig)
    fresh = np.array([float(r['base_died']) - float(r['teacher_died'])
                      for r in rows])
    n = len(orig)
    corr = np.corrcoef(orig, fresh)[0, 1]
    print(f'{a.corpus}: {n} rows')
    print(f'  original mean uplift : {orig.mean():+.4f} (selected at >=0.08)')
    print(f'  FRESH mean uplift    : {fresh.mean():+.4f}  '
          f'(shrinkage x{fresh.mean() / orig.mean():.2f})')
    print(f'  corr(orig, fresh)    : {corr:+.3f}')
    print(f'  fresh >= 0.08        : {100 * (fresh >= 0.08).mean():.1f}%')
    print(f'  fresh > 0            : {100 * (fresh > 0).mean():.1f}%')
    print(f'  fresh <= -0.08       : {100 * (fresh <= -0.08).mean():.1f}%')
    se = np.sqrt(2 * fresh.mean() * (1 - fresh.mean()) / 256) if 0 < fresh.mean() < 1 else 0
    print(f'  (R=256 per-row SE ~ {se:.3f} at the fresh mean death rate)')


if __name__ == '__main__':
    main()
