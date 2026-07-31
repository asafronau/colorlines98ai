"""Paired-seed bootstrap comparison of eval runs (review #5 protocol).

Same-seed 5k runs -> per-seed pairing kills cross-seed variance. For each
candidate vs base: bootstrap (10k resamples over seeds) of the differences in
mean, median, P5, P10, and <1000 rate.

    python -m alphatrain.scripts.paired_bootstrap \
        --base alphatrain/inference_cpp/data/pair_vh1.csv \
        --candidates alphatrain/inference_cpp/data/pair_m02.csv \
                     alphatrain/inference_cpp/data/pair_m04.csv
"""
import argparse
import csv

import numpy as np


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {int(r['seed']): int(r['score']) for r in rows}


def stats(x):
    return np.array([x.mean(), np.median(x), np.percentile(x, 5),
                     np.percentile(x, 10), (x < 1000).mean() * 100])


NAMES = ['mean', 'P50', 'P5', 'P10', '<1000%']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--candidates', nargs='+', required=True)
    p.add_argument('--n-boot', type=int, default=10000)
    a = p.parse_args()

    base = load(a.base)
    seeds = np.array(sorted(base.keys()))
    b = np.array([base[s] for s in seeds], dtype=np.float64)
    rng = np.random.default_rng(0)
    n = len(seeds)
    print(f'{n:,} paired seeds; {a.n_boot:,} bootstrap resamples')

    for cpath in a.candidates:
        cand = load(cpath)
        assert set(cand.keys()) == set(base.keys()), f'seed mismatch: {cpath}'
        c = np.array([cand[s] for s in seeds], dtype=np.float64)
        point = stats(c) - stats(b)
        diffs = np.empty((a.n_boot, len(NAMES)))
        for i in range(a.n_boot):
            idx = rng.integers(0, n, n)
            diffs[i] = stats(c[idx]) - stats(b[idx])
        lo = np.percentile(diffs, 2.5, axis=0)
        hi = np.percentile(diffs, 97.5, axis=0)
        name = cpath.split('/')[-1].replace('pair_', '').replace('.csv', '')
        print(f'\n=== {name} vs base ===')
        for j, m in enumerate(NAMES):
            sig = ('WIN ' if lo[j] > 0 else 'LOSS' if hi[j] < 0 else '    ') \
                if m != '<1000%' else \
                ('WIN ' if hi[j] < 0 else 'LOSS' if lo[j] > 0 else '    ')
            print(f'  {m:7s} {point[j]:+9.1f}  [{lo[j]:+9.1f}, {hi[j]:+9.1f}]  {sig}')


if __name__ == '__main__':
    main()
