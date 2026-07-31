"""Per-cell bootstrap analysis for the rowjudge/thirdjudge runs (R2 diag).

    python -m alphatrain.scripts.rowjudge_analysis \
        --meta alphatrain/inference_cpp/data/rowjudge_meta.csv \
        --results alphatrain/inference_cpp/data/rowjudge_results.csv
"""
import argparse
import csv

import numpy as np

from alphatrain.scripts.dagger_judge_analysis import boot_ci


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--meta', required=True)
    p.add_argument('--results', required=True)
    a = p.parse_args()
    with open(a.meta) as f:
        meta = list(csv.DictReader(f))
    with open(a.results) as f:
        res = list(csv.DictReader(f))
    assert len(meta) == len(res)
    seeds = np.array([int(r['original_seed']) for r in meta])
    cells = np.array([r['label'] for r in meta])
    up = np.array([float(r['base_died']) - float(r['teacher_died'])
                   for r in res])
    tup = np.array([float(r['teacher_turns']) - float(r['base_turns'])
                    for r in res])
    print(f'{"cell":14s} {"n":>4s} {"uplift":>8s} {"95% CI":>18s} {"turns":>7s}')
    for cell in sorted(set(cells)) + ['ALL']:
        m = np.ones(len(up), bool) if cell == 'ALL' else cells == cell
        lo, hi = boot_ci(up[m], seeds[m])
        print(f'{cell:14s} {m.sum():4d} {100 * up[m].mean():+7.2f}pp '
              f'[{100 * lo:+.2f}, {100 * hi:+.2f}] {tup[m].mean():+6.1f}')


if __name__ == '__main__':
    main()
