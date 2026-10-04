"""Pool several gate CSVs (seed banks) of one model: deaths, exposure, rate with Poisson CI, capped share.
Usage: python -m alphatrain.scripts.pooled_rate NAME=csv1,csv2 [NAME=csv1,csv2 ...]   (e.g. the two 2k gate banks of one model)"""
import csv
import sys

sys.path.insert(0, '.')
from alphatrain.scripts.survival_stats import poisson_ci

for arg in sys.argv[1:]:
    name, paths = arg.split('=', 1)
    rows = [r for p in paths.split(',') for r in csv.DictReader(open(p))]
    d = sum(1 for r in rows if int(r['capped']) == 0)
    e = sum(int(r['turns']) for r in rows)
    lo, hi = poisson_ci(d)
    capped = sum(int(r['capped']) for r in rows) / len(rows)
    print(f'{name:14s} {len(rows)} games, {d} deaths: {1e5 * d / e:.3f} [{1e5 * lo / e:.3f}, {1e5 * hi / e:.3f}] '
          f'per 100k turns, MTBF {e / d:,.0f}, capped {100 * capped:.1f}%')
