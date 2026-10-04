"""Hazard by game age with exact Poisson CIs (2 decimals): deaths and exposure (turns played) per age band.
Usage: python -m alphatrain.scripts.hazard_bands NAME=csv [NAME=csv ...]   (csv: seed,score,turns,capped from eval --scores-out)"""
import csv
import sys

sys.path.insert(0, '.')
from alphatrain.scripts.survival_stats import poisson_ci

BANDS = [(0, 5000), (5000, 20000), (20000, 50000), (50000, 100000)]
print(f'{"model":12s} ' + '  '.join(f'{f"{lo // 1000}-{hi // 1000}k":>24s}' for lo, hi in BANDS) + '   all ages')
for arg in sys.argv[1:]:
    name, path = arg.split('=', 1)
    rows = list(csv.DictReader(open(path)))
    cells, D, E = [], 0, 0
    for lo, hi in BANDS:
        deaths = sum(1 for r in rows if int(r['capped']) == 0 and lo <= int(r['turns']) < hi)
        expo = sum(max(0, min(int(r['turns']), hi) - lo) for r in rows)
        lo_ci, hi_ci = poisson_ci(deaths)
        cells.append(f'{deaths:4d}: {1e5 * deaths / expo:.2f} [{1e5 * lo_ci / expo:.2f},{1e5 * hi_ci / expo:.2f}]')
        D += deaths; E += expo
    lo_ci, hi_ci = poisson_ci(D)
    print(f'{name:12s} ' + '  '.join(f'{c:>24s}' for c in cells) + f'   {D}: {1e5 * D / E:.3f} [{1e5 * lo_ci / E:.3f}, {1e5 * hi_ci / E:.3f}]')
