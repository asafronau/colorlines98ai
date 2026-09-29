"""Survival statistics for capped greedy evals (survival analysis with right-censoring at the turn cap).

Every game contributes exposure (turns played) whether it died or reached the cap, so the death rate
uses every death, not just the few games that sit at P1/P5:
  death rate  = deaths / total turns played, reported per 100k turns with a 95% Poisson interval
  MTBF        = mean turns between deaths = 1 / death rate (infinite play <=> MTBF -> infinity)
  hazard      = the same rate within game-age buckets (is dying front-loaded?)
  S(t)        = share of games still alive at turn t (S(cap) = the capped share)

    python -m alphatrain.scripts.survival_stats [--cap 100000] NAME=alphatrain/inference_cpp/data/X.csv [NAME=...]
(--cap re-censors uncapped runs at that turn so they compare with capped ones.)
Input CSVs are written by the C++ engine (`eval --scores-out`): seed,score,turns,capped.
"""
import csv
import math
import sys

import numpy as np

BUCKETS = [(0, 5000), (5000, 20000), (20000, 50000), (50000, 100000)]
SURVIVAL_AT = [10000, 50000, 100000]


def poisson_ci(k):
    """95% interval for a Poisson count (Wilson-Hilferty approximation of the exact chi-square bounds)."""
    def wh(q, z):
        return q * (1 - 1 / (9 * q) + z / (3 * math.sqrt(q))) ** 3
    lo = 0.0 if k == 0 else wh(k, -1.959964)
    hi = wh(k + 1, 1.959964)
    return lo, hi


def main():
    head = (f'{"model":26s} {"n":>5s} {"deaths":>6s} {"deaths/100k turns [95% CI]":>28s} {"MTBF turns":>11s}  '
            + '  '.join(f'S({t // 1000}k)' for t in SURVIVAL_AT) + '   hazard per 100k turns by game age: '
            + ' '.join(f'{a // 1000}-{b // 1000}k' for a, b in BUCKETS))
    print(head)
    args, cap = sys.argv[1:], None
    if args[:1] == ['--cap']:
        cap, args = int(args[1]), args[2:]
    for arg in args:
        name, path = arg.split('=', 1)
        rows = list(csv.DictReader(open(path)))
        t = np.array([int(r['turns']) for r in rows])
        died = np.array([r['capped'] != '1' for r in rows])
        if cap is not None:  # game still alive at the cap -> censored there
            died &= t < cap
            t = np.minimum(t, cap)
        deaths, exposure = int(died.sum()), float(t.sum())
        rate = 1e5 * deaths / exposure
        lo, hi = (1e5 * x / exposure for x in poisson_ci(deaths))
        mtbf = exposure / deaths if deaths else float('inf')
        surv = [np.mean(t >= s) if s < t.max() else np.mean(~died | (t >= s)) for s in SURVIVAL_AT]
        haz = []
        for a, b in BUCKETS:
            exp_b = np.clip(t, a, b).sum() - a * len(t)          # turns played inside [a, b)
            d_b = int((died & (t >= a) & (t < b)).sum())
            haz.append(f'{1e5 * d_b / exp_b:7.1f}' if exp_b > 0 else '     - ')
        print(f'{name:26s} {len(t):5d} {deaths:6d} {rate:9.1f} [{lo:6.1f}, {hi:6.1f}]      {mtbf:11,.0f}  '
              + '  '.join(f'{s:6.1%}' for s in surv) + '   ' + ' '.join(haz))


if __name__ == '__main__':
    main()
