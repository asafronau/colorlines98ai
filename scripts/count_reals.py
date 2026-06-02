"""Aggregate REAL/curse/neutral forks across all logs/mine_*.json harvested so
far (live, before the harvest's own end-of-run consolidation). Prints the REAL
yield, the winner's-curse discard rate, and the CI widths of confirmed forks so
we can reason about whether bigger R would change anything."""
import os, sys, json, glob
import numpy as np

files = sorted(glob.glob('logs/mine_*.json'))
n_seed = n_band = n_flag = n_real = n_curse = 0
seeds_with_real = 0
real_rows, curse_rows = [], []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    n_seed += 1
    conf = d.get('confirms', {})
    band = d['meta'].get('band', [])
    n_band += len(band)
    seed = d['meta']['seed']
    had_real = False
    for r in d['rows']:
        if not r['flag']:
            continue
        n_flag += 1
        c = conf.get(str(r['depth']))
        if c and c['real']:
            n_real += 1; had_real = True
            real_rows.append((seed, r['depth'], c['gap'], c['lo'], c['hi'],
                              c['pol_cat'], c['best_cat']))
        else:
            n_curse += 1
            if c:
                curse_rows.append((seed, r['depth'], c['gap'], c['lo'], c['hi']))
    if had_real:
        seeds_with_real += 1

print(f"seeds mined:        {n_seed}")
print(f"band depths total:  {n_band}  ({n_band/max(n_seed,1):.1f}/seed)")
print(f"flagged in screen:  {n_flag}")
print(f"  -> REAL:          {n_real}  ({100*n_real/max(n_flag,1):.0f}% of flags)")
print(f"  -> curse:         {n_curse}  ({100*n_curse/max(n_flag,1):.0f}% of flags)")
print(f"seeds w/ >=1 REAL:  {seeds_with_real}/{n_seed} "
      f"({100*seeds_with_real/max(n_seed,1):.0f}%)")
print(f"REAL forks/seed:    {n_real/max(n_seed,1):.2f}")

if real_rows:
    gaps = np.array([r[2] for r in real_rows])
    widths = np.array([r[4]-r[3] for r in real_rows])
    los = np.array([r[3] for r in real_rows])
    print(f"\nREAL fork gaps (pp): min {gaps.min():.0f}  med {np.median(gaps):.0f}  "
          f"max {gaps.max():.0f}  mean {gaps.mean():.0f}")
    print(f"REAL CI widths (pp): min {widths.min():.0f}  med {np.median(widths):.0f}  "
          f"max {widths.max():.0f}")
    print(f"REAL CI low edge:    min {los.min():.1f}  med {np.median(los):.1f}  "
          f"(how far CI clears 0; small = marginal, would flip with noise)")
    marginal = sum(1 for r in real_rows if r[3] < 5)
    print(f"marginal REALs (CI low < 5pp): {marginal}/{len(real_rows)} "
          f"-- these are the ones tighter CIs would protect/kill")
    print("\ntop REAL forks by gap:")
    for s, dp, g, lo, hi, pc, bc in sorted(real_rows, key=lambda x: -x[2])[:12]:
        print(f"  seed {s} d{dp:>2}: pol {pc:4.0f}% -> best {bc:4.0f}%  "
              f"Δ{g:4.0f}pp  CI[{lo:4.0f},{hi:4.0f}]")

if curse_rows:
    cg = np.array([r[2] for r in curse_rows])
    print(f"\ncurse gaps on fresh seeds (pp): med {np.median(cg):.0f}  "
          f"(near 0 => screen noise, as expected)")
