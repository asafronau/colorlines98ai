"""Quantify how diluted the high-contrast forks are in the harvested labels.

For each mined anchor state we measure two within-state contrast signals:
  spread = max(catastrophe%) - min(catastrophe%) over its candidate moves
  gap    = policy-move catastrophe% - best-move catastrophe%  (what the policy
           reshaping actually cares about: how much safer the best alt is)
Then we bin STATES and LABELS by spread, so we can see what fraction of the
training mass is actually informative vs neutral. SE of a single R=100 rate is
~5pp, so a two-rate gap has SE ~7pp: spreads under ~10pp are mostly noise.
"""
import os, sys, glob, json
import numpy as np

files = sorted(glob.glob('logs/mine_*.json'))
spreads, gaps, ncands, labels_per_bin = [], [], [], []
bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 101)]
state_bin = {b: 0 for b in bins}
label_bin = {b: 0 for b in bins}
total_labels = 0

for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d['rows']:
        rates = [c[1] for c in r['cand_rates']]
        if not rates:
            continue
        spread = max(rates) - min(rates)
        spreads.append(spread)
        gaps.append(r['pol_cat'] - r['best_cat'])
        k = len(rates)
        ncands.append(k)
        total_labels += k
        for b in bins:
            if b[0] <= spread < b[1]:
                state_bin[b] += 1
                label_bin[b] += k
                break

spreads = np.array(spreads); gaps = np.array(gaps)
n_states = len(spreads)
print(f"anchor states: {n_states}   per-move labels: {total_labels}   "
      f"avg cands/state: {np.mean(ncands):.1f}\n")
print(f"within-state SPREAD (max-min catastrophe%):  "
      f"med {np.median(spreads):.0f}  mean {spreads.mean():.0f}  p90 "
      f"{np.percentile(spreads,90):.0f}  max {spreads.max():.0f}")
print(f"policy-vs-best GAP:                          "
      f"med {np.median(gaps):.0f}  mean {gaps.mean():.0f}  p90 "
      f"{np.percentile(gaps,90):.0f}  max {gaps.max():.0f}\n")

print(f"{'spread bin':>12} {'states':>8} {'state%':>7} {'labels':>8} {'label%':>7}")
for b in bins:
    name = f"{b[0]}-{b[1]}pp" if b[1] < 101 else f"{b[0]}+pp"
    print(f"{name:>12} "
          f"{state_bin[b]:>8} {100*state_bin[b]/n_states:>6.1f}% "
          f"{label_bin[b]:>8} {100*label_bin[b]/total_labels:>6.1f}%")

hi = sum(state_bin[b] for b in bins if b[0] >= 20)
hi_l = sum(label_bin[b] for b in bins if b[0] >= 20)
noise = sum(state_bin[b] for b in bins if b[1] <= 10)
print(f"\nHIGH-contrast (spread>=20pp): {hi} states "
      f"({100*hi/n_states:.0f}%), {hi_l} labels ({100*hi_l/total_labels:.0f}%)")
print(f"NOISE-floor   (spread<10pp):  {noise} states "
      f"({100*noise/n_states:.0f}%) -- mostly sampling noise, low true signal")
print(f"\n=> if loss is averaged uniformly over labels, the {100*hi_l/total_labels:.0f}% "
      f"high-contrast mass competes with {100*(1-hi_l/total_labels):.0f}% low-signal mass")
