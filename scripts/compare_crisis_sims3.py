"""Compare 200-sim vs 400-sim vs 800-sim crisis runs on the same 50 seeds.

Per-label (recovery/prevention) paired analysis. Extends compare_crisis_sims.py
to three sim levels.
"""
from __future__ import annotations
import glob, json, os

import numpy as np

RUNS = {
    200: 'data/crisis_v14_s200',
    400: 'data/crisis_v14_s400',
    800: 'data/crisis_v14_s800',
}


def load_run(directory, label):
    out = {}
    pattern = os.path.join(directory, f'game_seed*_{label}_*.json')
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            g = json.load(f)
        seed = g['original_seed']
        out[seed] = {
            'turns_survived': g['turns'] - g['replay_from_turn'],
            'capped': bool(g.get('capped', False)),
            'score': int(g.get('score', 0)),
        }
    return out


def percentiles(name, values):
    a = np.asarray(values, dtype=np.float64)
    return (f"{name}: n={len(a):>3d}  mean={a.mean():>5.0f}  "
            f"median={np.median(a):>4.0f}  "
            f"P10={np.percentile(a, 10):>4.0f}  "
            f"P25={np.percentile(a, 25):>4.0f}  "
            f"P75={np.percentile(a, 75):>4.0f}")


def main():
    for label in ['recovery', 'prevention']:
        runs = {sims: load_run(d, label) for sims, d in RUNS.items()}
        # Common seeds across all three
        common = sorted(set(runs[200]) & set(runs[400]) & set(runs[800]))

        print(f"\n{'=' * 78}")
        print(f"  {label.upper()} — n={len(common)} seeds in all three runs")
        print(f"{'=' * 78}")

        print(f"\n  Continue-turns survived (post-rewind):")
        for sims in [200, 400, 800]:
            vals = [runs[sims][s]['turns_survived'] for s in common]
            print(f"    {percentiles(f'sims={sims}', vals)}")

        print(f"\n  Cap@500 rate:")
        for sims in [200, 400, 800]:
            n_cap = sum(runs[sims][s]['capped'] for s in common)
            print(f"    sims={sims}: {n_cap}/{len(common)} "
                  f"({100*n_cap/len(common):.1f}%)")

        print(f"\n  Score during replay:")
        for sims in [200, 400, 800]:
            vals = [runs[sims][s]['score'] for s in common]
            print(f"    {percentiles(f'sims={sims}', vals)}")

        print(f"\n  Paired Δ continue-turns (vs sims=400 baseline):")
        for sims in [200, 800]:
            diffs = [runs[sims][s]['turns_survived']
                      - runs[400][s]['turns_survived'] for s in common]
            d = np.asarray(diffs, dtype=np.float64)
            wins = int((d > 0).sum())
            ties = int((d == 0).sum())
            losses = int((d < 0).sum())
            print(f"    sims={sims} − 400: mean={d.mean():+5.1f}  "
                  f"median={np.median(d):+4.0f}  "
                  f"SE={d.std(ddof=1)/np.sqrt(len(d)):>4.1f}  "
                  f"wins/ties/losses = {wins}/{ties}/{losses}")


if __name__ == '__main__':
    main()
