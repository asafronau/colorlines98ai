"""Compare crisis difficulty between V12 and V13 controlling for replay depth.

V13 used --policy-max-turns 12000 vs V12's 5000, so V13 crises can come from
much later game states (denser board → harder). This script filters both pools
to a common replay_from_turn range and compares continue_turns_survived to
see if the "V13 crisis is harder" effect is depth-driven.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _load(d, label, sample_n=None):
    files = sorted(glob.glob(os.path.join(d, f'game_seed*_{label}_*.json')))
    if sample_n and len(files) > sample_n:
        rng = np.random.default_rng(0)
        files = [files[i] for i in rng.choice(len(files), sample_n, replace=False)]
    rfrom, surv, capped = [], [], []
    for f in files:
        try:
            with open(f) as fp:
                g = json.load(fp)
            r = g.get('replay_from_turn')
            t = g.get('turns')
            if r is None or t is None:
                continue
            rfrom.append(r)
            surv.append(t - r)
            capped.append(g.get('capped', False))
        except (json.JSONDecodeError, OSError):
            continue
    return (np.asarray(rfrom), np.asarray(surv), np.asarray(capped))


def _report(name, rfrom, surv, capped, mask=None):
    if mask is None:
        mask = np.ones_like(rfrom, dtype=bool)
    r = rfrom[mask]
    s = surv[mask]
    c = capped[mask]
    if len(s) == 0:
        print(f"  {name}: n=0 (no files matching filter)")
        return
    print(f"  {name}: n={len(s)}")
    print(f"    replay_from_turn: mean={r.mean():.0f}  median={np.median(r):.0f}  "
          f"P10={np.percentile(r, 10):.0f}  P90={np.percentile(r, 90):.0f}  "
          f"max={r.max()}")
    print(f"    continue_turns survived: mean={s.mean():.0f}  "
          f"median={np.median(s):.0f}  "
          f"P5={np.percentile(s, 5):.0f}  P10={np.percentile(s, 10):.0f}  "
          f"P25={np.percentile(s, 25):.0f}  P75={np.percentile(s, 75):.0f}")
    print(f"    capped@500: {c.sum()}/{len(c)} ({100*c.mean():.1f}%)")
    print(f"    died early (<50 continue turns): "
          f"{(s < 50).sum()}/{len(s)} ({100*(s < 50).mean():.1f}%)")


def _compare(v12_dir, v13_dir, label, sample_n):
    print(f"\n========== {label.upper()} ==========")
    r12, s12, c12 = _load(v12_dir, label, sample_n)
    r13, s13, c13 = _load(v13_dir, label, sample_n)

    print("\n[UNFILTERED]")
    _report(f'V12 ({v12_dir})', r12, s12, c12)
    _report(f'V13 ({v13_dir})', r13, s13, c13)

    print("\n[FILTERED: replay_from_turn < 5000]")
    _report(f'V12 ({v12_dir})', r12, s12, c12, r12 < 5000)
    _report(f'V13 ({v13_dir})', r13, s13, c13, r13 < 5000)

    print("\n[V13 ONLY: replay_from_turn >= 5000 (the 'deep' subset)]")
    _report(f'V13 deep', r13, s13, c13, r13 >= 5000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--v12-dir', default='data/crisis_v12')
    p.add_argument('--v13-dir', default='data/crisis_v13')
    p.add_argument('--sample-n', type=int, default=2000,
                   help='Random sample size per pool (for speed). 0 = all.')
    args = p.parse_args()

    sample = args.sample_n if args.sample_n > 0 else None
    _compare(args.v12_dir, args.v13_dir, 'recovery', sample)
    _compare(args.v12_dir, args.v13_dir, 'prevention', sample)


if __name__ == '__main__':
    main()
