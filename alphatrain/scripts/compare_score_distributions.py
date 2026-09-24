"""Compare gameplay score *distributions* with independent resampling.

This script independently bootstraps each run's empirical distribution.
Matching seed IDs alone do not establish matching simulator protocols. Under
a matched protocol paired resampling is valid even if trajectories diverge;
its precision benefit depends on the covariance of the two outcomes.

Inputs may be C++ ``seed,score`` CSVs or eval-policy JSON files.  Optional seed
bounds select the same development range without treating matching seed IDs as
pairs.  When the CSV also contains turns, staged long-play survival is reported
at fixed horizons; this is more stable than interpreting a few extreme scores.
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np


SURVIVAL_THRESHOLDS = (1000, 2000, 5000, 10000, 20000)


def load_run(path, seed_start=None, seed_end=None):
    rows = []
    if path.endswith('.json'):
        raw = json.load(open(path))
        for seed_s, value in raw.items():
            seed = int(seed_s)
            score = value[0] if isinstance(value, (list, tuple)) else value
            rows.append((seed, float(score), None, None))
    else:
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                turns = float(row['turns']) if row.get('turns') else None
                capped = float(row['capped']) if row.get('capped') else None
                rows.append((int(row['seed']), float(row['score']),
                             turns, capped))
    if seed_start is not None:
        rows = [r for r in rows if r[0] >= seed_start]
    if seed_end is not None:
        rows = [r for r in rows if r[0] < seed_end]
    if not rows:
        raise ValueError(f'no scores selected from {path}')
    seeds = [r[0] for r in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f'duplicate seeds in {path}')
    run = {'scores': np.asarray([r[1] for r in rows], dtype=np.float64)}
    if all(r[2] is not None for r in rows):
        run['turns'] = np.asarray([r[2] for r in rows], dtype=np.float64)
    if all(r[3] is not None for r in rows):
        run['capped'] = np.asarray([r[3] for r in rows], dtype=np.float64)
    return run


def load_scores(path, seed_start=None, seed_end=None):
    """Backward-compatible score-only loader used by older analyses."""
    return load_run(path, seed_start, seed_end)['scores']


def stats(x):
    return {
        'mean': x.mean(),
        'p5': np.percentile(x, 5),
        'p10': np.percentile(x, 10),
        'p25': np.percentile(x, 25),
        'median': np.median(x),
        'p75': np.percentile(x, 75),
        'p90': np.percentile(x, 90),
        'p95': np.percentile(x, 95),
        'lt1000': np.mean(x < 1000),
        'gt10000': np.mean(x > 10000),
    }


def bootstrap_differences(a, b, n_boot, seed, chunk=100):
    """Independent empirical bootstrap of B-A distribution metrics."""
    rng = np.random.default_rng(seed)
    keys = ('mean', 'p10', 'median', 'p90', 'lt1000', 'gt10000')
    out = {k: [] for k in keys}
    for start in range(0, n_boot, chunk):
        c = min(chunk, n_boot - start)
        aa = a[rng.integers(0, len(a), size=(c, len(a)))]
        bb = b[rng.integers(0, len(b), size=(c, len(b)))]
        values_a = {
            'mean': aa.mean(1), 'p10': np.percentile(aa, 10, axis=1),
            'median': np.median(aa, axis=1),
            'p90': np.percentile(aa, 90, axis=1),
            'lt1000': (aa < 1000).mean(1),
            'gt10000': (aa > 10000).mean(1),
        }
        values_b = {
            'mean': bb.mean(1), 'p10': np.percentile(bb, 10, axis=1),
            'median': np.median(bb, axis=1),
            'p90': np.percentile(bb, 90, axis=1),
            'lt1000': (bb < 1000).mean(1),
            'gt10000': (bb > 10000).mean(1),
        }
        for k in keys:
            out[k].append(values_b[k] - values_a[k])
    return {k: np.concatenate(v) for k, v in out.items()}


def bootstrap_mean_difference(a, b, n_boot, seed, chunk=100):
    """Independent bootstrap for a scalar sample mean (B - A)."""
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, chunk):
        c = min(chunk, n_boot - start)
        aa = a[rng.integers(0, len(a), size=(c, len(a)))]
        bb = b[rng.integers(0, len(b), size=(c, len(b)))]
        out.append(bb.mean(1) - aa.mean(1))
    return np.concatenate(out)


def observed_survival(run, threshold):
    """Binary survival only when every game's status at this horizon is known.

    A game capped before the requested horizon is censored, not a death.
    Dropping those games would also bias the estimate, so decline to report
    the simple binary metric when any outcomes remain unknown.
    """
    if 'capped' in run and np.any(
            (run['capped'] > 0) & (run['turns'] < threshold)):
        return None
    return (run['turns'] >= threshold).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base')
    p.add_argument('candidate')
    p.add_argument('--seed-start', type=int)
    p.add_argument('--seed-end', type=int)
    p.add_argument('--bootstrap', type=int, default=10_000)
    p.add_argument('--bootstrap-seed', type=int, default=20260808)
    args = p.parse_args()

    run_a = load_run(args.base, args.seed_start, args.seed_end)
    run_b = load_run(args.candidate, args.seed_start, args.seed_end)
    a, b = run_a['scores'], run_b['scores']
    sa, sb = stats(a), stats(b)
    print('INDEPENDENT DISTRIBUTION COMPARISON (candidate - base)')
    print('Matching numeric seed IDs, if present, are intentionally not paired.')
    print(f'base n={len(a):,}: {args.base}')
    print(f'candidate n={len(b):,}: {args.candidate}')
    for name, s in (('base', sa), ('candidate', sb)):
        print(f'  {name:9s} mean={s["mean"]:.0f} median={s["median"]:.0f} '
              f'P5={s["p5"]:.0f} P10={s["p10"]:.0f} P90={s["p90"]:.0f} '
              f'P95={s["p95"]:.0f} <1000={100*s["lt1000"]:.2f}% '
              f'>10000={100*s["gt10000"]:.2f}%')

    boot = bootstrap_differences(
        a, b, args.bootstrap, args.bootstrap_seed)
    print(f'\n{args.bootstrap:,} independent bootstrap resamples; 95% CIs:')
    for k in ('mean', 'median', 'p10', 'p90'):
        point = sb[k] - sa[k]
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        print(f'  {k:8s}: {point:+.0f}  [{lo:+.0f}, {hi:+.0f}]')
    for k, label in (('lt1000', '<1000'), ('gt10000', '>10000')):
        point = 100 * (sb[k] - sa[k])
        lo, hi = 100 * np.percentile(boot[k], [2.5, 97.5])
        print(f'  {label:8s}: {point:+.2f}pp  [{lo:+.2f}, {hi:+.2f}]')
    if 'turns' in run_a and 'turns' in run_b:
        point = run_b['turns'].mean() - run_a['turns'].mean()
        d = bootstrap_mean_difference(
            run_a['turns'], run_b['turns'], args.bootstrap,
            args.bootstrap_seed + 1)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f'  mean turns: {point:+.0f}  [{lo:+.0f}, {hi:+.0f}] '
              f'(base={run_a["turns"].mean():.0f}, '
              f'candidate={run_b["turns"].mean():.0f})')
        print('  turn survival (candidate - base):')
        for i, threshold in enumerate(SURVIVAL_THRESHOLDS):
            survived_a = observed_survival(run_a, threshold)
            survived_b = observed_survival(run_b, threshold)
            if survived_a is None or survived_b is None:
                print(f'    >= {threshold:5d}: unavailable (games censored '
                      'before this horizon)')
                continue
            point = 100 * (survived_b.mean() - survived_a.mean())
            d = 100 * bootstrap_mean_difference(
                survived_a, survived_b, args.bootstrap,
                args.bootstrap_seed + 10 + i)
            lo, hi = np.percentile(d, [2.5, 97.5])
            print(f'    >= {threshold:5d}: {point:+.2f}pp  '
                  f'[{lo:+.2f}, {hi:+.2f}] '
                  f'(base={100*survived_a.mean():.2f}%, '
                  f'candidate={100*survived_b.mean():.2f}%)')
    if 'capped' in run_a and 'capped' in run_b:
        point = 100 * (run_b['capped'].mean() - run_a['capped'].mean())
        d = 100 * bootstrap_mean_difference(
            run_a['capped'], run_b['capped'], args.bootstrap,
            args.bootstrap_seed + 2)
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f'  cap rate:   {point:+.2f}pp  [{lo:+.2f}, {hi:+.2f}] '
              f'(base={100*run_a["capped"].mean():.2f}%, '
              f'candidate={100*run_b["capped"].mean():.2f}%)')


if __name__ == '__main__':
    main()
