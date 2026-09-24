"""Analyse native CRN rollout-judge results by immutable edit stratum."""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np


def mean_ci(values, rng, n_boot, chunk=1000):
    values = np.asarray(values, dtype=np.float64)
    draws = []
    for start in range(0, n_boot, chunk):
        count = min(chunk, n_boot - start)
        sample = values[rng.integers(
            0, len(values), size=(count, len(values)))]
        draws.append(sample.mean(1))
    return np.percentile(np.concatenate(draws), [2.5, 97.5])


def summarize(mask, death_uplift, turn_delta, top_share, rng, n_boot):
    d = death_uplift[mask]
    t = turn_delta[mask]
    q = top_share[mask]
    dlo, dhi = mean_ci(d, rng, n_boot)
    tlo, thi = mean_ci(t, rng, n_boot)
    return {
        'n': int(len(d)),
        'death_uplift': float(d.mean()),
        'death_uplift_ci95': [float(dlo), float(dhi)],
        'teacher_turn_delta': float(t.mean()),
        'teacher_turn_delta_ci95': [float(tlo), float(thi)],
        'genuine_fraction': float((d > 0.08).mean()),
        'phantom_fraction': float((d < -0.08).mean()),
        'tie_fraction': float((np.abs(d) <= 0.08).mean()),
        'positive_fraction': float((d > 0).mean()),
        'mean_top_share': float(q.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    parser.add_argument('--rows', required=True)
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20260818)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError('--bootstrap must be positive')

    with open(args.results, newline='') as handle:
        result_rows = list(csv.DictReader(handle))
    provenance = np.load(args.rows, allow_pickle=False)
    if len(result_rows) != len(provenance['rows']):
        raise ValueError('judge result and provenance row counts differ')
    state_ids = np.asarray([int(row['state']) for row in result_rows])
    if not np.array_equal(state_ids, np.arange(len(result_rows))):
        raise ValueError('judge results are not in sidecar state order')

    teacher_died = np.asarray(
        [float(row['teacher_died']) for row in result_rows])
    base_died = np.asarray([float(row['base_died']) for row in result_rows])
    teacher_turns = np.asarray(
        [float(row['teacher_turns']) for row in result_rows])
    base_turns = np.asarray([float(row['base_turns']) for row in result_rows])
    death_uplift = base_died - teacher_died
    turn_delta = teacher_turns - base_turns
    strata = provenance['stratum'].astype(str)
    top_share = provenance['top_share'].astype(np.float64)
    rng = np.random.default_rng(args.seed)

    report = {
        'schema_version': 1,
        'results': args.results,
        'rows': args.rows,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'positive_death_uplift_means_teacher_is_better': True,
        'overall': summarize(
            np.ones(len(strata), dtype=bool), death_uplift, turn_delta,
            top_share, rng, args.bootstrap),
        'strata': {},
        'top_share_bins': {},
    }
    # Preserve the exporter order rather than alphabetizing the scientific
    # strata into an accidental new presentation order.
    for name in dict.fromkeys(strata.tolist()):
        report['strata'][name] = summarize(
            strata == name, death_uplift, turn_delta, top_share, rng,
            args.bootstrap)

    # Search confidence is an obvious cheap candidate for filtering a much
    # larger causal mine.  Report fixed bins so a promising threshold is not
    # selected by repeatedly slicing the same rollout sample.
    share_bins = (
        ('lt_0.30', 0.0, 0.30),
        ('0.30_0.40', 0.30, 0.40),
        ('0.40_0.50', 0.40, 0.50),
        ('ge_0.50', 0.50, np.inf),
    )
    for name, lo, hi in share_bins:
        mask = (top_share >= lo) & (top_share < hi)
        if mask.any():
            report['top_share_bins'][name] = summarize(
                mask, death_uplift, turn_delta, top_share, rng,
                args.bootstrap)

    print(json.dumps(report, indent=2, sort_keys=False))
    if args.output:
        parent = os.path.dirname(args.output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = args.output + '.tmp'
        with open(tmp, 'w') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
        os.replace(tmp, args.output)


if __name__ == '__main__':
    main()
