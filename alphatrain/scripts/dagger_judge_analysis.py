"""Analyze redesigned Phase 0b judge results with cluster bootstrap CIs.

Primary endpoint (per ChatGPT review): population-weighted mean uplift in
died-within-H (base_died - teacher_died, positive = teacher move better) under
STUDENT continuation, with a source-seed cluster bootstrap 95% CI. Strata by
label (prevention/recovery) and teacher logit gap (<0.5, 0.5-1.0, >=1.0) are
descriptive; the estimand is NOT redefined around high-confidence states.

Decision bars (condition S):
  Strong GO : lower CI > +2pp
  Micro-GO  : lower CI > 0 and point estimate >= +1pp
  NO-GO     : upper CI < +1pp AND no advantage under T or burst conditions
  otherwise : gray zone — judgement call, not a forced verdict

    python -m alphatrain.scripts.dagger_judge_analysis \
        --meta alphatrain/inference_cpp/data/dagger_judge_meta.csv \
        --results-dir alphatrain/inference_cpp/data
"""
import argparse
import csv
import os

import numpy as np

CONDITIONS = ['S', 'T', 'L1', 'L2', 'L4', 'L8', 'L16']


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def boot_ci(uplift, seeds, n_boot=10000, seed=0):
    """Cluster bootstrap by source seed: resample seeds, keep all their states."""
    by_seed = {}
    for u, s in zip(uplift, seeds):
        by_seed.setdefault(s, []).append(u)
    groups = [np.array(v) for v in by_seed.values()]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    G = len(groups)
    for b in range(n_boot):
        pick = rng.integers(0, G, G)
        tot = np.concatenate([groups[i] for i in pick])
        means[b] = tot.mean()
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--meta',
                   default='alphatrain/inference_cpp/data/dagger_judge_meta.csv')
    p.add_argument('--results-dir', default='alphatrain/inference_cpp/data')
    p.add_argument('--n-boot', type=int, default=10000)
    a = p.parse_args()

    meta = read_csv(a.meta)
    m_seed = np.array([int(r['original_seed']) for r in meta])
    m_label = np.array([r['label'] for r in meta])
    m_gap = np.array([float(r['gap']) for r in meta])

    summary = {}
    for cond in CONDITIONS:
        path = os.path.join(a.results_dir, f'dagger_{cond}.csv')
        if not os.path.exists(path):
            continue
        rows = read_csv(path)
        if len(rows) != len(meta):
            print(f'{cond}: SKIP (rows {len(rows)} != meta {len(meta)})')
            continue
        td = np.array([float(r['teacher_died']) for r in rows])
        bd = np.array([float(r['base_died']) for r in rows])
        tt = np.array([float(r['teacher_turns']) for r in rows])
        bt = np.array([float(r['base_turns']) for r in rows])
        up = bd - td            # positive = teacher move survives more
        tup = tt - bt           # restricted mean turns uplift
        lo, hi = boot_ci(up, m_seed, a.n_boot)
        summary[cond] = (up.mean(), lo, hi)
        print(f'\n=== condition {cond} ===')
        print(f'  PRIMARY mean uplift: {100*up.mean():+.2f}pp  '
              f'[{100*lo:+.2f}, {100*hi:+.2f}] 95% CI  '
              f'(turns {tup.mean():+.1f})')
        for name, mask in [('prevention', m_label == 'prevention'),
                           ('recovery', m_label == 'recovery'),
                           ('gap<0.5', m_gap < 0.5),
                           ('0.5<=gap<1', (m_gap >= 0.5) & (m_gap < 1.0)),
                           ('gap>=1.0', m_gap >= 1.0)]:
            if mask.sum() == 0:
                continue
            slo, shi = boot_ci(up[mask], m_seed[mask], a.n_boot)
            print(f'  {name:11s} (n={mask.sum():4d}): {100*up[mask].mean():+.2f}pp '
                  f'[{100*slo:+.2f}, {100*shi:+.2f}]  '
                  f'(turns {tup[mask].mean():+.1f})')

    if 'S' not in summary:
        print('\ncondition S missing — no verdict yet')
        return
    mean, lo, hi = summary['S']
    others = [c for c in summary if c != 'S']
    any_other_adv = any(summary[c][1] > 0 for c in others)
    print('\n=== VERDICT (condition S primary) ===')
    print(f'S: {100*mean:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]')
    if lo > 0.02:
        print('STRONG GO: lower CI > +2pp under student continuation')
    elif lo > 0 and mean >= 0.01:
        print('MICRO-GO: lower CI > 0 and point estimate >= +1pp')
    elif hi < 0.01 and not any_other_adv:
        print('NO-GO: upper CI < +1pp and no advantage under any continuation')
    else:
        print('GRAY ZONE: judgement call — see per-condition table')


if __name__ == '__main__':
    main()
