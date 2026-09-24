"""Summarise rollout-judge results per relabel configuration, by stratum.
Reports genuine/phantom counts (|gap|>0.08), net genuine per 1k roots, and the
mean death-rate uplift with a bootstrap CI.  Excess = genuine - phantom is the
noise-corrected count (under a true tie both tails fire equally)."""
import argparse, csv, json
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', required=True)
    p.add_argument('--prefix', default='alphatrain/inference_cpp/data/relabel_pilot_')
    p.add_argument('--states', default='alphatrain/inference_cpp/data/relabel_pilot_states.bin')
    a = p.parse_args()
    meta = json.load(open(a.states + '.meta.json'))
    n_roots = len(meta)
    strat = np.array([m['stratum'] for m in meta])
    rng = np.random.default_rng(0)
    print(f'{"config":20s} {"strat":6s} {"roots":>5s} {"flips":>5s} {"genu":>4s} {"phan":>4s} '
          f'{"excess/1k":>9s} {"uplift":>7s} {"ci95":>16s}')
    for c in a.configs:
        idx = np.array(json.load(open(f'{a.prefix}{c}_flips.idx.json')))
        rows = list(csv.DictReader(open(f'{a.prefix}{c}_judge_r64h200.csv')))
        td = np.array([float(r['teacher_died']) for r in rows])
        bd = np.array([float(r['base_died']) for r in rows])
        gap = td - bd
        s = strat[idx]
        for name, m, nr in (('all', np.ones(len(idx), bool), n_roots),
                            ('tail', s == 'tail', (strat == 'tail').sum()),
                            ('broad', s == 'broad', (strat == 'broad').sum())):
            g = int((gap[m] < -0.08).sum()); ph = int((gap[m] > 0.08).sum())
            u = gap[m]
            boots = [u[rng.integers(0, len(u), len(u))].mean() for _ in range(2000)] if len(u) else [0]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f'{c:20s} {name:6s} {nr:5d} {m.sum():5d} {g:4d} {ph:4d} '
                  f'{1000*(g-ph)/nr:9.2f} {u.mean() if len(u) else 0:+7.3f} [{lo:+.3f},{hi:+.3f}]')


if __name__ == '__main__':
    main()
