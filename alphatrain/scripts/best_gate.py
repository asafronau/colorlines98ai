"""Pick the model with the lowest death rate among EVAL.md gate rows (for scripts/flywheel_loop.sh).

    python -m alphatrain.scripts.best_gate ta_S1_e4_a1.0 ta_S1_e4_a2.0 ta_S1_e4_a3.0
prints `<model> <rate> <ci_lo> <ci_hi>` for the best one (deaths per 100k turns) and one line per candidate on
stderr. Only the latest folded 2k gate row of each model (seeds 2,600,000-2,601,999) counts; a missing row is fatal.
"""
import argparse
import re
import sys

EVAL_MD = 'alphatrain/EVAL.md'


def gate_rate(rows, model, seeds, src):
    hits = [r for r in rows if r[1] == f'{model} [fold]' and r[3] == seeds]
    if not hits:
        sys.exit(f'FATAL: no folded gate row for {model} on seeds {seeds} in {src}')
    m = re.match(r'([\d.]+) \[([\d.]+), ([\d.]+)\]', hits[-1][20])
    if not m:
        sys.exit(f'FATAL: no death rate in the gate row of {model}: {hits[-1][20]!r}')
    return tuple(float(x) for x in m.groups())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='+')
    ap.add_argument('--seeds', default='2600000-2602000')
    ap.add_argument('--eval-md', default=EVAL_MD)
    a = ap.parse_args()
    rows = [[c.strip() for c in line.strip().strip('|').split('|')]
            for line in open(a.eval_md) if line.startswith('| 20')]
    rates = {m: gate_rate(rows, m, a.seeds, a.eval_md) for m in a.models}
    for m, (r, lo, hi) in rates.items():
        print(f'  {m}: {r} [{lo}, {hi}] deaths per 100k turns', file=sys.stderr)
    best = min(rates, key=lambda m: rates[m][0])
    print(best, *rates[best])


if __name__ == '__main__':
    main()
