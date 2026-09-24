"""Run (or ingest) a greedy policy eval and append one row to alphatrain/EVAL.md.

  run:  python -m alphatrain.scripts.eval_log run --model alphatrain/data/X.pt --desc "..." --seed-start 2600000 --seed-end 2601000
  csv:  python -m alphatrain.scripts.eval_log csv --csv alphatrain/inference_cpp/data/X.csv --model X --desc "..."
Percentiles use eval.cc's rule (sorted[int(p/100*n)]). Every row: date, model, description, seed range,
n, turn cap, mean, P1..P95, max, <500%, <1000%, >10k%, csv path."""
import argparse, csv, os, subprocess, sys, datetime, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_MD = os.path.join(ROOT, 'alphatrain', 'EVAL.md')
CPP = os.path.join(ROOT, 'alphatrain', 'inference_cpp')
HEADER = ('| date | model | description | seeds | n | cap | mean | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | max | <500 | <1000 | >10k | csv |\n'
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')


def stats(scores):
    s = np.sort(np.asarray(scores, dtype=np.int64)); n = len(s)
    pct = lambda p: int(s[min(int(p / 100.0 * n), n - 1)])
    return dict(n=n, mean=float(s.mean()), P1=pct(1), P5=pct(5), P10=pct(10), P25=pct(25), P50=pct(50), P75=pct(75),
                P90=pct(90), P95=pct(95), max=int(s[-1]), lt500=100 * (s < 500).mean(), lt1000=100 * (s < 1000).mean(), gt10k=100 * (s > 10000).mean())


def append_row(model, desc, seeds, cap, st, csv_rel):
    if not os.path.exists(EVAL_MD):
        open(EVAL_MD, 'w').write('# EVAL.md — greedy policy evaluations (policy-only, `inference_cpp/build/eval`, MPS fp16)\n\n'
                                 'One row per eval. Never compare per-seed across models; compare distributions. '
                                 'cap = --max-turns (1000000 = uncapped). Percentiles use eval.cc nearest-rank.\n\n' + HEADER)
    row = (f"| {datetime.date.today()} | {model} | {desc} | {seeds} | {st['n']} | {cap} | {st['mean']:.0f} | {st['P1']} | {st['P5']} | "
           f"{st['P10']} | {st['P25']} | {st['P50']} | {st['P75']} | {st['P90']} | {st['P95']} | {st['max']} | {st['lt500']:.1f}% | "
           f"{st['lt1000']:.1f}% | {st['gt10k']:.0f}% | {csv_rel} |\n")
    open(EVAL_MD, 'a').write(row); print(row.strip())


def load_csv(path):
    rows = list(csv.DictReader(open(path)))
    key = 'score' if 'score' in rows[0] else [k for k in rows[0] if k != 'seed'][0]
    seeds = [int(r['seed']) for r in rows]
    return [float(r[key]) for r in rows], (min(seeds), max(seeds) + 1)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--model', required=True); r.add_argument('--desc', default='')
    r.add_argument('--seed-start', type=int, default=2600000); r.add_argument('--seed-end', type=int, default=2601000)
    r.add_argument('--max-turns', type=int, default=1000000); r.add_argument('--batch', type=int, default=500); r.add_argument('--fp32', action='store_true'); r.add_argument('--canon', action='store_true'); r.add_argument('--tta', type=int, default=1)
    c = sub.add_parser('csv'); c.add_argument('--csv', required=True); c.add_argument('--model', required=True); c.add_argument('--desc', default='')
    c.add_argument('--cap', default='1000000')
    a = p.parse_args()
    if a.cmd == 'csv':
        scores, (s0, s1) = load_csv(a.csv)
        append_row(a.model, a.desc, f'{s0}-{s1}', a.cap, stats(scores), os.path.relpath(a.csv, ROOT)); return
    name = os.path.splitext(os.path.basename(a.model))[0]
    ts = os.path.join(CPP, 'data', name + '_ts.pt')
    if not os.path.exists(ts):
        subprocess.run([sys.executable, '-m', 'alphatrain.inference_cpp.export_ts', '--model', a.model, '--output', ts], cwd=ROOT, check=True)
    tag = ('uncapped' if a.max_turns >= 1000000 else f'cap{a.max_turns}') + ('_fp32' if a.fp32 else '') + ('_canon' if a.canon else '') + (f'_tta{a.tta}' if a.tta > 1 else '')
    out = os.path.join(CPP, 'data', f'{name}_{tag}_{a.seed_start}_{a.seed_end - a.seed_start}.csv')
    cmd = ['caffeinate', '-is', './build/eval', '--model', os.path.relpath(ts, CPP), '--device', 'mps', '--batch', str(a.batch),
           '--seed-start', str(a.seed_start), '--seed-end', str(a.seed_end), '--max-turns', str(a.max_turns), '--scores-out', os.path.relpath(out, CPP)] + (['--fp32'] if a.fp32 else []) + (['--canon'] if a.canon else []) + (['--tta', str(a.tta)] if a.tta > 1 else [])
    res = subprocess.run(cmd, cwd=CPP, capture_output=True, text=True)
    print('\n'.join(l for l in res.stdout.splitlines() if l.startswith(('done', 'scores', '  P1', '  <500', '  mean turns'))))
    if res.returncode != 0: print(res.stderr[-2000:]); sys.exit(res.returncode)
    scores, _ = load_csv(out)
    append_row(name + (' [fp32]' if a.fp32 else '') + (' [CANON]' if a.canon else '') + (f' [TTA-{a.tta}]' if a.tta > 1 else ''), a.desc, f'{a.seed_start}-{a.seed_end}', str(a.max_turns), stats(scores), os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
