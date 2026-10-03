"""Run (or ingest) a greedy policy eval and append one row to alphatrain/EVAL.md.

  run:  python -m alphatrain.scripts.eval_log run --model alphatrain/data/X.pt --desc "..."   (gate: seeds 2,600,000-2,601,999)
  csv:  python -m alphatrain.scripts.eval_log csv --csv alphatrain/inference_cpp/data/X.csv --model X --desc "..."
Protocol (2026-09-26): every eval is capped at --max-turns 100000 and judged by its percentiles and the share
of games that reach the cap; the mean is censored by the cap and is not the headline number.
Percentiles use eval.cc's rule (sorted[int(p/100*n)]). Every row: date, model, description, seed range,
n, turn cap, mean, P1..P95, max, <500%, <1000%, >10k%, capped%, csv path."""
import argparse, csv, os, subprocess, sys, datetime, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_MD = os.path.join(ROOT, 'alphatrain', 'EVAL.md')
CPP = os.path.join(ROOT, 'alphatrain', 'inference_cpp')
HEADER = ('| date | model | description | seeds | n | cap | mean | P1 | P5 | P10 | P25 | P50 | P75 | P90 | P95 | max | <500 | <1000 | >10k | capped | deaths/100k turns [95% CI] | MTBF turns | csv |\n'
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n')


def death_rate(capped, turns):
    """Deaths per 100k turns played (every game's turns count as exposure, capped or not), its 95% Poisson
    interval and MTBF. The per-turn hazard is flat over game age (HISTORY 256), so this one number
    determines the whole survival curve and is comparable across turn caps."""
    from alphatrain.scripts.survival_stats import poisson_ci
    deaths, exposure = int(np.sum(~np.asarray(capped))), float(np.sum(turns))
    lo, hi = poisson_ci(deaths)
    return (1e5 * deaths / exposure, 1e5 * lo / exposure, 1e5 * hi / exposure,
            exposure / deaths if deaths else float('inf'))


def stats(scores, capped, turns):
    s = np.sort(np.asarray(scores, dtype=np.int64)); n = len(s)
    pct = lambda p: int(s[min(int(p / 100.0 * n), n - 1)])
    rate, lo, hi, mtbf = death_rate(capped, turns)
    return dict(n=n, mean=float(s.mean()), P1=pct(1), P5=pct(5), P10=pct(10), P25=pct(25), P50=pct(50), P75=pct(75),
                P90=pct(90), P95=pct(95), max=int(s[-1]), lt500=100 * (s < 500).mean(), lt1000=100 * (s < 1000).mean(), gt10k=100 * (s > 10000).mean(),
                capped=100 * np.mean(capped), rate=rate, rate_lo=lo, rate_hi=hi, mtbf=mtbf)


def append_row(model, desc, seeds, cap, st, csv_rel):
    if not os.path.exists(EVAL_MD):
        open(EVAL_MD, 'w').write('# EVAL.md — greedy policy evaluations (policy-only, `inference_cpp/build/eval`, MPS fp16)\n\n'
                                 'One row per eval. Never compare per-seed across models; compare distributions. '
                                 'cap = --max-turns (1000000 = uncapped). Percentiles use eval.cc nearest-rank.\n\n' + HEADER)
    row = (f"| {datetime.date.today()} | {model} | {desc} | {seeds} | {st['n']} | {cap} | {st['mean']:.0f} | {st['P1']} | {st['P5']} | "
           f"{st['P10']} | {st['P25']} | {st['P50']} | {st['P75']} | {st['P90']} | {st['P95']} | {st['max']} | {st['lt500']:.1f}% | "
           f"{st['lt1000']:.1f}% | {st['gt10k']:.0f}% | {st['capped']:.1f}% | "
           f"{st['rate']:.2f} [{st['rate_lo']:.2f}, {st['rate_hi']:.2f}] | {st['mtbf']:,.0f} | {csv_rel} |\n")
    open(EVAL_MD, 'a').write(row); print(row.strip())


def load_csv(path):
    """Scores, per-game capped flags (0 when the CSV predates the column), turns and the seed range."""
    rows = list(csv.DictReader(open(path)))
    key = 'score' if 'score' in rows[0] else [k for k in rows[0] if k != 'seed'][0]
    seeds = [int(r['seed']) for r in rows]
    capped = [r.get('capped', '0') == '1' for r in rows]
    turns = [int(r['turns']) for r in rows]
    return [float(r[key]) for r in rows], capped, turns, (min(seeds), max(seeds) + 1)


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--model', required=True); r.add_argument('--desc', default='')
    r.add_argument('--seed-start', type=int, default=2600000); r.add_argument('--seed-end', type=int, default=2602000)
    r.add_argument('--max-turns', type=int, default=100000); r.add_argument('--batch', type=int, default=0); r.add_argument('--no-fold', action='store_true'); r.add_argument('--fp32', action='store_true'); r.add_argument('--canon', action='store_true'); r.add_argument('--tta', type=int, default=1); r.add_argument('--color-tta', type=int, default=1)
    c = sub.add_parser('csv'); c.add_argument('--csv', required=True); c.add_argument('--model', required=True); c.add_argument('--desc', default='')
    c.add_argument('--cap', default='100000')
    a = p.parse_args()
    if a.cmd == 'csv':
        scores, capped, turns, (s0, s1) = load_csv(a.csv)
        append_row(a.model, a.desc, f'{s0}-{s1}', a.cap, stats(scores, capped, turns), os.path.relpath(a.csv, ROOT)); return
    name = os.path.splitext(os.path.basename(a.model))[0]
    # Exports fold BatchNorm into the convolutions by default (HISTORY 257: same policy, ~1.3x faster
    # forward, different fp16 rounding); --no-fold reproduces the older unfolded exports.
    fold = not a.no_fold
    ts = os.path.join(CPP, 'data', name + ('_fold_ts.pt' if fold else '_ts.pt'))
    if not os.path.exists(ts):
        subprocess.run([sys.executable, '-m', 'alphatrain.inference_cpp.export_ts', '--model', a.model, '--output', ts]
                       + (['--fold-bn'] if fold else []), cwd=ROOT, check=True)
    # Throughput is flat in batch size (GPU-bound), so keep every game in flight: no refill tail.
    batch = a.batch or min(a.seed_end - a.seed_start, 4000)
    tag = ('uncapped' if a.max_turns >= 1000000 else f'cap{a.max_turns}') + ('_fp32' if a.fp32 else '') + ('_canon' if a.canon else '') + (f'_tta{a.tta}' if a.tta > 1 else '') + (f'_ctta{a.color_tta}' if a.color_tta > 1 else '') + ('_fold' if fold else '')
    out = os.path.join(CPP, 'data', f'{name}_{tag}_{a.seed_start}_{a.seed_end - a.seed_start}.csv')
    cmd = ['caffeinate', '-is', './build/eval', '--model', os.path.relpath(ts, CPP), '--device', 'mps', '--batch', str(batch),
           '--seed-start', str(a.seed_start), '--seed-end', str(a.seed_end), '--max-turns', str(a.max_turns), '--scores-out', os.path.relpath(out, CPP)] + (['--fp32'] if a.fp32 else []) + (['--canon'] if a.canon else []) + (['--tta', str(a.tta)] if a.tta > 1 else []) + (['--color-tta', str(a.color_tta)] if a.color_tta > 1 else [])
    res = subprocess.run(cmd, cwd=CPP, capture_output=True, text=True)
    print('\n'.join(l for l in res.stdout.splitlines() if l.startswith(('done', 'scores', '  P1', '  <500', '  mean turns'))))
    if res.returncode != 0: print(res.stderr[-2000:]); sys.exit(res.returncode)
    scores, capped, turns, _ = load_csv(out)
    append_row(name + (' [fp32]' if a.fp32 else '') + (' [CANON]' if a.canon else '') + (f' [TTA-{a.tta}]' if a.tta > 1 else '') + (f' [cTTA-{a.color_tta}]' if a.color_tta > 1 else '') + (' [fold]' if fold else ''), a.desc, f'{a.seed_start}-{a.seed_end}', str(a.max_turns), stats(scores, capped, turns), os.path.relpath(out, ROOT))


if __name__ == '__main__':
    main()
