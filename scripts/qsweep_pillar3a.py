"""Sweep q_weight for pillar3a + value_head_pillar3a at MCTS@100.

Saves each q's eval log to logs/qsweep_pillar3a_q{Q}.log and prints a
comparison table at the end.

Usage:
    python scripts/qsweep_pillar3a.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

Q_VALUES = [1.0, 1.5, 2.0, 2.5]
SEEDS = list(range(100))
MODEL = 'alphatrain/data/sharp_25_epoch_12.pt'
VALUE_HEAD = 'alphatrain/data/value_head_sharp25_ep12.pt'
SIMS = 100
MAX_TURNS = 20000
WORKERS = 16


def parse_log(text):
    """Extract MCTS (not Pol) mean, P50, %<1000, %>10000 from eval_parallel output.

    eval_parallel prints both Pol and MCTS blocks when --value-head-path is set.
    Layout:
        MEAN |  pol_mean |  mcts_mean |  +XX%
        Pol percentiles (100 games):
          P1=... P5=... P50=... ...
          <500: ... <1000: ... >5000: ... >10000: ...
        MCTS percentiles (100 games):
          P1=... P5=... P50=... ...
          <500: ... <1000: ... >5000: ... >10000: ...
    We want the MCTS values.
    """
    # MEAN row: 2nd \d+ is MCTS mean (1st is Pol)
    m_mean = re.search(r'^\s*MEAN\s*\|\s*(\d+)\s*\|\s*(\d+)', text, re.MULTILINE)
    mcts_mean = int(m_mean.group(2)) if m_mean else None

    # Find MCTS block start, then parse percentile + floor lines within it
    mcts_idx = text.find('MCTS percentiles')
    mcts_section = text[mcts_idx:] if mcts_idx >= 0 else ''
    m_p50 = re.search(r'P50=(\d+)', mcts_section)
    m_floor = re.search(r'<1000:\s*\d+\s*\(([\d.]+)%\)', mcts_section)
    m_above_10k = re.search(r'>10000:\s*\d+\s*\((\d+)%\)', mcts_section)
    return {
        'mean': mcts_mean,
        'p50': int(m_p50.group(1)) if m_p50 else None,
        'floor': float(m_floor.group(1)) if m_floor else None,
        'above_10k': int(m_above_10k.group(1)) if m_above_10k else None,
    }


def run_one(q):
    log_path = f'logs/qsweep_pillar3a_q{q}.log'
    cmd = [
        'python', '-m', 'alphatrain.scripts.eval_parallel',
        '--model', MODEL,
        '--value-head-path', VALUE_HEAD,
        '--simulations', str(SIMS),
        '--top-k', '30',
        '--batch-size', '8',
        '--device', 'mps',
        '--workers', str(WORKERS),
        '--max-turns', str(MAX_TURNS),
        '--q-weight', str(q),
        '--early-stop',
        '--seeds', *[str(s) for s in SEEDS],
    ]
    print(f"\n{'='*70}", flush=True)
    print(f"[q={q}] starting at {time.strftime('%H:%M:%S')}", flush=True)
    print(f"  log: {log_path}", flush=True)
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    with open(log_path, 'w') as f:
        # Stream stdout to log AND to parent
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  bufsize=1)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            # Forward only summary-ish lines to parent stdout
            if any(k in line for k in ('MEAN', 'percentile', '<1000',
                                       '>10000', 'P50=', 'seed=', 'eta=',
                                       'Eval ')):
                sys.stdout.write(f"  {line}")
                sys.stdout.flush()
        proc.wait()
    elapsed = time.time() - t0
    with open(log_path) as f:
        stats = parse_log(f.read())
    stats['q'] = q
    stats['elapsed_min'] = elapsed / 60
    print(f"[q={q}] done in {elapsed/60:.1f}min  "
          f"mean={stats['mean']}  P50={stats['p50']}  "
          f"floor={stats['floor']}%  >10K={stats['above_10k']}%",
          flush=True)
    return stats


def main():
    os.makedirs('logs', exist_ok=True)
    all_results = []
    for q in Q_VALUES:
        try:
            all_results.append(run_one(q))
        except KeyboardInterrupt:
            print("\nInterrupted — printing partial results.", flush=True)
            break

    print("\n" + "=" * 70, flush=True)
    print("Q-WEIGHT SWEEP SUMMARY (pillar3a + value_head_pillar3a, "
          f"MCTS@{SIMS}, {len(SEEDS)} seeds, cap={MAX_TURNS})", flush=True)
    print("=" * 70, flush=True)
    print(f"{'q':>5} | {'mean':>6} | {'P50':>6} | {'<1000':>6} | "
          f"{'>10K':>5} | {'min':>4}", flush=True)
    print('-' * 50, flush=True)
    for r in all_results:
        print(f"{r['q']:>5.1f} | {r['mean'] or '?':>6} | "
              f"{r['p50'] or '?':>6} | "
              f"{(str(r['floor']) + '%') if r['floor'] is not None else '?':>6} | "
              f"{(str(r['above_10k']) + '%') if r['above_10k'] is not None else '?':>5} | "
              f"{r['elapsed_min']:>4.1f}", flush=True)
    # Save tabular result
    summary_path = 'logs/qsweep_pillar3a_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"q,mean,p50,floor_pct,above_10k_pct,elapsed_min\n")
        for r in all_results:
            f.write(f"{r['q']},{r['mean']},{r['p50']},"
                    f"{r['floor']},{r['above_10k']},"
                    f"{r['elapsed_min']:.2f}\n")
    print(f"\nSaved CSV: {summary_path}", flush=True)


if __name__ == '__main__':
    main()
