"""Phase C: locate the earliest turn where failure trajectories start
diverging from success trajectories on density / clear-rate / policy features.

The Phase B verdict ruled out a local-action effect at flagged danger points.
ChatGPT's correction (2026-05-23): the divergence likely lives UPSTREAM, far
before the late danger points we sampled. By turn 250 in a 477-turn-death
trajectory the position may already be irrecoverable. The right question is
when the trajectory FIRST started looking different from a successful one.

For each turn t in {10, 25, 50, 100, 150, 200, 250, 300}, compute per-state
rolling-50-turn-forward metrics across all trajectories still alive at t+50,
then compare failure (19 confirmed deaths) vs success (24 long-survivors) at
each t. Output: per-metric per-turn means with 95% CIs and a "diverges?" flag.

Input data: logs/instrumented_failures_full/*.json,
            logs/instrumented_success_full/*.json
Output:     logs/phase_c/divergence_table.csv
            logs/phase_c/summary.txt
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


TURN_GRID = [10, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300]
WINDOW = 50  # forward window for rolling metrics


def load_trajectories(directory):
    out = []
    for path in sorted(glob.glob(os.path.join(directory, 'traj_seed*.json'))):
        with open(path) as f:
            g = json.load(f)
        # Only keep games long enough for any rolling-window analysis
        if len(g.get('metrics', [])) >= 25:
            out.append(g)
    return out


def compute_window_metrics(traj, t, window=WINDOW):
    """At turn t, compute rolling-window-forward metrics. Returns dict
    or None if not enough horizon left."""
    m = traj['metrics']
    if t + window > len(m):
        return None
    seg = m[t:t + window]
    # Clear rate: fraction of segment turns with cleared > 0
    n_clears = sum(1 for x in seg if x.get('cleared', 0) > 0)
    clear_rate = n_clears / window
    # Empties slope: linear regression slope
    emp = np.array([x['empties'] for x in seg], dtype=np.float64)
    lec = np.array([x.get('lec', x['empties']) for x in seg], dtype=np.float64)
    xs = np.arange(window, dtype=np.float64)
    emp_slope = float(np.polyfit(xs, emp, 1)[0])
    lec_slope = float(np.polyfit(xs, lec, 1)[0])
    # Score rate
    s0 = seg[0]['score']
    s_end = seg[-1]['score']
    score_rate = (s_end - s0) / window
    # Policy uncertainty
    top1_p_mean = float(np.mean([x['top1_p'] for x in seg]))
    gap_mean = float(np.mean([x['top1_top2_gap'] for x in seg]))
    # Empties level at t (not slope) — what's the board state right now
    empties_now = float(emp[0])
    lec_now = float(lec[0])
    return {
        'clear_rate_50': clear_rate,
        'empties_slope': emp_slope,
        'lec_slope': lec_slope,
        'score_rate': score_rate,
        'top1_p_mean': top1_p_mean,
        'top1_top2_gap_mean': gap_mean,
        'empties_now': empties_now,
        'lec_now': lec_now,
    }


def aggregate(values):
    """Return (mean, ci_low, ci_high, n) at 95%. Skips empty lists."""
    a = np.asarray(values, dtype=np.float64)
    n = len(a)
    if n == 0:
        return None
    mean = float(a.mean())
    if n == 1:
        return mean, mean, mean, 1
    se = float(a.std(ddof=1) / np.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def ci_separated(stats_a, stats_b):
    """Two-sided 95% CI overlap test. Returns True if CIs do not overlap."""
    if stats_a is None or stats_b is None:
        return False
    _, lo_a, hi_a, _ = stats_a
    _, lo_b, hi_b, _ = stats_b
    return hi_a < lo_b or hi_b < lo_a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fail-dir',
                         default='logs/instrumented_failures_full')
    parser.add_argument('--succ-dir',
                         default='logs/instrumented_success_full')
    parser.add_argument('--out-dir', default='logs/phase_c')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading trajectories...", flush=True)
    fail_trajs = load_trajectories(args.fail_dir)
    succ_trajs = load_trajectories(args.succ_dir)
    # Restrict to confirmed failures (final_score < 1000)
    fail_trajs = [t for t in fail_trajs if t.get('final_score', 0) < 1000]
    print(f"  failure trajectories: {len(fail_trajs)} (filtered to <1000)")
    print(f"  success trajectories: {len(succ_trajs)}")

    # Failure final-turn distribution — how many alive at each turn t?
    fail_lens = sorted([t['final_turn'] for t in fail_trajs])
    print(f"  failure final-turn percentiles: P10={fail_lens[int(0.1*len(fail_lens))]} "
          f"P50={fail_lens[len(fail_lens)//2]} P90={fail_lens[int(0.9*len(fail_lens))]} "
          f"max={fail_lens[-1]}")

    metrics_keys = ['clear_rate_50', 'empties_slope', 'lec_slope',
                     'score_rate', 'top1_p_mean', 'top1_top2_gap_mean',
                     'empties_now', 'lec_now']

    rows = []
    for t in TURN_GRID:
        f_vals = defaultdict(list)
        s_vals = defaultdict(list)
        for traj in fail_trajs:
            r = compute_window_metrics(traj, t)
            if r is None:
                continue
            for k in metrics_keys:
                f_vals[k].append(r[k])
        for traj in succ_trajs:
            r = compute_window_metrics(traj, t)
            if r is None:
                continue
            for k in metrics_keys:
                s_vals[k].append(r[k])
        n_f = len(f_vals.get('clear_rate_50', []))
        n_s = len(s_vals.get('clear_rate_50', []))
        rows.append({
            'turn': t,
            'n_fail': n_f,
            'n_succ': n_s,
            **{k: (aggregate(f_vals[k]), aggregate(s_vals[k]))
                 for k in metrics_keys}
        })

    # Print readable table per metric
    print()
    for mk in metrics_keys:
        print(f"\n=== Metric: {mk} ===")
        print(f"{'turn':>5} | {'n_f/n_s':>7} | {'fail mean':>10s} {'fail 95% CI':>22s} | "
              f"{'succ mean':>10s} {'succ 95% CI':>22s} | {'diff':>7s} | sep?")
        for r in rows:
            f_stats, s_stats = r[mk]
            n_f, n_s = r['n_fail'], r['n_succ']
            if f_stats is None or s_stats is None:
                print(f"{r['turn']:>5d} | {n_f:>3d}/{n_s:<3d} | "
                      f"{'—':>10s} | {'—':>10s}")
                continue
            f_m, f_lo, f_hi, _ = f_stats
            s_m, s_lo, s_hi, _ = s_stats
            sep = "✓ DIVERGED" if ci_separated(f_stats, s_stats) else ""
            print(f"{r['turn']:>5d} | {n_f:>3d}/{n_s:<3d} | "
                  f"{f_m:>10.3f} [{f_lo:>+8.3f}, {f_hi:>+8.3f}] | "
                  f"{s_m:>10.3f} [{s_lo:>+8.3f}, {s_hi:>+8.3f}] | "
                  f"{f_m - s_m:>+7.3f} | {sep}")

    # Save CSV for plotting / further analysis
    import csv
    csv_path = os.path.join(args.out_dir, 'divergence_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        cols = ['turn', 'n_fail', 'n_succ']
        for mk in metrics_keys:
            cols += [f'{mk}_fail_mean', f'{mk}_fail_lo', f'{mk}_fail_hi',
                      f'{mk}_succ_mean', f'{mk}_succ_lo', f'{mk}_succ_hi',
                      f'{mk}_diff', f'{mk}_diverged']
        writer.writerow(cols)
        for r in rows:
            row = [r['turn'], r['n_fail'], r['n_succ']]
            for mk in metrics_keys:
                f_stats, s_stats = r[mk]
                if f_stats is None or s_stats is None:
                    row += [''] * 8
                else:
                    f_m, f_lo, f_hi, _ = f_stats
                    s_m, s_lo, s_hi, _ = s_stats
                    diverged = ci_separated(f_stats, s_stats)
                    row += [f'{f_m:.4f}', f'{f_lo:.4f}', f'{f_hi:.4f}',
                            f'{s_m:.4f}', f'{s_lo:.4f}', f'{s_hi:.4f}',
                            f'{f_m - s_m:.4f}', int(diverged)]
            writer.writerow(row)
    print(f"\nSaved CSV: {csv_path}")

    # Verdict — earliest turn each metric diverges
    print(f"\n=== EARLIEST DIVERGENCE PER METRIC (95% CI separated) ===")
    for mk in metrics_keys:
        earliest = None
        for r in rows:
            f_stats, s_stats = r[mk]
            if ci_separated(f_stats, s_stats):
                earliest = r['turn']
                f_m = f_stats[0]
                s_m = s_stats[0]
                print(f"  {mk:<22s}: turn {earliest:>4d}  "
                      f"(fail={f_m:.3f}, succ={s_m:.3f}, "
                      f"diff={f_m - s_m:+.3f})")
                break
        if earliest is None:
            print(f"  {mk:<22s}: no significant divergence in tested range")


if __name__ == '__main__':
    main()
