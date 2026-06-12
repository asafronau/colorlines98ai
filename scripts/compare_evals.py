"""Paired comparison of two eval_policy per-seed score files.

Same seed list ⇒ each seed is its own control: per-seed deltas cancel the seed-luck
component that dominates aggregate-median noise. Reports mean/median delta with a
paired SE, win rate, and a sign-test p-value — typically 2-3x more sensitive than
eyeballing two aggregate medians at the same game count.

    PYTHONPATH=. python scripts/compare_evals.py \\
        logs/eval_scores/modelA_775000_775999.json \\
        logs/eval_scores/modelB_775000_775999.json
"""
import sys, json, math
import numpy as np


def main():
    fa, fb = sys.argv[1], sys.argv[2]
    A = json.load(open(fa))
    B = json.load(open(fb))
    common = sorted(set(A) & set(B), key=int)
    if len(common) < len(A) or len(common) < len(B):
        print(f"WARNING: only {len(common)} common seeds "
              f"(A={len(A)}, B={len(B)})")
    a = np.array([A[s][0] for s in common], dtype=np.float64)
    b = np.array([B[s][0] for s in common], dtype=np.float64)
    d = a - b
    n = len(d)
    corr = float(np.corrcoef(a, b)[0, 1])
    se_paired = d.std(ddof=1) / math.sqrt(n)
    se_unpaired = math.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    # two-sided sign test (normal approx)
    m = wins + losses
    z = (wins - m / 2) / math.sqrt(m / 4) if m else 0.0
    p_sign = math.erfc(abs(z) / math.sqrt(2))

    print(f"A = {fa}\nB = {fb}\n{n} paired seeds, corr(A,B)={corr:.3f}")
    print(f"\n  median A={np.median(a):.0f}  B={np.median(b):.0f}   "
          f"mean A={a.mean():.0f}  B={b.mean():.0f}")
    print(f"  mean delta (A-B) = {d.mean():+.0f}  ± {se_paired:.0f} (paired SE; "
          f"unpaired would be ±{se_unpaired:.0f})")
    print(f"  median delta     = {np.median(d):+.0f}")
    print(f"  A wins {wins} / loses {losses} / ties {n - m}   "
          f"sign-test p = {p_sign:.4f}")
    print(f"  delta percentiles: P10={np.percentile(d, 10):+.0f} "
          f"P25={np.percentile(d, 25):+.0f} P75={np.percentile(d, 75):+.0f} "
          f"P90={np.percentile(d, 90):+.0f}")


if __name__ == '__main__':
    main()
