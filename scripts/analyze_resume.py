"""Analysis of a resume_eval_parallel run.

Marginal floor comparison + catastrophe rates + bootstrap CIs. If raw rows
include the seed (6 cols), also runs the rigorous PAIRED test (common-RNG
per-seed): sign test + paired bootstrap, much lower variance than unpaired.
"""
import json
import sys
import numpy as np

REAL_SEED = -987654321
path = sys.argv[1]
d = json.load(open(path))
per = {tuple(r['t']): r for r in d['per_target']}
A = tuple(d['meta']['policy_target'])           # policy pick, e.g. (2,8)
B = next(t for t in per if t != A)              # alternative, e.g. (7,4)
rows = d['raw']
paired = len(rows[0]) == 6                       # seed present?

print(f"{path}\nA={A} (policy)   B={B} (alt)   mode="
      f"{'PAIRED' if paired else 'unpaired (no seed saved)'}\n")

if paired:
    da = {r[2]: r[3] for r in rows if (r[0], r[1]) == A and r[2] != REAL_SEED}
    db = {r[2]: r[3] for r in rows if (r[0], r[1]) == B and r[2] != REAL_SEED}
    seeds = sorted(set(da) & set(db))
    sa = np.array([da[s] for s in seeds])
    sb = np.array([db[s] for s in seeds])
else:
    def scores(t):
        s = [r[2] for r in rows if (r[0], r[1]) == t and r[2] >= 0]
        real = per[t].get('real')
        if real and real[0] in s:
            s.remove(real[0])
        return np.array(s)
    sa, sb = scores(A), scores(B)

n = len(sa)
print(f"n={n} paired seeds" if paired else f"n_A={len(sa)} n_B={len(sb)}")

print(f"\n{'metric':>6} {str(A):>12} {str(B):>12} {'B-A':>9} {'B/A-1':>7}")
for label, p in [('P1', 1), ('P5', 5), ('P10', 10), ('P25', 25),
                 ('P50', 50), ('P75', 75), ('P90', 90)]:
    va, vb = np.percentile(sa, p), np.percentile(sb, p)
    print(f"{label:>6} {va:>12.0f} {vb:>12.0f} {vb-va:>9.0f} "
          f"{100*(vb/va-1):>6.1f}%")
print(f"{'mean':>6} {sa.mean():>12.0f} {sb.mean():>12.0f} "
      f"{sb.mean()-sa.mean():>9.0f} {100*(sb.mean()/sa.mean()-1):>6.1f}%")

print("\nCatastrophe rates (% scoring <= threshold) — anti-gambling metric:")
for thr in (215, 500, 1000, 2000, 3000):
    pa, pb = 100*(sa <= thr).mean(), 100*(sb <= thr).mean()
    print(f"  <= {thr:>5}:  A={pa:>5.1f}%   B={pb:>5.1f}%   "
          f"(A {pa-pb:+.1f}pp)")

rng = np.random.default_rng(0)
Bz = 5000
if paired:
    diff = sb - sa
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    m = wins + losses
    z = (wins - m/2) / np.sqrt(m/4) if m else 0.0
    print(f"\nPaired sign test: B beats A on {wins}/{n} seeds "
          f"({100*wins/n:.1f}%), A beats B {losses} ({100*losses/n:.1f}%); "
          f"median diff(B-A)={np.median(diff):+.0f}; z={z:.2f}")
    print(f"Paired bootstrap 95% CI on (B-A) percentile diff ({Bz} resamples):")
    for label, p in [('P1', 1), ('P5', 5), ('P10', 10), ('P25', 25),
                     ('P50', 50), ('mean', None)]:
        dl = np.empty(Bz)
        for i in range(Bz):
            idx = rng.integers(0, n, n)
            ra, rb = sa[idx], sb[idx]
            dl[i] = (rb.mean()-ra.mean()) if p is None else \
                (np.percentile(rb, p)-np.percentile(ra, p))
        lo, hi = np.percentile(dl, [2.5, 97.5])
        obs = (sb.mean()-sa.mean()) if p is None else \
            (np.percentile(sb, p)-np.percentile(sa, p))
        sig = 'SIGNIFICANT' if (lo > 0 or hi < 0) else 'n.s.'
        print(f"  {label:>5}: B-A={obs:>+8.0f}  CI[{lo:>+8.0f},{hi:>+8.0f}] {sig}")
else:
    print(f"\nUnpaired bootstrap 95% CI on (B-A) percentile diff ({Bz}):")
    for label, p in [('P1', 1), ('P5', 5), ('P10', 10), ('P25', 25),
                     ('P50', 50), ('mean', None)]:
        dl = np.empty(Bz)
        for i in range(Bz):
            ra = sa[rng.integers(0, len(sa), len(sa))]
            rb = sb[rng.integers(0, len(sb), len(sb))]
            dl[i] = (rb.mean()-ra.mean()) if p is None else \
                (np.percentile(rb, p)-np.percentile(ra, p))
        lo, hi = np.percentile(dl, [2.5, 97.5])
        obs = (sb.mean()-sa.mean()) if p is None else \
            (np.percentile(sb, p)-np.percentile(sa, p))
        sig = 'SIGNIFICANT' if (lo > 0 or hi < 0) else 'n.s.'
        print(f"  {label:>5}: B-A={obs:>+8.0f}  CI[{lo:>+8.0f},{hi:>+8.0f}] {sig}")

print("\nReal-RNG single sample (CONFOUNDED — one point, not proof):")
print(f"  A={A}: score={per[A]['real'][0]} turns={per[A]['real'][1]}")
print(f"  B={B}: score={per[B]['real'][0]} turns={per[B]['real'][1]}")
