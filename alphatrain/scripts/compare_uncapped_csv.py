"""Compare greedy score-distribution CSVs (seed,score...) : mean/P50/P10/P5/<1000 with bootstrap CIs on the diff vs the first."""
import argparse, csv, numpy as np
def load(p):
    rows=list(csv.DictReader(open(p))); k='score' if 'score' in rows[0] else list(rows[0])[1]
    return np.array([float(r[k]) for r in rows])
def stats(s): return dict(n=len(s),mean=s.mean(),P50=np.median(s),P10=np.percentile(s,10),P5=np.percentile(s,5),lt1000=100*(s<1000).mean())
p=argparse.ArgumentParser(); p.add_argument('csvs',nargs='+'); a=p.parse_args()
rng=np.random.default_rng(0); ref=load(a.csvs[0])
print(f'{"file":60s} {"n":>5s} {"mean":>7s} {"P50":>7s} {"P10":>6s} {"P5":>6s} {"<1k%":>5s}  dmean% [ci95]')
for f in a.csvs:
    s=load(f); st=stats(s)
    if f==a.csvs[0]: d='  (ref)'
    else:
        b=[(s[rng.integers(0,len(s),len(s))].mean()-ref[rng.integers(0,len(ref),len(ref))].mean())/ref.mean() for _ in range(1000)]
        d=f'  {100*(s.mean()-ref.mean())/ref.mean():+.1f}% [{100*np.percentile(b,2.5):+.1f},{100*np.percentile(b,97.5):+.1f}]'
    print(f'{f.split("/")[-1]:60s} {st["n"]:5d} {st["mean"]:7.0f} {st["P50"]:7.0f} {st["P10"]:6.0f} {st["P5"]:6.0f} {st["lt1000"]:5.1f}{d}')
