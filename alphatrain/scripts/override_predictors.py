"""Can a single recorded search tell robust overrides from noise? Join override states to their recorded
search stats (visits, prior, Q of choice vs prior) and report precision/recall of simple filters for
'robust' (>=2/3 independent searches reproduce)."""
import glob, json, os, sys
import numpy as np
D = 'alphatrain/inference_cpp/data/override60_states.bin'
meta = json.load(open(D + '.meta.json')); robust = np.load(D + '.robust.npy'); noise = np.load(D + '.noise.npy')
games = {}
for f in glob.glob('alphatrain/data/selfsearch_18b96e40_pv_q2_c15_cap3k/game_seed*.json'):
    g = json.load(open(f)); games[g['seed']] = g['moves']
feat = []
for i, m in enumerate(meta):
    if not m['over']: continue
    mv = games[m['seed']][m['turn']]; acts = mv['cand_moves']; v = np.array(mv['cand_visits'], float); p = np.array(mv['cand_prior'], float)
    q = np.array(mv['cand_q'], float); qr = max(mv['q_max'] - mv['q_min'], 1e-9)
    ci, pi = acts.index(m['chosen']), acts.index(m['prior'])
    feat.append(dict(i=i, vshare=v[ci] / v.sum(), vratio=v[ci] / max(v[pi], 1), qm=(q[ci] - q[pi]) / qr if v[pi] > 0 else np.nan,
                     pch=p[ci], ppr=p[pi], robust=robust[i], noise=noise[i]))
F = {k: np.array([x[k] for x in feat]) for k in feat[0]}
base = F['robust'].mean(); print(f'{len(feat)} overrides; robust base rate {100*base:.0f}%, noise {100*F["noise"].mean():.0f}%')
for name, x in (('visit share of choice', F['vshare']), ('visits choice/prior', F['vratio']), ('norm Q choice-prior', F['qm']), ('prior prob of choice', F['pch']), ('prior prob of prior move', -F['ppr'])):
    ok = ~np.isnan(x); xs = x[ok]; r = F['robust'][ok]; nz = F['noise'][ok]
    qs = np.percentile(xs, [50, 75, 90])
    print(f'{name:28s} ' + '  '.join(f'top{100-int(pp)}%: prec {100*r[xs>=t].mean():.0f}% (noise {100*nz[xs>=t].mean():.0f}%) recall {100*r[xs>=t].sum()/r.sum():.0f}%' for pp, t in zip((50, 75, 90), qs)))
