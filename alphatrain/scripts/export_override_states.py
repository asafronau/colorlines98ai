"""From full-record search games: every override state (search chosen != recorded prior argmax) on every
--every-th move, plus --n-agree random agree states -> CLRJ (teacher=chosen, base=prior argmax) + meta."""
import argparse, glob, json, os, struct
import numpy as np
def flat(m): return (m['sr'] * 9 + m['sc']) * 81 + m['tr'] * 9 + m['tc']
p = argparse.ArgumentParser(); p.add_argument('--games-dir', required=True); p.add_argument('--out', required=True)
p.add_argument('--every', type=int, default=3); p.add_argument('--n-agree', type=int, default=2000); a = p.parse_args()
rows = []
for f in sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json'))):
    g = json.load(open(f))
    for i, m in enumerate(g['moves']):
        if i % a.every: continue
        ch = flat(m['chosen_move']); pa = m['cand_moves'][int(np.argmax(m['cand_prior']))]
        v = np.array(m['cand_visits'], float); q = np.array(m['cand_q'], float)
        rows.append(dict(board=m['board'], nb=m['next_balls'][:m['num_next']], chosen=ch, prior=pa, over=ch != pa,
                         top=float(v.max() / v.sum()), occ=int((np.array(m['board']) > 0).sum()), seed=g['seed'], turn=i))
rng = np.random.default_rng(0)
over = [r for r in rows if r['over']]; agree = [rows[i] for i in rng.choice([i for i, r in enumerate(rows) if not r['over']], a.n_agree, replace=False)]
sel = over + agree
with open(a.out, 'wb') as f:
    f.write(b'CLRJ'); f.write(struct.pack('<i', len(sel)))
    for r in sel:
        f.write(np.array(r['board'], np.int8).reshape(81).tobytes()); f.write(struct.pack('<i', len(r['nb'])))
        for t in range(3): f.write(struct.pack('<iii', r['nb'][t]['row'], r['nb'][t]['col'], r['nb'][t]['color']) if t < len(r['nb']) else struct.pack('<iii', 0, 0, 0))
        f.write(struct.pack('<iif', r['chosen'], r['prior'], r['top']))
json.dump([{k: r[k] for k in ('over', 'top', 'occ', 'seed', 'turn', 'chosen', 'prior')} for r in sel], open(a.out + '.meta.json', 'w'))
print(f'wrote {a.out}: {len(over)} override + {len(agree)} agree states')
