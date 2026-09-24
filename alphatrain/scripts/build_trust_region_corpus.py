"""Trust-region corpus: every state of the full-record search games; target = the BASE policy's own FP32
legal softmax (top-5 renormalized) in the canonical view, EXCEPT robust overrides (search choice reproduced
by >= --min-rep of K independent re-searches), whose target is the search's move (one-hot).
Train with soft CE (blend 1.0), --freeze-bn, --no-dihedral-augment, --no-color-augment, --augment-factor 1:
on 97% of rows the loss is minimized by staying exactly the base (no churn by construction)."""
import argparse, csv, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit


def flat(m): return (m['sr'] * 9 + m['sc']) * 81 + m['tr'] * 9 + m['tc']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--games-dir', required=True); p.add_argument('--base', required=True)
    p.add_argument('--override-states', required=True); p.add_argument('--relabels', nargs='+', required=True)
    p.add_argument('--min-rep', type=int, default=2); p.add_argument('--out', required=True)
    p.add_argument('--drop-reps', default='', help='comma list of re-search reproduction counts whose override rows are DROPPED (e.g. 1 = ambiguous 2-of-4)')
    p.add_argument('--override-target', choices=['onehot', 'pooled'], default='onehot',
                   help='onehot: robust overrides -> search move, others -> base soft; pooled: EVERY override row -> visits pooled over the recorded search + all re-searches (top-5)')
    a = p.parse_args(); drop = {int(x) for x in a.drop_reps.split(',') if x}
    meta = json.load(open(a.override_states + '.meta.json')); reps = np.zeros(len(meta), int); pooled = [dict() for _ in meta]
    for path in a.relabels:
        rows = list(csv.DictReader(open(path))); assert len(rows) == len(meta)
        reps += np.array([int(r['visit_argmax']) == m['chosen'] for r, m in zip(rows, meta)])
        for j, r in enumerate(rows):
            for k in range(30):
                c = r.get(f'cand{k}')
                if c is None or c.startswith('-1:'): continue
                act, vis = c.split(':')[:2]; pooled[j][int(act)] = pooled[j].get(int(act), 0) + int(vis)
    key = {(m['seed'], m['turn']): j for j, m in enumerate(meta)}
    robust = {(m['seed'], m['turn']): m['chosen'] for m, c in zip(meta, reps) if c >= a.min_rep}
    dropped = {(m['seed'], m['turn']) for m, c in zip(meta, reps) if c in drop}
    print(f'overrides {len(meta):,}: reps distribution {np.bincount(reps, minlength=len(a.relabels)+1).tolist()}; robust {len(robust):,}; dropped {len(dropped):,}; override target = {a.override_target}', flush=True)
    B, NP, NC, NN, OBS, TGT, POOL = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json'))):
        g = json.load(open(f))
        for i, m in enumerate(g['moves']):
            if (g['seed'], i) in dropped: continue
            pool = None
            if a.override_target == 'pooled' and (g['seed'], i) in key:
                pool = dict(pooled[key[(g['seed'], i)]])
                for act, vis in zip(m['cand_moves'], m['cand_visits']): pool[int(act)] = pool.get(int(act), 0) + int(vis)
            b = np.array(m['board'], np.int8); nb = m['next_balls'][:m['num_next']]
            npos = np.zeros((3, 2), np.int8); ncol = np.zeros(3, np.int8)
            for t, x in enumerate(nb[:3]): npos[t] = (x['row'], x['col']); ncol[t] = x['color']
            B.append(b); NP.append(npos); NC.append(ncol); NN.append(len(nb[:3])); TGT.append(robust.get((g['seed'], i), -1)); POOL.append(pool)
            OBS.append(build_observation(b, npos[:len(nb), 0].astype(np.int64), npos[:len(nb), 1].astype(np.int64), ncol[:len(nb)].astype(np.int64), len(nb)))
    n = len(B); print(f'{n:,} states; robust overrides {sum(t >= 0 for t in TGT):,}', flush=True)
    dev = torch.device('mps'); net, _ = load_model(a.base, dev, fp16=False)
    PI = np.zeros((n, 5), np.int64); PV = np.zeros((n, 5), np.float32); NZ = np.zeros(n, np.int64); MK = np.zeros(n, np.float32)
    obs = np.stack(OBS).astype(np.float32)
    with torch.inference_mode():
        for s in range(0, n, 4096):
            lg = net(torch.from_numpy(obs[s:s+4096]).to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                i = s + j
                if POOL[i] is not None:
                    it = sorted(POOL[i].items(), key=lambda kv: -kv[1])[:5]; tot = sum(v for _, v in it)
                    for q, (act, vis) in enumerate(it): PI[i, q] = act; PV[i, q] = vis / tot
                    NZ[i] = len(it); MK[i] = 1.0 if TGT[i] >= 0 else 0.0; continue
                if TGT[i] >= 0:
                    PI[i, 0] = TGT[i]; PV[i, 0] = 1.0; NZ[i] = 1; MK[i] = 1.0; continue
                cnt, idx, pri = _legal_priors_jit(B[i], lg[j].astype(np.float32), 5)
                k = int(cnt); pr = pri[:k] / pri[:k].sum(); PI[i, :k] = idx[:k]; PV[i, :k] = pr; NZ[i] = k
            if (s // 4096) % 40 == 0: print(f'  {s:,}/{n:,}', flush=True)
    perm = np.random.default_rng(0).permutation(n)
    out = {'boards': torch.from_numpy(np.stack(B)[perm]), 'next_pos': torch.from_numpy(np.stack(NP)[perm]), 'next_col': torch.from_numpy(np.stack(NC)[perm]),
           'n_next': torch.from_numpy(np.array(NN, np.int8)[perm]), 'pol_indices': torch.from_numpy(PI[perm]), 'pol_values': torch.from_numpy(PV[perm]),
           'pol_nnz': torch.from_numpy(NZ[perm]), 'disagree_mask': torch.from_numpy(MK[perm]),
           'disagree_mask_protocol': f'1 = robust override (>= {a.min_rep}/{len(a.relabels)} re-searches) with one-hot search target; 0 = base FP32 soft target',
           'num_channels': 18, 'max_score': 0.0, 'value_mode': 'policy_slim', 'gamma': 0.99, 'num_value_bins': 64}
    torch.save(out, a.out); print(f'wrote {a.out}: {n:,} rows, {int(MK.sum()):,} robust rows (mask=1); pooled rows {sum(x is not None for x in POOL):,}; target top-1 mean {PV.max(1).mean():.3f}')


if __name__ == '__main__':
    main()
