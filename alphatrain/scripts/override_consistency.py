"""Are the search's overrides reproducible? Given the exported states (recorded search choice vs base prior)
and K independent re-searches (mcts_relabel --seed-salt), measure per state how often an independent search
picks the recorded choice / the prior / something else, the Q margin choice-vs-prior, and classify overrides
as ROBUST (>= k of K independent searches reproduce the non-prior choice) or NOISE. Optionally score models'
greedy argmax on each class."""
import argparse, csv, json, struct, sys
import numpy as np


def load_relabel(path):
    rows = list(csv.DictReader(open(path))); out = []
    for r in rows:
        cands = {}
        for k in range(30):
            c = r.get(f'cand{k}')
            if c is None or c.startswith('-1:'): continue
            a, v, p, q = c.split(':'); cands[int(a)] = (int(v), float(q))
        out.append((int(r['visit_argmax']), cands, float(r['q_min']), float(r['q_max'])))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--states', required=True); ap.add_argument('--relabels', nargs='+', required=True)
    ap.add_argument('--robust-k', type=int, default=2); ap.add_argument('--models', nargs='*', default=[]); ap.add_argument('--fp16', action='store_true'); a = ap.parse_args()
    meta = json.load(open(a.states + '.meta.json')); n = len(meta)
    over = np.array([m['over'] for m in meta]); chosen = np.array([m['chosen'] for m in meta]); prior = np.array([m['prior'] for m in meta])
    occ = np.array([m['occ'] for m in meta])
    R = [load_relabel(p) for p in a.relabels]; K = len(R)
    arg = np.array([[R[k][i][0] for i in range(n)] for k in range(K)])          # (K, n)
    rep_choice = (arg == chosen).sum(0); rep_prior = (arg == prior).sum(0)
    print(f'{n} states ({over.sum()} overrides, {(~over).sum()} agree); K={K} independent searches, 400 sims each\n')
    for name, m in (('OVERRIDE rows', over), ('AGREE rows', ~over)):
        print(f'{name}: independent search picks recorded choice {100*(arg[:, m] == chosen[m]).mean():.1f}%, '
              f'the prior move {100*(arg[:, m] == prior[m]).mean():.1f}%, other {100*((arg[:, m] != chosen[m]) & (arg[:, m] != prior[m])).mean():.1f}%')
    pair = np.mean([(arg[i] == arg[j]) for i in range(K) for j in range(i + 1, K)], axis=0)
    print(f'pairwise agreement of independent searches: override rows {100*pair[over].mean():.1f}%, agree rows {100*pair[~over].mean():.1f}%')
    dist = np.bincount(rep_choice[over], minlength=K + 1)
    print(f'override rows: # of {K} independent searches reproducing the recorded choice: ' + ', '.join(f'{k}:{dist[k]}' for k in range(K + 1)))
    # Q margin choice - prior, normalized by q range, where both visited
    margins = []
    for i in np.where(over)[0]:
        for k in range(K):
            c = R[k][i][1]; rng_ = max(R[k][i][3] - R[k][i][2], 1e-9)
            if chosen[i] in c and prior[i] in c and c[chosen[i]][0] > 0 and c[prior[i]][0] > 0:
                margins.append((c[chosen[i]][1] - c[prior[i]][1]) / rng_)
    margins = np.array(margins)
    print(f'override rows, normalized Q(choice)-Q(prior) in independent searches: n={len(margins)} P25 {np.percentile(margins,25):+.3f} P50 {np.median(margins):+.3f} P75 {np.percentile(margins,75):+.3f}; >0 in {100*(margins>0).mean():.0f}%')
    robust = over & (rep_choice >= a.robust_k); noise = over & (rep_choice == 0)
    print(f'ROBUST overrides (>= {a.robust_k}/{K} reproduce): {robust.sum()} ({100*robust.sum()/over.sum():.0f}% of overrides); NOISE (0/{K}): {noise.sum()} ({100*noise.sum()/over.sum():.0f}%)')
    print(f'  occupancy mean: robust {occ[robust].mean():.1f}, noise {occ[noise].mean():.1f}, agree {occ[~over].mean():.1f}')
    np.save(a.states + '.robust.npy', robust); np.save(a.states + '.noise.npy', noise)
    if a.models:
        import torch
        sys.path.insert(0, '.')
        from alphatrain.observation import build_observation
        from alphatrain.evaluate import load_model
        from alphatrain.mcts import _legal_priors_jit
        with open(a.states, 'rb') as f:
            f.read(8); recs = []
            for _ in range(n):
                b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
                nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)]; f.read(12); recs.append((b, k, nb))
        obs = torch.from_numpy(np.stack([build_observation(b, np.array([x[0] for x in nb[:k]]), np.array([x[1] for x in nb[:k]]), np.array([x[2] for x in nb[:k]]), k) for b, k, nb in recs]).astype(np.float32))
        dev = torch.device('mps')
        print(f'\n{"model":40s} {"adopt ROBUST":>12s} {"adopt NOISE":>11s} {"keep AGREE":>10s}')
        for mp in a.models:
            net, _ = load_model(mp, dev, fp16=a.fp16); am = np.zeros(n, np.int64)
            with torch.inference_mode():
                for s in range(0, n, 2048):
                    xb = obs[s:s+2048].to(dev); lg = net(xb.half() if a.fp16 else xb).float().cpu().numpy()
                    for j in range(lg.shape[0]):
                        c, idx, _ = _legal_priors_jit(recs[s + j][0], lg[j].astype(np.float32), 1); am[s + j] = idx[0] if c else -1
            hit = am == chosen
            print(f'{mp.split("/")[-1][:40]:40s} {100*hit[robust].mean():12.1f} {100*hit[noise].mean():11.1f} {100*hit[~over].mean():10.1f}')


if __name__ == '__main__':
    main()
