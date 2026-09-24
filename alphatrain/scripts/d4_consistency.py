"""Symmetry consistency of policy checkpoints on the same states: per state, the legal argmax in each of
the 8 exact board symmetries (observation rebuilt from the transformed board/preview, logits mapped back),
and the argmax of the 8-view logit average (TTA). Reports, per stratum of the state file's meta:
all-8-views-agree %, mean agreement of views 1-7 with view 0, and how often TTA changes view 0's move."""
import argparse, json, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
from alphatrain.scripts.tta_check import views, ACT


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--states', required=True); ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--strata-key', default='stratum'); a = ap.parse_args()
    meta = json.load(open(a.states + '.meta.json')); n = len(meta)
    strat = np.array([str(m.get(a.strata_key, 'all')) for m in meta])
    recs = []
    with open(a.states, 'rb') as f:
        f.read(8)
        for _ in range(n):
            b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
            nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)][:k]; f.read(12); recs.append((b, nb))
    obs = torch.from_numpy(np.stack([o for b, nb in recs for o in views(b, nb)]).astype(np.float32))
    dev = torch.device('mps'); groups = ['all'] + sorted(set(strat.tolist()) - {'all'})
    print(f'{n} states; strata {dict(zip(*np.unique(strat, return_counts=True)))}')
    print(f'{"model":38s} {"stratum":8s} {"all 8 agree":>11s} {"view k==view0":>13s} {"TTA != view0":>12s}')
    for mp in a.models:
        net, _ = load_model(mp, dev, fp16=False); pv = np.zeros((n, 8), np.int64); tta = np.zeros(n, np.int64)
        with torch.inference_mode():
            for s in range(0, n * 8, 2048):
                lg = net(obs[s:s+2048].to(dev)).float().cpu().numpy().reshape(-1, 8, 6561)
                for j in range(lg.shape[0]):
                    i = s // 8 + j; back = np.stack([lg[j, v, ACT[v]] for v in range(8)])
                    for v in range(8):
                        c, idx, _ = _legal_priors_jit(recs[i][0], back[v].copy(), 1); pv[i, v] = idx[0] if c else -1
                    c, idx, _ = _legal_priors_jit(recs[i][0], back.mean(0).copy(), 1); tta[i] = idx[0] if c else -1
        for g in groups:
            m = np.ones(n, bool) if g == 'all' else strat == g
            print(f'{mp.split("/")[-1][:38]:38s} {g:8s} {100*(pv[m] == pv[m, :1]).all(1).mean():10.1f}% {100*(pv[m, 1:] == pv[m, :1]).mean():12.1f}% {100*(tta[m] != pv[m, 0]).mean():11.1f}%', flush=True)


if __name__ == '__main__':
    main()
