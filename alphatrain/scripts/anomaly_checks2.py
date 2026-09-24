"""Anomaly hunt, part 2 (what the net uses / respects):
 D. color symmetry: argmax agreement under 8 random relabelings of colors 1..7 (exact game symmetry)
 E. input ablations: how often the argmax changes when an input group is zeroed (preview, area, line channels)"""
import argparse, json, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
ap = argparse.ArgumentParser(); ap.add_argument('--states', default='alphatrain/inference_cpp/data/relabel_pilot_states.bin')
ap.add_argument('--models', nargs='+', required=True); a = ap.parse_args()
recs = []
with open(a.states, 'rb') as f:
    f.read(4); n = struct.unpack('<i', f.read(4))[0]
    for _ in range(n):
        b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
        nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)][:k]; f.read(12); recs.append((b, nb))
rng = np.random.default_rng(3); perms = [np.concatenate([[0], rng.permutation(7) + 1]) for _ in range(8)]; perms[0] = np.arange(8)
def obs_of(b, nb, perm):
    return build_observation(perm[b].astype(np.int8), np.array([x[0] for x in nb], np.int64), np.array([x[1] for x in nb], np.int64), np.array([perm[x[2]] for x in nb], np.int64), len(nb))
O = np.stack([[obs_of(b, nb, p) for p in perms] for b, nb in recs]).astype(np.float32)   # (n, 8, 18, 9, 9)
dev = torch.device('mps')
def run(net, X):
    out = np.zeros(len(X), np.int64); L = []
    with torch.inference_mode():
        for s in range(0, len(X), 4096): L.append(net(torch.from_numpy(X[s:s+4096]).to(dev)).float().cpu().numpy())
    return np.concatenate(L)
def am(board, lg):
    c, i, _ = _legal_priors_jit(board, lg.astype(np.float32), 1); return i[0] if c else -1
groups = {'preview (ch 8-11)': [8, 9, 10, 11], 'component area (ch 12)': [12], 'line H/V/D1/D2 (ch 13-16)': [13, 14, 15, 16], 'line max (ch 17)': [17], 'empty plane (ch 7)': [7]}
for mp in a.models:
    net, _ = load_model(mp, dev, fp16=False)
    L = run(net, O.reshape(-1, 18, 9, 9)).reshape(n, 8, 6561)
    arg = np.array([[am(recs[i][0], L[i, p]) for p in range(8)] for i in range(n)])
    tta = np.array([am(recs[i][0], L[i].mean(0)) for i in range(n)])
    print(f'{mp.split("/")[-1]}\n  D. color relabel: perm-k == identity {100*(arg[:, 1:] == arg[:, :1]).mean():.1f}%; all 8 agree {100*(arg == arg[:, :1]).all(1).mean():.1f}%; color-TTA changes move {100*(tta != arg[:, 0]).mean():.1f}%')
    base = O[:, 0]
    for g, chs in groups.items():
        X = base.copy(); X[:, chs] = 0.0
        L2 = run(net, X); a2 = np.array([am(recs[i][0], L2[i]) for i in range(n)])
        print(f'  E. zero {g:28s}: argmax changes on {100*(a2 != arg[:, 0]).mean():5.1f}% of states')
