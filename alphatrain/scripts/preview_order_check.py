"""Anomaly check: the 3 preview balls occupy channels 8/9/10 by generation order, which is meaningless in the
game (positions are distinct). How often does the argmax change when only the ORDER of the preview balls is
permuted (6 orders)? And what does averaging over the 6 orders change?"""
import itertools, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
st = sys.argv[1]; mp = sys.argv[2]
recs = []
with open(st, 'rb') as f:
    f.read(4); n = struct.unpack('<i', f.read(4))[0]
    for _ in range(n):
        b = np.frombuffer(f.read(81), np.int8).reshape(9, 9).copy(); k = struct.unpack('<i', f.read(4))[0]
        nb = [struct.unpack('<iii', f.read(12)) for _ in range(3)][:k]; f.read(12); recs.append((b, nb))
recs = [r for r in recs if len(r[1]) == 3]
orders = list(itertools.permutations(range(3)))
X = np.stack([[build_observation(b, np.array([nb[o][0] for o in od], np.int64), np.array([nb[o][1] for o in od], np.int64), np.array([nb[o][2] for o in od], np.int64), 3) for od in orders] for b, nb in recs]).astype(np.float32)
net, _ = load_model(mp, torch.device('mps'), fp16=False); L = []
with torch.inference_mode():
    for s in range(0, len(X) * 6, 4096): L.append(net(torch.from_numpy(X.reshape(-1, 18, 9, 9)[s:s+4096]).to('mps')).float().cpu().numpy())
L = np.concatenate(L).reshape(len(recs), 6, 6561)
am = lambda b, lg: (lambda c, i, _: i[0] if c else -1)(*_legal_priors_jit(b, lg.astype(np.float32), 1))
arg = np.array([[am(recs[i][0], L[i, o]) for o in range(6)] for i in range(len(recs))])
avg = np.array([am(recs[i][0], L[i].mean(0)) for i in range(len(recs))])
print(f'{len(recs):,} states with 3 preview balls: argmax under a different preview ORDER == original {100*(arg[:, 1:] == arg[:, :1]).mean():.1f}%; '
      f'all 6 orders agree {100*(arg == arg[:, :1]).all(1).mean():.1f}%; order-averaging changes the move {100*(avg != arg[:, 0]).mean():.1f}%')
