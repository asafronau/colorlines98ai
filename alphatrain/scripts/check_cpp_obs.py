"""Eval/deploy-path check: C++ Game::BuildObs + LegalMask (obs_dump) vs the training reference
(build_observation + _legal_priors_jit legality) on 20k random r2_bulk states + 10k student near-death states."""
import struct, subprocess, sys, os
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.mcts import _legal_priors_jit
S = sys.argv[1]; rng = np.random.default_rng(2)
d = torch.load('alphatrain/data/r2_bulk.pt', map_location='cpu', weights_only=True, mmap=True)
ix = np.sort(rng.choice(d['boards'].shape[0], 20000, replace=False))
recs = [(d['boards'][i].numpy().reshape(81), [tuple(int(v) for v in (d['next_pos'][i][t][0], d['next_pos'][i][t][1], d['next_col'][i][t])) for t in range(int(d['n_next'][i]))]) for i in ix]
with open('alphatrain/inference_cpp/data/greedy_tail200_18b96e40_2600k.bin', 'rb') as f:
    f.read(4); n = struct.unpack('<i', f.read(4))[0]; blobs = [f.read(133) for _ in range(n)]
for j in np.sort(rng.choice(n, 10000, replace=False)):
    b = blobs[j]; k = struct.unpack('<i', b[81:85])[0]
    recs.append((np.frombuffer(b[:81], np.int8).copy(), [struct.unpack('<iii', b[85 + 12 * t:97 + 12 * t]) for t in range(k)]))
st = f'{S}/objcheck_states.bin'
with open(st, 'wb') as f:
    f.write(b'CLRJ'); f.write(struct.pack('<i', len(recs)))
    for b, nb in recs:
        f.write(b.astype(np.int8).tobytes()); f.write(struct.pack('<i', len(nb)))
        for t in range(3): f.write(struct.pack('<iii', *nb[t]) if t < len(nb) else struct.pack('<iii', 0, 0, 0))
        f.write(struct.pack('<iif', 0, 0, 0.0))
subprocess.run(['alphatrain/inference_cpp/build/obs_dump', st, f'{S}/objcheck_obs.f32', f'{S}/objcheck_legal.u8'], check=True)
co = np.fromfile(f'{S}/objcheck_obs.f32', np.float32).reshape(len(recs), 18, 9, 9); cl = np.fromfile(f'{S}/objcheck_legal.u8', np.uint8).reshape(len(recs), 6561)
bad_ch = np.zeros(18); bad_legal = 0; occ = []
for i, (b, nb) in enumerate(recs):
    ref = build_observation(b.reshape(9, 9), np.array([x[0] for x in nb], np.int64), np.array([x[1] for x in nb], np.int64), np.array([x[2] for x in nb], np.int64), len(nb))
    bad_ch += (np.abs(ref - co[i]) > 1e-5).any(-1).any(-1)
    c, idx, _ = _legal_priors_jit(b.reshape(9, 9), np.zeros(6561, np.float32), 6561)
    ls = np.zeros(6561, bool); ls[idx[:c]] = True
    if not np.array_equal(ls, cl[i].astype(bool)): bad_legal += 1
    occ.append((b > 0).sum())
print(f'{len(recs):,} states (occupancy P50 {np.median(occ):.0f}, max {max(occ)}):')
print('  obs channels differing (C++ vs training reference):', {c: int(v) for c, v in enumerate(bad_ch) if v} or 'none')
print(f'  legal mask differing: {bad_legal}')
