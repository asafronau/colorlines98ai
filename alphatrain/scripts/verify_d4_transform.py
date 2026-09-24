"""Ground-truth check of the training-time dihedral augmentation. For each of the dataset's 8 transforms,
compare (a) the dataset's LUT transform of the observation of a board with (b) the observation built from
the actually-transformed board+preview, channel by channel; and check the policy LUT maps moves the same way.
Usage: verify_d4_transform.py <path/to/dataset.py> [n_states]"""
import importlib.util, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.scripts.tta_check import CELL, ACT

path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
spec = importlib.util.spec_from_file_location('ds_under_test', path); ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
d = torch.load('alphatrain/data/r2_bulk.pt', map_location='cpu', weights_only=True, mmap=True)
idx = np.sort(np.random.default_rng(0).choice(d['boards'].shape[0], N, replace=False))
B = d['boards'][idx].numpy(); NP = d['next_pos'][idx].numpy(); NC = d['next_col'][idx].numpy(); NN = d['n_next'][idx].numpy()


def obs_of(board, npos, ncol, k):
    return build_observation(board, npos[:k, 0].astype(np.int64), npos[:k, 1].astype(np.int64), ncol[:k].astype(np.int64), int(k))


obs0 = torch.from_numpy(np.stack([obs_of(B[i], NP[i], NC[i], NN[i]) for i in range(N)]))
has_line = hasattr(ds, '_LINE_LUTS') or hasattr(ds, '_build_line_direction_luts')
obs_luts, pol_luts = ds._build_dihedral_luts()
line_luts = ds._build_line_direction_luts() if hasattr(ds, '_build_line_direction_luts') else None
print(f'{path}\n  line-direction LUTs present: {line_luts is not None}')
for t in range(8):
    inv = torch.from_numpy(np.argsort(np.asarray(obs_luts[t])))   # dataset stores old->new; transform indexes new<-old
    if line_luts is not None:
        linv = torch.from_numpy(np.argsort(np.asarray(line_luts[t])))
        try:
            got = ds._transform_observation(obs0.clone(), inv, linv)
        except TypeError:
            got = ds._transform_observation(obs0.clone(), inv)
    else:
        got = obs0.reshape(N, 18, 81)[:, :, inv].reshape(N, 18, 9, 9)
    # which geometric transform is t?  match the empty plane (channel 7)
    best = None
    for v in range(8):
        b2 = np.zeros((N, 81), np.int8)
        for i in range(N): b2[i, CELL[v]] = B[i].reshape(81)
        e7 = torch.from_numpy((b2 == 0).astype(np.float32).reshape(N, 9, 9))
        if torch.equal(got[:, 7], e7): best = v; break
    if best is None:
        print(f'  t={t}: NO geometric transform matches the empty plane!'); continue
    v = best; gt = []
    for i in range(N):
        b2 = np.zeros(81, np.int8); b2[CELL[v]] = B[i].reshape(81)
        np2 = NP[i].copy()
        for j in range(int(NN[i])):
            c = CELL[v][int(NP[i][j, 0]) * 9 + int(NP[i][j, 1])]; np2[j] = (c // 9, c % 9)
        gt.append(obs_of(b2.reshape(9, 9), np2, NC[i], NN[i]))
    gt = torch.from_numpy(np.stack(gt))
    bad = [(ch, round(float((got[:, ch] - gt[:, ch]).abs().max()), 3), round(100 * float(((got[:, ch] - gt[:, ch]).abs() > 1e-5).any(-1).any(-1).float().mean()), 1)) for ch in range(18)]
    bad = [x for x in bad if x[1] > 1e-5]
    pol_ok = np.mean(np.asarray(pol_luts[t])[np.arange(6561)] == ACT[v]) if len(np.asarray(pol_luts[t])) == 6561 else float('nan')
    print(f'  t={t} == geometric view {v}: policy LUT agrees with ground truth on {100*pol_ok:.1f}% of moves; '
          f'obs channels wrong: {bad if bad else "none"}  (channel, max|err|, % of states affected)')
