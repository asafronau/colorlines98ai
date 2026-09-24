"""Argmax-match of checkpoints against r3_tail rows, split by disagree_mask (0 = r2_bulk,
0<m<1 = relabelled agree/decayed, m>=1 = relabelled flips within 60 turns of death)."""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model
p = argparse.ArgumentParser(); p.add_argument('--models', nargs='+', required=True); p.add_argument('--n-bulk', type=int, default=60000); p.add_argument('--n-new', type=int, default=60000)
a = p.parse_args()
d = torch.load('alphatrain/data/r3_tail.pt', map_location='cpu', weights_only=True, mmap=True)
mask = d['disagree_mask'].numpy(); rng = np.random.default_rng(0)
idx = np.sort(np.concatenate([rng.choice(np.where(mask == 0)[0], a.n_bulk, replace=False), rng.choice(np.where(mask > 0)[0], min(a.n_new, (mask > 0).sum()), replace=False)]))
boards = d['boards'][idx].numpy(); npos = d['next_pos'][idx].numpy(); ncol = d['next_col'][idx].numpy(); nn_ = d['n_next'][idx].numpy(); pi = d['pol_indices'][idx].numpy(); pv = d['pol_values'][idx].numpy(); m = mask[idx]
tgt = pi[np.arange(len(idx)), pv.argmax(1)]
obs = np.zeros((len(idx), 18, 9, 9), np.float32)
for i in range(len(idx)): obs[i] = build_observation(boards[i], npos[i, :, 0].astype(np.int64), npos[i, :, 1].astype(np.int64), ncol[i].astype(np.int64), int(nn_[i]))
dev = torch.device('mps'); obs_t = torch.from_numpy(obs)
groups = [('r2_bulk (mask 0)', m == 0), ('relabel agree/decayed (0<m<1)', (m > 0) & (m < 1)), ('relabel flips <60t (m=1)', m >= 1)]
print(f'{"model":42s} ' + ' '.join(f'{g[0]:>30s}' for g in groups))
for mp in a.models:
    net, _ = load_model(mp, dev, fp16=False); arg = np.zeros(len(idx), np.int64)
    with torch.inference_mode():
        for s in range(0, len(idx), 2048):
            lg = net(obs_t[s:s+2048].to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                k = pi[s + j][pv[s + j] > 0]; arg[s + j] = k[lg[j][k].argmax()]
    match = arg == tgt
    print(f'{mp.split("/")[-1]:42s} ' + ' '.join(f'{100*match[g[1]].mean():29.2f}%' for g in groups))
