"""Same random 1M r2_bulk states, two label sets: original hard labels vs e40's 8-view-averaged move
(TTA sidecar). For a small from-scratch born-again A/B."""
import argparse
import numpy as np, torch
p = argparse.ArgumentParser(); p.add_argument('--n', type=int, default=1_000_000); p.add_argument('--out-dir', required=True); a = p.parse_args()
d = torch.load('alphatrain/data/r2_bulk.pt', map_location='cpu', weights_only=False, mmap=True); z = np.load('alphatrain/data/r2_bulk_tta8_e40.npz')
N = d['boards'].shape[0]; idx = np.sort(np.random.default_rng(11).choice(N, a.n, replace=False)); t = torch.from_numpy(idx)
base = {k: (v[t].clone() if hasattr(v, 'shape') and v.ndim >= 1 and v.shape[0] == N else v) for k, v in d.items()}
torch.save(base, f'{a.out_dir}/ab1m_orig.pt')
tta = dict(base); PI = z['tta_idx'][idx].astype(np.int64); PV = z['tta_val'][idx].astype(np.float32); PV /= PV.sum(1, keepdims=True)
tta['pol_indices'] = torch.from_numpy(PI); tta['pol_values'] = torch.from_numpy(PV); tta['pol_nnz'] = torch.from_numpy((PV > 0).sum(1).astype(np.int64))
torch.save(tta, f'{a.out_dir}/ab1m_tta.pt')
print(f'wrote ab1m_orig.pt / ab1m_tta.pt: {a.n:,} identical states; labels differ on {100*(z["tta_idx"][idx,0] != z["orig_arg"][idx]).mean():.1f}% of rows')
