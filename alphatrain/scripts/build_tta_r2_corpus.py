"""Build training tensors from a TTA relabel sidecar (relabel_tta_tensor.py):
  --mode tta    : every row's target = the 8-view-averaged decision (top-5, softmax over the 5)
  --mode hybrid : original label where the original target is decisive (top-share >= --tau), else TTA"""
import argparse
import numpy as np, torch
p = argparse.ArgumentParser(); p.add_argument('--src', required=True); p.add_argument('--sidecar', required=True)
p.add_argument('--mode', choices=['tta', 'hybrid'], required=True); p.add_argument('--tau', type=float, default=0.4); p.add_argument('--out', required=True)
a = p.parse_args()
d = torch.load(a.src, map_location='cpu', weights_only=False); z = np.load(a.sidecar)
PI = z['tta_idx'].astype(np.int64); PV = z['tta_val'].astype(np.float32); PV = PV / PV.sum(1, keepdims=True)
if a.mode == 'hybrid':
    keep = z['orig_top'] >= a.tau
    PI[keep] = d['pol_indices'].numpy()[keep]; PV[keep] = d['pol_values'].numpy()[keep]
    print(f'hybrid: original label kept on {100*keep.mean():.1f}% of rows (top-share >= {a.tau}), TTA elsewhere')
out = dict(d); out['pol_indices'] = torch.from_numpy(PI); out['pol_values'] = torch.from_numpy(PV)
out['pol_nnz'] = torch.from_numpy((PV > 0).sum(1).astype(np.int64))
torch.save(out, a.out); print(f'wrote {a.out}: {len(PI):,} rows, mode {a.mode}')
