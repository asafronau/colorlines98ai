"""Self-distillation control: every row's target = the base model's own single-view (view-0) argmax from the
relabel sidecar (fp16 deployed model). Zero new information: a fine-tune on this measures the pure cost of the
fine-tuning procedure."""
import argparse
import numpy as np, torch
p = argparse.ArgumentParser(); p.add_argument('--src', required=True); p.add_argument('--sidecar', required=True); p.add_argument('--out', required=True); a = p.parse_args()
d = torch.load(a.src, map_location='cpu', weights_only=False); z = np.load(a.sidecar); n = len(z['view0_arg'])
PI = np.zeros((n, 5), np.int64); PV = np.zeros((n, 5), np.float32); PI[:, 0] = z['view0_arg']; PV[:, 0] = 1.0
out = dict(d); out['pol_indices'] = torch.from_numpy(PI); out['pol_values'] = torch.from_numpy(PV); out['pol_nnz'] = torch.ones(n, dtype=torch.int64)
torch.save(out, a.out); print(f'wrote {a.out}: {n:,} rows, target = base view-0 argmax')
