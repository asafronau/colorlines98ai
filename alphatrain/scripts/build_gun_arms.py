"""Smoking-gun arms (content vs concentration) from r3_tail.pt at 1/12.5 scale:
  A control : 1,000,000 random r2_bulk rows (mask 0)
  B content : A + 16,000 random relabelled rows with their r3_tail masks (same masked-row share as r3_tail)
  C weight  : A + 16,000 MORE random r2_bulk rows given B's mask values (same concentration, benign labels)
"""
import argparse, torch, numpy as np
p = argparse.ArgumentParser(); p.add_argument('--src', default='alphatrain/data/r3_tail.pt'); p.add_argument('--out-dir', required=True)
p.add_argument('--n-bulk', type=int, default=1_000_000); p.add_argument('--n-new', type=int, default=16_000); a = p.parse_args()
d = torch.load(a.src, map_location='cpu', weights_only=True, mmap=True); mask = d['disagree_mask'].numpy(); rng = np.random.default_rng(0)
bulk = np.where(mask == 0)[0]; new = np.where(mask > 0)[0]
pick_bulk = rng.choice(bulk, a.n_bulk + a.n_new, replace=False); ctrl, extra = pick_bulk[:a.n_bulk], pick_bulk[a.n_bulk:]
pick_new = rng.choice(new, a.n_new, replace=False)
keys = ['boards', 'next_pos', 'next_col', 'n_next', 'pol_indices', 'pol_values', 'pol_nnz', 'disagree_mask']
def build(idx, mask_override=None, name=''):
    idx = np.sort(idx); out = {k: d[k][idx].clone() for k in keys}
    if mask_override is not None: out['disagree_mask'] = torch.from_numpy(mask_override.astype(np.float32))
    for k in ('num_channels', 'max_score', 'value_mode', 'gamma', 'num_value_bins'): out[k] = d[k]
    perm = torch.randperm(len(idx), generator=torch.Generator().manual_seed(1))
    for k in keys: out[k] = out[k][perm].contiguous()
    torch.save(out, f'{a.out_dir}/gun_{name}.pt'); print(name, len(idx), 'rows, mask sum', float(out['disagree_mask'].sum()))
build(ctrl, np.zeros(a.n_bulk), 'A_control')
mB = np.concatenate([np.zeros(a.n_bulk), mask[pick_new]]); build(np.concatenate([ctrl, pick_new]), None, 'B_content')  # keeps r3_tail masks (0 for ctrl)
build(np.concatenate([ctrl, extra]), np.concatenate([np.zeros(a.n_bulk), mask[pick_new]]), 'C_weight')
