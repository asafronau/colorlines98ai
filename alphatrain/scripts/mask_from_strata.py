"""Attach disagree_mask=1 to crisis-provenance rows (amplification dial for
--disagree-gamma), from the strata sidecar. Provenance-based = augmentation-
invariant.

    python -m alphatrain.scripts.mask_from_strata h_big
"""
import sys

import numpy as np
import torch

name = sys.argv[1]
p = f'alphatrain/data/{name}.pt'
strata = np.load(f'alphatrain/data/{name}_strata.npz')['strata']
d = torch.load(p, map_location='cpu', weights_only=False)
mask = torch.from_numpy(
    np.char.startswith(strata.astype(str), 'c_').astype(np.int8))
assert mask.shape[0] == d['boards'].shape[0]
d['disagree_mask'] = mask
torch.save(d, p)
t = torch.load(p, weights_only=True)
print(f'{name}: crisis mask on {int(mask.sum()):,}/{len(mask):,} rows '
      f'({100 * mask.float().mean():.1f}%); weights_only OK')
