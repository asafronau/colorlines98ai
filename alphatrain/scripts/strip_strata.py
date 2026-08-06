"""Move the (numpy) strata array out of hard-CE tensors into .npz sidecars so
the trainer's weights_only=True load works. Verifies the safe load after.

    python -m alphatrain.scripts.strip_strata
"""
import sys

import numpy as np
import torch

for name in (sys.argv[1:] or ('h_all', 'h70', 'h44')):
    p = f'alphatrain/data/{name}.pt'
    d = torch.load(p, map_location='cpu', weights_only=False)
    if 'strata' in d:
        np.savez(f'alphatrain/data/{name}_strata.npz', strata=d.pop('strata'))
        torch.save(d, p)
    t = torch.load(p, weights_only=True)
    print(f'{name}: weights_only OK ({t["boards"].shape[0]:,} rows)')
