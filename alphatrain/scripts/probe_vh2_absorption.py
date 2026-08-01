"""Mechanism probe: what did vh2 do with the 676 rows that created it?

  1. vh2's fp16 legal argmax on the round-1 winning rows: absorbed (plays the
     teacher move) / kept vh1's move / third.
  2. Exports the still-disagreeing rows for judging under vh2 continuation
     (does their advantage persist after the merge?).

    python -m alphatrain.scripts.probe_vh2_absorption
"""
import struct

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.diag_dagger1_fp16 import fp16_argmax_margin

CPP = 'alphatrain/inference_cpp/data'


def main():
    c = torch.load('alphatrain/data/advfilt.pt', map_location='cpu',
                   weights_only=False)
    n = c['boards'].shape[0]
    t_mv = c['tgt_idx'][:, 0].numpy()
    v1_mv = c['vh1_move'].numpy()
    tmp = 'alphatrain/data/advfilt.pt.probe.tmp'
    torch.save({'boards': c['boards'], 'next_pos': c['next_pos'],
                'next_col': c['next_col'], 'n_next': c['n_next'],
                'pol_indices': torch.zeros((n, 5), dtype=torch.int64),
                'pol_values': torch.zeros((n, 5), dtype=torch.float32),
                'max_score': 0.0}, tmp)
    ds = TensorDatasetGPU(tmp, augment=False, color_augment=False,
                          augment_factor=1, device='mps')
    dev = torch.device('mps')
    net, _ = load_model('alphatrain/data/small128_vh2.pt', dev, fp16=True)
    arg, _ = fp16_argmax_margin(net, ds, n, dev)

    absorbed = arg == t_mv
    kept = arg == v1_mv
    third = ~absorbed & ~kept
    print(f'\n{n} round-1 winning rows under vh2 (fp16 legal argmax):')
    print(f'  ABSORBED (plays teacher move): {100 * absorbed.mean():.1f}%')
    print(f'  kept vh1 move                : {100 * kept.mean():.1f}%')
    print(f'  third move                   : {100 * third.mean():.1f}%')

    rows = np.where(~absorbed)[0]
    boards = c['boards'].numpy().reshape(n, 81)
    npos, ncol, nn = c['next_pos'].numpy(), c['next_col'].numpy(), c['n_next'].numpy()
    with open(f'{CPP}/probe_vh2_states.bin', 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(rows)))
        for i in rows:
            f.write(boards[i].astype(np.int8).tobytes())
            f.write(struct.pack('<i', int(nn[i])))
            for t in range(3):
                f.write(struct.pack('<iii', int(npos[i, t, 0]),
                                    int(npos[i, t, 1]), int(ncol[i, t])))
            f.write(struct.pack('<iif', int(t_mv[i]), int(arg[i]),
                                float(c['weight'][i])))
    print(f'\nwrote {CPP}/probe_vh2_states.bin: {len(rows)} still-contested '
          f'rows (teacher vs vh2-current move, judge under vh2)')
    import os
    os.remove(tmp)


if __name__ == '__main__':
    main()
