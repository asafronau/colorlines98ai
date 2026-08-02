"""Final-candidate experiment (review #6): re-derive the round-1 vector AT vh2.

Stage 'export': vh2's fp16 legal argmax on the 676 round-1 rows -> judge bin
(teacher move vs vh2's current move; absorbed rows skipped).
Stage 'corpus': from the fresh R=256 rejudge, keep rows with fresh uplift >=
0.08, weight = FRESH estimate, base move = vh2's move.

    python -m alphatrain.scripts.build_r1_at_vh2 --stage export
    python -m alphatrain.scripts.build_r1_at_vh2 --stage corpus
"""
import argparse
import csv
import struct

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.diag_dagger1_fp16 import fp16_argmax_margin

CPP = 'alphatrain/inference_cpp/data'
SRC = 'alphatrain/data/advfilt.pt'


def vh2_argmax(c, n):
    tmp = SRC + '.r1v2.tmp'
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
    import os
    os.remove(tmp)
    return arg


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['export', 'corpus'], required=True)
    a = p.parse_args()
    c = torch.load(SRC, map_location='cpu', weights_only=False)
    n = c['boards'].shape[0]
    t_mv = c['tgt_idx'][:, 0].numpy()
    arg = vh2_argmax(c, n)
    rows = np.where((arg >= 0) & (arg != t_mv))[0]

    if a.stage == 'export':
        boards = c['boards'].numpy().reshape(n, 81)
        npos, ncol, nn = (c['next_pos'].numpy(), c['next_col'].numpy(),
                          c['n_next'].numpy())
        with open(f'{CPP}/r1vh2_judge.bin', 'wb') as f:
            f.write(b'CLRJ')
            f.write(struct.pack('<i', len(rows)))
            for i in rows:
                f.write(boards[i].astype(np.int8).tobytes())
                f.write(struct.pack('<i', int(nn[i])))
                for t in range(3):
                    f.write(struct.pack('<iii', int(npos[i, t, 0]),
                                        int(npos[i, t, 1]), int(ncol[i, t])))
                f.write(struct.pack('<iif', int(t_mv[i]), int(arg[i]), 0.0))
        np.save(f'{CPP}/r1vh2_rows.npy', rows)
        print(f'{CPP}/r1vh2_judge.bin: {len(rows)} contested rows '
              f'({n - len(rows)} absorbed/invalid skipped)')
    else:
        rows_saved = np.load(f'{CPP}/r1vh2_rows.npy')
        with open(f'{CPP}/r1vh2_results.csv') as f:
            res = list(csv.DictReader(f))
        assert len(res) == len(rows_saved)
        fresh = np.array([float(r['base_died']) - float(r['teacher_died'])
                          for r in res])
        keep = fresh >= 0.08
        sel = rows_saved[keep]
        idx = torch.from_numpy(sel)
        out = {
            'boards': c['boards'][idx], 'next_pos': c['next_pos'][idx],
            'next_col': c['next_col'][idx], 'n_next': c['n_next'][idx],
            'tgt_idx': c['tgt_idx'][idx], 'tgt_prob': c['tgt_prob'][idx],
            'vh1_move': torch.from_numpy(arg[sel].astype(np.int64)),
            'weight': torch.from_numpy(fresh[keep].astype(np.float32)),
            'seed': torch.from_numpy(sel.astype(np.int64)),
            '_stats': {'n_seeds': len(sel),
                       'min_margin': 'r1 rows, FRESH vh2 uplift >= 0.08'},
        }
        torch.save(out, 'alphatrain/data/advfilt_r1vh2.pt')
        print(f'advfilt_r1vh2.pt: {len(sel)} rows (fresh mean '
              f'{fresh[keep].mean():.3f}; from {len(rows_saved)} contested)')


if __name__ == '__main__':
    main()
