"""Export a train_crisis_ft corpus (advfilt*.pt) as a CLRJ judge bin, using the
STORED teacher/base moves — for independent rejudging with fresh seeds.

    python -m alphatrain.scripts.export_ftcorpus_judge \
        --corpus alphatrain/data/advfilt2.pt \
        --out alphatrain/inference_cpp/data/rejudge_r2.bin
"""
import argparse
import struct

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    c = torch.load(a.corpus, map_location='cpu', weights_only=False)
    n = c['boards'].shape[0]
    t_mv = c['tgt_idx'][:, 0].numpy()
    b_mv = c['vh1_move'].numpy()
    boards = c['boards'].numpy().reshape(n, 81)
    npos, ncol, nn = c['next_pos'].numpy(), c['next_col'].numpy(), c['n_next'].numpy()
    w = c['weight'].numpy()
    with open(a.out, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', n))
        for i in range(n):
            f.write(boards[i].astype(np.int8).tobytes())
            f.write(struct.pack('<i', int(nn[i])))
            for t in range(3):
                f.write(struct.pack('<iii', int(npos[i, t, 0]),
                                    int(npos[i, t, 1]), int(ncol[i, t])))
            f.write(struct.pack('<iif', int(t_mv[i]), int(b_mv[i]), float(w[i])))
    print(f'{a.out}: {n} rows (original weight mean {w.mean():.3f})')


if __name__ == '__main__':
    main()
