"""Export the 27-feature linear leaf-value evaluator for the C++ MCTS.

Writes:
  data/feature_value.bin  : magic 'CLFV' + 27 coefs + 27 means + 27 stds + 1 bias (all <f4)
  data/golden_feature.bin : magic 'CLFG' + int32 K + per case:
        board[81] <f4 (0..7), int32 n_next, next[9] <f4 (r,c,col x3),
        feats[25] <f4 (board_features_with_next), V <f4 (_evaluate_features_linear)

The C++ feature_value.cc must reproduce feats + V bit-close. Uses the SAME
weights file the validated MCTS run used (feature_value_weights_2y_nb.npz, 27-feat).

    python -m alphatrain.inference_cpp.export_feature_weights
"""
import os, struct
import numpy as np

from alphatrain.mcts import _evaluate_features_linear
from alphatrain.scripts.mine_death_features import board_features_with_next

WEIGHTS = 'alphatrain/data/feature_value_weights_2y_nb.npz'
OUTW = 'alphatrain/inference_cpp/data/feature_value.bin'
OUTG = 'alphatrain/inference_cpp/data/golden_feature.bin'


def rand_board(rng, fill):
    b = np.zeros((9, 9), dtype=np.int8)
    for r in range(9):
        for c in range(9):
            if rng.random() < fill:
                b[r, c] = rng.integers(1, 8)
    return b


def rand_next(rng, board, allow_occupied):
    if allow_occupied:
        cells = [(r, c) for r in range(9) for c in range(9)]
    else:
        cells = [(r, c) for r in range(9) for c in range(9) if board[r, c] == 0]
    rng.shuffle(cells)
    n = min(3, len(cells))
    return [(cells[i][0], cells[i][1], int(rng.integers(1, 8))) for i in range(n)]


def main():
    d = np.load(WEIGHTS)
    coefs = d['coefs'].astype(np.float32)
    means = d['means'].astype(np.float32)
    stds = d['stds'].astype(np.float32)
    bias = np.float32(d['bias'])
    os.makedirs(os.path.dirname(OUTW), exist_ok=True)
    with open(OUTW, 'wb') as f:
        f.write(b'CLFV')
        f.write(coefs.astype('<f4').tobytes())
        f.write(means.astype('<f4').tobytes())
        f.write(stds.astype('<f4').tobytes())
        f.write(struct.pack('<f', float(bias)))

    rng = np.random.default_rng(777)
    cases = []
    for k in range(60):
        board = rand_board(rng, rng.uniform(0.1, 0.9))
        nb = rand_next(rng, board, allow_occupied=(k % 2 == 0))  # half exercise blocked spawns
        nr = np.array([x[0] for x in nb] + [0] * (3 - len(nb)), dtype=np.int64)
        nc = np.array([x[1] for x in nb] + [0] * (3 - len(nb)), dtype=np.int64)
        ncol = np.array([x[2] for x in nb] + [0] * (3 - len(nb)), dtype=np.int64)
        nn = len(nb)
        feats = np.array(board_features_with_next(board, nr, nc, ncol, nn), dtype=np.float64)
        V = float(_evaluate_features_linear(board, nr, nc, ncol, nn, coefs, means, stds, bias))
        cases.append((board, nb, feats, V))

    with open(OUTG, 'wb') as f:
        f.write(b'CLFG')
        f.write(struct.pack('<i', len(cases)))
        for board, nb, feats, V in cases:
            f.write(board.astype('<f4').tobytes())
            f.write(struct.pack('<i', len(nb)))
            flat = np.zeros(9, dtype='<f4')
            for i, (r, c, col) in enumerate(nb):
                flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2] = r, c, col
            f.write(flat.tobytes())
            f.write(feats.astype('<f4').tobytes())
            f.write(struct.pack('<f', V))

    print(f'wrote {OUTW} (27 coefs/means/stds + bias)')
    print(f'wrote {OUTG} ({len(cases)} cases)')
    print('sample V:', [round(c[3], 3) for c in cases[:8]])


if __name__ == '__main__':
    main()
