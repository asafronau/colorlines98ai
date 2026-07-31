"""Dump golden vectors for the C++ game engine: obs, legal mask, line-clear.

These are the DETERMINISTIC kernels (no RNG), so the C++ port must reproduce
them bit-for-bit. Output: data/golden_game.bin (format read by game_test.cc).

    python -m alphatrain.inference_cpp.export_game_golden
"""
import os, struct
import numpy as np

from game.board import ColorLinesGame, _clear_lines_at
from game.config import BOARD_SIZE
from alphatrain.observation import build_observation

OUT = 'alphatrain/inference_cpp/data/golden_game.bin'


def legal_mask(game):
    m = np.zeros(81 * 81, dtype=np.float32)
    for (sr, sc), (tr, tc) in game.get_legal_moves():
        m[(sr * 9 + sc) * 81 + (tr * 9 + tc)] = 1.0
    return m


def obs_for(board, next_balls):
    nr = np.zeros(3, dtype=np.int64); nc = np.zeros(3, dtype=np.int64)
    ncol = np.zeros(3, dtype=np.int64)
    nn = min(len(next_balls), 3)
    for i in range(nn):
        (r, c), col = next_balls[i]
        nr[i], nc[i], ncol[i] = r, c, col
    return build_observation(board.astype(np.int8), nr, nc, ncol, nn)


def rand_board(rng, fill):
    b = np.zeros((9, 9), dtype=np.int8)
    for r in range(9):
        for c in range(9):
            if rng.random() < fill:
                b[r, c] = rng.integers(1, 8)
    return b


def rand_next(rng, board):
    empty = [(r, c) for r in range(9) for c in range(9) if board[r, c] == 0]
    rng.shuffle(empty)
    nb = []
    for i in range(min(3, len(empty))):
        nb.append((empty[i], int(rng.integers(1, 8))))
    return nb


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rng = np.random.default_rng(12345)

    obs_cases, clear_cases = [], []

    # obs + legal cases: varied densities
    for _ in range(40):
        fill = rng.uniform(0.15, 0.85)
        board = rand_board(rng, fill)
        nb = rand_next(rng, board)
        g = ColorLinesGame()
        g.reset(board=board.copy(), next_balls=list(nb))
        obs_cases.append((board, nb, obs_for(board, nb), legal_mask(g)))

    # clear cases: plant lines of various lengths + a no-line control
    for length in (5, 5, 6, 7, 4):  # 4 = control (no clear)
        board = np.zeros((9, 9), dtype=np.int8)
        r0 = int(rng.integers(0, 9)); c0 = int(rng.integers(0, 9 - length + 1))
        color = int(rng.integers(1, 8))
        for c in range(c0, c0 + length):
            board[r0, c] = color
        # sprinkle noise elsewhere
        for _ in range(10):
            rr, cc = int(rng.integers(0, 9)), int(rng.integers(0, 9))
            if board[rr, cc] == 0 and not (rr == r0 and c0 <= cc < c0 + length):
                board[rr, cc] = int(rng.integers(1, 8))
        bin_ = board.copy()
        cleared = _clear_lines_at(board.copy(), r0, c0)  # on a copy
        bout = board.copy()
        _clear_lines_at(bout, r0, c0)
        clear_cases.append((bin_, r0, c0, cleared, bout))

    with open(OUT, 'wb') as f:
        f.write(b'CLGM')
        f.write(struct.pack('<i', len(obs_cases)))
        for board, nb, obs, legal in obs_cases:
            f.write(board.astype('<f4').tobytes())
            f.write(struct.pack('<i', min(len(nb), 3)))
            flat = np.zeros(9, dtype='<f4')
            for i in range(min(len(nb), 3)):
                (r, c), col = nb[i]
                flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2] = r, c, col
            f.write(flat.tobytes())
            f.write(obs.astype('<f4').tobytes())
            f.write(legal.astype('<f4').tobytes())
        f.write(struct.pack('<i', len(clear_cases)))
        for bin_, r, c, cleared, bout in clear_cases:
            f.write(bin_.astype('<f4').tobytes())
            f.write(struct.pack('<iii', r, c, cleared))
            f.write(bout.astype('<f4').tobytes())

    print(f'wrote {OUT}: {len(obs_cases)} obs/legal cases, '
          f'{len(clear_cases)} clear cases')
    print('clear cleared-counts:', [cc[3] for cc in clear_cases])


if __name__ == '__main__':
    main()
