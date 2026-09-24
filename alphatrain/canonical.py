"""Canonical form of a Color Lines position under the game's exact symmetries:
the 8 board rotations/reflections (D4) x relabelings of the 7 colors (S7).

For each D4 view v: transform board + preview cells; relabel colors by order of first
appearance scanning the transformed board (cells 0..80), then the preview balls sorted
by cell; key = relabeled board (81 bytes) + (cell, color) of the sorted preview balls,
padded with (255, 0). The canonical view v* is the smallest key (ties -> smallest v).
A net that only ever sees canonical inputs is exactly invariant to both symmetry groups.
Mirrors inference_cpp/src/canonical.h — keep the two identical (tests compare them)."""
import numpy as np
from numba import njit

_M = 8
_F = [lambda r, c: (r, c), lambda r, c: (c, _M - r), lambda r, c: (_M - r, _M - c), lambda r, c: (_M - c, r),
      lambda r, c: (r, _M - c), lambda r, c: (_M - r, c), lambda r, c: (c, r), lambda r, c: (_M - c, _M - r)]
CELL = np.array([[f(r, c)[0] * 9 + f(r, c)[1] for r in range(9) for c in range(9)] for f in _F], dtype=np.int64)
ACT = (CELL[:, :, None] * 81 + CELL[:, None, :]).reshape(8, 6561)   # original action -> action in view v


@njit(cache=True)
def _canon_one(board, pcell, pcol, n, cell_map):
    best_key = np.full(87, 255, dtype=np.int64); best_v = -1
    best_board = np.zeros(81, dtype=np.int8); best_pc = np.zeros(3, dtype=np.int64); best_pcol = np.zeros(3, dtype=np.int64)
    for v in range(8):
        bv = np.zeros(81, dtype=np.int8)
        for i in range(81):
            bv[cell_map[v, i]] = board[i]
        pc = np.zeros(3, dtype=np.int64); pk = np.zeros(3, dtype=np.int64)
        for t in range(n):
            pc[t] = cell_map[v, pcell[t]]; pk[t] = pcol[t]
        # sort preview balls by transformed cell (n <= 3: insertion sort)
        for a in range(1, n):
            j = a
            while j > 0 and pc[j - 1] > pc[j]:
                tmp = pc[j]; pc[j] = pc[j - 1]; pc[j - 1] = tmp
                tmp = pk[j]; pk[j] = pk[j - 1]; pk[j - 1] = tmp
                j -= 1
        cmap = np.zeros(8, dtype=np.int64); nxt = 1
        for i in range(81):
            c = bv[i]
            if c > 0 and cmap[c] == 0:
                cmap[c] = nxt; nxt += 1
        for t in range(n):
            c = pk[t]
            if c > 0 and cmap[c] == 0:
                cmap[c] = nxt; nxt += 1
        for c in range(1, 8):
            if cmap[c] == 0:
                cmap[c] = nxt; nxt += 1
        key = np.full(87, 255, dtype=np.int64)
        for i in range(81):
            key[i] = cmap[bv[i]] if bv[i] > 0 else 0
        for t in range(3):
            if t < n:
                key[81 + 2 * t] = pc[t]; key[82 + 2 * t] = cmap[pk[t]]
            else:
                key[81 + 2 * t] = 255; key[82 + 2 * t] = 0
        better = False
        for i in range(87):
            if key[i] != best_key[i]:
                better = key[i] < best_key[i]
                break
        if best_v < 0 or better:
            best_v = v
            for i in range(87):
                best_key[i] = key[i]
            for i in range(81):
                best_board[i] = key[i]
            for t in range(3):
                best_pc[t] = pc[t] if t < n else 0
                best_pcol[t] = cmap[pk[t]] if t < n else 0
    return best_v, best_board, best_pc, best_pcol


def canonicalize(board, next_balls):
    """board: (9,9) or (81,) int array; next_balls: list of (r, c, color).
    Returns (v, canon_board (9,9) int8, canon_next [(r, c, color)...] sorted by cell)."""
    b = np.asarray(board, dtype=np.int8).reshape(81)
    n = len(next_balls)
    pcell = np.zeros(3, dtype=np.int64); pcol = np.zeros(3, dtype=np.int64)
    for t, (r, c, col) in enumerate(next_balls):
        pcell[t] = r * 9 + c; pcol[t] = col
    v, cb, pc, pk = _canon_one(b, pcell, pcol, n, CELL)
    return int(v), cb.reshape(9, 9).copy(), [(int(pc[t]) // 9, int(pc[t]) % 9, int(pk[t])) for t in range(n)]
