"""Observation builder for AlphaTrain.

Builds a multi-channel tensor representation of a Color Lines 98 board state.
Designed for CNN input — includes tactical features that standard CNNs can't
learn efficiently from raw pixels.

Channels:
  0-6:  One-hot color planes (7 colors)
  7:    Empty cells
  8-10: Next ball positions (color/7.0 per ball)
  11:   Next ball mask (1.0 where a next ball will spawn)
  12:   Component area heatmap (empty cell = component_size / 81)
  13-16: Line potential (H, V, D1, D2) — same-color count per direction
  17:   Max line length at each cell (max across 4 directions)

Total: 18 channels, (18, 9, 9) float32
"""

import numpy as np
from numba import njit
from game.config import BOARD_SIZE, NUM_COLORS

NUM_CHANNELS = 18


@njit(cache=True)
def _line_length_at(board, r, c, dr, dc):
    """Count same-color balls in one direction from (r,c), including (r,c)."""
    color = board[r, c]
    if color == 0:
        return 0
    count = 1
    # Forward
    nr, nc = r + dr, c + dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        count += 1
        nr += dr
        nc += dc
    # Backward
    nr, nc = r - dr, c - dc
    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
        count += 1
        nr -= dr
        nc -= dc
    return count


@njit(cache=True)
def _component_sizes(board):
    """Compute connected component sizes for empty cells.

    Returns (9, 9) int32 array where each empty cell contains the size
    of its connected component. Ball cells are 0.
    """
    labels = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    sizes = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    queue_r = np.empty(81, dtype=np.int32)
    queue_c = np.empty(81, dtype=np.int32)
    current = 0

    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if board[sr, sc] != 0 or labels[sr, sc] != 0:
                continue
            current += 1
            labels[sr, sc] = current
            queue_r[0] = sr
            queue_c[0] = sc
            head = 0
            tail = 1
            while head < tail:
                r = queue_r[head]
                c = queue_c[head]
                head += 1
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr, nc] == 0 and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            queue_r[tail] = nr
                            queue_c[tail] = nc
                            tail += 1

    # Count sizes per component
    comp_sizes = np.zeros(current + 1, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if labels[r, c] > 0:
                comp_sizes[labels[r, c]] += 1

    # Map each cell to its component size
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if labels[r, c] > 0:
                sizes[r, c] = comp_sizes[labels[r, c]]

    return sizes


@njit(cache=True)
def build_line_potentials_batch(boards, obs_out):
    """Compute line potential channels 13-17 for N boards in one JIT call.

    obs_out: (N, 18, 9, 9) — writes channels 13-17 in-place.
    Eliminates the Python loop that was calling _line_length_at per cell.
    """
    N = boards.shape[0]
    dirs_dr = (0, 1, 1, 1)
    dirs_dc = (1, 0, 1, -1)
    for bi in range(N):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = boards[bi, r, c]
                if color == 0:
                    continue
                max_len = 0
                for di in range(4):
                    dr = dirs_dr[di]
                    dc = dirs_dc[di]
                    length = _line_length_at(boards[bi], r, c, dr, dc)
                    obs_out[bi, 13 + di, r, c] = length / 9.0
                    if length > max_len:
                        max_len = length
                obs_out[bi, 17, r, c] = max_len / 9.0


@njit(cache=True)
def build_observation(board, next_r, next_c, next_color, n_next):
    """Build (18, 9, 9) observation tensor from board state.

    All channels are computed in a single JIT call for maximum performance.
    """
    obs = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Channels 0-6: one-hot color planes
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r, c]
            if v == 0:
                obs[7, r, c] = 1.0  # Channel 7: empty
            else:
                obs[v - 1, r, c] = 1.0  # Channels 0-6: colors

    # Channels 8-10: next ball color, Channel 11: next ball mask
    for i in range(n_next):
        obs[8 + i, next_r[i], next_c[i]] = next_color[i] / 7.0
        obs[11, next_r[i], next_c[i]] = 1.0

    # Channel 12: component area heatmap
    comp_sizes = _component_sizes(board)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if comp_sizes[r, c] > 0:
                obs[12, r, c] = comp_sizes[r, c] / 81.0

    # Channels 13-16: line potentials (H, V, D1, D2)
    # Channel 17: max line length
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))  # H, V, D1, D2
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                continue
            max_len = 0
            for di in range(4):
                dr, dc = dirs[di][0], dirs[di][1]
                length = _line_length_at(board, r, c, dr, dc)
                obs[13 + di, r, c] = length / 9.0  # normalized to [0, 1]
                if length > max_len:
                    max_len = length
            obs[17, r, c] = max_len / 9.0

    return obs
