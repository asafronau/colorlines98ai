"""Afterstate computation for pairwise value training.

An afterstate is the board AFTER executing a move (ball moved + lines cleared)
but BEFORE new balls spawn. This is deterministic given (board, move).

The pairwise approach trains the value head to rank afterstates:
V(good_afterstate) > V(bad_afterstate) + margin
"""

import numpy as np
from numba import njit
from game.config import BOARD_SIZE, MIN_LINE_LENGTH


@njit(cache=True)
def compute_afterstate(board, sr, sc, tr, tc):
    """Execute move on board copy, return (afterstate_board, score_delta).

    Moves ball from (sr,sc) to (tr,tc), clears any lines formed.
    Does NOT spawn new balls (that's stochastic).
    """
    after = board.copy()
    after[tr, tc] = after[sr, sc]
    after[sr, sc] = 0

    # Check lines through target
    color = after[tr, tc]
    if color == 0:
        return after, 0

    clear_r = np.empty(81, dtype=np.int64)
    clear_c = np.empty(81, dtype=np.int64)
    n_clear = 0

    dirs_dr = (0, 1, 1, 1)
    dirs_dc = (1, 0, 1, -1)

    for di in range(4):
        dr = dirs_dr[di]
        dc = dirs_dc[di]
        line_r = np.empty(9, dtype=np.int64)
        line_c = np.empty(9, dtype=np.int64)
        line_r[0] = tr
        line_c[0] = tc
        n = 1
        r, c = tr + dr, tc + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and after[r, c] == color:
            line_r[n] = r
            line_c[n] = c
            n += 1
            r += dr
            c += dc
        r, c = tr - dr, tc - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and after[r, c] == color:
            line_r[n] = r
            line_c[n] = c
            n += 1
            r -= dr
            c -= dc
        if n >= MIN_LINE_LENGTH:
            for i in range(n):
                already = False
                for j in range(n_clear):
                    if clear_r[j] == line_r[i] and clear_c[j] == line_c[i]:
                        already = True
                        break
                if not already:
                    clear_r[n_clear] = line_r[i]
                    clear_c[n_clear] = line_c[i]
                    n_clear += 1

    for i in range(n_clear):
        after[clear_r[i], clear_c[i]] = 0

    # Score: n * (n - 4) for n >= 5
    score = n_clear * (n_clear - 4) if n_clear >= MIN_LINE_LENGTH else 0
    return after, score
