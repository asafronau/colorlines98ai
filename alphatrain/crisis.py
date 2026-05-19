"""Fast structural crisis detection for Color Lines boards."""

import numpy as np
from numba import njit


@njit(cache=True)
def crisis_features(board):
    """Return (empty, components, largest_component, avg_reach, low_mob_balls).

    avg_reach is the average number of empty target cells reachable by each
    ball, using empty connected components. It is the strongest mined
    near-death feature and combines emptiness with board connectivity.
    """
    labels = np.zeros((9, 9), dtype=np.int8)
    queue = np.empty(81, dtype=np.int32)
    comp_count = np.zeros(82, dtype=np.int32)

    current = np.int8(0)
    empty = 0
    largest = 0

    for sr in range(9):
        for sc in range(9):
            if board[sr, sc] != 0 or labels[sr, sc] != 0:
                continue

            current += 1
            labels[sr, sc] = current
            queue[0] = sr * 9 + sc
            head = 0
            tail = 1

            while head < tail:
                pos = queue[head]
                head += 1
                r = pos // 9
                c = pos % 9
                for d in range(4):
                    if d == 0:
                        nr = r
                        nc = c + 1
                    elif d == 1:
                        nr = r
                        nc = c - 1
                    elif d == 2:
                        nr = r + 1
                        nc = c
                    else:
                        nr = r - 1
                        nc = c
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        if board[nr, nc] == 0 and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            queue[tail] = nr * 9 + nc
                            tail += 1

            comp_count[current] = tail
            empty += tail
            if tail > largest:
                largest = tail

    mobility = 0
    balls = 0
    low_mobility = 0

    for r in range(9):
        for c in range(9):
            if board[r, c] == 0:
                continue

            balls += 1
            reachable = 0
            seen = np.zeros(82, dtype=np.int8)
            for d in range(4):
                if d == 0:
                    nr = r
                    nc = c + 1
                elif d == 1:
                    nr = r
                    nc = c - 1
                elif d == 2:
                    nr = r + 1
                    nc = c
                else:
                    nr = r - 1
                    nc = c
                if 0 <= nr < 9 and 0 <= nc < 9:
                    lbl = labels[nr, nc]
                    if lbl > 0 and seen[lbl] == 0:
                        seen[lbl] = 1
                        reachable += comp_count[lbl]

            mobility += reachable
            if reachable < 5:
                low_mobility += 1

    avg_reach = 0.0
    if balls > 0:
        avg_reach = mobility / balls

    return empty, int(current), largest, avg_reach, low_mobility


def crisis_level(board, prevention_avg_reach=10.0, prevention_components=5,
                 emergency_avg_reach=5.0, emergency_empty=20,
                 emergency_components=6):
    """Classify a board as None, 'prevention', or 'emergency'."""
    empty, n_components, _largest, avg_reach, _low_mob = crisis_features(board)
    if avg_reach < emergency_avg_reach:
        return 'emergency'
    if empty < emergency_empty and n_components >= emergency_components:
        return 'emergency'
    if avg_reach < prevention_avg_reach and \
            n_components >= prevention_components:
        return 'prevention'
    return None
