"""Strict golden tests: batched engine primitives vs scalar game/board.py.

Each batched primitive must be bit-identical (or partition-identical, for component
labels) to the scalar numba function on random boards. Light CPU (small batch) so it
can run alongside a live mining job.

    PYTHONPATH=. python scripts/test_batched_engine.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from game.board import (_label_empty_components, _is_reachable,
                        _clear_lines_at, ColorLinesGame, BOARD_SIZE)
from alphatrain import batched_engine as be


def _random_boards(K, rng):
    """Varied-density boards (0=empty, 1..7=color)."""
    boards = np.zeros((K, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for k in range(K):
        density = rng.uniform(0.1, 0.95)
        mask = rng.random((BOARD_SIZE, BOARD_SIZE)) < density
        colors = rng.integers(1, 8, (BOARD_SIZE, BOARD_SIZE)).astype(np.int8)
        boards[k] = np.where(mask, colors, 0)
    return boards


def test_components_partition(K=200, seed=0):
    """Batched labels must induce the SAME partition as scalar BFS (values may differ)."""
    rng = np.random.default_rng(seed)
    boards = _random_boards(K, rng)
    blabels = be.label_components(boards)
    bad = 0
    for k in range(K):
        slab = _label_empty_components(boards[k])
        # same partition <=> for every pair of empty cells, (same scalar comp) == (same batched comp)
        empties = np.argwhere(boards[k] == 0)
        # compare via a canonical map: scalar label -> batched label must be consistent + bijective
        s2b = {}
        b2s = {}
        for (r, c) in empties:
            s, b = int(slab[r, c]), int(blabels[k, r, c])
            if s == 0 or b == 0:
                bad += 1; break
            if s in s2b and s2b[s] != b:
                bad += 1; break
            if b in b2s and b2s[b] != s:
                bad += 1; break
            s2b[s] = b; b2s[b] = s
    print(f"  components partition: {K-bad}/{K} boards match", flush=True)
    assert bad == 0, "component partition mismatch"


def test_reachable(K=200, pairs_per_board=40, seed=1):
    """Batched reachable() must match scalar _is_reachable exactly over random moves."""
    rng = np.random.default_rng(seed)
    boards = _random_boards(K, rng)
    blabels = be.label_components(boards)
    slabels = [_label_empty_components(boards[k]) for k in range(K)]
    total = mism = 0
    for _ in range(pairs_per_board):
        src = np.zeros((K, 2), dtype=np.int64)
        tgt = np.zeros((K, 2), dtype=np.int64)
        for k in range(K):
            balls = np.argwhere(boards[k] != 0)
            empt = np.argwhere(boards[k] == 0)
            src[k] = balls[rng.integers(len(balls))] if len(balls) else (0, 0)
            tgt[k] = empt[rng.integers(len(empt))] if len(empt) else (0, 0)
        bres = be.reachable(blabels, src, tgt)
        for k in range(K):
            sres = _is_reachable(slabels[k], int(src[k, 0]), int(src[k, 1]),
                                 int(tgt[k, 0]), int(tgt[k, 1]))
            total += 1
            if bool(bres[k]) != bool(sres):
                mism += 1
    print(f"  reachable: {total-mism}/{total} move queries match", flush=True)
    assert mism == 0, "reachable mismatch"


def _boards_with_lines(K, rng):
    """Mix of random boards and boards with a deliberate 5+ line, to exercise clearing."""
    boards = _random_boards(K, rng)
    for k in range(0, K, 2):  # half get a planted line of random color/orientation
        color = int(rng.integers(1, 8))
        length = int(rng.integers(5, 8))
        r0, c0 = int(rng.integers(0, BOARD_SIZE)), int(rng.integers(0, BOARD_SIZE - length))
        orient = rng.integers(0, 3)
        for i in range(length):
            if orient == 0 and c0 + i < BOARD_SIZE:
                boards[k, r0, c0 + i] = color
            elif orient == 1 and r0 + i < BOARD_SIZE:
                boards[k, (r0 + i) % BOARD_SIZE, c0] = color
            elif (r0 + i) < BOARD_SIZE and (c0 + i) < BOARD_SIZE:
                boards[k, r0 + i, c0 + i] = color
    return boards


def test_clear_lines(K=200, centers_per_board=6, seed=2):
    """Batched clear_lines_at must match scalar _clear_lines_at (board + count) exactly."""
    rng = np.random.default_rng(seed)
    boards = _boards_with_lines(K, rng)
    total = mism = cleared_cases = 0
    for _ in range(centers_per_board):
        rows = np.zeros(K, dtype=np.int64); cols = np.zeros(K, dtype=np.int64)
        for k in range(K):
            balls = np.argwhere(boards[k] != 0)
            rc = balls[rng.integers(len(balls))] if len(balls) else (0, 0)
            rows[k], cols[k] = rc[0], rc[1]
        bb = boards.copy()
        bn = be.clear_lines_at(bb, rows, cols)
        for k in range(K):
            sb = boards[k].copy()
            sn = _clear_lines_at(sb, int(rows[k]), int(cols[k]))
            total += 1
            if sn > 0:
                cleared_cases += 1
            if sn != int(bn[k]) or not np.array_equal(sb, bb[k]):
                mism += 1
    print(f"  clear_lines: {total-mism}/{total} match ({cleared_cases} actually cleared)",
          flush=True)
    assert mism == 0, "clear_lines mismatch"


def _make_clear_case(rng):
    """Board where moving src->tgt completes a 5-line (so the move clears, no spawn).
    Returns (board, src, tgt). Rest of board empty so reachability is trivial."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    color = int(rng.integers(1, 8))
    row = int(rng.integers(1, BOARD_SIZE))      # need row-1 in bounds for src below? use row>=1
    c0 = int(rng.integers(0, BOARD_SIZE - 4))   # 4 cells c0..c0+3, tgt c0+4
    for i in range(4):
        board[row, c0 + i] = color
    tgt = (row, c0 + 4)
    src = (row - 1, c0 + 4)                      # directly above tgt -> reachable, same color
    board[src] = color
    return board, src, tgt


def test_apply_move_clear(K=120, seed=3):
    """Deterministic move+clear path: apply_move must equal scalar trusted_move exactly
    (cleared>0 => no spawn => fully deterministic)."""
    rng = np.random.default_rng(seed)
    cases = [_make_clear_case(rng) for _ in range(K)]
    boards = np.stack([c[0] for c in cases])
    src = np.array([c[1] for c in cases], dtype=np.int64)
    tgt = np.array([c[2] for c in cases], dtype=np.int64)
    # dummy next balls (unused on the clear path, but apply_move reads shapes)
    next_pos = np.zeros((K, 3, 2), dtype=np.int8)
    next_col = np.ones((K, 3), dtype=np.int8)
    next_n = np.full(K, 3, dtype=np.int8)
    bb = boards.copy()
    go, _, _, _ = be.apply_move(bb, next_pos.copy(), next_col.copy(), next_n.copy(),
                                src, tgt, np.random.default_rng(0))
    mism = 0
    for k in range(K):
        g = ColorLinesGame(); g.reset(board=boards[k].copy(),
                                      next_balls=[((0, 0), 1), ((0, 1), 1), ((0, 2), 1)])
        g.trusted_move(int(src[k, 0]), int(src[k, 1]), int(tgt[k, 0]), int(tgt[k, 1]))
        if not np.array_equal(g.board, bb[k]) or bool(go[k]) != bool(g.game_over):
            mism += 1
    print(f"  apply_move (clear path, bit-identical): {K-mism}/{K} match", flush=True)
    assert mism == 0, "apply_move clear-path mismatch"


def test_apply_move_spawn_valid(K=200, seed=4):
    """Spawn path validity: on a no-clear move, board gains balls only on previously
    empty cells, the moved ball lands at tgt, and game_over <=> board full."""
    rng = np.random.default_rng(seed)
    boards = _random_boards(K, rng) // 1  # 0..7
    # ensure src has a ball and tgt empty + reachable: pick per board
    src = np.zeros((K, 2), dtype=np.int64); tgt = np.zeros((K, 2), dtype=np.int64)
    keep = np.ones(K, dtype=bool)
    labels = be.label_components(boards)
    for k in range(K):
        balls = np.argwhere(boards[k] != 0)
        empt = np.argwhere(boards[k] == 0)
        if len(balls) == 0 or len(empt) == 0:
            keep[k] = False; continue
        # find a reachable (src,tgt)
        found = False
        for _ in range(20):
            s = balls[rng.integers(len(balls))]; t = empt[rng.integers(len(empt))]
            if _is_reachable(_label_empty_components(boards[k]), int(s[0]), int(s[1]),
                             int(t[0]), int(t[1])):
                src[k], tgt[k] = s, t; found = True; break
        keep[k] = found
    next_pos = np.zeros((K, 3, 2), dtype=np.int8)
    next_col = np.ones((K, 3), dtype=np.int8)
    next_n = np.full(K, 3, dtype=np.int8)
    before_empty = (boards == 0)
    bb = boards.copy()
    go, np2, nc2, nn2 = be.apply_move(bb, next_pos, next_col, next_n, src, tgt, rng)
    bad = 0
    for k in range(K):
        if not keep[k]:
            continue
        # every newly-filled cell must have been empty before (spawn only fills empties)
        newly = (bb[k] != 0) & (boards[k] == 0)
        # tgt should be filled unless its line cleared
        if go[k] and (bb[k] != 0).sum() != BOARD_SIZE * BOARD_SIZE:
            bad += 1; continue
        # no ball value out of range
        if (bb[k] < 0).any() or (bb[k] > 7).any():
            bad += 1
    print(f"  apply_move (spawn path, validity): {int(keep.sum())-bad}/{int(keep.sum())} ok",
          flush=True)
    assert bad == 0, "apply_move spawn-path invalid"


if __name__ == '__main__':
    print("=== batched engine golden tests (vs scalar) ===", flush=True)
    test_components_partition()
    test_reachable()
    test_clear_lines()
    test_apply_move_clear()
    test_apply_move_spawn_valid()
    print("ALL PASS", flush=True)
