"""Batched Color Lines engine primitives (numpy, vectorized over K boards).

Foundation for the array-based batched-tree MCTS (docs/batched_mcts_plan.md). Each
primitive must be BIT-IDENTICAL to its scalar game/board.py + alphatrain/mcts.py
counterpart for the same inputs — verified by scripts/test_batched_engine.py.

Board batch: int8 [K, 9, 9], 0 = empty, 1..7 = ball color.
"""
import numpy as np
from game.config import MIN_LINE_LENGTH, BALLS_PER_TURN, NUM_COLORS

BOARD_SIZE = 9
_NCELL = BOARD_SIZE * BOARD_SIZE
_BIG = np.int16(32767)
# cell id grid (1..81), used to seed connected-component labels
_CELL_ID = (np.arange(BOARD_SIZE * BOARD_SIZE, dtype=np.int16).reshape(
    BOARD_SIZE, BOARD_SIZE) + 1)


def label_components(boards):
    """Connected components of EMPTY cells, batched.

    Returns labels int16 [K,9,9]: 0 where a ball sits, else a component id shared
    by all empty cells in the same 4-connected region. Label VALUES differ from the
    scalar BFS (here = min cell-id in the component) but the PARTITION is identical,
    which is all reachability needs. Iterative min-label propagation to convergence.
    """
    boards = np.asarray(boards)
    empty = boards == 0
    labels = np.where(empty, _CELL_ID, np.int16(0))
    for _ in range(BOARD_SIZE * BOARD_SIZE):
        # Balls = _BIG so min() ignores them; empties carry their current label.
        work = np.where(empty, labels, _BIG)
        m = work.copy()
        # min with the 4 neighbours (shift; borders padded with _BIG)
        m[:, 1:, :] = np.minimum(m[:, 1:, :], work[:, :-1, :])   # up
        m[:, :-1, :] = np.minimum(m[:, :-1, :], work[:, 1:, :])  # down
        m[:, :, 1:] = np.minimum(m[:, :, 1:], work[:, :, :-1])   # left
        m[:, :, :-1] = np.minimum(m[:, :, :-1], work[:, :, 1:])  # right
        new = np.where(empty, m, np.int16(0))
        if np.array_equal(new, labels):
            break
        labels = new
    return labels


_LINE_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def clear_lines_at(boards, rows, cols, active=None):
    """Clear 5+ lines through (rows[k], cols[k]) on each board, IN PLACE.

    Bit-identical to scalar _clear_lines_at: 4 directions, contiguous same-color run
    each way, mark the whole line (incl. center) when length >= MIN_LINE_LENGTH, union
    across directions, set marked cells to 0. Returns cleared-count int64 [K].
    `active` (bool [K], default all) gates which trees are processed — inactive trees
    are untouched (center may be anything).
    """
    boards = np.asarray(boards)
    K = boards.shape[0]
    kar = np.arange(K)
    rows = np.asarray(rows); cols = np.asarray(cols)
    color = boards[kar, rows, cols].astype(np.int16)     # [K]
    active = (color > 0) if active is None else (active & (color > 0))
    # Pad by P=8 (max line offset) with a -1 border that never matches any color,
    # so neighbour gathers need no clip / in-bounds mask (np.clip was the hot spot).
    P = BOARD_SIZE - 1
    SZP = BOARD_SIZE + 2 * P
    pad = np.full((K, SZP, SZP), -1, dtype=np.int16)
    pad[:, P:P + BOARD_SIZE, P:P + BOARD_SIZE] = boards
    rp, cp = rows + P, cols + P                          # center in padded coords
    clear_pad = np.zeros((K, SZP, SZP), dtype=bool)
    kar2 = kar[:, None]
    color2 = color[:, None]
    active2 = active[:, None]
    offs = np.arange(1, BOARD_SIZE)[None, :]             # [1,8] line offsets

    for dr, dc in _LINE_DIRS:
        length = active.astype(np.int32)                 # center counts as 1 if active
        cells = []                                       # (rr, cc, run) padded coords, [K,8]
        for sign in (1, -1):
            rr = rp[:, None] + (sign * dr) * offs          # [K,8]
            cc = cp[:, None] + (sign * dc) * offs
            same = (pad[kar2, rr, cc] == color2) & active2
            run = np.cumprod(same, axis=1).astype(bool)    # cumulative AND along offsets
            length += run.sum(axis=1)
            cells.append((rr, cc, run))
        long = length >= MIN_LINE_LENGTH                 # [K], implies active
        clear_pad[kar, rp, cp] |= long
        long2 = long[:, None]
        for rr, cc, run in cells:
            clear_pad[kar2, rr, cc] |= (run & long2)       # distinct cells per row -> safe

    clear_mask = clear_pad[:, P:P + BOARD_SIZE, P:P + BOARD_SIZE]
    n_clear = clear_mask.reshape(K, -1).sum(axis=1).astype(np.int64)
    boards[clear_mask] = 0
    return n_clear


def _random_empty_order(boards, rng):
    """Per tree, cells ranked by a uniform-random key with non-empty cells last.
    order[k] is a permutation of 0..80; the first n_empty[k] entries are a uniform
    random selection of distinct empty cells. Returns order [K,81], n_empty [K]."""
    K = boards.shape[0]
    empty = (boards == 0).reshape(K, _NCELL)
    keys = np.where(empty, rng.random((K, _NCELL)), -1.0)
    order = np.argsort(-keys, axis=1)        # empties first (random), then balls
    return order, empty.sum(axis=1)


def apply_move(boards, next_pos, next_col, next_n, src, tgt, rng,
               num_colors=NUM_COLORS, active=None):
    """Batched trusted_move: move ball src->tgt, clear at tgt; if nothing cleared,
    spawn this turn's balls (displaced to a random empty if blocked), clear at each
    landing, regenerate next balls, and flag boards that filled up. IN PLACE on boards.

    `active` (bool [K], default all) gates which trees move — inactive trees are
    untouched and return game_over=False with their next balls unchanged.

    Deterministic parts (move + line clears) are bit-identical to scalar trusted_move;
    spawn DRAWS use `rng` (a numpy Generator) and need not match scalar (open-loop
    determinization). Score is intentionally not tracked (leaf value = feature eval).

    next_pos int [K,3,2], next_col int [K,3], next_n int [K]. Returns
    (game_over bool[K], new_pos[K,3,2], new_col[K,3], new_n[K]).
    """
    boards = np.asarray(boards)
    K = boards.shape[0]
    kar = np.arange(K)
    act0 = np.ones(K, dtype=bool) if active is None else np.asarray(active)
    src = np.asarray(src); tgt = np.asarray(tgt)
    sr, sc, tr, tc = src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1]
    color = np.where(act0, boards[kar, sr, sc], 0).astype(boards.dtype)
    mv = kar[act0]
    boards[mv, sr[act0], sc[act0]] = 0
    boards[mv, tr[act0], tc[act0]] = color[act0]
    cleared = clear_lines_at(boards, tr, tc, active=act0)   # [K]
    spawn = act0 & (cleared == 0)                       # spawn only if active & no clear

    for i in range(next_pos.shape[1]):                  # up to BALLS_PER_TURN balls
        act = spawn & (i < next_n)
        if not act.any():
            continue
        pr = next_pos[:, i, 0].astype(np.int64)
        pc = next_pos[:, i, 1].astype(np.int64)
        pcol = next_col[:, i].astype(np.int8)
        intended_empty = boards[kar, pr, pc] == 0
        land_r, land_c = pr.copy(), pc.copy()
        disp = act & ~intended_empty                    # blocked -> displace
        if disp.any():
            order, n_emp = _random_empty_order(boards, rng)
            cell = order[:, 0]                           # a uniform random empty cell
            has_empty = n_emp > 0
            use = disp & has_empty
            land_r = np.where(use, cell // BOARD_SIZE, land_r)
            land_c = np.where(use, cell % BOARD_SIZE, land_c)
            disp = use
        place = (act & intended_empty) | disp
        kk = kar[place]
        boards[kk, land_r[place], land_c[place]] = pcol[place]
        clear_lines_at(boards, land_r, land_c, active=place)

    game_over = np.zeros(K, dtype=bool)
    new_pos, new_col, new_n = (next_pos.copy(), next_col.copy(), next_n.copy())
    if spawn.any():
        filled = (boards.reshape(K, _NCELL) != 0).all(axis=1)
        game_over = spawn & filled
        # regenerate next balls for spawn trees (open-loop sampler)
        order, n_emp = _random_empty_order(boards, rng)
        want = np.minimum(BALLS_PER_TURN, n_emp).astype(next_n.dtype)
        cols = rng.integers(1, num_colors + 1, size=(K, BALLS_PER_TURN)).astype(next_col.dtype)
        for i in range(BALLS_PER_TURN):
            sel = spawn & (i < want)
            cell = order[:, i]
            new_pos[sel, i, 0] = (cell // BOARD_SIZE)[sel]
            new_pos[sel, i, 1] = (cell % BOARD_SIZE)[sel]
            new_col[sel, i] = cols[sel, i]
        new_n[spawn] = want[spawn]
    return game_over, new_pos, new_col, new_n


def reachable_many(labels, src, tgt):
    """Reachability for W candidate moves per tree. labels [K,9,9],
    src/tgt int [K,W,2]. Returns bool [K,W] (same rule as reachable())."""
    labels = np.asarray(labels)
    K, W = src.shape[0], src.shape[1]
    kk = np.arange(K)[:, None]                       # [K,1] broadcast
    tr, tc = tgt[..., 0], tgt[..., 1]
    tgt_label = labels[kk, tr, tc]                    # [K,W]
    valid = tgt_label > 0
    sr, sc = src[..., 0], src[..., 1]
    out = np.zeros((K, W), dtype=bool)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = sr + dr, sc + dc
        inb = (nr >= 0) & (nr < BOARD_SIZE) & (nc >= 0) & (nc < BOARD_SIZE)
        nrc = np.clip(nr, 0, BOARD_SIZE - 1)
        ncc = np.clip(nc, 0, BOARD_SIZE - 1)
        out |= inb & valid & (labels[kk, nrc, ncc] == tgt_label)
    return out


def _gather_rc(labels, rc):
    """labels[k, rc[k,0], rc[k,1]] for each k. rc int [K,2]."""
    k = np.arange(labels.shape[0])
    return labels[k, rc[:, 0], rc[:, 1]]


def reachable(labels, src_rc, tgt_rc):
    """Batched _is_reachable: tgt empty AND some 4-neighbour of src shares tgt's
    component. labels [K,9,9], src_rc/tgt_rc int [K,2]. Returns bool [K]."""
    labels = np.asarray(labels)
    src_rc = np.asarray(src_rc)
    tgt_rc = np.asarray(tgt_rc)
    K = labels.shape[0]
    k = np.arange(K)
    tgt_label = labels[k, tgt_rc[:, 0], tgt_rc[:, 1]]      # int16 [K]
    out = np.zeros(K, dtype=bool)
    valid_tgt = tgt_label > 0                              # tgt must be empty
    sr, sc = src_rc[:, 0], src_rc[:, 1]
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = sr + dr, sc + dc
        inb = (nr >= 0) & (nr < BOARD_SIZE) & (nc >= 0) & (nc < BOARD_SIZE)
        nr_c = np.clip(nr, 0, BOARD_SIZE - 1)
        nc_c = np.clip(nc, 0, BOARD_SIZE - 1)
        nlabel = labels[k, nr_c, nc_c]
        out |= inb & valid_tgt & (nlabel == tgt_label)
    return out
