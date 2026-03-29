"""Fast heuristic player — Numba JIT-accelerated move evaluation.

Evaluates all legal moves by temporarily modifying the board,
computing line potential at the target, and restoring. All hot
loops are JIT-compiled for near-C performance.
"""

import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList
from .config import BOARD_SIZE, MIN_LINE_LENGTH
from .board import (ColorLinesGame, _label_empty_components, _get_source_mask,
                    _get_target_mask, _clear_lines_at, _get_empty_array,
                    _count_empty)

# Default heuristic weights (17 params):
# Line: [clear_mult, clear_base, partial_pow2, partial_linear, break_penalty]
# Combo: [combo_bonus, line4_bonus]
# Spatial: [center_dist, edge_bonus, same_color_n, diff_color_pen, empty_n]
# Anti-frag: [hole_penalty, connected_region, empty_count]
# Source: [source_center_bonus, move_distance]
# Starting from Phase 11 CMA-ES values for line weights, spatial starts at 0
DEFAULT_WEIGHTS = np.array([
    14.6, 109.4, 5.7, 1.38, 2.4,   # w[0..4]: line (CMA-ES tuned)
    0.0, 0.0,                        # w[5..6]: combo, line4
    0.0, 0.0, 0.0, 0.0, 0.0,       # w[7..11]: spatial
    0.0, 0.0, 0.0,                   # w[12..14]: anti-frag
    0.0, 0.0,                        # w[15..16]: source
], dtype=np.float64)

# Mutable module-level weights used by get_best_move_fast / get_softmax_move_fast.
# Updated by set_weights() for CMA-ES optimization.
_ACTIVE_WEIGHTS = DEFAULT_WEIGHTS.copy()

# Default rollout softmax temperature (CMA-ES optimized)
_ACTIVE_TEMPERATURE = 3.23

# ML oracle weights (None = disabled, set via enable_ml_oracle())
_ML_WEIGHTS = None
_ML_BLEND = 0.05  # Blend factor: low value = tiebreaker, high = ML dominates (breaks deep rollouts)


def enable_ml_oracle(weights_path='checkpoints/linear_oracle.npz', blend=0.3):
    """Enable ML oracle blending in heuristic evaluation."""
    global _ML_WEIGHTS, _ML_BLEND
    data = np.load(weights_path)
    _ML_WEIGHTS = data['weights'].astype(np.float64)
    _ML_BLEND = blend
    print(f"ML oracle enabled: {len(_ML_WEIGHTS)} weights, blend={blend}", flush=True)


def disable_ml_oracle():
    """Disable ML oracle, use pure heuristic."""
    global _ML_WEIGHTS
    _ML_WEIGHTS = None


def set_weights(weights, temperature=None):
    """Set heuristic weights for CMA-ES optimization.

    Args:
        weights: array of 5 floats [clear_mult, clear_base, partial_pow2,
                 partial_linear, break_penalty]
        temperature: optional softmax temperature override
    """
    global _ACTIVE_WEIGHTS, _ACTIVE_TEMPERATURE
    _ACTIVE_WEIGHTS = np.array(weights, dtype=np.float64)
    if temperature is not None:
        _ACTIVE_TEMPERATURE = float(temperature)


def get_weights():
    """Get current active weights."""
    return _ACTIVE_WEIGHTS.copy(), _ACTIVE_TEMPERATURE


# Precompute direction arrays for numba
_DIRS_DR = np.array([0, 1, 1, 1], dtype=np.int8)
_DIRS_DC = np.array([1, 0, 1, -1], dtype=np.int8)


@njit(cache=True)
def _line_length(board, r, c, color, dr, dc):
    """Line length through (r,c) in direction (dr,dc)."""
    length = 1
    cr, cc = r + dr, c + dc
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
        length += 1
        cr += dr
        cc += dc
    cr, cc = r - dr, c - dc
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
        length += 1
        cr -= dr
        cc -= dc
    return length


@njit(cache=True)
def _max_line_at(board, r, c, color):
    """Max line length through (r,c) in any of 4 directions."""
    best = 1
    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        length = _line_length(board, r, c, color, dr, dc)
        if length > best:
            best = length
    return best


@njit(cache=True)
def _empty_extends(board, r, c, color, dr, dc):
    """Count empty cells at both ends of a same-color run through (r,c)."""
    extends = 0
    # Go to positive end of the run
    cr, cc = r + dr, c + dc
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
        cr += dr
        cc += dc
    # Count empties past positive end
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == 0:
        extends += 1
        cr += dr
        cc += dc
    # Go to negative end
    cr, cc = r - dr, c - dc
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
        cr -= dr
        cc -= dc
    while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == 0:
        extends += 1
        cr -= dr
        cc -= dc
    return extends


@njit(cache=True)
def _score_for_clear(n):
    """Game score for clearing n balls."""
    if n < MIN_LINE_LENGTH:
        return 0
    return n * (n - 4)


@njit(cache=True)
def _total_clearable(board, r, c, color):
    """Count total unique balls that would be cleared by lines of 5+ through (r,c).

    Unlike _max_line_at which only returns the longest single line, this counts
    ALL balls across ALL intersecting lines >= 5 (e.g., a cross of 5h + 5v = 9 balls).
    """
    clear_r = np.empty(36, dtype=np.int8)
    clear_c = np.empty(36, dtype=np.int8)
    n_clear = 0

    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        # Count line length in this direction
        line_r = np.empty(9, dtype=np.int8)
        line_c = np.empty(9, dtype=np.int8)
        line_r[0] = r
        line_c[0] = c
        n = 1
        cr, cc = r + dr, c + dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
            line_r[n] = cr
            line_c[n] = cc
            n += 1
            cr += dr
            cc += dc
        cr, cc = r - dr, c - dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
            line_r[n] = cr
            line_c[n] = cc
            n += 1
            cr -= dr
            cc -= dc
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
    return n_clear


@njit(cache=True)
def _count_empty_jit(board):
    """Count empty cells on board (JIT version for use in evaluate)."""
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                n += 1
    return n


@njit(cache=True)
def _flood_fill_size(board, start_r, start_c):
    """Count connected empty cells from a starting empty cell. Fast BFS."""
    if board[start_r, start_c] != 0:
        return 0
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    queue_r = np.empty(81, dtype=np.int8)
    queue_c = np.empty(81, dtype=np.int8)
    queue_r[0] = start_r
    queue_c[0] = start_c
    visited[start_r, start_c] = 1
    head = 0
    tail = 1
    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if board[nr, nc] == 0 and visited[nr, nc] == 0:
                    visited[nr, nc] = 1
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1
    return tail


@njit(cache=True)
def _evaluate_move_w(board, sr, sc, tr, tc, color, w, empty_total=-1):
    """Super heuristic with ~20 tunable weights for CMA-ES discovery.

    Line features (w[0..4]):
      w[0]:  clear_multiplier — scales game points for clearing
      w[1]:  clear_base — flat bonus for any clear
      w[2]:  partial_pow2 — (length-1)^2 for extendable partials
      w[3]:  partial_linear — (length-1) for dead-end partials
      w[4]:  break_penalty — penalty for disrupting source-side partials

    Combo features (w[5..6]):
      w[5]:  combo_bonus — bonus for multi-line clears (crosses)
      w[6]:  line4_bonus — bonus for creating a 4-in-a-row (one away from clearing)

    Spatial features (w[7..11]):
      w[7]:  center_distance — reward placing away from center (highway rule)
      w[8]:  edge_bonus — bonus for placing on the edge of the board
      w[9]:  same_color_neighbors — reward clustering same color (zoning)
      w[10]: diff_color_penalty — penalty for adjacent different colors (anti-scatter)
      w[11]: empty_neighbors — reward having empty neighbors (breathing room)

    Anti-fragmentation (w[12..14]):
      w[12]: hole_penalty — penalty for trapping cells into dead pockets
      w[13]: connected_region — reward for keeping large connected empty regions
      w[14]: empty_count — reward total empty cells (survival pressure)

    Source features (w[15..16]):
      w[15]: source_center_bonus — bonus for REMOVING a ball from the center
      w[16]: move_distance — reward/penalty for move distance (prefer short moves?)
    """
    board[sr, sc] = 0
    board[tr, tc] = color

    score = 0.0

    # === LINE FEATURES ===

    # 1) Clearable line at target
    max_line = _max_line_at(board, tr, tc, color)
    if max_line >= MIN_LINE_LENGTH:
        score += _score_for_clear(max_line) * w[0] + w[1]

    # 2) Combo: cheap cross detection (count directions with lines >= 5)
    if w[5] != 0.0:
        n_clear_dirs = 0
        for di in range(4):
            if _line_length(board, tr, tc, color, _DIRS_DR[di], _DIRS_DC[di]) >= MIN_LINE_LENGTH:
                n_clear_dirs += 1
        if n_clear_dirs >= 2:
            score += n_clear_dirs * w[5]

    # 3) Partial line potential at target
    has_line4 = False
    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        length = _line_length(board, tr, tc, color, dr, dc)
        if length == 4:
            has_line4 = True
        if length >= 2:
            ext = _empty_extends(board, tr, tc, color, dr, dc)
            if length + ext >= MIN_LINE_LENGTH:
                score += (length - 1) ** 2 * w[2]
            else:
                score += (length - 1) * w[3]

    # 4) Line-of-4 bonus (one away from clearing — very valuable)
    if has_line4:
        score += w[6]

    # 5) Penalty for breaking partial lines at source
    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        for sign in (1, -1):
            nr = sr + sign * dr
            nc = sc + sign * dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
                broken_len = _line_length(board, nr, nc, color, dr, dc)
                old_len = broken_len + 1
                if old_len >= 3:
                    score -= (old_len - 1) * w[4]

    # === SPATIAL FEATURES (skip if all weights zero) ===
    if w[7] != 0.0 or w[8] != 0.0 or w[9] != 0.0 or w[10] != 0.0 or w[11] != 0.0:
        center = BOARD_SIZE // 2
        dist_to_center = abs(tr - center) + abs(tc - center)
        score += dist_to_center * w[7]
        on_edge = 1 if (tr == 0 or tr == BOARD_SIZE - 1 or tc == 0 or tc == BOARD_SIZE - 1) else 0
        score += on_edge * w[8]
        same_color_n = 0
        diff_color_n = 0
        empty_n = 0
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = tr + dr, tc + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                v = board[nr, nc]
                if v == 0:
                    empty_n += 1
                elif v == color:
                    same_color_n += 1
                else:
                    diff_color_n += 1
        score += same_color_n * w[9]
        score -= diff_color_n * w[10]
        score += empty_n * w[11]

    # === ANTI-FRAGMENTATION (skip if all weights zero) ===
    if w[12] != 0.0 or w[13] != 0.0 or w[14] != 0.0:
        if w[12] != 0.0:
            holes_created = 0
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = tr + dr, tc + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
                    surrounded = True
                    for dr2, dc2 in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        nr2, nc2 = nr + dr2, nc + dc2
                        if 0 <= nr2 < BOARD_SIZE and 0 <= nc2 < BOARD_SIZE and board[nr2, nc2] == 0:
                            surrounded = False
                            break
                    if surrounded:
                        holes_created += 1
            score -= holes_created * w[12]
        if w[13] != 0.0:
            local_empty = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = tr + dr, tc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
                        local_empty += 1
            score += local_empty * w[13]
        if w[14] != 0.0:
            if empty_total < 0:
                empty_total = _count_empty_jit(board)
            score += empty_total * w[14]

    # === SOURCE FEATURES (skip if all weights zero) ===
    if w[15] != 0.0 or w[16] != 0.0:
        center = BOARD_SIZE // 2
        src_dist = abs(sr - center) + abs(sc - center)
        score += (center - src_dist) * w[15]
        move_dist = abs(tr - sr) + abs(tc - sc)
        score += move_dist * w[16]

    # Restore
    board[sr, sc] = color
    board[tr, tc] = 0
    return score


@njit(cache=True)
def _evaluate_move(board, sr, sc, tr, tc, color):
    """Evaluate a move with default weights (CMA-ES optimized + spatial features)."""
    return _evaluate_move_w(board, sr, sc, tr, tc, color,
                            np.array([14.6, 109.4, 5.7, 1.38, 2.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


@njit(cache=True)
def _evaluate_move_with_next(board, sr, sc, tr, tc, color,
                              next_r, next_c, next_color, n_next):
    """Evaluate a move with next-ball awareness.

    Same as _evaluate_move, plus:
    4) Bonus if next-ball spawns complete/extend lines
    5) Penalty for moving to a next-ball spawn position
    6) Extra bonus for clearing (clears skip spawns entirely)
    """
    board[sr, sc] = 0
    board[tr, tc] = color

    score = 0.0
    clears_line = False

    # 1) Clearable line at target?
    max_line = _max_line_at(board, tr, tc, color)
    if max_line >= MIN_LINE_LENGTH:
        score += _score_for_clear(max_line) * 10.0 + 100.0
        clears_line = True

    # 2) Partial line potential at target
    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        length = _line_length(board, tr, tc, color, dr, dc)
        if length >= 2:
            ext = _empty_extends(board, tr, tc, color, dr, dc)
            if length + ext >= MIN_LINE_LENGTH:
                score += (length - 1) ** 2 * 2.0
            else:
                score += (length - 1) * 0.3

    # 3) Penalty for breaking partial lines at source
    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        for sign in (1, -1):
            nr = sr + sign * dr
            nc = sc + sign * dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == color:
                broken_len = _line_length(board, nr, nc, color, dr, dc)
                old_len = broken_len + 1
                if old_len >= 3:
                    score -= (old_len - 1) * 1.5

    # 4) Next-ball synergy: if this move doesn't clear, spawns happen.
    #    Check if spawned balls would create/extend lines.
    if not clears_line and n_next > 0:
        for i in range(n_next):
            nr = next_r[i]
            nc = next_c[i]
            nc_color = next_color[i]
            # Only score if spawn position is still empty (ball might land there)
            if board[nr, nc] == 0:
                # Check line length if this ball spawns
                spawn_line = _max_line_at_color(board, nr, nc, nc_color)
                if spawn_line + 1 >= MIN_LINE_LENGTH:
                    # Next ball will complete a clearable line!
                    score += _score_for_clear(spawn_line + 1) * 5.0 + 30.0
                elif spawn_line >= 2:
                    # Next ball extends a partial line (3 or 4 in a row)
                    score += spawn_line * 3.0

    # 5) Penalty: moving to a cell where a ball is about to spawn
    if not clears_line:
        for i in range(n_next):
            if tr == next_r[i] and tc == next_c[i]:
                score -= 15.0
                break

    # 6) Extra clear bonus: clearing skips spawns (keeps board emptier)
    if clears_line:
        score += n_next * 5.0

    # Restore
    board[sr, sc] = color
    board[tr, tc] = 0
    return score


@njit(cache=True)
def _max_line_at_color(board, r, c, color):
    """Max line length through (r,c) counting only existing same-color neighbors.
    Does NOT count the cell itself (used for spawn prediction)."""
    best = 0
    for di in range(4):
        dr = _DIRS_DR[di]
        dc = _DIRS_DC[di]
        length = 0
        # Positive direction
        cr, cc = r + dr, c + dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
            length += 1
            cr += dr
            cc += dc
        # Negative direction
        cr, cc = r - dr, c - dc
        while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
            length += 1
            cr -= dr
            cc -= dc
        if length > best:
            best = length
    return best


@njit(cache=True)
def _find_best_move_with_next(board, source_cells, target_masks,
                               next_r, next_c, next_color, n_next):
    """Find best move using next-ball-aware evaluation."""
    best_score = -1e9
    best_sr = 0
    best_sc = 0
    best_tr = 0
    best_tc = 0

    for si in range(source_cells.shape[0]):
        sr = source_cells[si, 0]
        sc = source_cells[si, 1]
        color = board[sr, sc]

        for tr in range(BOARD_SIZE):
            for tc in range(BOARD_SIZE):
                if target_masks[si, tr, tc] == 0.0:
                    continue
                score = _evaluate_move_with_next(
                    board, sr, sc, tr, tc, color,
                    next_r, next_c, next_color, n_next)
                if score > best_score:
                    best_score = score
                    best_sr = sr
                    best_sc = sc
                    best_tr = tr
                    best_tc = tc

    return best_sr, best_sc, best_tr, best_tc, best_score


@njit(cache=True)
def _find_best_move_w(board, source_cells, target_masks, w):
    """Find best move with tunable weights."""
    best_score = -1e9
    best_sr = 0
    best_sc = 0
    best_tr = 0
    best_tc = 0
    empty_total = _count_empty_jit(board)

    for si in range(source_cells.shape[0]):
        sr = source_cells[si, 0]
        sc = source_cells[si, 1]
        color = board[sr, sc]

        for tr in range(BOARD_SIZE):
            for tc in range(BOARD_SIZE):
                if target_masks[si, tr, tc] == 0.0:
                    continue
                score = _evaluate_move_w(board, sr, sc, tr, tc, color, w, empty_total)
                if score > best_score:
                    best_score = score
                    best_sr = sr
                    best_sc = sc
                    best_tr = tr
                    best_tc = tc

    return best_sr, best_sc, best_tr, best_tc, best_score


@njit(cache=True)
def _find_best_move(board, source_cells, target_masks):
    """Find best move with default weights (CMA-ES optimized + spatial features)."""
    return _find_best_move_w(board, source_cells, target_masks,
                             np.array([14.6, 109.4, 5.7, 1.38, 2.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


@njit(cache=True)
def _score_all_moves_w(board, source_cells, target_masks, moves_out, scores_out, w):
    """Score all legal moves with tunable weights."""
    empty_total = _count_empty_jit(board)
    n = 0
    for si in range(source_cells.shape[0]):
        sr = source_cells[si, 0]
        sc = source_cells[si, 1]
        color = board[sr, sc]
        for tr in range(BOARD_SIZE):
            for tc in range(BOARD_SIZE):
                if target_masks[si, tr, tc] == 0.0:
                    continue
                moves_out[n, 0] = sr
                moves_out[n, 1] = sc
                moves_out[n, 2] = tr
                moves_out[n, 3] = tc
                scores_out[n] = _evaluate_move_w(board, sr, sc, tr, tc, color, w, empty_total)
                n += 1
    return n


@njit(cache=True)
def _score_all_moves(board, source_cells, target_masks, moves_out, scores_out):
    """Score all legal moves with default weights (CMA-ES optimized + spatial features)."""
    return _score_all_moves_w(board, source_cells, target_masks, moves_out, scores_out,
                              np.array([14.6, 109.4, 5.7, 1.38, 2.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


# Pre-allocated buffers for _score_all_moves (max 81 sources × 80 targets)
_MOVES_BUF = np.empty((6480, 4), dtype=np.intp)
_SCORES_BUF = np.empty(6480, dtype=np.float64)


def _get_sources_and_targets(game):
    """Compute sources array and target masks. Shared by best/softmax move functions."""
    game._ensure_cc()
    from .board import _get_source_mask, _get_target_mask
    source_mask = _get_source_mask(game.board)
    # Build sources array without np.argwhere (avoid tuple conversion overhead)
    sources = np.empty((81, 2), dtype=np.intp)
    n_src = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if source_mask[r, c] > 0:
                sources[n_src, 0] = r
                sources[n_src, 1] = c
                n_src += 1
    if n_src == 0:
        return None, None
    sources = sources[:n_src]
    target_masks = np.zeros((n_src, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    cc = game._cc_labels
    for i in range(n_src):
        target_masks[i] = _get_target_mask(cc, sources[i, 0], sources[i, 1])
    return sources, target_masks


def get_best_move_fast(game: ColorLinesGame, use_next_balls: bool = False):
    """Fast best-move selection using JIT-compiled evaluation.

    If ML oracle is enabled, blends heuristic + ML scores for move selection.
    """
    sources, target_masks = _get_sources_and_targets(game)
    if sources is None:
        return None

    if _ML_WEIGHTS is not None:
        # Hybrid: heuristic + ML oracle (batched JIT)
        from .features import score_all_moves_linear
        next_r = np.zeros(3, dtype=np.intp)
        next_c = np.zeros(3, dtype=np.intp)
        next_color = np.zeros(3, dtype=np.intp)
        n_next = len(game.next_balls)
        for i, ((r, c), col) in enumerate(game.next_balls):
            next_r[i] = r
            next_c[i] = c
            next_color[i] = col

        n = _score_all_moves_w(game.board, sources, target_masks,
                                _MOVES_BUF, _SCORES_BUF, _ACTIVE_WEIGHTS)
        if n == 0:
            return None

        # Get ML scores for all moves (single JIT call)
        ml_scores = np.empty(n, dtype=np.float64)
        score_all_moves_linear(game.board, _MOVES_BUF, n,
                                next_r, next_c, next_color, n_next,
                                _ML_WEIGHTS, ml_scores)

        # Normalize and blend
        h = _SCORES_BUF[:n].copy()
        h_norm = (h - h.mean()) / (h.std() + 1e-8)
        m_norm = (ml_scores - ml_scores.mean()) / (ml_scores.std() + 1e-8)
        combined = h_norm + m_norm * _ML_BLEND

        best_idx = int(np.argmax(combined))
        m = _MOVES_BUF[best_idx]
        return ((int(m[0]), int(m[1])), (int(m[2]), int(m[3])))

    if use_next_balls and len(game.next_balls) > 0:
        n_next = len(game.next_balls)
        next_r = np.zeros(3, dtype=np.intp)
        next_c = np.zeros(3, dtype=np.intp)
        next_color = np.zeros(3, dtype=np.intp)
        for i, ((r, c), col) in enumerate(game.next_balls):
            next_r[i] = r
            next_c[i] = c
            next_color[i] = col
        sr, sc, tr, tc, _ = _find_best_move_with_next(
            game.board, sources, target_masks,
            next_r, next_c, next_color, n_next)
    else:
        sr, sc, tr, tc, _ = _find_best_move_w(game.board, sources, target_masks,
                                               _ACTIVE_WEIGHTS)

    return ((int(sr), int(sc)), (int(tr), int(tc)))


def get_softmax_move_fast(game: ColorLinesGame, temperature: float, rng):
    """Select a move by sampling from softmax over heuristic scores.

    Uses _ACTIVE_WEIGHTS for scoring. temperature controls exploration.
    """
    sources, target_masks = _get_sources_and_targets(game)
    if sources is None:
        return None

    n = _score_all_moves_w(game.board, sources, target_masks, _MOVES_BUF, _SCORES_BUF,
                           _ACTIVE_WEIGHTS)
    if n == 0:
        return None

    scores = _SCORES_BUF[:n].copy()
    scores /= temperature
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()

    idx = rng.choice(n, p=probs)
    m = _MOVES_BUF[idx]
    return ((int(m[0]), int(m[1])), (int(m[2]), int(m[3])))


# ── JIT Rollout ────────────────────────────────────────────────────────
# Full rollout in compiled code — no Python re-entry.

@njit(cache=True)
def _jit_rollout(board, next_r, next_c, next_color, n_next,
                 sr, sc, tr, tc, depth, temperature, weights, seed):
    """Run a full rollout in JIT. Returns score gained.

    Executes initial move (sr,sc)->(tr,tc), then plays `depth` random moves
    using softmax over heuristic scores. All game logic (line clearing,
    ball spawning, pathfinding) is inlined — no Python callbacks.
    """
    b = board.copy()
    next_r = next_r.copy()
    next_c = next_c.copy()
    next_color = next_color.copy()
    np.random.seed(seed)
    score = 0
    game_over = False

    # --- Execute initial move ---
    color = b[sr, sc]
    b[sr, sc] = 0
    b[tr, tc] = color

    cleared = _clear_lines_at(b, tr, tc)
    if cleared > 0:
        pts = cleared * (cleared - 4) if cleared >= 5 else 0
        score += pts
    else:
        # Spawn next balls
        for i in range(n_next):
            r, c = next_r[i], next_c[i]
            if b[r, c] == 0:
                b[r, c] = next_color[i]
            else:
                empty = _get_empty_array(b)
                if len(empty) > 0:
                    idx = np.random.randint(0, len(empty))
                    b[empty[idx, 0], empty[idx, 1]] = next_color[i]
        # Check lines at spawn positions
        for i in range(n_next):
            r, c = next_r[i], next_c[i]
            if b[r, c] != 0:
                sc2 = _clear_lines_at(b, r, c)
                if sc2 > 0:
                    pts = sc2 * (sc2 - 4) if sc2 >= 5 else 0
                    score += pts
        # Generate new next balls
        empty = _get_empty_array(b)
        n_empty = len(empty)
        if n_empty == 0:
            game_over = True
        else:
            nn = min(3, n_empty)
            for i in range(3):
                if i < nn:
                    idx = np.random.randint(0, n_empty)
                    next_r[i] = empty[idx, 0]
                    next_c[i] = empty[idx, 1]
                    next_color[i] = np.random.randint(1, 8)
                else:
                    next_r[i] = 0
                    next_c[i] = 0
                    next_color[i] = 0
            n_next = nn

    # Pre-allocate rollout buffers
    moves_buf = np.empty((6480, 4), dtype=np.intp)
    scores_buf = np.empty(6480, dtype=np.float64)

    # --- Rollout loop ---
    for _step in range(depth):
        if game_over:
            break

        # Get sources and targets
        cc = _label_empty_components(b)
        src_mask = _get_source_mask(b)

        # Score all moves
        empty_total = _count_empty_jit(b)
        n_moves = 0
        for si_r in range(9):
            for si_c in range(9):
                if src_mask[si_r, si_c] == 0.0:
                    continue
                tgt_mask = _get_target_mask(cc, si_r, si_c)
                c2 = b[si_r, si_c]
                for t_r in range(9):
                    for t_c in range(9):
                        if tgt_mask[t_r, t_c] == 0.0:
                            continue
                        moves_buf[n_moves, 0] = si_r
                        moves_buf[n_moves, 1] = si_c
                        moves_buf[n_moves, 2] = t_r
                        moves_buf[n_moves, 3] = t_c
                        scores_buf[n_moves] = _evaluate_move_w(
                            b, si_r, si_c, t_r, t_c, c2, weights, empty_total)
                        n_moves += 1

        if n_moves == 0:
            break

        # Softmax sampling
        max_s = scores_buf[0]
        for i in range(1, n_moves):
            if scores_buf[i] > max_s:
                max_s = scores_buf[i]
        total = 0.0
        for i in range(n_moves):
            scores_buf[i] = np.exp((scores_buf[i] - max_s) / temperature)
            total += scores_buf[i]

        threshold = np.random.random() * total
        cumsum = 0.0
        chosen = 0
        for i in range(n_moves):
            cumsum += scores_buf[i]
            if cumsum >= threshold:
                chosen = i
                break

        # Execute chosen move
        m_sr = moves_buf[chosen, 0]
        m_sc = moves_buf[chosen, 1]
        m_tr = moves_buf[chosen, 2]
        m_tc = moves_buf[chosen, 3]
        m_color = b[m_sr, m_sc]
        b[m_sr, m_sc] = 0
        b[m_tr, m_tc] = m_color

        cleared = _clear_lines_at(b, m_tr, m_tc)
        if cleared > 0:
            pts = cleared * (cleared - 4) if cleared >= 5 else 0
            score += pts
        else:
            # Spawn
            for i in range(n_next):
                r, c = next_r[i], next_c[i]
                if b[r, c] == 0:
                    b[r, c] = next_color[i]
                else:
                    empty = _get_empty_array(b)
                    if len(empty) > 0:
                        idx = np.random.randint(0, len(empty))
                        b[empty[idx, 0], empty[idx, 1]] = next_color[i]
            # Check spawn clears
            for i in range(n_next):
                r, c = next_r[i], next_c[i]
                if b[r, c] != 0:
                    sc2 = _clear_lines_at(b, r, c)
                    if sc2 > 0:
                        pts = sc2 * (sc2 - 4) if sc2 >= 5 else 0
                        score += pts
            # New next balls
            empty = _get_empty_array(b)
            n_empty = len(empty)
            if n_empty == 0:
                game_over = True
            else:
                nn = min(3, n_empty)
                for i in range(3):
                    if i < nn:
                        idx = np.random.randint(0, n_empty)
                        next_r[i] = empty[idx, 0]
                        next_c[i] = empty[idx, 1]
                        next_color[i] = np.random.randint(1, 8)
                    else:
                        next_r[i] = 0
                        next_c[i] = 0
                        next_color[i] = 0
                n_next = nn

    return float(score)


def do_jit_rollout(game, source, target, depth, temperature, weights, seed):
    """Python wrapper for _jit_rollout. Extracts arrays from game object."""
    next_r = np.zeros(3, dtype=np.int64)
    next_c = np.zeros(3, dtype=np.int64)
    next_color = np.zeros(3, dtype=np.int64)
    n_next = len(game.next_balls)
    for i, ((r, c), col) in enumerate(game.next_balls):
        next_r[i] = r
        next_c[i] = c
        next_color[i] = col
    return _jit_rollout(game.board, next_r, next_c, next_color, n_next,
                        source[0], source[1], target[0], target[1],
                        depth, temperature, weights, seed)
