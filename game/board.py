"""Color Lines 98 game engine.

9x9 board, 7 colors, lines of 5+ to clear.
Each turn: player moves a ball along a free path, then 3 new balls spawn
(unless the move cleared a line).

Hot paths are JIT-compiled with Numba for speed.
"""

import numpy as np
from numba import njit
from typing import Optional
from .config import BOARD_SIZE, NUM_COLORS, BALLS_PER_TURN, MIN_LINE_LENGTH

# Directions for line checking: horizontal, vertical, two diagonals
LINE_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def calculate_score(num_balls: int) -> int:
    """Score formula: n * (n - 4) for n >= 5. So 5->5, 6->12, 7->21, ..."""
    if num_balls < MIN_LINE_LENGTH:
        return 0
    return num_balls * (num_balls - 4)


# ── Numba JIT kernels ────────────────────────────────────────────────

@njit(cache=True)
def _label_empty_components(board):
    """Label connected components of empty cells. 0=ball, 1+=component id."""
    labels = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    queue_r = np.empty(81, dtype=np.int8)
    queue_c = np.empty(81, dtype=np.int8)
    current = np.int8(0)

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
    return labels


@njit(cache=True)
def _count_empty(board):
    """Count empty cells on board."""
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                n += 1
    return n


@njit(cache=True)
def _get_empty_array(board):
    """Return (N, 2) int64 array of (row, col) for empty cells. JIT replacement for np.argwhere."""
    out = np.empty((81, 2), dtype=np.int64)
    n = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                out[n, 0] = r
                out[n, 1] = c
                n += 1
    return out[:n]


@njit(cache=True)
def _get_source_mask(board):
    """Return float32 (9,9) mask: 1.0 where a ball has ≥1 adjacent empty cell."""
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                continue
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr, nc] == 0:
                    mask[r, c] = 1.0
                    break
    return mask


@njit(cache=True)
def _get_target_mask(cc_labels, sr, sc):
    """Return float32 (9,9) mask: 1.0 for empty cells reachable from source.

    Checks ALL adjacent empty cells — a ball can reach any connected component
    that touches it, not just the first one found.
    """
    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Collect all unique component labels adjacent to the source
    labels = np.zeros(4, dtype=np.int8)
    n_labels = 0
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and cc_labels[nr, nc] > 0:
            lbl = cc_labels[nr, nc]
            # Check if already seen
            found = False
            for i in range(n_labels):
                if labels[i] == lbl:
                    found = True
                    break
            if not found:
                labels[n_labels] = lbl
                n_labels += 1
    if n_labels == 0:
        return mask
    # Mark all cells belonging to any reachable component
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if cc_labels[r, c] > 0:
                for i in range(n_labels):
                    if cc_labels[r, c] == labels[i]:
                        mask[r, c] = 1.0
                        break
    return mask


@njit(cache=True)
def _find_lines_at(board, row, col):
    """Find total balls to clear in lines of 5+ through (row,col). Returns count."""
    color = board[row, col]
    if color == 0:
        return 0
    # Use a flat array to track positions to clear (max 4 lines × 9 = 36)
    clear_r = np.empty(36, dtype=np.int8)
    clear_c = np.empty(36, dtype=np.int8)
    n_clear = 0

    dirs_dr = np.array([0, 1, 1, 1], dtype=np.int8)
    dirs_dc = np.array([1, 0, 1, -1], dtype=np.int8)

    for di in range(4):
        dr = dirs_dr[di]
        dc = dirs_dc[di]
        # Collect all positions in this line
        line_r = np.empty(9, dtype=np.int8)
        line_c = np.empty(9, dtype=np.int8)
        line_r[0] = row
        line_c[0] = col
        n = 1
        # Positive direction
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            line_r[n] = r
            line_c[n] = c
            n += 1
            r += dr
            c += dc
        # Negative direction
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            line_r[n] = r
            line_c[n] = c
            n += 1
            r -= dr
            c -= dc
        if n >= MIN_LINE_LENGTH:
            for i in range(n):
                # Add if not already present
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
def _clear_lines_at(board, row, col):
    """Clear lines through (row,col), return number of balls cleared."""
    color = board[row, col]
    if color == 0:
        return 0

    clear_r = np.empty(36, dtype=np.int8)
    clear_c = np.empty(36, dtype=np.int8)
    n_clear = 0

    dirs_dr = np.array([0, 1, 1, 1], dtype=np.int8)
    dirs_dc = np.array([1, 0, 1, -1], dtype=np.int8)

    for di in range(4):
        dr = dirs_dr[di]
        dc = dirs_dc[di]
        line_r = np.empty(9, dtype=np.int8)
        line_c = np.empty(9, dtype=np.int8)
        line_r[0] = row
        line_c[0] = col
        n = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            line_r[n] = r
            line_c[n] = c
            n += 1
            r += dr
            c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
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

    # Actually clear
    for i in range(n_clear):
        board[clear_r[i], clear_c[i]] = 0

    return n_clear


@njit(cache=True)
def _is_reachable(cc_labels, sr, sc, tr, tc):
    """Check if target is reachable from source via empty cells."""
    if cc_labels[tr, tc] == 0:
        return False
    target_label = cc_labels[tr, tc]
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if cc_labels[nr, nc] == target_label:
                return True
    return False


@njit(cache=True)
def _get_observation(board, next_r, next_c, next_color, n_next):
    """Build 13-channel observation tensor.

    Channels 0-6: one-hot color planes
    Channel 7: empty cells
    Channels 8-10: next ball color (normalized)
    Channel 11: next ball mask
    Channel 12: largest connected empty region (reachability)
    """
    obs = np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r, c]
            if v == 0:
                obs[7, r, c] = 1.0
            else:
                obs[v - 1, r, c] = 1.0
    for i in range(n_next):
        obs[8 + i, next_r[i], next_c[i]] = next_color[i] / 7.0
        obs[11, next_r[i], next_c[i]] = 1.0

    # Channel 12: component area heatmap — each empty cell = component_size / 81
    # Bright = large open region, dim = small trapped pocket
    labels = _label_empty_components(board)
    counts = np.zeros(82, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            lbl = labels[r, c]
            if lbl > 0:
                counts[lbl] += 1
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            lbl = labels[r, c]
            if lbl > 0:
                obs[12, r, c] = counts[lbl] / 81.0

    return obs


# ── Game class ────────────────────────────────────────────────────────

class ColorLinesGame:
    """Color Lines 98 game state and logic."""

    def __init__(self, seed: Optional[int] = None, num_colors: int = NUM_COLORS):
        self.rng = np.random.default_rng(seed)
        self.num_colors = num_colors
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.next_balls: list[tuple[tuple[int, int], int]] = []
        self.score = 0
        self.game_over = False
        self.turns = 0
        self._cc_labels: Optional[np.ndarray] = None

    def reset(self, board: Optional[np.ndarray] = None,
              next_balls: Optional[list] = None) -> None:
        if board is not None:
            self.board = board.copy()
        else:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.score = 0
        self.game_over = False
        self.turns = 0
        if next_balls is not None:
            self.next_balls = list(next_balls)
        else:
            self._generate_next_balls()
        if board is None:
            self._spawn_balls()
            self._generate_next_balls()
        self._cc_labels = None

    def clone(self, rng=None) -> 'ColorLinesGame':
        g = ColorLinesGame.__new__(ColorLinesGame)
        g.rng = rng if rng is not None else np.random.default_rng()
        g.num_colors = self.num_colors
        g.board = self.board.copy()
        g.next_balls = list(self.next_balls)
        g.score = self.score
        g.game_over = self.game_over
        g.turns = self.turns
        g._cc_labels = None
        return g

    # ── CC cache ──────────────────────────────────────────────────────

    def _ensure_cc(self):
        if self._cc_labels is None:
            self._cc_labels = _label_empty_components(self.board)

    # ── Ball spawning ─────────────────────────────────────────────────

    def _get_empty_cells(self) -> list[tuple[int, int]]:
        """Legacy Python list version — use _get_empty_array JIT for hot paths."""
        empty = _get_empty_array(self.board)
        return [(int(r), int(c)) for r, c in empty]

    def _generate_next_balls(self):
        empty = _get_empty_array(self.board)
        n_empty = len(empty)
        if n_empty == 0:
            self.next_balls = []
            return
        n = min(BALLS_PER_TURN, n_empty)
        indices = self.rng.choice(n_empty, size=n, replace=False)
        colors = self.rng.integers(1, self.num_colors + 1, size=n)
        self.next_balls = [
            ((int(empty[indices[i], 0]), int(empty[indices[i], 1])), int(colors[i]))
            for i in range(n)
        ]

    def _spawn_balls(self) -> int:
        spawned = 0
        for (row, col), color in self.next_balls:
            if self.board[row, col] == 0:
                self.board[row, col] = color
                spawned += 1
            else:
                empty = _get_empty_array(self.board)
                if len(empty) > 0:
                    idx = self.rng.integers(0, len(empty))
                    self.board[empty[idx, 0], empty[idx, 1]] = color
                    spawned += 1
        self._cc_labels = None
        return spawned

    # ── Masks ─────────────────────────────────────────────────────────

    def get_source_mask(self) -> np.ndarray:
        return _get_source_mask(self.board)

    def get_target_mask(self, source: tuple[int, int]) -> np.ndarray:
        self._ensure_cc()
        return _get_target_mask(self._cc_labels, source[0], source[1])

    def get_legal_moves(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        self._ensure_cc()
        moves = []
        source_mask = self.get_source_mask()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if source_mask[r, c] == 0:
                    continue
                target_mask = self.get_target_mask((r, c))
                for tr in range(BOARD_SIZE):
                    for tc in range(BOARD_SIZE):
                        if target_mask[tr, tc] > 0:
                            moves.append(((r, c), (tr, tc)))
        return moves

    # ── Move execution ────────────────────────────────────────────────

    def move(self, source: tuple[int, int], target: tuple[int, int]) -> dict:
        result = {'valid': False, 'score': 0, 'cleared': 0, 'game_over': self.game_over}

        if self.game_over:
            return result
        sr, sc = source
        tr, tc = target
        if self.board[sr, sc] == 0 or self.board[tr, tc] != 0:
            return result

        self._ensure_cc()
        if not _is_reachable(self._cc_labels, sr, sc, tr, tc):
            return result

        # Execute
        color = self.board[sr, sc]
        self.board[sr, sc] = 0
        self.board[tr, tc] = color
        self._cc_labels = None
        result['valid'] = True
        self.turns += 1

        # Check lines at target
        cleared = _clear_lines_at(self.board, tr, tc)
        if cleared > 0:
            pts = calculate_score(cleared)
            self.score += pts
            result['score'] = pts
            result['cleared'] = cleared
        else:
            # No line cleared → spawn new balls
            self._spawn_balls()
            # Check if spawned balls created lines
            for (br, bc), _ in self.next_balls:
                if self.board[br, bc] != 0:
                    spawn_cleared = _clear_lines_at(self.board, br, bc)
                    if spawn_cleared > 0:
                        pts = calculate_score(spawn_cleared)
                        self.score += pts
                        result['score'] += pts
                        result['cleared'] += spawn_cleared
            self._generate_next_balls()
            # Check game over
            if _count_empty(self.board) == 0:
                self.game_over = True

        result['game_over'] = self.game_over
        return result

    def fast_move(self, source: tuple[int, int], target: tuple[int, int]) -> tuple[bool, int, bool]:
        """Lightweight move for rollouts. Returns (valid, score, game_over).

        Same game logic as move() but avoids dict allocation.
        """
        if self.game_over:
            return (False, 0, True)
        sr, sc = source
        tr, tc = target
        if self.board[sr, sc] == 0 or self.board[tr, tc] != 0:
            return (False, 0, self.game_over)

        self._ensure_cc()
        if not _is_reachable(self._cc_labels, sr, sc, tr, tc):
            return (False, 0, self.game_over)

        # Execute
        color = self.board[sr, sc]
        self.board[sr, sc] = 0
        self.board[tr, tc] = color
        self._cc_labels = None
        self.turns += 1

        total_pts = 0
        cleared = _clear_lines_at(self.board, tr, tc)
        if cleared > 0:
            total_pts = calculate_score(cleared)
            self.score += total_pts
        else:
            self._spawn_balls()
            for (br, bc), _ in self.next_balls:
                if self.board[br, bc] != 0:
                    spawn_cleared = _clear_lines_at(self.board, br, bc)
                    if spawn_cleared > 0:
                        pts = calculate_score(spawn_cleared)
                        self.score += pts
                        total_pts += pts
            self._generate_next_balls()
            if _count_empty(self.board) == 0:
                self.game_over = True

        return (True, total_pts, self.game_over)

    def trusted_move(self, sr, sc, tr, tc):
        """Execute a move known to be legal. Skips validation for speed.

        For use in MCTS/search where legal moves are pre-computed.
        Caller guarantees: source has a ball, target is empty, path exists.
        """
        # Execute
        color = self.board[sr, sc]
        self.board[sr, sc] = 0
        self.board[tr, tc] = color
        self._cc_labels = None
        self.turns += 1

        cleared = _clear_lines_at(self.board, tr, tc)
        if cleared > 0:
            self.score += calculate_score(cleared)
        else:
            self._spawn_balls()
            for (br, bc), _ in self.next_balls:
                if self.board[br, bc] != 0:
                    spawn_cleared = _clear_lines_at(self.board, br, bc)
                    if spawn_cleared > 0:
                        self.score += calculate_score(spawn_cleared)
            self._generate_next_balls()
            if _count_empty(self.board) == 0:
                self.game_over = True

    # ── Observation ───────────────────────────────────────────────────

    def get_observation(self) -> np.ndarray:
        n = min(len(self.next_balls), 3)
        next_r = np.zeros(3, dtype=np.int64)
        next_c = np.zeros(3, dtype=np.int64)
        next_color = np.zeros(3, dtype=np.int64)
        for i in range(n):
            (row, col), color = self.next_balls[i]
            next_r[i] = row
            next_c[i] = col
            next_color[i] = color
        return _get_observation(self.board, next_r, next_c, next_color, n)

    # ── Display ───────────────────────────────────────────────────────

    def render(self) -> str:
        COLOR_CHARS = '.RGBYCMP'
        lines = ['  ' + ' '.join(str(i) for i in range(BOARD_SIZE))]
        lines.append('  ' + '-' * (BOARD_SIZE * 2 - 1))
        for row in range(BOARD_SIZE):
            cells = []
            for col in range(BOARD_SIZE):
                cells.append(COLOR_CHARS[self.board[row, col]])
            lines.append(f'{row}|' + ' '.join(cells))
        lines.append(f'Score: {self.score} | Turn: {self.turns}')
        return '\n'.join(lines)
