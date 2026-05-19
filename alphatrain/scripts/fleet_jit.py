"""JIT-vectorized step for the fleet miner.

Eliminates the Python per-game loop in `phase1_oracle_fleet.py`. Each
training step previously did ~512 Python iterations of:
  - _get_legal_priors_flat (returns Python dict)
  - max(...) Python lookup
  - game.move(...) (Python method calling JIT helpers)

Combined Python overhead was ~60-70ms/step on Colab L4 (the GPU forward
took ~8ms — GPU was idle 80%+ of the time). This module replaces that loop
with a single numba prange call that runs the full move logic per game in
JIT-compiled, parallel CPU code.

Layout: all per-game state is in flat numpy arrays, indexed by game slot.
A single `step_fleet_jit()` call advances all live games by one move.

Functions:
  - step_fleet_jit(boards, next_pos, next_col, n_next, scores, turns,
                   game_overs, completion_flags, pol_logits, rng_states)
    Applies the argmax legal move per game, handles line clears, spawn,
    game-over detection. completion_flags is the output: 0=alive,
    1=died, 2=capped (NOTE: capping is handled by the caller; this
    function only flags game_overs, the caller checks turns >= horizon).

Note on RNG determinism: per-game RNG is a small 64-bit LCG seeded by
caller. Sequences differ from the original numpy PCG64, but the rollout
is still deterministic given a seed. Common-RNG correctness within a
single mining run is preserved (same seed → same trajectory).
"""

import numpy as np
from numba import njit, prange

# Constants (match game/board.py)
BOARD_SIZE = 9
MIN_LINE_LENGTH = 5
NUM_COLORS = 7
BALLS_PER_TURN = 3


@njit(cache=True, inline='always')
def _lcg_u32(state_arr, i):
    """Knuth LCG; updates state_arr[i] in place and returns a uint32."""
    s = state_arr[i] * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    state_arr[i] = s
    return np.uint32(s >> np.uint64(33))


@njit(cache=True, inline='always')
def _lcg_rand_range(state_arr, i, hi):
    """Uniform int in [0, hi). Caller ensures hi > 0."""
    return _lcg_u32(state_arr, i) % np.uint32(hi)


@njit(cache=True)
def _label_components_local(board):
    """Label empty-cell components. Returns (9,9) int8."""
    labels = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    qr = np.empty(81, dtype=np.int8)
    qc = np.empty(81, dtype=np.int8)
    cur = np.int8(0)
    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if board[sr, sc] != 0 or labels[sr, sc] != 0:
                continue
            cur += 1
            labels[sr, sc] = cur
            qr[0] = sr; qc[0] = sc
            head = 0; tail = 1
            while head < tail:
                r = qr[head]; c = qc[head]; head += 1
                for d in range(4):
                    if d == 0: nr, nc = r, c+1
                    elif d == 1: nr, nc = r, c-1
                    elif d == 2: nr, nc = r+1, c
                    else: nr, nc = r-1, c
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr, nc] == 0 and labels[nr, nc] == 0:
                            labels[nr, nc] = cur
                            qr[tail] = nr; qc[tail] = nc; tail += 1
    return labels


@njit(cache=True)
def _argmax_legal_jit(board, pol_logits):
    """Find the legal move with max pol_logits. Returns flat action or -1."""
    labels = _label_components_local(board)
    best_action = np.int32(-1)
    best_score = np.float32(-1e30)
    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if board[sr, sc] == 0:
                continue
            # Collect unique component labels adjacent to (sr,sc)
            adj_comps = np.zeros(4, dtype=np.int8)
            n_adj = 0
            for d in range(4):
                if d == 0: nr, nc = sr, sc+1
                elif d == 1: nr, nc = sr, sc-1
                elif d == 2: nr, nc = sr+1, sc
                else: nr, nc = sr-1, sc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if labels[nr, nc] > 0:
                        seen = False
                        for j in range(n_adj):
                            if adj_comps[j] == labels[nr, nc]:
                                seen = True; break
                        if not seen:
                            adj_comps[n_adj] = labels[nr, nc]; n_adj += 1
            if n_adj == 0:
                continue
            # Score all reachable targets
            src_idx = sr * BOARD_SIZE + sc
            for tr in range(BOARD_SIZE):
                for tc in range(BOARD_SIZE):
                    if labels[tr, tc] == 0:
                        continue
                    reachable = False
                    for j in range(n_adj):
                        if labels[tr, tc] == adj_comps[j]:
                            reachable = True; break
                    if not reachable:
                        continue
                    action = src_idx * 81 + tr * BOARD_SIZE + tc
                    s = pol_logits[action]
                    if s > best_score:
                        best_score = s
                        best_action = action
    return best_action


@njit(cache=True)
def _clear_lines_at_jit(board, row, col):
    """Clear lines of 5+ through (row,col). Returns cleared count."""
    color = board[row, col]
    if color == 0:
        return 0
    clear_r = np.empty(36, dtype=np.int8)
    clear_c = np.empty(36, dtype=np.int8)
    n_clear = 0
    dirs_dr = np.array([0, 1, 1, 1], dtype=np.int8)
    dirs_dc = np.array([1, 0, 1, -1], dtype=np.int8)
    for di in range(4):
        dr = dirs_dr[di]; dc = dirs_dc[di]
        line_r = np.empty(9, dtype=np.int8)
        line_c = np.empty(9, dtype=np.int8)
        line_r[0] = row; line_c[0] = col
        n = 1
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            line_r[n] = r; line_c[n] = c; n += 1
            r += dr; c += dc
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == color:
            line_r[n] = r; line_c[n] = c; n += 1
            r -= dr; c -= dc
        if n >= MIN_LINE_LENGTH:
            for i in range(n):
                already = False
                for j in range(n_clear):
                    if clear_r[j] == line_r[i] and clear_c[j] == line_c[i]:
                        already = True; break
                if not already:
                    clear_r[n_clear] = line_r[i]
                    clear_c[n_clear] = line_c[i]
                    n_clear += 1
    for i in range(n_clear):
        board[clear_r[i], clear_c[i]] = 0
    return n_clear


@njit(cache=True, inline='always')
def _score_for_clear(n):
    """n*(n-4) for n>=5, else 0."""
    if n < MIN_LINE_LENGTH:
        return 0
    return n * (n - 4)


@njit(cache=True)
def _spawn_and_regen_next(board, next_pos, next_col, rng_states, b):
    """Spawn the existing next_balls onto board, then generate a fresh
    preview of 3 next balls. Returns:
      - total_cleared from cascade clears
      - total_score from cascades
      - new_n_next (3 unless board full)
    Modifies board, next_pos[b], next_col[b], rng_states[b] in place.
    """
    # Spawn existing next_balls (those in next_pos[b], next_col[b])
    total_cleared = 0
    total_score = 0
    for i in range(BALLS_PER_TURN):
        r = next_pos[b, i, 0]; c = next_pos[b, i, 1]
        col = next_col[b, i]
        if col == 0:
            continue  # padding slot
        if board[r, c] == 0:
            board[r, c] = col
        else:
            # Cell occupied — pick a random empty cell
            # First scan empties (max 81)
            empties_r = np.empty(81, dtype=np.int8)
            empties_c = np.empty(81, dtype=np.int8)
            n_emp = 0
            for rr in range(BOARD_SIZE):
                for cc in range(BOARD_SIZE):
                    if board[rr, cc] == 0:
                        empties_r[n_emp] = rr
                        empties_c[n_emp] = cc
                        n_emp += 1
            if n_emp == 0:
                continue  # board full, can't spawn
            pick = _lcg_rand_range(rng_states, b, n_emp)
            board[empties_r[pick], empties_c[pick]] = col
        # Check line clears at the cell where this ball ended up
        if board[r, c] == col:
            cleared = _clear_lines_at_jit(board, r, c)
            if cleared > 0:
                total_cleared += cleared
                total_score += _score_for_clear(cleared)

    # Generate fresh next_balls preview (3 distinct empty cells + colors)
    empties_r = np.empty(81, dtype=np.int8)
    empties_c = np.empty(81, dtype=np.int8)
    n_emp = 0
    for rr in range(BOARD_SIZE):
        for cc in range(BOARD_SIZE):
            if board[rr, cc] == 0:
                empties_r[n_emp] = rr
                empties_c[n_emp] = cc
                n_emp += 1
    new_n = min(BALLS_PER_TURN, n_emp)
    # Fisher-Yates partial shuffle
    for k in range(new_n):
        idx = k + _lcg_rand_range(rng_states, b, n_emp - k)
        # Swap empties[k] and empties[idx]
        tr = empties_r[k]; tc = empties_c[k]
        empties_r[k] = empties_r[idx]; empties_c[k] = empties_c[idx]
        empties_r[idx] = tr; empties_c[idx] = tc
        next_pos[b, k, 0] = empties_r[k]
        next_pos[b, k, 1] = empties_c[k]
        col = np.int8(1 + (_lcg_u32(rng_states, b) % np.uint32(NUM_COLORS)))
        next_col[b, k] = col
    # Pad unused slots with 0
    for k in range(new_n, BALLS_PER_TURN):
        next_pos[b, k, 0] = 0
        next_pos[b, k, 1] = 0
        next_col[b, k] = 0
    return total_score, new_n


@njit(cache=True, parallel=True)
def step_fleet_jit(boards, next_pos, next_col, n_next, scores, turns,
                    game_overs, completion, pol_logits, rng_states):
    """Apply one step per game in parallel.

    All inputs are mutated in place. `completion` output:
        0 = still alive
        1 = died this step (no legal move OR board full)
        Caller separately marks 2 = capped when turns >= horizon.

    boards: (M, 9, 9) int8
    next_pos: (M, 3, 2) int8
    next_col: (M, 3) int8
    n_next: (M,) int8
    scores: (M,) int32
    turns: (M,) int32
    game_overs: (M,) bool
    completion: (M,) int8 output
    pol_logits: (M, 6561) float32
    rng_states: (M,) uint64
    """
    M = boards.shape[0]
    for b in prange(M):
        completion[b] = 0
        if game_overs[b]:
            completion[b] = 1
            continue
        # Pick argmax legal move
        best = _argmax_legal_jit(boards[b], pol_logits[b])
        if best < 0:
            game_overs[b] = True
            completion[b] = 1
            continue
        sr = best // (81 * BOARD_SIZE)
        sc = (best // 81) % BOARD_SIZE
        tr = (best % 81) // BOARD_SIZE
        tc = best % BOARD_SIZE
        color = boards[b, sr, sc]
        boards[b, sr, sc] = 0
        boards[b, tr, tc] = color
        turns[b] += 1
        # Clear lines at target
        cleared = _clear_lines_at_jit(boards[b], tr, tc)
        if cleared > 0:
            scores[b] += _score_for_clear(cleared)
        else:
            # Spawn + regen
            extra_score, new_n = _spawn_and_regen_next(
                boards[b], next_pos, next_col, rng_states, b)
            scores[b] += extra_score
            n_next[b] = new_n
            # Game over if board full
            n_emp = 0
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if boards[b, r, c] == 0:
                        n_emp += 1
            if n_emp == 0:
                game_overs[b] = True
                completion[b] = 1


@njit(cache=True, parallel=True)
def build_obs_fleet_jit(boards, next_pos, next_col, n_next, out):
    """Batched observation builder. Replaces the Python loop over
    _build_obs_for_game(). Channels 0-17 match alphatrain.observation.

    out: (M, 18, 9, 9) float32 — written in place.
    """
    M = boards.shape[0]
    for b in prange(M):
        # Zero
        for ch in range(18):
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    out[b, ch, r, c] = 0.0
        # Channels 0-6 + 7
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                v = boards[b, r, c]
                if v == 0:
                    out[b, 7, r, c] = 1.0
                else:
                    out[b, v - 1, r, c] = 1.0
        # Channels 8-10 + 11
        for i in range(min(n_next[b], 3)):
            r = next_pos[b, i, 0]; c = next_pos[b, i, 1]
            col = next_col[b, i]
            if col == 0:
                continue
            out[b, 8 + i, r, c] = col / 7.0
            out[b, 11, r, c] = 1.0
        # Channel 12: component area heatmap
        labels = _label_components_local(boards[b])
        counts = np.zeros(82, dtype=np.int32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if labels[r, c] > 0:
                    counts[labels[r, c]] += 1
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if labels[r, c] > 0:
                    out[b, 12, r, c] = counts[labels[r, c]] / 81.0
        # Channels 13-16: line potentials (H, V, D1, D2). Channel 17: max line length.
        dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if boards[b, r, c] == 0:
                    continue
                max_len = 0
                for di in range(4):
                    dr = dirs[di][0]; dc = dirs[di][1]
                    color = boards[b, r, c]
                    n = 1
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and boards[b, rr, cc] == color:
                        n += 1; rr += dr; cc += dc
                    rr, cc = r - dr, c - dc
                    while 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and boards[b, rr, cc] == color:
                        n += 1; rr -= dr; cc -= dc
                    out[b, 13 + di, r, c] = n / 9.0
                    if n > max_len:
                        max_len = n
                out[b, 17, r, c] = max_len / 9.0
