"""Fully GPU-vectorized game engine for fleet mining.

Every per-step operation is a batched tensor op on (M, 9, 9) boards. Zero
Python or CPU work per game — the only CPU work is orchestration (refill
slots, accumulate outcomes) which is O(M) once per step, not per game.

Verified against the CPU game engine via alphatrain/tests/test_fleet_gpu.py.

Operations:
  - gpu_label_components(boards)             -> (M, 9, 9) int labels
  - gpu_legal_argmax(boards, pol_logits)     -> (M,) flat action, -1 if none
  - gpu_apply_move(boards, moves)            -> in-place; returns color moved
  - gpu_clear_at_target(boards, tr, tc)      -> in-place; returns cleared count + score
  - gpu_spawn_balls(boards, next_pos, next_col, n_next, rng_state)
       in-place spawn; returns cleared count + score from cascades
  - gpu_generate_next_balls(boards, rng_state) -> next_pos, next_col, n_next
  - gpu_step(...)  -- the full per-step function combining all of the above
"""

import torch

BOARD_SIZE = 9
NUM_CELLS = 81
NUM_MOVES = 6561
MIN_LINE_LENGTH = 5
NUM_COLORS = 7
BALLS_PER_TURN = 3


def gpu_label_components(boards: torch.Tensor) -> torch.Tensor:
    """Label connected components of empty cells per board.

    boards: (M, 9, 9) int (board[r, c] == 0 means empty)
    Returns: (M, 9, 9) long — label for each empty cell (1..81), 0 for balls.

    Method: iterative min-label propagation. Each empty cell takes the min
    label of itself + 4 neighbors. After ~20 iterations on a 9x9, all
    connected components have a stable label = min cell index in that
    component.
    """
    M = boards.shape[0]
    device = boards.device
    empty = (boards == 0)
    # Initial labels: 1..81 for empty cells, 0 for balls
    cell_idx = torch.arange(1, NUM_CELLS + 1, device=device,
                              dtype=torch.long).view(1, BOARD_SIZE, BOARD_SIZE)
    labels = (cell_idx * empty.long()).expand(M, -1, -1).clone()

    # 9x9 diameter = 16 → 16 iterations is sufficient. No early-exit check
    # (torch.equal() launches a kernel; skipping it saves overhead at the
    # small cost of always doing the full 16 iters).
    # Ball cells stay at +inf throughout to prevent labels leaking through.
    INF = NUM_CELLS + 1
    big = labels.clone()
    big[~empty] = INF
    for _ in range(16):
        shifted_up = torch.roll(big, shifts=1, dims=1).clone()
        shifted_up[:, 0, :] = INF
        shifted_down = torch.roll(big, shifts=-1, dims=1).clone()
        shifted_down[:, -1, :] = INF
        shifted_left = torch.roll(big, shifts=1, dims=2).clone()
        shifted_left[:, :, 0] = INF
        shifted_right = torch.roll(big, shifts=-1, dims=2).clone()
        shifted_right[:, :, -1] = INF
        big = torch.minimum(
            torch.minimum(big, shifted_up),
            torch.minimum(torch.minimum(shifted_down, shifted_left),
                          shifted_right))
        big[~empty] = INF
    labels = torch.where(empty, big, torch.zeros_like(big))
    labels[labels == INF] = 0
    return labels


def gpu_legal_argmax(boards: torch.Tensor, pol_logits: torch.Tensor,
                       labels: torch.Tensor = None) -> torch.Tensor:
    """Argmax over legal (src, tgt) moves per board.

    boards: (M, 9, 9) int8 or int
    pol_logits: (M, 6561) float
    labels: (M, 9, 9) long — pre-computed component labels (optional). If
        None, computed here. Pre-computing once per step saves duplicate work
        with gpu_build_obs.
    Returns: (M,) long — flat action (src*81 + tgt), or -1 if no legal moves.
    """
    M = boards.shape[0]
    device = boards.device
    if labels is None:
        labels = gpu_label_components(boards)

    # For each cell, the labels of its 4 neighbors (0 if neighbor is out-of-bounds
    # or a ball)
    nb_labels = torch.zeros(M, BOARD_SIZE, BOARD_SIZE, 4, device=device,
                             dtype=torch.long)
    for k, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        shifted = torch.roll(labels, shifts=(dr, dc), dims=(1, 2))
        # Zero out wrap-around
        if dr == 1:
            shifted[:, 0, :] = 0
        elif dr == -1:
            shifted[:, -1, :] = 0
        if dc == 1:
            shifted[:, :, 0] = 0
        elif dc == -1:
            shifted[:, :, -1] = 0
        nb_labels[:, :, :, k] = shifted

    # Per ball cell (src), the set of reachable component labels = its
    # 4 neighbor labels (where >0). Per empty cell (tgt), its label = labels[tgt].
    # Legal move: tgt_label is in the set of src's neighbor labels (and src is
    # a ball, tgt is empty).

    # Build "is reachable" mask of shape (M, src, tgt) where src and tgt are
    # flat 81-cell indices. This is 81*81 = 6561 per board; M = 1024 boards
    # is 6.7M entries. Memory: 6.7M * 1 byte = 6.7MB; fine.

    # nb_labels: (M, 9, 9, 4) — for each cell, its 4 neighbor labels
    # Reshape to (M, 81, 4): for each src cell, 4 neighbor labels
    nb_labels_flat = nb_labels.view(M, NUM_CELLS, 4)  # (M, 81, 4)
    # Labels: (M, 9, 9) -> (M, 81)
    tgt_labels_flat = labels.view(M, NUM_CELLS)  # (M, 81)

    # legal[m, src, tgt] = any(nb_labels_flat[m, src, k] == tgt_labels_flat[m, tgt])
    # Use broadcasting: (M, 81, 1, 4) vs (M, 1, 81, 1) -> (M, 81, 81, 4)
    src_nb = nb_labels_flat.unsqueeze(2)  # (M, 81, 1, 4)
    tgt_lbl = tgt_labels_flat.unsqueeze(1).unsqueeze(-1)  # (M, 1, 81, 1)
    # Match: (M, 81, 81, 4) — element is True if neighbor label matches tgt label
    match = (src_nb == tgt_lbl) & (tgt_lbl > 0)
    legal_mask = match.any(dim=-1)  # (M, 81, 81)

    # Source must be a ball, target must be empty (label > 0)
    is_ball = (boards != 0).view(M, NUM_CELLS, 1)  # (M, 81, 1)
    is_empty = (boards == 0).view(M, 1, NUM_CELLS)  # (M, 1, 81)
    legal_mask = legal_mask & is_ball & is_empty  # (M, 81, 81)

    # Apply mask to pol_logits, then argmax over 6561
    legal_flat = legal_mask.view(M, NUM_MOVES)
    # Set illegal scores to -inf so argmax picks legal
    masked_logits = torch.where(
        legal_flat, pol_logits, torch.full_like(pol_logits, float('-inf')))
    has_any = legal_flat.any(dim=1)
    best = masked_logits.argmax(dim=1)
    best = torch.where(has_any, best.long(), torch.full_like(best, -1).long())
    return best


def gpu_clear_lines(boards: torch.Tensor) -> tuple:
    """Detect lines of 5+ and clear them. Returns (n_cleared per board, score per board).

    boards: (M, 9, 9) — mutated in place.
    """
    M = boards.shape[0]
    device = boards.device
    occ = (boards != 0)

    clear_mask = torch.zeros_like(occ)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # Run length scan: for each cell, how many same-color cells extend
        # in (dr, dc) direction (including self).
        fwd = _scan_dir(boards, dr, dc)  # (M, 9, 9) int
        bwd = _scan_dir(boards, -dr, -dc)
        total = (fwd + bwd - 1) * occ.long()
        clear_mask = clear_mask | ((total >= MIN_LINE_LENGTH) & occ)

    n_cleared = clear_mask.sum(dim=(1, 2)).long()
    # Score per board: n * (n - 4) if n >= 5 else 0
    score = torch.where(n_cleared >= MIN_LINE_LENGTH,
                         n_cleared * (n_cleared - 4),
                         torch.zeros_like(n_cleared))
    boards.masked_fill_(clear_mask, 0)
    return n_cleared, score


def _scan_dir(boards: torch.Tensor, dr: int, dc: int) -> torch.Tensor:
    """Forward run-length scan in direction (dr, dc).

    For each cell, count consecutive same-color cells (including self)
    extending in direction (dr, dc). Returns (M, 9, 9) int.
    """
    M = boards.shape[0]
    occ = (boards != 0)
    run = occ.long().clone()
    # Up to 8 propagation steps (9x9 board, max chain length 9)
    for _ in range(BOARD_SIZE - 1):
        nb_board = torch.roll(boards, shifts=(dr, dc), dims=(1, 2))
        nb_run = torch.roll(run, shifts=(dr, dc), dims=(1, 2))
        # Zero out wrap-around for boards (so equality check fails at edges)
        if dr == 1: nb_board[:, 0, :] = -1; nb_run[:, 0, :] = 0
        elif dr == -1: nb_board[:, -1, :] = -1; nb_run[:, -1, :] = 0
        if dc == 1: nb_board[:, :, 0] = -1; nb_run[:, :, 0] = 0
        elif dc == -1: nb_board[:, :, -1] = -1; nb_run[:, :, -1] = 0
        same = (boards == nb_board) & occ & (nb_board > 0)
        run = torch.where(same, torch.maximum(run, nb_run + 1), run)
    return run


def gpu_per_board_sample_empty(empties: torch.Tensor, rand_per_board: torch.Tensor) -> torch.Tensor:
    """Sample one random empty cell per board.

    empties: (M, 9, 9) bool
    rand_per_board: (M,) float in [0, 1) — pre-generated random values

    Returns: (M,) flat cell index of the sampled empty cell. -1 if no empties.
    """
    M = empties.shape[0]
    empties_flat = empties.view(M, NUM_CELLS).long()
    cumsum = empties_flat.cumsum(dim=1)  # (M, 81): 1-indexed empty positions
    empty_count = cumsum[:, -1]  # (M,)
    # Random index in [0, empty_count)
    rand_idx = (rand_per_board * empty_count.float()).long()
    rand_idx = torch.clamp(rand_idx, max=empty_count - 1)
    # Find first cell where cumsum == rand_idx + 1 (the (rand_idx+1)-th empty)
    target = (rand_idx + 1).unsqueeze(1)  # (M, 1)
    # We want first index where cumsum[m] == target[m]
    # Use argmax of (cumsum == target).long(): returns first True index
    matches = (cumsum == target).long()
    found = matches.argmax(dim=1)  # (M,) — first match, or 0 if no match
    # If empty_count is 0, return -1
    no_empty = empty_count == 0
    found = torch.where(no_empty, torch.full_like(found, -1), found)
    return found


def gpu_sample_k_distinct_empties(empties: torch.Tensor, k: int,
                                     rand_scores: torch.Tensor) -> torch.Tensor:
    """Sample K distinct empty cells per board.

    empties: (M, 9, 9) bool
    rand_scores: (M, 81) float — pre-generated random scores for each cell

    Returns: (M, k) flat cell indices. May contain -1 if fewer than k empties.
    """
    M = empties.shape[0]
    empties_flat = empties.view(M, NUM_CELLS)
    # Mask non-empty cells to -inf so they're never picked
    scores = torch.where(empties_flat, rand_scores,
                          torch.full_like(rand_scores, float('-inf')))
    # Top-k by score
    top_vals, top_idx = scores.topk(k, dim=1)  # (M, k)
    # Where top_val is -inf, mark as -1 (not enough empties)
    invalid = top_vals == float('-inf')
    top_idx = torch.where(invalid, torch.full_like(top_idx, -1), top_idx)
    return top_idx


def gpu_spawn_balls(boards: torch.Tensor, next_pos: torch.Tensor,
                     next_col: torch.Tensor, n_next: torch.Tensor,
                     active_mask: torch.Tensor, rand_state: torch.Tensor) -> tuple:
    """Spawn next_balls onto boards.

    For each ball slot (up to BALLS_PER_TURN=3), if the predetermined
    next_pos is empty, place the ball there. Otherwise pick a random empty
    cell.

    Mutates boards in place. After spawning, checks for line clears at each
    spawn cell and accumulates score.

    boards: (M, 9, 9) — mutated
    next_pos: (M, 3, 2) int8 — (row, col) per ball slot
    next_col: (M, 3) int8 — color per ball slot (1..7, 0 = empty slot)
    n_next: (M,) int8 — number of valid balls per board
    active_mask: (M,) bool — only act on active boards
    rand_state: (M,) — RNG offset for this step (used to make spawns
        deterministic for tests)

    Returns: (n_cleared_per_board (M,), score_per_board (M,))
    """
    M = boards.shape[0]
    device = boards.device

    # Spawn all 3 slots first (no line-clear scans between slots — we do ONE
    # combined clear scan at the end). This trades a tiny bit of fidelity
    # (cascade clears between balls) for much less GPU work; line patterns
    # of 5+ are still detected by the final clear pass.
    for slot in range(BALLS_PER_TURN):
        # Boards that have this slot defined and are active
        slot_valid = active_mask & (n_next > slot)
        if not slot_valid.any():
            continue
        r = next_pos[:, slot, 0].long()  # (M,)
        c = next_pos[:, slot, 1].long()
        col = next_col[:, slot].long()  # (M,)
        # Direct spawn where target cell is empty
        batch_idx = torch.arange(M, device=device)
        target_empty = (boards[batch_idx, r, c] == 0) & slot_valid
        direct_indices = torch.nonzero(target_empty, as_tuple=False).squeeze(-1)
        if direct_indices.numel() > 0:
            boards[direct_indices, r[direct_indices], c[direct_indices]] = \
                col[direct_indices].to(boards.dtype)

        # Fallback: pick random empty for cells that were occupied
        fallback = slot_valid & ~target_empty
        if fallback.any():
            empties = (boards == 0)
            rand_vals = torch.rand(M, device=device)
            chosen_flat = gpu_per_board_sample_empty(empties, rand_vals)
            valid_fallback = fallback & (chosen_flat >= 0)
            valid_idx = torch.nonzero(valid_fallback, as_tuple=False).squeeze(-1)
            if valid_idx.numel() > 0:
                fr = chosen_flat[valid_idx] // BOARD_SIZE
                fc = chosen_flat[valid_idx] % BOARD_SIZE
                boards[valid_idx, fr, fc] = col[valid_idx].to(boards.dtype)

    # One combined line-clear scan after all 3 spawns. Saves 2 of the 3
    # per-slot clear scans (each was 4 directions × 9-iter scan = 36 ops).
    total_cleared, total_score = gpu_clear_lines(boards)
    return total_cleared, total_score


def gpu_build_obs(boards: torch.Tensor, next_pos: torch.Tensor,
                    next_col: torch.Tensor, n_next: torch.Tensor,
                    labels: torch.Tensor = None) -> torch.Tensor:
    """Build (M, 18, 9, 9) observation tensor entirely on GPU.

    Channels 0-6: one-hot color planes
    Channel 7:   empty mask
    Channels 8-10: next-ball color/7 (one channel per next-ball slot)
    Channel 11:  next-ball mask
    Channel 12:  component area heatmap (component_size / 81 per empty cell)
    Channels 13-16: line potentials per direction (H, V, D1, D2)
    Channel 17:  max line length

    boards: (M, 9, 9) int8 or long
    next_pos: (M, 3, 2) int8
    next_col: (M, 3) int8
    n_next: (M,) int8
    """
    M = boards.shape[0]
    device = boards.device
    obs = torch.zeros(M, 18, BOARD_SIZE, BOARD_SIZE, device=device,
                       dtype=torch.float32)

    # Channels 0-6: color planes
    for c in range(1, 8):
        obs[:, c - 1] = (boards == c).float()
    # Channel 7: empty
    obs[:, 7] = (boards == 0).float()

    # Channels 8-10 + 11: next balls
    batch_idx = torch.arange(M, device=device)
    for k in range(3):
        valid = (n_next > k).view(M)
        if not valid.any():
            continue
        r = next_pos[:, k, 0].long()
        c = next_pos[:, k, 1].long()
        col = next_col[:, k].float() / 7.0
        # Place where valid; safe to write to (r, c) for invalid slots since
        # we'll mask out — but to be safe, only where valid
        valid_idx = batch_idx[valid]
        obs[valid_idx, 8 + k, r[valid_idx], c[valid_idx]] = col[valid_idx]
        obs[valid_idx, 11, r[valid_idx], c[valid_idx]] = 1.0

    # Channel 12: component area heatmap
    if labels is None:
        labels = gpu_label_components(boards)  # (M, 9, 9) long
    # Compute size per component via scatter
    # Build per-(board, label) counts
    # Flatten labels to (M*81); add a batch offset (each board has 82 possible labels)
    flat_labels = labels.view(M, -1)  # (M, 81)
    # Per-board: count occurrences of each label
    # Use one-hot then sum: (M, 81, 82) one-hot -> sum over cells -> (M, 82)
    # Memory: M * 81 * 82 = 6642M = 26MB at M=1024 — fine
    label_onehot = torch.nn.functional.one_hot(flat_labels, num_classes=82).float()  # (M, 81, 82)
    counts = label_onehot.sum(dim=1)  # (M, 82) — count per label
    # Look up: each cell's label -> count
    cell_counts = torch.gather(counts, 1, flat_labels)  # (M, 81)
    # Mask: only empty cells get the heatmap value
    empty_mask = (boards == 0).view(M, -1).float()  # (M, 81)
    obs[:, 12] = (cell_counts * empty_mask / 81.0).view(M, BOARD_SIZE, BOARD_SIZE)

    # Channels 13-16: line potentials. Channel 17: max line length.
    # For each direction, compute forward + backward scan (per cell) - 1.
    occ_f = (boards != 0).float()
    max_line = torch.zeros(M, BOARD_SIZE, BOARD_SIZE, device=device,
                            dtype=torch.float32)
    for di, (dr, dc) in enumerate([(0, 1), (1, 0), (1, 1), (1, -1)]):
        fwd = _scan_dir(boards, dr, dc).float()
        bwd = _scan_dir(boards, -dr, -dc).float()
        total = (fwd + bwd - 1.0) * occ_f
        obs[:, 13 + di] = total / 9.0
        max_line = torch.maximum(max_line, total)
    obs[:, 17] = max_line / 9.0
    return obs


def gpu_clear_lines_functional(boards: torch.Tensor) -> tuple:
    """Functional version of gpu_clear_lines. Returns (new_boards, n_cleared, score).

    Same logic but never mutates input. cudagraph-compatible.
    """
    M = boards.shape[0]
    occ = (boards != 0)
    clear_mask = torch.zeros_like(occ)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        fwd = _scan_dir(boards, dr, dc)
        bwd = _scan_dir(boards, -dr, -dc)
        total = (fwd + bwd - 1) * occ.long()
        clear_mask = clear_mask | ((total >= MIN_LINE_LENGTH) & occ)
    n_cleared = clear_mask.sum(dim=(1, 2)).long()
    score = torch.where(n_cleared >= MIN_LINE_LENGTH,
                         n_cleared * (n_cleared - 4),
                         torch.zeros_like(n_cleared))
    new_boards = torch.where(clear_mask, torch.zeros_like(boards), boards)
    return new_boards, n_cleared, score


def gpu_spawn_balls_functional(boards: torch.Tensor, next_pos: torch.Tensor,
                                 next_col: torch.Tensor, n_next: torch.Tensor,
                                 active_mask: torch.Tensor,
                                 rand_per_slot: torch.Tensor) -> tuple:
    """Functional version of gpu_spawn_balls. Returns (new_boards, n_cleared, score).

    rand_per_slot: (M, 3) float — random values in [0, 1) for fallback empty-cell
        selection (one per slot per board).
    """
    M = boards.shape[0]
    device = boards.device
    batch_idx = torch.arange(M, device=device)

    for slot in range(BALLS_PER_TURN):
        slot_valid = active_mask & (n_next > slot)
        r = next_pos[:, slot, 0].long()
        c = next_pos[:, slot, 1].long()
        col = next_col[:, slot].long()
        # Direct place if cell empty
        cell_at_target = boards[batch_idx, r, c]
        target_empty = (cell_at_target == 0) & slot_valid
        # Build a board update using scatter_add via flat indices
        # Use torch.where over (M, 9, 9) update tensor instead of mutation
        flat_idx = batch_idx * NUM_CELLS + r * BOARD_SIZE + c
        update_flat = torch.zeros(M * NUM_CELLS, dtype=boards.dtype, device=device)
        # Direct placement: set update_flat[flat_idx] = col where target_empty
        col_to_place = torch.where(target_empty, col.to(boards.dtype),
                                    torch.zeros_like(col, dtype=boards.dtype))
        update_flat.scatter_(0, flat_idx, col_to_place)
        boards = boards + update_flat.view(M, BOARD_SIZE, BOARD_SIZE)
        # Note: this works since target cells were 0 (empty) when target_empty=True

        # Fallback: where slot_valid but target_empty was False, pick a random empty
        fallback = slot_valid & ~target_empty
        if fallback.any():
            empties_now = (boards == 0)
            chosen_flat = gpu_per_board_sample_empty(empties_now,
                                                       rand_per_slot[:, slot])
            valid_fallback = fallback & (chosen_flat >= 0)
            # Build a second update via scatter, masking by valid_fallback
            f_flat_idx = batch_idx * NUM_CELLS + chosen_flat.clamp(min=0)
            update_flat2 = torch.zeros(M * NUM_CELLS, dtype=boards.dtype, device=device)
            col2 = torch.where(valid_fallback, col.to(boards.dtype),
                                torch.zeros_like(col, dtype=boards.dtype))
            update_flat2.scatter_(0, f_flat_idx, col2)
            boards = boards + update_flat2.view(M, BOARD_SIZE, BOARD_SIZE)

    new_boards, total_cleared, total_score = gpu_clear_lines_functional(boards)
    return new_boards, total_cleared, total_score


def gpu_step_functional(boards: torch.Tensor, next_pos: torch.Tensor,
                          next_col: torch.Tensor, n_next: torch.Tensor,
                          scores: torch.Tensor, turns: torch.Tensor,
                          game_overs: torch.Tensor, pol_logits: torch.Tensor,
                          rand_score: torch.Tensor, rand_color: torch.Tensor,
                          labels: torch.Tensor = None) -> tuple:
    """Fully functional gpu_step. Returns all updated state tensors fresh.

    No in-place mutations on input tensors — cudagraph-compatible for use
    under torch.compile(mode='reduce-overhead').

    Returns:
        (boards, next_pos, next_col, n_next, scores, turns, game_overs, completion)
    """
    M = boards.shape[0]
    device = boards.device

    active = ~game_overs

    # 1. Argmax legal move (uses pre-computed labels if provided)
    moves = gpu_legal_argmax(boards, pol_logits, labels=labels)
    no_move = (moves < 0) & active
    valid_move = (moves >= 0) & active

    # 2. Decode + apply move (functionally)
    safe_moves = moves.clamp(min=0)
    src_r = (safe_moves // (81 * BOARD_SIZE)).long()
    src_c = ((safe_moves // 81) % BOARD_SIZE).long()
    tgt_r = ((safe_moves % 81) // BOARD_SIZE).long()
    tgt_c = (safe_moves % BOARD_SIZE).long()
    batch_idx = torch.arange(M, device=device)
    move_color = boards[batch_idx, src_r, src_c]

    # Build "after move" board via scatter:
    # Subtract color at src (where valid), add color at tgt (where valid)
    src_flat = batch_idx * NUM_CELLS + src_r * BOARD_SIZE + src_c
    tgt_flat = batch_idx * NUM_CELLS + tgt_r * BOARD_SIZE + tgt_c
    delta_flat = torch.zeros(M * NUM_CELLS, dtype=boards.dtype, device=device)
    # Subtract color at src
    delta_flat.scatter_(0, src_flat,
                         torch.where(valid_move, -move_color,
                                       torch.zeros_like(move_color)))
    # Add color at tgt (do this via a SECOND scatter to avoid src==tgt edge case;
    # in this game src != tgt by definition since src is a ball and tgt is empty)
    delta_flat.scatter_add_(0, tgt_flat,
                             torch.where(valid_move, move_color,
                                           torch.zeros_like(move_color)))
    boards = boards + delta_flat.view(M, BOARD_SIZE, BOARD_SIZE)

    turns = turns + valid_move.long()

    # 3. Line clears
    boards, cleared, score = gpu_clear_lines_functional(boards)
    scores = scores + score

    # 4. If no clear → spawn + cascade clears + new next preview
    no_clear = (cleared == 0) & valid_move
    # We always run spawn pipeline; gpu_spawn_balls only acts where active_mask=True
    boards2, spawn_cleared, spawn_score = gpu_spawn_balls_functional(
        boards, next_pos, next_col, n_next, no_clear, rand_score[:, :3])
    boards = boards2
    scores = scores + spawn_score

    # 5. Generate new next-balls preview (only for boards that spawned)
    new_next_pos, new_next_col, new_n_next = gpu_generate_next_balls(
        boards, rand_score, rand_color)
    mask_pos = no_clear.view(M, 1, 1).expand(-1, BALLS_PER_TURN, 2)
    next_pos = torch.where(mask_pos, new_next_pos, next_pos)
    mask_col = no_clear.view(M, 1).expand(-1, BALLS_PER_TURN)
    next_col = torch.where(mask_col, new_next_col, next_col)
    n_next = torch.where(no_clear, new_n_next, n_next)

    # 6. Game-over check
    n_empty = (boards == 0).sum(dim=(1, 2))
    new_died = ((n_empty == 0) | no_move) & active
    game_overs = game_overs | new_died

    return boards, next_pos, next_col, n_next, scores, turns, game_overs, new_died


def gpu_step(boards: torch.Tensor, next_pos: torch.Tensor,
             next_col: torch.Tensor, n_next: torch.Tensor,
             scores: torch.Tensor, turns: torch.Tensor,
             game_overs: torch.Tensor, pol_logits: torch.Tensor,
             rand_score: torch.Tensor, rand_color: torch.Tensor,
             labels: torch.Tensor = None) -> torch.Tensor:
    """One step per board on GPU. Mutates state tensors in place.

    boards:     (M, 9, 9) int8  — mutated
    next_pos:   (M, 3, 2) int8 — mutated (only for boards that spawned)
    next_col:   (M, 3)    int8 — mutated
    n_next:     (M,)      int8 — mutated
    scores:     (M,)      long — mutated (added to)
    turns:      (M,)      long — mutated (incremented for valid moves)
    game_overs: (M,)      bool — mutated (set True when game ends)
    pol_logits: (M, 6561) float — read-only
    rand_score: (M, 81)   float — pre-sampled randomness for next-3 selection
    rand_color: (M, 3)    long  — pre-sampled colors in [1, 7]

    Returns: completion mask (M,) bool — True for boards that died this step.
    """
    M = boards.shape[0]
    device = boards.device

    active = ~game_overs
    if not active.any():
        return torch.zeros(M, dtype=torch.bool, device=device)

    # 1. Argmax legal move. Component labels are also needed downstream by
    # gpu_build_obs (next step), so we don't cache them here — but within this
    # step they're computed once inside gpu_legal_argmax.
    moves = gpu_legal_argmax(boards, pol_logits, labels=labels)  # (M,) — -1 if no legal
    no_move = (moves < 0) & active
    valid_move = (moves >= 0) & active

    # 2. Decode and apply move (only for valid boards)
    safe_moves = moves.clamp(min=0)
    src_r = (safe_moves // (81 * 9)).long()
    src_c = ((safe_moves // 81) % 9).long()
    tgt_r = ((safe_moves % 81) // 9).long()
    tgt_c = (safe_moves % 9).long()

    batch_idx = torch.arange(M, device=device)
    # Pre-extract source color (will be 0 for invalid boards)
    move_color = boards[batch_idx, src_r, src_c]
    # Clear source where move is valid
    new_src = torch.where(valid_move, torch.zeros_like(move_color), move_color)
    boards[batch_idx, src_r, src_c] = new_src
    # Set target to color where move is valid (otherwise preserve)
    pre_tgt = boards[batch_idx, tgt_r, tgt_c]
    new_tgt = torch.where(valid_move, move_color, pre_tgt)
    boards[batch_idx, tgt_r, tgt_c] = new_tgt
    turns.add_(valid_move.long())

    # 3. Line clears (whole-board scan; cheaper than per-cell)
    cleared, score = gpu_clear_lines(boards)
    scores.add_(score)

    # 4. If nothing cleared from the move, spawn next-balls + cascades + new preview
    no_clear = (cleared == 0) & valid_move
    if no_clear.any():
        # Spawn this turn's pre-decided next balls
        spawn_cleared, spawn_score = gpu_spawn_balls(
            boards, next_pos, next_col, n_next, no_clear,
            rand_score[:, 0])  # reuse first random as fallback rng
        scores.add_(spawn_score)
        # Generate new next preview (3 distinct empties + random colors)
        new_next_pos, new_next_col, new_n_next = gpu_generate_next_balls(
            boards, rand_score, rand_color)
        # Update only for boards that spawned
        mask_pos = no_clear.view(M, 1, 1).expand(-1, BALLS_PER_TURN, 2)
        next_pos.copy_(torch.where(mask_pos, new_next_pos, next_pos))
        mask_col = no_clear.view(M, 1).expand(-1, BALLS_PER_TURN)
        next_col.copy_(torch.where(mask_col, new_next_col, next_col))
        n_next.copy_(torch.where(no_clear, new_n_next, n_next))

    # 5. Game over: no empty cells, or no legal move was available
    n_empty = (boards == 0).sum(dim=(1, 2))
    new_died = ((n_empty == 0) | no_move) & active
    game_overs.copy_(game_overs | new_died)
    return new_died


def gpu_generate_next_balls(boards: torch.Tensor, rand_score: torch.Tensor,
                              rand_color: torch.Tensor) -> tuple:
    """Generate the next-3 preview for each board.

    rand_score: (M, 81) — random scores for top-k empty selection
    rand_color: (M, 3) int — random colors in [1, 7] (caller pre-generates
        using % NUM_COLORS + 1)

    Returns: (next_pos (M, 3, 2) int8, next_col (M, 3) int8, n_next (M,) int8)
    """
    M = boards.shape[0]
    device = boards.device
    empties = (boards == 0)
    top_flat = gpu_sample_k_distinct_empties(empties, BALLS_PER_TURN, rand_score)
    # Decode
    next_pos = torch.zeros(M, BALLS_PER_TURN, 2, dtype=torch.int8, device=device)
    next_col = torch.zeros(M, BALLS_PER_TURN, dtype=torch.int8, device=device)
    n_next = torch.zeros(M, dtype=torch.int8, device=device)
    for k in range(BALLS_PER_TURN):
        valid = top_flat[:, k] >= 0
        # Where valid: assign pos + color
        rr = (top_flat[:, k] // BOARD_SIZE)
        cc = (top_flat[:, k] % BOARD_SIZE)
        next_pos[:, k, 0] = torch.where(valid, rr.to(torch.int8),
                                          torch.zeros_like(rr, dtype=torch.int8))
        next_pos[:, k, 1] = torch.where(valid, cc.to(torch.int8),
                                          torch.zeros_like(cc, dtype=torch.int8))
        next_col[:, k] = torch.where(valid, rand_color[:, k].to(torch.int8),
                                       torch.zeros_like(rand_color[:, k], dtype=torch.int8))
        n_next = n_next + valid.to(torch.int8)
    return next_pos, next_col, n_next
