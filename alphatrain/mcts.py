"""Neural MCTS for Color Lines 98.

AlphaZero-style Monte Carlo Tree Search using the trained ResNet:
- Policy head provides move priors (which moves to explore)
- Value head evaluates leaf positions (replaces heuristic rollouts)
- PUCT selection balances exploration vs exploitation
- Virtual loss batching: collect B leaves per batch, one NN forward pass
- Determinized: each simulation samples independent ball spawns via game.clone()
- MuZero-style Q normalization for score-based (not win/loss) values

Performance optimizations:
- Flat integer action keys (avoids tuple-of-tuples overhead)
- Shared RNG across simulations (avoids 5us np.random.default_rng per clone)
- Inlined PUCT selection (avoids method call + property overhead)
- Direct _legal_priors_jit → dict with int keys (no numpy decomposition)

Nodes represent post-spawn game states — matching the NN's training
distribution. 400 simulations average over spawn outcomes.

Usage:
    python -m alphatrain.evaluate --player mcts --simulations 400 --games 20
"""

import math
import numpy as np
import torch
from numba import njit

from alphatrain.observation import build_observation
from game.board import _label_empty_components, _is_reachable

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561

VIRTUAL_LOSS = 1.0


@njit(cache=True)
def _evaluate_board(board):
    """Fast board evaluation for MCTS leaf nodes.

    Returns a score where higher = healthier board.
    Captures: empty count, largest connected region, line potential,
    and partition penalty.
    """
    empty = 0
    max_region = 0
    n_regions = 0
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    queue_r = np.empty(81, dtype=np.int8)
    queue_c = np.empty(81, dtype=np.int8)

    # BFS to find connected empty regions
    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if board[sr, sc] == 0:
                empty += 1
                if visited[sr, sc] == 0:
                    # Flood fill this region
                    n_regions += 1
                    region_size = 0
                    visited[sr, sc] = 1
                    queue_r[0] = sr
                    queue_c[0] = sc
                    head, tail = 0, 1
                    while head < tail:
                        r, c = queue_r[head], queue_c[head]
                        head += 1
                        region_size += 1
                        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                if board[nr, nc] == 0 and visited[nr, nc] == 0:
                                    visited[nr, nc] = 1
                                    queue_r[tail] = nr
                                    queue_c[tail] = nc
                                    tail += 1
                    if region_size > max_region:
                        max_region = region_size

    # Line potential: count balls in partial lines (3-4 in a row)
    line_potential = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = board[r, c]
            if color == 0:
                continue
            # Check horizontal and vertical only (avoid double-counting)
            for dr, dc in ((0, 1), (1, 0)):
                length = 1
                cr, cc = r + dr, c + dc
                while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and board[cr, cc] == color:
                    length += 1
                    cr += dr
                    cc += dc
                if 3 <= length <= 4:
                    line_potential += length

    # Score: empty space + connectivity bonus + line potential - partition penalty
    # Partition: if the board has multiple disconnected empty regions,
    # balls can't move freely between them
    partition_penalty = 0
    if n_regions > 1:
        # Penalty proportional to how fragmented the board is
        # (empty - max_region) = cells in non-largest regions
        partition_penalty = (empty - max_region) * 2

    return float(empty + max_region + line_potential - partition_penalty)


@njit(cache=True)
def _legal_priors_jit(board, pol_logits, top_k):
    """Compute top-K legal move priors entirely in JIT.

    Returns (count, flat_indices[top_k], priors[top_k]).

    Optimized: BFS labels + precomputed per-component cell lists so each
    source ball iterates only its reachable cells (not all 81). Top-K is
    maintained inline via a sorted insertion buffer, and softmax is computed
    over only the final K values instead of all legal moves.
    """
    # Label connected components of empty cells and collect per-component
    # cell lists in a single BFS pass
    labels = np.zeros((9, 9), dtype=np.int8)
    queue = np.empty(162, dtype=np.int32)
    current = np.int8(0)
    comp_start = np.empty(82, dtype=np.int32)
    comp_count = np.empty(82, dtype=np.int32)
    comp_flat = np.empty(81, dtype=np.int32)
    comp_count[:] = 0
    n_empty = 0

    for sr in range(9):
        for sc in range(9):
            if board[sr, sc] != 0 or labels[sr, sc] != 0:
                continue
            current += 1
            labels[sr, sc] = current
            comp_start[current] = n_empty
            queue[0] = sr * 9 + sc
            head = 0
            tail = 1
            comp_flat[n_empty] = sr * 9 + sc
            n_empty += 1
            while head < tail:
                pos = queue[head]
                head += 1
                r = pos // 9
                c = pos % 9
                for d in range(4):
                    if d == 0:
                        nr, nc = r, c + 1
                    elif d == 1:
                        nr, nc = r, c - 1
                    elif d == 2:
                        nr, nc = r + 1, c
                    else:
                        nr, nc = r - 1, c
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        if board[nr, nc] == 0 and labels[nr, nc] == 0:
                            labels[nr, nc] = current
                            queue[tail] = nr * 9 + nc
                            tail += 1
                            comp_flat[n_empty] = nr * 9 + nc
                            n_empty += 1
            comp_count[current] = n_empty - comp_start[current]

    # Collect legal moves with inline top-K maintenance
    # Instead of collecting ALL moves then sorting, we maintain a sorted buffer
    # of the K best logits. Softmax is deferred to only these K values.
    k = top_k
    topk_idx = np.empty(k, dtype=np.int32)
    topk_log = np.empty(k, dtype=np.float32)
    n_top = 0
    min_top = np.float32(-1e30)
    n_moves = 0

    for sr in range(9):
        for sc in range(9):
            if board[sr, sc] == 0:
                continue
            # Find reachable component IDs (deduplicated)
            adj_comp = np.empty(4, dtype=np.int8)
            n_adj = 0
            for d in range(4):
                if d == 0:
                    nr, nc = sr, sc + 1
                elif d == 1:
                    nr, nc = sr, sc - 1
                elif d == 2:
                    nr, nc = sr + 1, sc
                else:
                    nr, nc = sr - 1, sc
                if 0 <= nr < 9 and 0 <= nc < 9:
                    lbl = labels[nr, nc]
                    if lbl > 0:
                        found = False
                        for j in range(n_adj):
                            if adj_comp[j] == lbl:
                                found = True
                                break
                        if not found:
                            adj_comp[n_adj] = lbl
                            n_adj += 1

            if n_adj == 0:
                continue

            base = (sr * 9 + sc) * 81

            # Iterate only cells in reachable components
            for ci in range(n_adj):
                comp_id = adj_comp[ci]
                start = comp_start[comp_id]
                count = comp_count[comp_id]
                for j in range(count):
                    tgt = comp_flat[start + j]
                    idx = base + tgt
                    logit = pol_logits[idx]
                    n_moves += 1

                    # Maintain top-K sorted buffer (ascending, smallest first)
                    if n_top < k:
                        pos = n_top
                        while pos > 0 and topk_log[pos - 1] > logit:
                            topk_idx[pos] = topk_idx[pos - 1]
                            topk_log[pos] = topk_log[pos - 1]
                            pos -= 1
                        topk_idx[pos] = idx
                        topk_log[pos] = logit
                        n_top += 1
                        if n_top == k:
                            min_top = topk_log[0]
                    elif logit > min_top:
                        # Evict smallest, shift and insert
                        pos = 0
                        while pos < k - 1 and topk_log[pos + 1] < logit:
                            topk_idx[pos] = topk_idx[pos + 1]
                            topk_log[pos] = topk_log[pos + 1]
                            pos += 1
                        topk_idx[pos] = idx
                        topk_log[pos] = logit
                        min_top = topk_log[0]

    if n_moves == 0:
        return 0, topk_idx[:0], topk_log[:0]

    # Softmax over just the K selected values
    actual_k = n_top
    max_l = topk_log[0]
    for i in range(1, actual_k):
        if topk_log[i] > max_l:
            max_l = topk_log[i]
    total = np.float32(0.0)
    priors = np.empty(actual_k, dtype=np.float32)
    for i in range(actual_k):
        p = np.exp(topk_log[i] - max_l)
        priors[i] = p
        total += p
    for i in range(actual_k):
        priors[i] /= total

    return actual_k, topk_idx[:actual_k], priors


def _flat_to_action(flat_idx):
    """Decode flat index to ((sr, sc), (tr, tc)) action tuple."""
    src_flat = flat_idx // 81
    tgt_flat = flat_idx % 81
    return ((src_flat // 9, src_flat % 9), (tgt_flat // 9, tgt_flat % 9))


class Node:
    """MCTS tree node."""
    __slots__ = ('children', 'visit_count', 'value_sum', 'prior')

    def __init__(self, prior=0.0):
        self.children = {}  # flat_action_int -> Node
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expanded(self):
        return len(self.children) > 0


# Pre-allocated buffers for _build_obs_for_game (avoid per-call allocation)
_obs_nr = np.zeros(3, dtype=np.intp)
_obs_nc = np.zeros(3, dtype=np.intp)
_obs_ncol = np.zeros(3, dtype=np.intp)


def _build_obs_for_game(game):
    """Build observation tensor from game state."""
    nr, nc, ncol = _obs_nr, _obs_nc, _obs_ncol
    nb = game.next_balls
    nn = len(nb)
    if nn > 3:
        nn = 3
    if nn >= 1:
        pos_col = nb[0]
        nr[0] = pos_col[0][0]; nc[0] = pos_col[0][1]; ncol[0] = pos_col[1]
    else:
        nr[0] = 0; nc[0] = 0; ncol[0] = 0
    if nn >= 2:
        pos_col = nb[1]
        nr[1] = pos_col[0][0]; nc[1] = pos_col[0][1]; ncol[1] = pos_col[1]
    else:
        nr[1] = 0; nc[1] = 0; ncol[1] = 0
    if nn >= 3:
        pos_col = nb[2]
        nr[2] = pos_col[0][0]; nc[2] = pos_col[0][1]; ncol[2] = pos_col[1]
    else:
        nr[2] = 0; nc[2] = 0; ncol[2] = 0
    return build_observation(game.board, nr, nc, ncol, nn)


def _get_legal_priors(game, pol_logits_np, top_k):
    """Extract legal move priors from policy logits, keep top-K.

    Uses JIT-compiled function for speed (computes connected components,
    legal moves, softmax, and top-K entirely in numba).

    Returns dict mapping ((sr,sc),(tr,tc)) -> prior (legacy tuple format).
    """
    k, flat_idx, priors = _legal_priors_jit(
        game.board, pol_logits_np, top_k)
    if k == 0:
        return {}
    # Vectorized index -> action conversion (avoids Python loop)
    idx = flat_idx[:k].astype(np.int32)
    src_flat = idx // 81
    tgt_flat = idx % 81
    sr = src_flat // 9
    sc = src_flat % 9
    tr = tgt_flat // 9
    tc = tgt_flat % 9
    pri = priors[:k]
    return {((int(sr[i]), int(sc[i])), (int(tr[i]), int(tc[i]))): float(pri[i])
            for i in range(k)}


def _get_legal_priors_flat(board, pol_logits_np, top_k):
    """Extract legal move priors as flat-index dict {int: float}.

    Same as _get_legal_priors but avoids tuple-of-tuples key overhead.
    Returns dict mapping flat_action_int -> prior.
    """
    k, flat_idx, priors = _legal_priors_jit(board, pol_logits_np, top_k)
    if k == 0:
        return {}
    # Direct int keys — no numpy decomposition needed
    return {int(flat_idx[i]): float(priors[i]) for i in range(k)}


class MCTS:
    """Neural MCTS with PUCT selection, virtual loss batching, and value-head
    leaf evaluation.

    Args:
        net: AlphaTrainNet model (eval mode, on device)
        device: torch device
        num_simulations: simulations per search (default 400)
        c_puct: exploration constant (default 2.5)
        top_k: max children per node (default 30)
        batch_size: leaves per batched NN eval (default 16)
    """

    def __init__(self, net=None, device=None, max_score=30000.0,
                 num_simulations=400, c_puct=2.5, top_k=30, batch_size=16,
                 inference_client=None, dynamic_sims=False,
                 heuristic_value=False, value_net=None,
                 terminal_value=None, override_threshold=0.0):
        self.net = net
        self.device = device
        self.max_score = max_score
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.batch_size = batch_size
        self.inference_client = inference_client
        self.dynamic_sims = dynamic_sims
        self.heuristic_value = heuristic_value
        self.value_net = value_net
        self.terminal_value = terminal_value
        self.override_threshold = override_threshold
        self._fp16 = False
        self._sim_rng = None  # SimpleRng, set per search
        # Pre-allocate obs buffer for server mode (reused across searches)
        if inference_client is not None:
            self._obs_np_buf = np.empty(
                (batch_size, 18, 9, 9), dtype=np.float32)
        if net is not None and device is not None:
            # Detect fp16 model (JIT traced or .half())
            try:
                p = next(net.parameters())
                self._fp16 = (p.dtype == torch.float16)
            except (StopIteration, AttributeError):
                pass
            dtype = torch.float16 if self._fp16 else torch.float32
            self._obs_buf = torch.empty(batch_size, 18, 9, 9,
                                        device=device, dtype=dtype)

    def _nn_evaluate_single(self, game):
        """Single NN forward pass -> (priors dict, value scalar).

        Uses inference_client if available, otherwise direct net call.
        When value_net is set, uses it for value instead of the policy
        model's value head.
        Returns priors as {flat_int: prior}.
        """
        obs_np = _build_obs_for_game(game)
        if self.inference_client is not None:
            pol_np, value = self.inference_client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, self.top_k)
            # If value_net is loaded in-process, override server's value
            if self.value_net is not None:
                obs_t = torch.from_numpy(obs_np).unsqueeze(0)
                vnet_device = next(self.value_net.parameters()).device
                vnet_dtype = next(self.value_net.parameters()).dtype
                obs_t = obs_t.to(device=vnet_device, dtype=vnet_dtype)
                with torch.inference_mode():
                    vnet_logits = self.value_net(obs_t)
                    value = torch.sigmoid(vnet_logits.squeeze(-1)).item()
            return priors, value
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        if self._fp16:
            obs = obs.half()
        with torch.inference_mode():
            pol_logits, val_logits = self.net(obs)
            if self.value_net is not None:
                vnet_dtype = next(self.value_net.parameters()).dtype
                vnet_obs = obs.to(dtype=vnet_dtype)
                vnet_logits = self.value_net(vnet_obs)
                value = torch.sigmoid(vnet_logits.squeeze(-1)).item()
            else:
                value = self.net.predict_value(
                    val_logits, max_val=self.max_score).item()
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol_np, self.top_k)
        return priors, value

    def _select_child(self, node, min_q, max_q):
        """PUCT selection with MuZero-style Q normalization."""
        best_score = float('-inf')
        best_action = None
        best_child = None
        sqrt_parent = math.sqrt(node.visit_count)
        q_range = max_q - min_q

        for action, child in node.children.items():
            if child.visit_count > 0:
                q = child.value_sum / child.visit_count
                q_norm = (q - min_q) / q_range if q_range > 0 else 0.5
            else:
                q_norm = 0.5
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_norm + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def search(self, game, temperature=0.0, dirichlet_alpha=0.0,
               dirichlet_weight=0.0, return_policy=False,
               force_full_search=False):
        """Run batched MCTS with virtual loss from current game state.

        Args:
            temperature: move selection temperature (0=argmax, >0=proportional
                to visit counts). Used for self-play exploration.
            dirichlet_alpha: Dirichlet noise parameter for root priors (0=off).
            dirichlet_weight: weight for Dirichlet noise (e.g., 0.25).
            return_policy: if True, return (action, policy_target) where
                policy_target is a (6561,) array of normalized visit counts.

        Returns:
            action tuple, or (action, policy_target) if return_policy=True.
        """
        root = Node()

        # Seed sim_rng deterministically from full game state.
        # Hash full board + next_balls + score + turns for unique seed.
        import hashlib
        h = hashlib.md5(game.board.tobytes())
        for pos, color in game.next_balls:
            h.update(bytes([pos[0], pos[1], color]))
        h.update(game.score.to_bytes(4, 'little', signed=False))
        h.update(game.turns.to_bytes(4, 'little', signed=False))
        state_seed = int.from_bytes(h.digest()[:8], 'little')
        from game.rng import SimpleRng
        self._sim_rng = SimpleRng(state_seed)

        # Expand root
        priors, root_value = self._nn_evaluate_single(game)
        if not priors:
            return (None, None) if return_policy else None
        for action, prior in priors.items():
            root.children[action] = Node(prior=prior)
        root.visit_count = 1
        root.value_sum = root_value

        # Measure policy confidence BEFORE Dirichlet noise
        # (Dirichlet reduces P_max by ~25%, masking true confidence)
        raw_max_prior = max(c.prior for c in root.children.values()) if root.children else 0.0

        # Dirichlet noise at root for exploration
        if dirichlet_alpha > 0 and dirichlet_weight > 0:
            noise = self._sim_rng.dirichlet(
                [dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = ((1 - dirichlet_weight) * child.prior
                               + dirichlet_weight * noise[i])

        min_q = root_value
        max_q = root_value

        # Localize frequently accessed attributes for speed
        c_puct = self.c_puct
        top_k = self.top_k
        batch_size = self.batch_size
        num_sims = self.num_simulations

        # Dynamic sims: reduce search for positions where policy is confident.
        # Check raw prior (before Dirichlet) — reflects true policy confidence.
        # With top_k=30 softmax, P_max typically ranges 0.05-0.70.
        if self.dynamic_sims and root.children and not force_full_search:
            if raw_max_prior > 0.5:
                num_sims = max(50, num_sims // 10)
            elif raw_max_prior > 0.3:
                num_sims = max(100, num_sims // 4)
        self._last_effective_sims = num_sims
        self._last_max_prior = raw_max_prior
        sim_rng = self._sim_rng
        use_server = self.inference_client is not None
        if use_server:
            obs_np_buf = self._obs_np_buf

        sims_done = 0
        while sims_done < num_sims:
            bs = min(batch_size, num_sims - sims_done)
            batch_paths = []
            batch_games = []
            batch_leaf_nodes = []
            batch_game_over = []
            obs_count = 0

            # === SELECT B leaves with virtual loss ===
            for _ in range(bs):
                node = root
                sim_game = game.clone(rng=sim_rng)
                path = [node]

                while node.children and not sim_game.game_over:
                    # Open-loop PUCT: filter to moves legal on
                    # this sim's actual board (stochastic spawns
                    # may differ from the sim that expanded this node).
                    # Check: src occupied, tgt empty, path reachable.
                    best_score = -1e30
                    best_action = 0
                    best_child = None
                    sqrt_parent = math.sqrt(node.visit_count)
                    q_range = max_q - min_q
                    board = sim_game.board
                    cc_labels = _label_empty_components(board)
                    for act_i, child in node.children.items():
                        src_f = act_i // 81
                        tgt_f = act_i % 81
                        sr, sc = src_f // 9, src_f % 9
                        tr, tc = tgt_f // 9, tgt_f % 9
                        if board[sr, sc] == 0:
                            continue
                        if board[tr, tc] != 0:
                            continue
                        if not _is_reachable(cc_labels, sr, sc, tr, tc):
                            continue
                        vc = child.visit_count
                        if vc > 0:
                            q = child.value_sum / vc
                            q_norm = (q - min_q) / q_range if q_range > 0 else 0.5
                        else:
                            q_norm = 0.5
                        u = c_puct * child.prior * sqrt_parent / (1 + vc)
                        score = q_norm + u
                        if score > best_score:
                            best_score = score
                            best_action = act_i
                            best_child = child

                    if best_child is None:
                        break  # no legal children on this board

                    src_flat = best_action // 81
                    tgt_flat = best_action % 81
                    sim_game.trusted_move(
                        src_flat // 9, src_flat % 9,
                        tgt_flat // 9, tgt_flat % 9)
                    path.append(best_child)
                    node = best_child

                for n in path:
                    n.visit_count += 1
                    n.value_sum -= VIRTUAL_LOSS

                batch_paths.append(path)
                batch_leaf_nodes.append(node)
                batch_games.append(sim_game)

                if sim_game.game_over:
                    batch_game_over.append(True)
                else:
                    batch_game_over.append(False)
                    obs = _build_obs_for_game(sim_game)
                    if use_server:
                        obs_np_buf[obs_count] = obs
                    else:
                        t = torch.from_numpy(obs)
                        if self._fp16:
                            t = t.half()
                        self._obs_buf[obs_count] = t
                    obs_count += 1

            # === BATCH EVALUATE ===
            if obs_count > 0:
                if use_server:
                    pol_np, val_np = self.inference_client.evaluate_batch(
                        obs_np_buf, obs_count)
                    # No copy needed — shared memory is safe until next
                    # evaluate_batch call (worker is single-threaded)
                else:
                    with torch.inference_mode():
                        pol_logits, val_logits = self.net(
                            self._obs_buf[:obs_count])
                        values_t = self.net.predict_value(
                            val_logits, max_val=self.max_score)
                    pol_np = pol_logits.float().cpu().numpy()  # fp32 for JIT
                    val_np = values_t.cpu().numpy()

            # === VALUE NET BATCH EVAL (if separate value network) ===
            vnet_values = None
            if self.value_net is not None and obs_count > 0:
                vnet_device = next(self.value_net.parameters()).device
                vnet_dtype = next(self.value_net.parameters()).dtype
                if use_server:
                    vnet_obs = torch.from_numpy(
                        obs_np_buf[:obs_count].copy()).to(
                        device=vnet_device, dtype=vnet_dtype)
                else:
                    vnet_obs = self._obs_buf[:obs_count].to(
                        device=vnet_device, dtype=vnet_dtype)
                with torch.inference_mode():
                    vnet_logits = self.value_net(vnet_obs)
                    vnet_values = torch.sigmoid(
                        vnet_logits.squeeze(-1)).float().cpu().numpy()

            # === EXPAND + BACKUP ===
            nn_idx = 0
            for b in range(bs):
                path = batch_paths[b]

                if batch_game_over[b]:
                    if self.terminal_value is not None:
                        value = self.terminal_value
                    elif self.value_net is not None:
                        value = 0.0
                    else:
                        value = float(batch_games[b].score)
                elif self.heuristic_value:
                    value = _evaluate_board(batch_games[b].board)
                else:
                    # Expand with policy priors
                    node = batch_leaf_nodes[b]
                    k, flat_idx, pri = _legal_priors_jit(
                        batch_games[b].board, pol_np[nn_idx], top_k)
                    ch = node.children
                    for i in range(k):
                        action_key = int(flat_idx[i])
                        if action_key not in ch:
                            ch[action_key] = Node(prior=float(pri[i]))
                    # Value: use separate value_net if available
                    if vnet_values is not None:
                        value = float(vnet_values[nn_idx])
                    else:
                        value = float(val_np[nn_idx])
                    nn_idx += 1

                min_q = min(min_q, value)
                max_q = max(max_q, value)

                for n in path:
                    n.value_sum += VIRTUAL_LOSS + value

            sims_done += bs

        # Build policy target from visit counts
        if return_policy or temperature > 0:
            actions = list(root.children.keys())
            visits = np.array([root.children[a].visit_count
                               for a in actions], dtype=np.float32)

        if return_policy:
            # Build full 6561-dim policy target
            # Actions are already flat indices — write directly
            policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
            visit_sum = visits.sum()
            if visit_sum > 0:
                for i, a in enumerate(actions):
                    policy_target[a] = visits[i] / visit_sum

        # Select move
        if temperature > 0 and len(root.children) > 0:
            # Temperature-weighted sampling
            adjusted = visits ** (1.0 / temperature)
            probs = adjusted / adjusted.sum()
            # Weighted sampling using our deterministic RNG
            u = self._sim_rng.next_f64()
            cumsum = 0.0
            chosen_idx = len(actions) - 1  # fallback to last
            for ci in range(len(actions)):
                cumsum += probs[ci]
                if u < cumsum:
                    chosen_idx = ci
                    break
            flat_action = actions[chosen_idx]
        else:
            # Greedy argmax
            flat_action = max(root.children.items(),
                              key=lambda x: x[1].visit_count)[0]

        # Override threshold: only override policy's top move if MCTS
        # has significantly more visits on a different move.
        # Prevents destructive coin-flip overrides on near-ties.
        if self.override_threshold > 0 and temperature == 0:
            # Find policy's preferred move (highest prior)
            policy_action = max(root.children.items(),
                                key=lambda x: x[1].prior)[0]
            mcts_action = max(root.children.items(),
                              key=lambda x: x[1].visit_count)[0]
            if mcts_action != policy_action:
                pol_visits = root.children[policy_action].visit_count
                mcts_visits = root.children[mcts_action].visit_count
                if mcts_visits <= pol_visits * (1 + self.override_threshold):
                    flat_action = policy_action  # trust policy on near-ties

        # Decode flat action to tuple format for callers
        action = _flat_to_action(flat_action)

        self._last_root = root
        self._last_min_q = min_q
        self._last_max_q = max_q

        if return_policy:
            return action, policy_target
        return action


def make_mcts_player(net, device, max_score=30000.0,
                     num_simulations=400, c_puct=2.5, top_k=30,
                     batch_size=16, value_net=None, terminal_value=None,
                     override_threshold=0.0):
    """Create MCTS player function for use with evaluate."""
    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=num_simulations,
                c_puct=c_puct, top_k=top_k, batch_size=batch_size,
                value_net=value_net, terminal_value=terminal_value,
                override_threshold=override_threshold)

    def player(game):
        return mcts.search(game)

    return player
