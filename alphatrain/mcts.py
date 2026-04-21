"""Neural MCTS for Color Lines 98.

AlphaZero-style Monte Carlo Tree Search using the trained ResNet:
- Policy head provides move priors (which moves to explore)
- PUCT selection balances exploration vs exploitation (policy-only, no value)
- Virtual loss batching: collect B leaves per batch, one NN forward pass
- Determinized: each simulation samples independent ball spawns via game.clone()

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

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561


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
    __slots__ = ('children', 'visit_count', 'prior')

    def __init__(self, prior=0.0):
        self.children = {}  # flat_action_int -> Node
        self.visit_count = 0
        self.prior = prior

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
    """Neural MCTS with PUCT selection and virtual loss batching.

    Policy-only: no value head. PUCT score = c_puct * prior * sqrt(N_parent) / (1 + N_child).

    Args:
        net: AlphaTrainNet model (eval mode, on device)
        device: torch device
        num_simulations: simulations per search (default 400)
        c_puct: exploration constant (default 2.5)
        top_k: max children per node (default 30)
        batch_size: leaves per batched NN eval (default 16)
    """

    def __init__(self, net=None, device=None,
                 num_simulations=400, c_puct=2.5, top_k=30, batch_size=16,
                 inference_client=None, dynamic_sims=False):
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.batch_size = batch_size
        self.inference_client = inference_client
        self.dynamic_sims = dynamic_sims
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
        """Single NN forward pass -> priors dict.

        Uses inference_client if available, otherwise direct net call.
        Returns priors as {flat_int: prior}.
        """
        obs_np = _build_obs_for_game(game)
        if self.inference_client is not None:
            pol_np = self.inference_client.evaluate(obs_np)
            return _get_legal_priors_flat(game.board, pol_np, self.top_k)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        if self._fp16:
            obs = obs.half()
        with torch.inference_mode():
            pol_logits, _ = self.net(obs)
        pol_np = pol_logits[0].float().cpu().numpy()
        return _get_legal_priors_flat(game.board, pol_np, self.top_k)

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

        # Seed sim_rng from game state for reproducibility across modes.
        # Same board position → same MCTS simulation spawns → identical trees
        # regardless of local vs server execution.
        # NOTE: Python hash() is randomized per-process (PYTHONHASHSEED),
        # so we use a deterministic hash from raw board bytes instead.
        board_bytes = game.board.tobytes()
        state_seed = int.from_bytes(board_bytes[:8], 'little')
        state_seed = (state_seed ^ (game.score * 31) ^ (game.turns * 7)) & 0xFFFFFFFF
        from game.rng import SimpleRng
        self._sim_rng = SimpleRng(state_seed)

        # Expand root
        priors = self._nn_evaluate_single(game)
        if not priors:
            return (None, None) if return_policy else None
        for action, prior in priors.items():
            root.children[action] = Node(prior=prior)
        root.visit_count = 1

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
                    # Inline PUCT selection (policy-only, no Q component)
                    best_score = -1e30
                    best_action = 0
                    best_child = None
                    sqrt_parent = math.sqrt(node.visit_count)
                    for act_i, child in node.children.items():
                        score = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
                        if score > best_score:
                            best_score = score
                            best_action = act_i
                            best_child = child

                    # Decode flat action and execute trusted move
                    # (move is guaranteed legal — from _legal_priors_jit)
                    src_flat = best_action // 81
                    tgt_flat = best_action % 81
                    sim_game.trusted_move(
                        src_flat // 9, src_flat % 9,
                        tgt_flat // 9, tgt_flat % 9)
                    path.append(best_child)
                    node = best_child

                for n in path:
                    n.visit_count += 1

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
                    pol_np = self.inference_client.evaluate_batch(
                        obs_np_buf, obs_count)
                else:
                    with torch.inference_mode():
                        pol_logits, _ = self.net(self._obs_buf[:obs_count])
                    pol_np = pol_logits.float().cpu().numpy()

            # === EXPAND ===
            nn_idx = 0
            for b in range(bs):
                if not batch_game_over[b]:
                    node = batch_leaf_nodes[b]
                    # Inline expand: JIT -> Nodes directly (no intermediate dict)
                    k, flat_idx, pri = _legal_priors_jit(
                        batch_games[b].board, pol_np[nn_idx], top_k)
                    ch = node.children
                    for i in range(k):
                        ch[int(flat_idx[i])] = Node(prior=float(pri[i]))
                    nn_idx += 1

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

        # Decode flat action to tuple format for callers
        action = _flat_to_action(flat_action)

        if return_policy:
            return action, policy_target
        return action


def make_mcts_player(net, device, num_simulations=400, c_puct=2.5,
                     top_k=30, batch_size=16):
    """Create MCTS player function for use with evaluate."""
    mcts = MCTS(net, device,
                num_simulations=num_simulations,
                c_puct=c_puct, top_k=top_k, batch_size=batch_size)

    def player(game):
        return mcts.search(game)

    return player
