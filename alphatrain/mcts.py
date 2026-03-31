"""Neural MCTS for Color Lines 98.

AlphaZero-style Monte Carlo Tree Search using the trained ResNet:
- Policy head provides move priors (which moves to explore)
- Value head evaluates leaf positions (replaces heuristic rollouts)
- PUCT selection balances exploration vs exploitation
- Virtual loss batching: collect B leaves per batch, one NN forward pass
- Determinized: each simulation samples independent ball spawns via game.clone()
- MuZero-style Q normalization for score-based (not win/loss) values

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

VIRTUAL_LOSS = 1.0


@njit(cache=True)
def _legal_priors_jit(board, pol_logits, top_k):
    """Compute top-K legal move priors entirely in JIT.

    Returns (count, flat_indices[top_k], priors[top_k]).
    """
    # Label connected components of empty cells
    labels = np.zeros((9, 9), dtype=np.int8)
    queue = np.empty(162, dtype=np.int32)
    current = np.int8(0)
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

    # Collect legal moves and logits
    move_idx = np.empty(6561, dtype=np.int32)
    move_log = np.empty(6561, dtype=np.float32)
    n_moves = 0

    for sr in range(9):
        for sc in range(9):
            if board[sr, sc] == 0:
                continue
            # Find reachable component labels
            adj = np.zeros(82, dtype=np.int8)
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
                        adj[lbl] = 1

            base = (sr * 9 + sc) * 81
            for tr in range(9):
                for tc in range(9):
                    if labels[tr, tc] > 0 and adj[labels[tr, tc]] == 1:
                        idx = base + tr * 9 + tc
                        move_idx[n_moves] = idx
                        move_log[n_moves] = pol_logits[idx]
                        n_moves += 1

    if n_moves == 0:
        return 0, move_idx[:0], move_log[:0]

    # Softmax
    max_l = move_log[0]
    for i in range(1, n_moves):
        if move_log[i] > max_l:
            max_l = move_log[i]
    total = np.float32(0.0)
    probs = np.empty(n_moves, dtype=np.float32)
    for i in range(n_moves):
        p = np.exp(move_log[i] - max_l)
        probs[i] = p
        total += p
    for i in range(n_moves):
        probs[i] /= total

    # Top-K selection
    k = min(top_k, n_moves)
    out_idx = np.empty(k, dtype=np.int32)
    out_pri = np.empty(k, dtype=np.float32)
    used = np.zeros(n_moves, dtype=np.int8)
    total_p = np.float32(0.0)
    for j in range(k):
        best_i = -1
        best_v = np.float32(-1.0)
        for i in range(n_moves):
            if used[i] == 0 and probs[i] > best_v:
                best_v = probs[i]
                best_i = i
        used[best_i] = 1
        out_idx[j] = move_idx[best_i]
        out_pri[j] = probs[best_i]
        total_p += probs[best_i]
    # Renormalize
    if total_p > 0:
        for i in range(k):
            out_pri[i] /= total_p

    return k, out_idx, out_pri


class Node:
    """MCTS tree node."""
    __slots__ = ('children', 'visit_count', 'value_sum', 'prior')

    def __init__(self, prior=0.0):
        self.children = {}  # (source, target) -> Node
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


def _build_obs_for_game(game):
    """Build observation tensor from game state."""
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    return build_observation(game.board, nr, nc, ncol, nn)


def _get_legal_priors(game, pol_logits_np, top_k):
    """Extract legal move priors from policy logits, keep top-K.

    Uses JIT-compiled function for speed (computes connected components,
    legal moves, softmax, and top-K entirely in numba).
    """
    k, flat_idx, priors = _legal_priors_jit(
        game.board, pol_logits_np, top_k)
    if k == 0:
        return {}
    # Vectorized index → action conversion (avoids Python loop)
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
                 inference_client=None):
        self.net = net
        self.device = device
        self.max_score = max_score
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.batch_size = batch_size
        self.inference_client = inference_client
        self._fp16 = False
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
        """
        obs_np = _build_obs_for_game(game)
        if self.inference_client is not None:
            pol_np, value = self.inference_client.evaluate(obs_np)
            priors = _get_legal_priors(game, pol_np, self.top_k)
            return priors, value
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)
        if self._fp16:
            obs = obs.half()
        with torch.inference_mode():
            pol_logits, val_logits = self.net(obs)
            value = self.net.predict_value(val_logits, max_val=self.max_score).item()
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors(game, pol_np, self.top_k)
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
                q = child.q_value
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
               dirichlet_weight=0.0, return_policy=False):
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

        # Expand root
        priors, root_value = self._nn_evaluate_single(game)
        if not priors:
            return (None, None) if return_policy else None
        for action, prior in priors.items():
            root.children[action] = Node(prior=prior)
        root.visit_count = 1
        root.value_sum = root_value

        # Dirichlet noise at root for exploration
        if dirichlet_alpha > 0 and dirichlet_weight > 0:
            noise = np.random.dirichlet(
                [dirichlet_alpha] * len(root.children))
            for i, child in enumerate(root.children.values()):
                child.prior = ((1 - dirichlet_weight) * child.prior
                               + dirichlet_weight * noise[i])

        min_q = root_value
        max_q = root_value

        # Pre-allocate obs buffer for server mode
        if self.inference_client is not None:
            obs_np_buf = np.empty(
                (self.batch_size, 18, 9, 9), dtype=np.float32)

        sims_done = 0
        while sims_done < self.num_simulations:
            bs = min(self.batch_size, self.num_simulations - sims_done)
            batch_paths = []
            batch_games = []
            batch_leaf_nodes = []
            batch_game_over = []
            obs_count = 0

            # === SELECT B leaves with virtual loss ===
            for _ in range(bs):
                node = root
                sim_game = game.clone()
                path = [node]

                while node.expanded() and not sim_game.game_over:
                    action, child = self._select_child(node, min_q, max_q)
                    sim_game.move(action[0], action[1])
                    path.append(child)
                    node = child

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
                    if self.inference_client is not None:
                        obs_np_buf[obs_count] = obs
                    else:
                        t = torch.from_numpy(obs)
                        if self._fp16:
                            t = t.half()
                        self._obs_buf[obs_count] = t
                    obs_count += 1

            # === BATCH EVALUATE ===
            if obs_count > 0:
                if self.inference_client is not None:
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

            # === EXPAND + BACKUP ===
            nn_idx = 0
            for b in range(bs):
                path = batch_paths[b]

                if batch_game_over[b]:
                    value = float(batch_games[b].score)
                else:
                    value = float(val_np[nn_idx])
                    node = batch_leaf_nodes[b]
                    priors = _get_legal_priors(
                        batch_games[b], pol_np[nn_idx], self.top_k)
                    for act, prior in priors.items():
                        node.children[act] = Node(prior=prior)
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
            policy_target = np.zeros(NUM_MOVES, dtype=np.float32)
            visit_sum = visits.sum()
            if visit_sum > 0:
                for i, a in enumerate(actions):
                    (sr, sc), (tr, tc) = a
                    idx = (sr * 9 + sc) * 81 + tr * 9 + tc
                    policy_target[idx] = visits[i] / visit_sum

        # Select move
        if temperature > 0 and len(root.children) > 0:
            # Temperature-weighted sampling
            adjusted = visits ** (1.0 / temperature)
            probs = adjusted / adjusted.sum()
            chosen_idx = np.random.choice(len(actions), p=probs)
            action = actions[chosen_idx]
        else:
            # Greedy argmax
            action = max(root.children.items(),
                         key=lambda x: x[1].visit_count)[0]

        if return_policy:
            return action, policy_target
        return action


def make_mcts_player(net, device, max_score=30000.0,
                     num_simulations=400, c_puct=2.5, top_k=30,
                     batch_size=16):
    """Create MCTS player function for use with evaluate."""
    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=num_simulations,
                c_puct=c_puct, top_k=top_k, batch_size=batch_size)

    def player(game):
        return mcts.search(game)

    return player
