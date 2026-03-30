"""Afterstate MCTS for Color Lines 98 (Open-Loop).

Rigorous stochastic MCTS with afterstate evaluation:
- Tree nodes represent game states (post-spawn)
- At leaf expansion: get policy priors from game state, create children
- Leaf VALUE is computed from the afterstate (pre-spawn, deterministic)
  which is what the NN was trained on
- Spawns happen naturally via game.clone() + game.move() during traversal
- 400 simulations average over spawn distributions (Monte Carlo sampling)

This is the formal "Open-Loop MCTS" for stochastic games: the tree
traverses through random spawns, and Q-values converge to the true
expected value across the spawn distribution.

Usage:
    python -m alphatrain.evaluate --player mcts --simulations 400 --games 20
"""

import math
import numpy as np
import torch

from alphatrain.observation import build_observation
from alphatrain.afterstate import compute_afterstate

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561

VIRTUAL_LOSS = 1.0


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


def _build_afterstate_obs(afterstate_board, parent_next_balls):
    """Build 18-channel observation for an afterstate.

    Uses the afterstate board and the parent state's next_balls
    (visible balls that will spawn onto this board).
    """
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(parent_next_balls), 3)
    for i, ((r, c), col) in enumerate(parent_next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    return build_observation(afterstate_board, nr, nc, ncol, nn)


def _get_legal_priors(game, pol_logits_np, top_k):
    """Extract legal move priors from policy logits, keep top-K."""
    source_mask = game.get_source_mask()
    src_rows, src_cols = np.where(source_mask > 0)

    if len(src_rows) == 0:
        return {}

    all_sr, all_sc, all_tr, all_tc = [], [], [], []
    for i in range(len(src_rows)):
        sr, sc = int(src_rows[i]), int(src_cols[i])
        target_mask = game.get_target_mask((sr, sc))
        tgt_rows, tgt_cols = np.where(target_mask > 0)
        n = len(tgt_rows)
        if n > 0:
            all_sr.append(np.full(n, sr, dtype=np.int32))
            all_sc.append(np.full(n, sc, dtype=np.int32))
            all_tr.append(tgt_rows.astype(np.int32))
            all_tc.append(tgt_cols.astype(np.int32))

    if not all_sr:
        return {}

    sr_arr = np.concatenate(all_sr)
    sc_arr = np.concatenate(all_sc)
    tr_arr = np.concatenate(all_tr)
    tc_arr = np.concatenate(all_tc)

    indices = (sr_arr * 9 + sc_arr) * 81 + tr_arr * 9 + tc_arr
    logits = pol_logits_np[indices]
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    if len(probs) > top_k:
        top_idx = np.argpartition(probs, -top_k)[-top_k:]
        sr_arr = sr_arr[top_idx]
        sc_arr = sc_arr[top_idx]
        tr_arr = tr_arr[top_idx]
        tc_arr = tc_arr[top_idx]
        probs = probs[top_idx]
        probs /= probs.sum()

    return {((int(sr_arr[i]), int(sc_arr[i])), (int(tr_arr[i]), int(tc_arr[i]))): float(probs[i])
            for i in range(len(probs))}


class MCTS:
    """Open-loop afterstate MCTS with virtual loss batching.

    Leaf evaluation uses afterstates (deterministic, what the NN was
    trained on). Tree traversal goes through spawns via game.clone().

    Args:
        net: AlphaTrainNet model (eval mode, on device)
        device: torch device
        num_simulations: simulations per search (default 400)
        c_puct: exploration constant (default 2.5)
        top_k: max children per node (default 30)
        batch_size: max leaves per NN batch (default 16)
    """

    def __init__(self, net, device, max_score=30000.0,
                 num_simulations=400, c_puct=2.5, top_k=30, batch_size=16):
        self.net = net
        self.device = device
        self.max_score = max_score
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k = top_k
        self.batch_size = batch_size
        self._obs_buf = torch.empty(batch_size, 18, 9, 9, device=device)

    def _expand_node(self, node, game):
        """Expand node: get policy priors, create children."""
        obs = torch.from_numpy(
            _build_obs_for_game(game)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pol_logits, _ = self.net(obs)
        priors = _get_legal_priors(
            game, pol_logits[0].cpu().numpy(), self.top_k)
        for action, prior in priors.items():
            node.children[action] = Node(prior=prior)

    def _evaluate_afterstate(self, game, action):
        """Compute afterstate value: score_delta + NN prediction.

        The afterstate is the board after move + line clears, before spawns.
        This is deterministic and is what the NN was trained on.
        """
        (sr, sc), (tr, tc) = action
        board_int64 = game.board.astype(np.int64)
        after_board, score_delta = compute_afterstate(
            board_int64, sr, sc, tr, tc)
        obs = _build_afterstate_obs(
            after_board.astype(np.int8), game.next_balls)
        return obs, int(score_delta)

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

    def search(self, game):
        """Run open-loop afterstate MCTS from current game state."""
        root = Node()

        # Expand root
        self._expand_node(root, game)
        if not root.children:
            return None
        root.visit_count = 1

        min_q = 0.0
        max_q = self.max_score

        sims_done = 0
        while sims_done < self.num_simulations:
            bs = min(self.batch_size, self.num_simulations - sims_done)
            batch_paths = []
            batch_actions = []  # the action that led to each leaf
            batch_games = []    # parent game state at each leaf
            batch_game_over = []
            obs_count = 0

            # === SELECT B leaves with virtual loss ===
            for _ in range(bs):
                node = root
                sim_game = game.clone()
                path = [node]
                last_action = None

                # Descend while expanded
                while node.expanded() and not sim_game.game_over:
                    action, child = self._select_child(node, min_q, max_q)
                    last_action = action
                    # Save pre-move game for afterstate computation at leaf
                    parent_game_board = sim_game.board.copy()
                    parent_next_balls = list(sim_game.next_balls)
                    # Execute move (includes spawns — for tree traversal)
                    sim_game.move(action[0], action[1])
                    path.append(child)
                    node = child

                # Apply virtual loss
                for n in path:
                    n.visit_count += 1
                    n.value_sum -= VIRTUAL_LOSS

                batch_paths.append(path)
                batch_games.append(sim_game)

                if sim_game.game_over:
                    batch_game_over.append(True)
                    batch_actions.append(None)
                else:
                    batch_game_over.append(False)
                    # Compute afterstate observation for the leaf
                    # (the move that led here, evaluated on pre-spawn board)
                    (sr, sc), (tr, tc) = last_action
                    after_board, score_delta = compute_afterstate(
                        parent_game_board.astype(np.int64), sr, sc, tr, tc)
                    self._obs_buf[obs_count] = torch.from_numpy(
                        _build_afterstate_obs(
                            after_board.astype(np.int8), parent_next_balls))
                    batch_actions.append(
                        (score_delta, obs_count))
                    obs_count += 1

            # === BATCH EVALUATE afterstates + EXPAND leaves ===
            nn_values = None
            if obs_count > 0:
                with torch.no_grad():
                    _, val_logits = self.net(self._obs_buf[:obs_count])
                    nn_values = self.net.predict_value(
                        val_logits, max_val=self.max_score).cpu().numpy()

            for b in range(bs):
                path = batch_paths[b]
                leaf = path[-1]

                if batch_game_over[b]:
                    value = 0.0
                else:
                    score_delta, obs_idx = batch_actions[b]
                    value = int(score_delta) + float(nn_values[obs_idx])
                    # Expand the leaf for future traversals
                    if not leaf.expanded():
                        self._expand_node(leaf, batch_games[b])

                min_q = min(min_q, value)
                max_q = max(max_q, value)

                # Undo virtual loss, apply real value
                for n in path:
                    n.value_sum += VIRTUAL_LOSS + value

            sims_done += bs

        # Return most-visited child
        return max(root.children.items(),
                   key=lambda x: x[1].visit_count)[0]


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
