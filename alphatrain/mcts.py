"""Neural MCTS for Color Lines 98.

AlphaZero-style Monte Carlo Tree Search using the trained ResNet:
- Policy head provides move priors (which moves to explore)
- Value head evaluates leaf positions (replaces heuristic rollouts)
- PUCT selection balances exploration vs exploitation
- Determinized: each simulation samples independent ball spawns via game.clone()
- MuZero-style Q normalization for score-based (not win/loss) values

Usage:
    python -m alphatrain.evaluate --player mcts --simulations 200 --games 20
"""

import math
import numpy as np
import torch

from alphatrain.observation import build_observation

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561


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
    """Extract legal move priors from policy logits, keep top-K."""
    source_mask = game.get_source_mask()
    moves = []
    logits = []

    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if source_mask[sr, sc] == 0:
                continue
            target_mask = game.get_target_mask((sr, sc))
            for tr in range(BOARD_SIZE):
                for tc in range(BOARD_SIZE):
                    if target_mask[tr, tc] > 0:
                        idx = (sr * 9 + sc) * 81 + tr * 9 + tc
                        moves.append(((sr, sc), (tr, tc)))
                        logits.append(pol_logits_np[idx])

    if not moves:
        return {}

    logits = np.array(logits, dtype=np.float32)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # Keep top-K by prior
    if len(moves) > top_k:
        top_idx = np.argpartition(probs, -top_k)[-top_k:]
        moves = [moves[i] for i in top_idx]
        probs = probs[top_idx]
        probs /= probs.sum()

    return {moves[i]: float(probs[i]) for i in range(len(moves))}


class MCTS:
    """Neural MCTS with PUCT selection and value-head leaf evaluation.

    Args:
        net: AlphaTrainNet model (eval mode, on device)
        device: torch device
        num_simulations: simulations per search (default 200)
        c_puct: exploration constant (default 2.5)
        top_k: max children per node (default 30)
    """

    def __init__(self, net, device, max_score=30000.0,
                 num_simulations=200, c_puct=2.5, top_k=30):
        self.net = net
        self.device = device
        self.max_score = max_score
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.top_k = top_k

    def _nn_evaluate(self, game):
        """Single NN forward pass -> (priors dict, value scalar)."""
        obs = torch.from_numpy(
            _build_obs_for_game(game)
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pol_logits, val_logits = self.net(obs)
            value = self.net.predict_value(val_logits, max_val=self.max_score).item()
        priors = _get_legal_priors(
            game, pol_logits[0].cpu().numpy(), self.top_k)
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
                q_norm = 0.5  # optimistic prior for unvisited
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_norm + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def search(self, game):
        """Run MCTS from current game state, return best (source, target)."""
        root = Node()

        # Expand root
        priors, root_value = self._nn_evaluate(game)
        if not priors:
            return None
        for action, prior in priors.items():
            root.children[action] = Node(prior=prior)
        root.visit_count = 1
        root.value_sum = root_value

        min_q = root_value
        max_q = root_value

        for _ in range(self.num_simulations):
            node = root
            sim_game = game.clone()
            path = [node]

            # Selection: descend while expanded
            while node.expanded() and not sim_game.game_over:
                action, child = self._select_child(node, min_q, max_q)
                sim_game.move(action[0], action[1])
                path.append(child)
                node = child

            # Leaf evaluation
            if sim_game.game_over:
                value = float(sim_game.score)
            else:
                priors, value = self._nn_evaluate(sim_game)
                for act, prior in priors.items():
                    node.children[act] = Node(prior=prior)

            # Update Q bounds
            min_q = min(min_q, value)
            max_q = max(max_q, value)

            # Backup
            for n in path:
                n.visit_count += 1
                n.value_sum += value

        # Return most-visited child
        best_action = max(root.children.items(),
                          key=lambda x: x[1].visit_count)[0]
        return best_action


def make_mcts_player(net, device, max_score=30000.0,
                     num_simulations=200, c_puct=2.5, top_k=30):
    """Create MCTS player function for use with evaluate."""
    mcts = MCTS(net, device, max_score=max_score,
                num_simulations=num_simulations,
                c_puct=c_puct, top_k=top_k)

    def player(game):
        return mcts.search(game)

    return player
