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

from alphatrain.observation import build_observation

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


def _get_legal_priors(game, pol_logits_np, top_k):
    """Extract legal move priors from policy logits, keep top-K.

    Vectorized: builds all legal moves at once via numpy indexing.
    """
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

    def _nn_evaluate_single(self, game):
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
                q_norm = 0.5
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = q_norm + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def search(self, game):
        """Run batched MCTS with virtual loss from current game state."""
        root = Node()

        # Expand root with single eval
        priors, root_value = self._nn_evaluate_single(game)
        if not priors:
            return None
        for action, prior in priors.items():
            root.children[action] = Node(prior=prior)
        root.visit_count = 1
        root.value_sum = root_value

        min_q = root_value
        max_q = root_value

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

                # Descend while expanded
                while node.expanded() and not sim_game.game_over:
                    action, child = self._select_child(node, min_q, max_q)
                    sim_game.move(action[0], action[1])
                    path.append(child)
                    node = child

                # Apply virtual loss to diversify subsequent selections
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
                    self._obs_buf[obs_count] = torch.from_numpy(
                        _build_obs_for_game(sim_game))
                    obs_count += 1

            # === BATCH EVALUATE all non-terminal leaves ===
            if obs_count > 0:
                with torch.no_grad():
                    pol_logits, val_logits = self.net(self._obs_buf[:obs_count])
                    values_t = self.net.predict_value(
                        val_logits, max_val=self.max_score)
                pol_np = pol_logits.cpu().numpy()
                val_np = values_t.cpu().numpy()

            # === EXPAND + UNDO VIRTUAL LOSS + BACKUP ===
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
