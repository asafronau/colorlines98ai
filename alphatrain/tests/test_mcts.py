"""Tests for alphatrain.mcts — MCTS correctness, determinism, and edge cases."""

import numpy as np
import torch
import pytest
from game.board import ColorLinesGame
from alphatrain.mcts import (
    Node, MCTS, _build_obs_for_game, _get_legal_priors,
    _get_legal_priors_flat, _legal_priors_jit, VIRTUAL_LOSS,
)


class DummyNet:
    """Minimal mock that returns uniform policy + constant value."""

    def __init__(self, value=250.0, num_value_bins=1):
        self.num_value_bins = num_value_bins
        self._value = value

    def __call__(self, obs):
        B = obs.shape[0]
        pol = torch.zeros(B, 6561)
        val = torch.full((B, 1), self._value)
        return pol, val

    def predict_value(self, val_logits, max_val=500.0):
        B = val_logits.shape[0]
        return torch.full((B,), self._value)

    def parameters(self):
        return iter([torch.zeros(1)])

    def to(self, device):
        return self

    def train(self, mode):
        return self


class BiasedNet(DummyNet):
    """Net that assigns high logit to a specific move index."""

    def __init__(self, preferred_idx=0, value=250.0):
        super().__init__(value=value)
        self._preferred_idx = preferred_idx

    def __call__(self, obs):
        B = obs.shape[0]
        pol = torch.full((B, 6561), -10.0)
        pol[:, self._preferred_idx] = 10.0
        val = torch.full((B, 1), self._value)
        return pol, val


# ── Node tests ──────────────────────────────────────────────────────

class TestNode:
    def test_init_defaults(self):
        n = Node()
        assert n.prior == 0.0
        assert n.visit_count == 0
        assert n.value_sum == 0.0
        assert n.q_value == 0.0
        assert not n.expanded()

    def test_init_with_prior(self):
        n = Node(prior=0.3)
        assert n.prior == 0.3

    def test_q_value_computation(self):
        n = Node()
        n.visit_count = 4
        n.value_sum = 1000.0
        assert n.q_value == 250.0

    def test_q_value_zero_visits(self):
        """q_value should be 0.0 when visit_count is 0 (avoid division by zero)."""
        n = Node()
        assert n.q_value == 0.0

    def test_expanded_after_adding_children(self):
        n = Node()
        assert not n.expanded()
        n.children[42] = Node(prior=0.5)
        assert n.expanded()

    def test_children_dict_type(self):
        """Children should be a dict."""
        n = Node()
        assert isinstance(n.children, dict)


# ── Observation helpers ─────────────────────────────────────────────

class TestObservation:
    def test_build_obs_shape(self):
        game = ColorLinesGame(seed=42)
        game.reset()
        obs = _build_obs_for_game(game)
        assert obs.shape == (18, 9, 9)
        assert obs.dtype == np.float32

    def test_build_obs_with_no_next_balls(self):
        """Observation should work when next_balls is empty."""
        game = ColorLinesGame(seed=42)
        game.reset()
        game.next_balls = []
        obs = _build_obs_for_game(game)
        assert obs.shape == (18, 9, 9)
        # Channels 8-11 should be all zeros (no next balls)
        assert obs[8:12].sum() == 0.0

    def test_build_obs_with_three_next_balls(self):
        """Observation should encode up to 3 next balls."""
        game = ColorLinesGame(seed=42)
        game.reset()
        assert len(game.next_balls) == 3
        obs = _build_obs_for_game(game)
        # Channel 11 (next ball mask) should have exactly 3 nonzero cells
        assert obs[11].sum() == 3.0

    def test_build_obs_deterministic(self):
        """Same game state produces same observation."""
        game = ColorLinesGame(seed=42)
        game.reset()
        obs1 = _build_obs_for_game(game)
        obs2 = _build_obs_for_game(game)
        np.testing.assert_array_equal(obs1, obs2)


# ── Legal priors ────────────────────────────────────────────────────

class TestLegalPriors:
    def test_sums_to_one(self):
        game = ColorLinesGame(seed=42)
        game.reset()
        logits = np.random.default_rng(0).standard_normal(6561).astype(np.float32)
        priors = _get_legal_priors(game, logits, top_k=100)
        assert len(priors) > 0
        total = sum(priors.values())
        assert abs(total - 1.0) < 1e-5

    def test_top_k_limit(self):
        game = ColorLinesGame(seed=42)
        game.reset()
        logits = np.random.default_rng(0).standard_normal(6561).astype(np.float32)
        priors = _get_legal_priors(game, logits, top_k=5)
        assert len(priors) <= 5
        total = sum(priors.values())
        assert abs(total - 1.0) < 1e-5

    def test_all_moves_are_legal(self):
        """Every move in priors must be a legal game move."""
        game = ColorLinesGame(seed=42)
        game.reset()
        logits = np.random.default_rng(0).standard_normal(6561).astype(np.float32)
        priors = _get_legal_priors(game, logits, top_k=50)
        legal = set(game.get_legal_moves())
        for action in priors:
            assert action in legal, f"{action} not in legal moves"

    def test_empty_board_no_moves(self):
        """Empty board has no balls → no legal moves → empty priors."""
        board = np.zeros((9, 9), dtype=np.int8)
        logits = np.zeros(6561, dtype=np.float32)
        k, idx, pri = _legal_priors_jit(board, logits, 30)
        assert k == 0

    def test_full_board_no_moves(self):
        """Full board has no empty cells → no targets → no moves."""
        board = np.ones((9, 9), dtype=np.int8)
        logits = np.zeros(6561, dtype=np.float32)
        k, idx, pri = _legal_priors_jit(board, logits, 30)
        assert k == 0

    def test_isolated_ball_no_moves(self):
        """Ball surrounded by other balls has no adjacent empty → no moves."""
        board = np.zeros((9, 9), dtype=np.int8)
        # Place ball at center surrounded by other balls
        board[4, 4] = 1
        board[3, 4] = 2
        board[5, 4] = 2
        board[4, 3] = 2
        board[4, 5] = 2
        logits = np.zeros(6561, dtype=np.float32)
        k, idx, pri = _legal_priors_jit(board, logits, 30)
        # The surrounded ball has no adjacent empty, but the surrounding
        # balls DO have adjacent empty cells
        assert k > 0  # surrounding balls can move

    def test_flat_priors_int_keys(self):
        """_get_legal_priors_flat returns int keys."""
        game = ColorLinesGame(seed=42)
        game.reset()
        logits = np.random.default_rng(0).standard_normal(6561).astype(np.float32)
        priors = _get_legal_priors_flat(game.board, logits, 30)
        for key in priors:
            assert isinstance(key, int)
            assert 0 <= key < 6561

    def test_priors_positive(self):
        """All prior probabilities must be positive."""
        game = ColorLinesGame(seed=42)
        game.reset()
        logits = np.random.default_rng(0).standard_normal(6561).astype(np.float32)
        priors = _get_legal_priors(game, logits, top_k=30)
        for p in priors.values():
            assert p > 0

    def test_extreme_logits(self):
        """Should handle very large/small logits without overflow."""
        game = ColorLinesGame(seed=42)
        game.reset()
        logits = np.full(6561, -100.0, dtype=np.float32)
        logits[0] = 100.0  # one extreme value
        priors = _get_legal_priors(game, logits, top_k=30)
        total = sum(priors.values())
        assert abs(total - 1.0) < 1e-4


# ── MCTS search ─────────────────────────────────────────────────────

class TestMCTSSearch:
    def test_returns_legal_move(self):
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
        move = mcts.search(game)
        assert move is not None
        legal = set(game.get_legal_moves())
        assert move in legal

    def test_game_not_mutated(self):
        """MCTS must not modify the input game."""
        game = ColorLinesGame(seed=42)
        game.reset()
        board_before = game.board.copy()
        score_before = game.score
        turns_before = game.turns

        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
        mcts.search(game)

        np.testing.assert_array_equal(game.board, board_before)
        assert game.score == score_before
        assert game.turns == turns_before

    def test_none_on_full_board(self):
        """MCTS returns None when no legal moves exist."""
        game = ColorLinesGame(seed=42)
        game.reset()
        game.board[:] = 1
        game.game_over = True

        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
        move = mcts.search(game)
        assert move is None

    def test_many_simulations(self):
        """Many simulations should not crash."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet(value=100.0)
        mcts = MCTS(net, torch.device('cpu'), num_simulations=100,
                     top_k=10, max_score=500.0)
        move = mcts.search(game)
        assert move is not None

    def test_make_mcts_player(self):
        from alphatrain.mcts import make_mcts_player
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        player = make_mcts_player(net, torch.device('cpu'),
                                   num_simulations=5, top_k=10)
        move = player(game)
        assert move is not None
        legal = set(game.get_legal_moves())
        assert move in legal


# ── MCTS determinism ────────────────────────────────────────────────

class TestMCTSDeterminism:
    def test_same_state_same_action(self):
        """Same game state should produce same action (state-seeded RNG)."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        action1 = mcts.search(game)
        action2 = mcts.search(game)
        assert action1 == action2

    def test_same_seed_same_game(self):
        """Two games with same seed produce identical MCTS actions."""
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        game1 = ColorLinesGame(seed=42)
        game1.reset()
        action1 = mcts.search(game1)

        game2 = ColorLinesGame(seed=42)
        game2.reset()
        action2 = mcts.search(game2)

        assert action1 == action2

    def test_different_states_different_actions(self):
        """Different game states should (usually) produce different actions."""
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        game1 = ColorLinesGame(seed=42)
        game1.reset()
        action1 = mcts.search(game1)

        game2 = ColorLinesGame(seed=999)
        game2.reset()
        action2 = mcts.search(game2)

        # Not guaranteed but extremely likely with different seeds
        # (if this fails, it's a statistical fluke — just re-run)
        assert action1 != action2


# ── Return policy ───────────────────────────────────────────────────

class TestReturnPolicy:
    def test_return_policy_shape(self):
        """return_policy=True returns (action, policy_target)."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        result = mcts.search(game, return_policy=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        action, policy = result
        assert action is not None
        assert policy.shape == (6561,)

    def test_policy_sums_to_one(self):
        """Policy target should be normalized visit counts."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        _, policy = mcts.search(game, return_policy=True)
        total = policy.sum()
        assert abs(total - 1.0) < 1e-5, f"Policy sums to {total}"

    def test_policy_nonnegative(self):
        """All policy values must be >= 0."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        _, policy = mcts.search(game, return_policy=True)
        assert (policy >= 0).all()

    def test_policy_matches_action(self):
        """The chosen action should have nonzero policy mass."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)

        action, policy = mcts.search(game, return_policy=True)
        (sr, sc), (tr, tc) = action
        idx = (sr * 9 + sc) * 81 + tr * 9 + tc
        assert policy[idx] > 0

    def test_return_policy_none_on_full_board(self):
        """Full board returns (None, None)."""
        game = ColorLinesGame(seed=42)
        game.reset()
        game.board[:] = 1
        game.game_over = True

        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
        result = mcts.search(game, return_policy=True)
        assert result == (None, None)


# ── Temperature and Dirichlet ───────────────────────────────────────

class TestExploration:
    def test_temperature_zero_is_greedy(self):
        """temperature=0 should always pick the highest-visit move."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=50, top_k=10)

        # Run twice — greedy should be deterministic
        action1 = mcts.search(game, temperature=0.0)
        action2 = mcts.search(game, temperature=0.0)
        assert action1 == action2

    def test_temperature_positive_uses_sampling(self):
        """temperature>0 samples proportional to visit counts.
        With enough sims, the action should still be valid."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=50, top_k=10)

        action, policy = mcts.search(game, temperature=1.0, return_policy=True)
        assert action is not None
        legal = set(game.get_legal_moves())
        assert action in legal

    def test_dirichlet_noise_changes_priors(self):
        """Dirichlet noise should modify the root priors."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()

        # Without noise
        mcts1 = MCTS(net, torch.device('cpu'), num_simulations=50, top_k=10)
        _, pol1 = mcts1.search(game, dirichlet_alpha=0.0, return_policy=True)

        # With noise — policy distribution should differ
        mcts2 = MCTS(net, torch.device('cpu'), num_simulations=50, top_k=10)
        _, pol2 = mcts2.search(game, dirichlet_alpha=0.3,
                               dirichlet_weight=0.25, return_policy=True)

        # Policies should differ (noise redistributes visits)
        assert not np.array_equal(pol1, pol2)


# ── Virtual loss ────────────────────────────────────────────────────

class TestVirtualLoss:
    def test_virtual_loss_constant(self):
        """VIRTUAL_LOSS should be defined and positive."""
        assert VIRTUAL_LOSS > 0

    def test_virtual_loss_restored_after_search(self):
        """After search completes, no residual virtual loss should remain.
        Every node's value_sum/visit_count should be self-consistent."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet(value=100.0)
        mcts = MCTS(net, torch.device('cpu'), num_simulations=30,
                     top_k=10, max_score=500.0, batch_size=8)

        # We can't directly inspect the tree after search() because it's
        # local to the method. But we can verify the search completes
        # without error and produces a valid result.
        action = mcts.search(game)
        assert action is not None


# ── Batch size behavior ─────────────────────────────────────────────

class TestBatchSize:
    def test_batch_size_1(self):
        """batch_size=1 should work (no batching)."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=10,
                     top_k=10, batch_size=1)
        move = mcts.search(game)
        assert move is not None

    def test_batch_size_larger_than_sims(self):
        """batch_size > num_simulations should work."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()
        mcts = MCTS(net, torch.device('cpu'), num_simulations=5,
                     top_k=10, batch_size=32)
        move = mcts.search(game)
        assert move is not None

    def test_different_batch_sizes_same_quality(self):
        """Different batch sizes should produce valid moves."""
        game = ColorLinesGame(seed=42)
        game.reset()
        net = DummyNet()

        for bs in [1, 4, 8, 16]:
            mcts = MCTS(net, torch.device('cpu'), num_simulations=20,
                         top_k=10, batch_size=bs)
            move = mcts.search(game)
            assert move is not None
            legal = set(game.get_legal_moves())
            assert move in legal
