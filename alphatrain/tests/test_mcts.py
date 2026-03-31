"""Tests for alphatrain.mcts — Afterstate MCTS correctness."""

import numpy as np
import torch
import pytest
from game.board import ColorLinesGame
from alphatrain.mcts import Node, MCTS, _build_obs_for_game, _get_legal_priors


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

    def to(self, device):
        return self

    def train(self, mode):
        return self


# -- Node tests --

def test_node_init():
    n = Node(prior=0.3)
    assert n.prior == 0.3
    assert n.visit_count == 0
    assert n.q_value == 0.0
    assert not n.expanded()


def test_node_q_value():
    n = Node()
    n.visit_count = 4
    n.value_sum = 1000.0
    assert n.q_value == 250.0


# -- Observation helpers --

def test_build_obs_shape():
    game = ColorLinesGame(seed=42)
    game.reset()
    obs = _build_obs_for_game(game)
    assert obs.shape == (18, 9, 9)
    assert obs.dtype == np.float32


def test_node_expanded():
    n = Node()
    assert not n.expanded()
    n.children[((0, 0), (1, 1))] = Node(prior=0.5)
    assert n.expanded()


# -- Legal priors --

def test_get_legal_priors_sums_to_one():
    game = ColorLinesGame(seed=42)
    game.reset()
    logits = np.random.randn(6561).astype(np.float32)
    priors = _get_legal_priors(game, logits, top_k=100)
    assert len(priors) > 0
    total = sum(priors.values())
    assert abs(total - 1.0) < 1e-5


def test_get_legal_priors_top_k():
    game = ColorLinesGame(seed=42)
    game.reset()
    logits = np.random.randn(6561).astype(np.float32)
    priors = _get_legal_priors(game, logits, top_k=5)
    assert len(priors) <= 5
    total = sum(priors.values())
    assert abs(total - 1.0) < 1e-5


def test_get_legal_priors_all_valid():
    """Every move in priors must be a legal game move."""
    game = ColorLinesGame(seed=42)
    game.reset()
    logits = np.random.randn(6561).astype(np.float32)
    priors = _get_legal_priors(game, logits, top_k=50)
    legal = set(game.get_legal_moves())
    for action in priors:
        assert action in legal, f"{action} not in legal moves"


# -- MCTS search --

def test_mcts_returns_legal_move():
    game = ColorLinesGame(seed=42)
    game.reset()
    net = DummyNet()
    mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
    move = mcts.search(game)
    assert move is not None
    legal = set(game.get_legal_moves())
    assert move in legal


def test_mcts_visits_root_children():
    game = ColorLinesGame(seed=42)
    game.reset()
    net = DummyNet()
    mcts = MCTS(net, torch.device('cpu'), num_simulations=20, top_k=10)
    move = mcts.search(game)
    assert move is not None


def test_mcts_game_not_mutated():
    """MCTS must not modify the input game."""
    game = ColorLinesGame(seed=42)
    game.reset()
    board_before = game.board.copy()
    score_before = game.score
    turns_before = game.turns

    net = DummyNet()
    mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
    mcts.search(game)

    assert np.array_equal(game.board, board_before)
    assert game.score == score_before
    assert game.turns == turns_before


def test_mcts_none_on_full_board():
    """MCTS returns None when no legal moves exist."""
    game = ColorLinesGame(seed=42)
    game.reset()
    game.board[:] = 1
    game.game_over = True

    net = DummyNet()
    mcts = MCTS(net, torch.device('cpu'), num_simulations=10, top_k=10)
    move = mcts.search(game)
    assert move is None


def test_mcts_multiple_simulations():
    """More simulations should not crash."""
    game = ColorLinesGame(seed=42)
    game.reset()
    net = DummyNet(value=100.0)
    mcts = MCTS(net, torch.device('cpu'), num_simulations=50,
                top_k=10, max_score=500.0)
    move = mcts.search(game)
    assert move is not None


def test_make_mcts_player():
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
