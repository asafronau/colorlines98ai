"""Unit tests for observation builder.

All tests are deterministic — no random state.
Tests verify correctness of each channel independently.
"""

import numpy as np
import pytest
from alphatrain.observation import (
    build_observation, _line_length_at, _component_sizes,
    NUM_CHANNELS, BOARD_SIZE
)


def _empty_board():
    return np.zeros((9, 9), dtype=np.int64)


def _no_next():
    return (np.zeros(3, dtype=np.intp),
            np.zeros(3, dtype=np.intp),
            np.zeros(3, dtype=np.intp),
            0)


class TestLineLength:
    def test_single_ball(self):
        board = _empty_board()
        board[4, 4] = 1
        assert _line_length_at(board, 4, 4, 0, 1) == 1  # horizontal
        assert _line_length_at(board, 4, 4, 1, 0) == 1  # vertical

    def test_horizontal_line_of_3(self):
        board = _empty_board()
        board[4, 3] = 2
        board[4, 4] = 2
        board[4, 5] = 2
        assert _line_length_at(board, 4, 4, 0, 1) == 3
        assert _line_length_at(board, 4, 3, 0, 1) == 3
        assert _line_length_at(board, 4, 5, 0, 1) == 3

    def test_vertical_line_of_5(self):
        board = _empty_board()
        for r in range(5):
            board[r, 0] = 3
        assert _line_length_at(board, 2, 0, 1, 0) == 5

    def test_diagonal_line(self):
        board = _empty_board()
        for i in range(4):
            board[i, i] = 5
        assert _line_length_at(board, 1, 1, 1, 1) == 4

    def test_different_colors_dont_count(self):
        board = _empty_board()
        board[4, 3] = 1
        board[4, 4] = 2
        board[4, 5] = 1
        assert _line_length_at(board, 4, 4, 0, 1) == 1

    def test_empty_cell_returns_zero(self):
        board = _empty_board()
        assert _line_length_at(board, 4, 4, 0, 1) == 0

    def test_edge_line(self):
        board = _empty_board()
        for c in range(9):
            board[0, c] = 7
        assert _line_length_at(board, 0, 4, 0, 1) == 9


class TestComponentSizes:
    def test_empty_board(self):
        board = _empty_board()
        sizes = _component_sizes(board)
        assert sizes[4, 4] == 81  # one big component

    def test_single_ball_splits_nothing(self):
        board = _empty_board()
        board[4, 4] = 1
        sizes = _component_sizes(board)
        assert sizes[4, 4] == 0  # ball cell = 0
        assert sizes[0, 0] == 80  # all empty cells in one component

    def test_wall_splits_board(self):
        board = _empty_board()
        # Vertical wall at column 4
        for r in range(9):
            board[r, 4] = 1
        sizes = _component_sizes(board)
        # Left side: 9*4 = 36, right side: 9*4 = 36
        assert sizes[0, 0] == 36
        assert sizes[0, 8] == 36
        assert sizes[0, 4] == 0  # wall cell

    def test_isolated_cell(self):
        board = _empty_board()
        # Surround (4,4) with balls
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            board[4 + dr, 4 + dc] = 1
        sizes = _component_sizes(board)
        assert sizes[4, 4] == 1  # isolated single cell


class TestBuildObservation:
    def test_output_shape(self):
        board = _empty_board()
        nr, nc, ncol, nn = _no_next()
        obs = build_observation(board, nr, nc, ncol, nn)
        assert obs.shape == (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)

    def test_output_dtype(self):
        board = _empty_board()
        nr, nc, ncol, nn = _no_next()
        obs = build_observation(board, nr, nc, ncol, nn)
        assert obs.dtype == np.float32

    def test_empty_board_channels(self):
        board = _empty_board()
        nr, nc, ncol, nn = _no_next()
        obs = build_observation(board, nr, nc, ncol, nn)
        # All color channels should be 0
        for c in range(7):
            assert obs[c].sum() == 0
        # Empty channel should be all 1s
        assert obs[7].sum() == 81
        # Component area should be 81/81 = 1.0 everywhere
        assert np.allclose(obs[12], 1.0)
        # Line potentials should be 0 (no balls)
        for c in range(13, 18):
            assert obs[c].sum() == 0

    def test_color_channels_one_hot(self):
        board = _empty_board()
        board[0, 0] = 3  # color 3
        board[8, 8] = 7  # color 7
        nr, nc, ncol, nn = _no_next()
        obs = build_observation(board, nr, nc, ncol, nn)
        assert obs[2, 0, 0] == 1.0  # color 3 → channel 2
        assert obs[6, 8, 8] == 1.0  # color 7 → channel 6
        assert obs[7, 0, 0] == 0.0  # not empty
        assert obs[7, 4, 4] == 1.0  # empty

    def test_next_balls(self):
        board = _empty_board()
        nr = np.array([1, 3, 5], dtype=np.intp)
        nc = np.array([2, 4, 6], dtype=np.intp)
        ncol = np.array([2, 5, 7], dtype=np.intp)
        obs = build_observation(board, nr, nc, ncol, 3)
        assert obs[8, 1, 2] == pytest.approx(2 / 7.0)
        assert obs[9, 3, 4] == pytest.approx(5 / 7.0)
        assert obs[10, 5, 6] == pytest.approx(7 / 7.0)
        assert obs[11, 1, 2] == 1.0
        assert obs[11, 3, 4] == 1.0
        assert obs[11, 5, 6] == 1.0
        assert obs[11, 0, 0] == 0.0  # no next ball here

    def test_line_potential_horizontal(self):
        board = _empty_board()
        board[4, 3] = 1
        board[4, 4] = 1
        board[4, 5] = 1
        nr, nc, ncol, nn = _no_next()
        obs = build_observation(board, nr, nc, ncol, nn)
        # Channel 13 = horizontal line length
        assert obs[13, 4, 4] == pytest.approx(3 / 9.0)
        # Channel 14 = vertical (single ball)
        assert obs[14, 4, 4] == pytest.approx(1 / 9.0)
        # Channel 17 = max line (3)
        assert obs[17, 4, 4] == pytest.approx(3 / 9.0)

    def test_line_potential_diagonal(self):
        board = _empty_board()
        for i in range(5):
            board[i, i] = 4
        nr, nc, ncol, nn = _no_next()
        obs = build_observation(board, nr, nc, ncol, nn)
        # Channel 15 = diagonal (1,1) direction
        assert obs[15, 2, 2] == pytest.approx(5 / 9.0)
        # Channel 17 = max line (5)
        assert obs[17, 2, 2] == pytest.approx(5 / 9.0)

    def test_deterministic(self):
        """Same input always produces same output."""
        board = _empty_board()
        board[0, 0] = 1
        board[4, 4] = 3
        nr, nc, ncol, nn = _no_next()
        obs1 = build_observation(board, nr, nc, ncol, nn)
        obs2 = build_observation(board, nr, nc, ncol, nn)
        assert np.array_equal(obs1, obs2)
