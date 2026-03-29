"""Tests for afterstate computation."""

import numpy as np
import pytest
from alphatrain.afterstate import compute_afterstate
from game.config import BOARD_SIZE


def test_simple_move_no_clear():
    """Move a ball with no line formed."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    board[0, 0] = 1
    after, score = compute_afterstate(board, 0, 0, 4, 4)
    assert after[0, 0] == 0
    assert after[4, 4] == 1
    assert score == 0


def test_move_forms_line_of_5():
    """Move a ball to complete a horizontal line of 5."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    # Place 4 balls in a row
    for c in range(4):
        board[0, c] = 3
    # Place the 5th ball elsewhere
    board[5, 5] = 3
    after, score = compute_afterstate(board, 5, 5, 0, 4)
    # All 5 balls should be cleared
    assert after[0, 0] == 0
    assert after[0, 4] == 0
    assert after[5, 5] == 0
    assert score == 5  # 5 * (5-4) = 5


def test_move_forms_line_of_6():
    """Line of 6 scores 12."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    for c in range(5):
        board[0, c] = 2
    board[8, 8] = 2
    after, score = compute_afterstate(board, 8, 8, 0, 5)
    assert score == 12  # 6 * (6-4) = 12
    for c in range(6):
        assert after[0, c] == 0


def test_no_clear_preserves_other_balls():
    """Non-clearing move preserves all other balls."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    board[0, 0] = 1
    board[3, 3] = 5
    board[7, 7] = 2
    after, score = compute_afterstate(board, 0, 0, 1, 1)
    assert after[3, 3] == 5
    assert after[7, 7] == 2
    assert after[1, 1] == 1
    assert score == 0


def test_original_board_unchanged():
    """compute_afterstate must not modify the input board."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    board[0, 0] = 1
    board_copy = board.copy()
    compute_afterstate(board, 0, 0, 4, 4)
    assert np.array_equal(board, board_copy)


def test_cross_clear():
    """Move completing lines in two directions."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int64)
    # Horizontal: row 4, cols 0-3
    for c in range(4):
        board[4, c] = 1
    # Vertical: rows 0-3, col 4
    for r in range(4):
        board[r, 4] = 1
    # Move ball to (4, 4) to complete both lines
    board[8, 8] = 1
    after, score = compute_afterstate(board, 8, 8, 4, 4)
    # Both lines cleared (5 horizontal + 5 vertical - 1 shared = 9 balls)
    assert score == 9 * (9 - 4)  # 45
    assert after[4, 4] == 0
