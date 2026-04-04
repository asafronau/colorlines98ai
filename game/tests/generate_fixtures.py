"""Generate deterministic test fixtures for cross-language validation.

Outputs JSON fixtures that capture the exact behavior of every game engine
function. A Rust implementation must match these outputs precisely.

Usage:
    python -m game.tests.generate_fixtures --output game/tests/fixtures.json
"""

import json
import argparse
import numpy as np
from game.board import (
    ColorLinesGame, calculate_score,
    _label_empty_components, _count_empty, _get_empty_array,
    _get_source_mask, _get_target_mask, _find_lines_at, _clear_lines_at,
    _is_reachable,
)


def _board_to_list(board):
    return board.tolist()


def _generate_score_fixtures():
    """calculate_score for various inputs."""
    fixtures = []
    for n in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]:
        fixtures.append({
            'function': 'calculate_score',
            'input': {'num_balls': n},
            'expected': calculate_score(n),
        })
    return fixtures


def _make_board(cells):
    board = np.zeros((9, 9), dtype=np.int8)
    for (r, c), color in cells.items():
        board[r, c] = color
    return board


def _generate_component_fixtures():
    """_label_empty_components for various boards."""
    fixtures = []

    # Empty board
    board = np.zeros((9, 9), dtype=np.int8)
    labels = _label_empty_components(board)
    fixtures.append({
        'function': 'label_empty_components',
        'input': {'board': _board_to_list(board)},
        'expected': {'labels': _board_to_list(labels)},
    })

    # Full board
    board = np.ones((9, 9), dtype=np.int8)
    labels = _label_empty_components(board)
    fixtures.append({
        'function': 'label_empty_components',
        'input': {'board': _board_to_list(board)},
        'expected': {'labels': _board_to_list(labels)},
    })

    # Vertical wall splitting board
    board = np.zeros((9, 9), dtype=np.int8)
    board[:, 4] = 1
    labels = _label_empty_components(board)
    fixtures.append({
        'function': 'label_empty_components',
        'input': {'board': _board_to_list(board)},
        'expected': {'labels': _board_to_list(labels)},
    })

    # Isolated cell
    board = np.zeros((9, 9), dtype=np.int8)
    board[3, 3:6] = 1
    board[4, 3] = 1
    board[4, 5] = 1
    board[5, 3:6] = 1
    labels = _label_empty_components(board)
    fixtures.append({
        'function': 'label_empty_components',
        'input': {'board': _board_to_list(board)},
        'expected': {'labels': _board_to_list(labels)},
    })

    return fixtures


def _generate_line_fixtures():
    """_find_lines_at and _clear_lines_at for various scenarios."""
    fixtures = []

    scenarios = [
        ('horizontal_5', {(4, c): 2 for c in range(5)}, 4, 2),
        ('vertical_5', {(r, 4): 3 for r in range(5)}, 2, 4),
        ('diagonal_5', {(i, i): 1 for i in range(5)}, 2, 2),
        ('anti_diagonal_5', {(i, 4 - i): 1 for i in range(5)}, 2, 2),
        ('horizontal_6', {(4, c): 2 for c in range(6)}, 4, 3),
        ('cross_9', {**{(4, c): 2 for c in range(5)},
                     **{(r, 2): 2 for r in range(5)}}, 4, 2),
        ('no_line_4', {(4, c): 2 for c in range(4)}, 4, 1),
        ('empty_cell', {}, 4, 4),
    ]

    for name, cells, row, col in scenarios:
        board = _make_board(cells)
        count = _find_lines_at(board, row, col)

        board_copy = board.copy()
        cleared = _clear_lines_at(board_copy, row, col)

        fixtures.append({
            'function': 'find_lines_at',
            'name': name,
            'input': {
                'board': _board_to_list(board),
                'row': row, 'col': col,
            },
            'expected': {'count': int(count)},
        })
        fixtures.append({
            'function': 'clear_lines_at',
            'name': name,
            'input': {
                'board': _board_to_list(board),
                'row': row, 'col': col,
            },
            'expected': {
                'cleared': int(cleared),
                'board_after': _board_to_list(board_copy),
            },
        })

    return fixtures


def _generate_reachability_fixtures():
    """_is_reachable and _get_target_mask fixtures."""
    fixtures = []

    # Board with wall
    board = np.zeros((9, 9), dtype=np.int8)
    board[:, 4] = 1  # vertical wall
    board[4, 0] = 1  # source ball on left
    labels = _label_empty_components(board)

    # Reachable (same side)
    fixtures.append({
        'function': 'is_reachable',
        'input': {
            'labels': _board_to_list(labels),
            'sr': 4, 'sc': 0, 'tr': 0, 'tc': 0,
        },
        'expected': bool(_is_reachable(labels, 4, 0, 0, 0)),
    })
    # Unreachable (other side)
    fixtures.append({
        'function': 'is_reachable',
        'input': {
            'labels': _board_to_list(labels),
            'sr': 4, 'sc': 0, 'tr': 0, 'tc': 8,
        },
        'expected': bool(_is_reachable(labels, 4, 0, 0, 8)),
    })

    # Target mask
    mask = _get_target_mask(labels, 4, 0)
    fixtures.append({
        'function': 'get_target_mask',
        'input': {
            'labels': _board_to_list(labels),
            'sr': 4, 'sc': 0,
        },
        'expected': {'mask': _board_to_list(mask)},
    })

    return fixtures


def _generate_game_fixtures():
    """Full game replay fixtures with deterministic seed."""
    fixtures = []

    for seed in [42, 100, 777]:
        game = ColorLinesGame(seed=seed)
        game.reset()

        board_after_reset = game.board.copy()
        next_balls_after_reset = list(game.next_balls)

        # Play 10 moves with first-legal-move strategy
        moves = []
        for turn in range(10):
            if game.game_over:
                break

            source_mask = game.get_source_mask()
            action = None
            for sr in range(9):
                for sc in range(9):
                    if source_mask[sr, sc] == 0:
                        continue
                    target_mask = game.get_target_mask((sr, sc))
                    for tr in range(9):
                        for tc in range(9):
                            if target_mask[tr, tc] > 0:
                                action = ((sr, sc), (tr, tc))
                                break
                        if action:
                            break
                    if action:
                        break
                if action:
                    break

            if action is None:
                break

            result = game.move(action[0], action[1])
            moves.append({
                'source': list(action[0]),
                'target': list(action[1]),
                'valid': result['valid'],
                'cleared': result.get('cleared', 0),
                'score_after': game.score,
                'board_after': _board_to_list(game.board),
                'game_over': game.game_over,
            })

        fixtures.append({
            'function': 'game_replay',
            'seed': seed,
            'board_after_reset': _board_to_list(board_after_reset),
            'next_balls_after_reset': [
                {'row': r, 'col': c, 'color': int(col)}
                for (r, c), col in next_balls_after_reset
            ],
            'moves': moves,
            'final_score': game.score,
            'final_turns': game.turns,
            'game_over': game.game_over,
        })

    return fixtures


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='game/tests/fixtures.json')
    args = p.parse_args()

    all_fixtures = {
        'score': _generate_score_fixtures(),
        'components': _generate_component_fixtures(),
        'lines': _generate_line_fixtures(),
        'reachability': _generate_reachability_fixtures(),
        'game_replay': _generate_game_fixtures(),
    }

    total = sum(len(v) for v in all_fixtures.values())
    with open(args.output, 'w') as f:
        json.dump(all_fixtures, f, indent=2)

    print(f"Generated {total} fixtures to {args.output}", flush=True)
    for k, v in all_fixtures.items():
        print(f"  {k}: {len(v)}", flush=True)


if __name__ == '__main__':
    main()
