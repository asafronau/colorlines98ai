"""Validate selfplay game JSONs: every move must be legal.

For each move, checks the SAVED board state:
- src has a ball (color 1-7)
- tgt is empty (0)
- A path exists from src to tgt through empty cells (BFS)
- Board values are all 0-7

Crashes on first invalid state with detailed diagnostics.

Usage:
    python -m alphatrain.scripts.validate_games data/selfplay_v8_s1600 --max-games 50
"""

import os
import json
import argparse
import numpy as np
from numba import njit

BOARD_SIZE = 9


@njit(cache=True)
def has_path(board, sr, sc, tr, tc):
    """BFS: is there a path from (sr,sc) to (tr,tc) through empty cells?"""
    if board[sr, sc] == 0 or board[tr, tc] != 0:
        return False
    visited = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.bool_)
    queue_r = np.empty(81, dtype=np.int8)
    queue_c = np.empty(81, dtype=np.int8)
    # Start BFS from src's neighbors (src itself has a ball)
    head, tail = 0, 0
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if nr == tr and nc == tc:
                return True
            if board[nr, nc] == 0 and not visited[nr, nc]:
                visited[nr, nc] = True
                queue_r[tail] = nr
                queue_c[tail] = nc
                tail += 1
    while head < tail:
        r, c = queue_r[head], queue_c[head]
        head += 1
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                if nr == tr and nc == tc:
                    return True
                if board[nr, nc] == 0 and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1
    return False


def validate_game(path):
    """Check every move in a game JSON is legal on its saved board."""
    data = json.load(open(path))
    moves = data['moves']
    fname = os.path.basename(path)

    for turn, move_data in enumerate(moves):
        board = np.array(move_data['board'], dtype=np.int8)

        assert board.min() >= 0 and board.max() <= 7, \
            f"{fname} turn {turn}: bad board values {board.min()}-{board.max()}"

        cm = move_data['chosen_move']
        sr, sc, tr, tc = cm['sr'], cm['sc'], cm['tr'], cm['tc']

        assert board[sr, sc] > 0, \
            f"{fname} turn {turn}: src ({sr},{sc}) is empty"

        assert board[tr, tc] == 0, \
            f"{fname} turn {turn}: tgt ({tr},{tc}) occupied (color={board[tr, tc]})"

        assert has_path(board, sr, sc, tr, tc), \
            f"{fname} turn {turn}: no path from ({sr},{sc}) to ({tr},{tc})"

    return len(moves)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('game_dir', help='Directory with game JSON files')
    p.add_argument('--max-games', type=int, default=50)
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.game_dir) if f.endswith('.json'))
    files = files[:args.max_games]

    print(f"Validating {len(files)} games from {args.game_dir}", flush=True)

    # Warm up numba
    has_path(np.zeros((9, 9), dtype=np.int8), 0, 0, 0, 1)

    total_turns = 0
    for i, fname in enumerate(files):
        fpath = os.path.join(args.game_dir, fname)
        turns = validate_game(fpath)
        total_turns += turns
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] {total_turns:,} turns validated",
                  flush=True)

    print(f"\nALL VALID: {len(files)} games, {total_turns:,} turns", flush=True)


if __name__ == '__main__':
    main()
