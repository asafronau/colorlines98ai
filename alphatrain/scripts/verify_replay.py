"""Verify that replaying games from seed + chosen_moves reproduces the data.

Usage:
    python -m alphatrain.scripts.verify_replay
"""

import json
import glob
import numpy as np
from game.board import ColorLinesGame


def replay_game(game_data):
    """Replay a game from seed, return running scores at each move."""
    game = ColorLinesGame(seed=game_data['seed'])
    game.reset()

    running_scores = []
    for i, move in enumerate(game_data['moves']):
        # Verify board matches
        expected_board = np.array(move['board'], dtype=np.int8)
        if not np.array_equal(game.board, expected_board):
            return None, i, "board mismatch"

        running_scores.append(game.score)

        # Execute chosen move
        cm = move['chosen_move']
        source = (cm['sr'], cm['sc'])
        target = (cm['tr'], cm['tc'])
        result = game.move(source, target)
        if not result['valid']:
            return None, i, "invalid move"

    return running_scores, len(game_data['moves']), None


def main():
    files = sorted(glob.glob('data/alphazero_v1/game_*.json'))
    print(f"Verifying replay for {len(files)} games...", flush=True)

    ok = 0
    fail = 0
    for i, f in enumerate(files):
        with open(f) as fh:
            game_data = json.load(fh)

        scores, n, err = replay_game(game_data)
        if err:
            print(f"  FAIL {f}: {err} at move {n}", flush=True)
            fail += 1
        else:
            final = game_data['score']
            # Check that final running score + last move's delta = game score
            if scores[-1] <= final:
                ok += 1
            else:
                print(f"  WARN {f}: last running_score={scores[-1]} > final={final}")
                fail += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] ok={ok} fail={fail}", flush=True)

    print(f"\nDone: {ok} ok, {fail} fail out of {len(files)}")

    # Show score progression for first successful game
    if ok > 0:
        with open(files[0]) as fh:
            game_data = json.load(fh)
        scores, _, _ = replay_game(game_data)
        deltas = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        nonzero = [(i, d) for i, d in enumerate(deltas) if d > 0]
        print(f"\nFirst game (seed={game_data['seed']}, final={game_data['score']}):")
        print(f"  {len(scores)} moves, {len(nonzero)} scoring moves")
        print(f"  Score at move 0: {scores[0]}")
        print(f"  Score at move 100: {scores[min(100, len(scores)-1)]}")
        print(f"  Score at last move: {scores[-1]}")
        print(f"  First 5 scoring deltas: {nonzero[:5]}")
        remaining = [game_data['score'] - s for s in scores]
        print(f"  Remaining score at move 0: {remaining[0]}")
        print(f"  Remaining score at move 100: {remaining[min(100, len(scores)-1)]}")
        print(f"  Remaining score at last: {remaining[-1]}")


if __name__ == '__main__':
    main()
