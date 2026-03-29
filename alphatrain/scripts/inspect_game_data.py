"""Inspect game data format for TD target computation.

Usage:
    python -m alphatrain.scripts.inspect_game_data
"""

import json
import glob
import numpy as np


def main():
    files = sorted(glob.glob('data/alphazero_v1/game_*.json'))
    print(f"Found {len(files)} game files")

    # Look at first game
    with open(files[0]) as f:
        game = json.load(f)

    print(f"\nGame keys: {list(game.keys())}")
    print(f"Number of moves: {len(game['moves'])}")

    # Look at first move
    m = game['moves'][0]
    print(f"\nMove keys: {list(m.keys())}")
    print(f"  game_score: {m.get('game_score')}")
    print(f"  score: {m.get('score')}")
    print(f"  turn: {m.get('turn')}")
    print(f"  board shape: {np.array(m['board']).shape}")
    print(f"  next_balls: {m.get('next_balls', [])[:2]}...")
    print(f"  top_moves: {len(m.get('top_moves', []))} moves")
    print(f"  top_scores: {m.get('top_scores', [])[:5]}...")

    # Look at score progression in first game
    scores = [m.get('game_score', m.get('score', 0)) for m in game['moves']]
    print(f"\nScore progression (first game): {scores[:10]}... -> {scores[-1]}")

    # Check if there's a per-turn score or just final
    if 'score' in game['moves'][0] and 'game_score' in game['moves'][0]:
        print(f"\nBoth 'score' and 'game_score' present")
        print(f"  move[0]: score={game['moves'][0]['score']}, game_score={game['moves'][0]['game_score']}")
        print(f"  move[5]: score={game['moves'][5]['score']}, game_score={game['moves'][5]['game_score']}")
        print(f"  move[-1]: score={game['moves'][-1]['score']}, game_score={game['moves'][-1]['game_score']}")

    # Check if score changes per turn (running score vs final)
    unique_scores = len(set(scores))
    print(f"\nUnique score values across {len(scores)} moves: {unique_scores}")
    if unique_scores == 1:
        print("  -> All moves have SAME score (final game_score). No per-turn deltas available.")
    else:
        print("  -> Scores differ per move (running score available).")
        deltas = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        nonzero = [d for d in deltas if d > 0]
        print(f"  Non-zero deltas: {len(nonzero)}/{len(deltas)}")
        print(f"  Delta examples: {nonzero[:10]}")

    # Stats across all games
    final_scores = []
    n_moves = []
    for f in files:
        with open(f) as fh:
            g = json.load(fh)
        n_moves.append(len(g['moves']))
        scores = [m.get('game_score', m.get('score', 0)) for m in g['moves']]
        final_scores.append(scores[-1])

    print(f"\nAcross {len(files)} games:")
    print(f"  Scores: mean={np.mean(final_scores):.0f}, "
          f"std={np.std(final_scores):.0f}, "
          f"min={np.min(final_scores)}, max={np.max(final_scores)}")
    print(f"  Moves/game: mean={np.mean(n_moves):.0f}, "
          f"min={np.min(n_moves)}, max={np.max(n_moves)}")


if __name__ == '__main__':
    main()
