"""Benchmark Python game engine — compare with Rust."""
import time
from game.board import ColorLinesGame

N_GAMES = 500

t0 = time.perf_counter()
total_turns = 0
total_score = 0

for seed in range(N_GAMES):
    g = ColorLinesGame(seed=seed)
    g.reset()
    while not g.game_over:
        moves = g.get_legal_moves()
        if not moves:
            break
        g.move(moves[0][0], moves[0][1])
    total_turns += g.turns
    total_score += g.score

elapsed = time.perf_counter() - t0
print(f"Python: {N_GAMES} games in {elapsed:.2f}s")
print(f"  {N_GAMES / elapsed:.0f} games/s")
print(f"  total turns: {total_turns}, mean score: {total_score / N_GAMES:.0f}")
print(f"  {elapsed / total_turns * 1e6:.1f} us/turn")
