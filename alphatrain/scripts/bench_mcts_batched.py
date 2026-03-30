"""Benchmark batched vs sequential MCTS.

Usage:
    python -m alphatrain.scripts.bench_mcts_batched
"""

import time
import torch
from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS
from game.board import ColorLinesGame


def bench_search(mcts, game, n_moves=10):
    """Time n_moves MCTS searches from the same position."""
    # Warmup
    mcts.search(game)
    t0 = time.perf_counter()
    for _ in range(n_moves):
        mcts.search(game)
    elapsed = (time.perf_counter() - t0) / n_moves
    return elapsed


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, max_score = load_model('alphatrain/data/alphatrain_td_best.pt', device)

    game = ColorLinesGame(seed=42)
    game.reset()

    print(f"\n=== Batched MCTS benchmark (400 sims, device={device}) ===\n", flush=True)

    for bs in [1, 4, 8, 16, 32]:
        mcts = MCTS(net, device, max_score=max_score, num_simulations=400,
                     c_puct=2.5, top_k=30, batch_size=bs)
        t = bench_search(mcts, game, n_moves=5)
        est_game = t * 300 / 60
        print(f"  batch_size={bs:>2}: {t*1000:.0f}ms/move, "
              f"~{est_game:.1f} min/game", flush=True)

    # Full game benchmark with batch_size=16
    print(f"\n=== Full game benchmark (seed=42, 400 sims, bs=16) ===\n", flush=True)
    mcts = MCTS(net, device, max_score=max_score, num_simulations=400,
                 c_puct=2.5, top_k=30, batch_size=16)

    def player(g):
        return mcts.search(g)

    game2 = ColorLinesGame(seed=42)
    game2.reset()
    t0 = time.time()
    turns = 0
    while not game2.game_over:
        move = player(game2)
        if move is None:
            break
        result = game2.move(move[0], move[1])
        if not result['valid']:
            break
        turns += 1
        if turns % 50 == 0:
            elapsed = time.time() - t0
            print(f"  turn {turns}, score={game2.score}, "
                  f"{elapsed:.0f}s ({elapsed/turns:.2f}s/turn)", flush=True)
    elapsed = time.time() - t0
    print(f"\n  Final: score={game2.score}, turns={turns}, "
          f"{elapsed:.0f}s ({elapsed/turns:.2f}s/turn)", flush=True)


if __name__ == '__main__':
    main()
