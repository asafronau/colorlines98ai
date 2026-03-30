"""Benchmark afterstate MCTS vs old determinized MCTS.

Usage:
    python -m alphatrain.scripts.bench_afterstate_mcts
"""

import time
import torch
from alphatrain.evaluate import load_model, play_game
from alphatrain.mcts import MCTS, make_mcts_player
from game.board import ColorLinesGame


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, max_score = load_model('alphatrain/data/alphatrain_td_best.pt', device)

    game = ColorLinesGame(seed=42)
    game.reset()

    print(f"\n=== Afterstate MCTS benchmark (device={device}) ===\n", flush=True)

    # Benchmark per-move speed
    mcts = MCTS(net, device, max_score=max_score, num_simulations=400,
                c_puct=2.5, top_k=30, batch_size=16)

    # Warmup
    mcts.search(game)

    n_moves = 10
    t0 = time.perf_counter()
    for _ in range(n_moves):
        mcts.search(game)
    per_move = (time.perf_counter() - t0) / n_moves
    print(f"  Per move: {per_move*1000:.0f}ms (400 sims)", flush=True)
    print(f"  Estimated per game: {per_move * 300 / 60:.1f} min (300 turns)",
          flush=True)

    # Full game
    print(f"\n=== Full game (seed=42, 400 sims) ===\n", flush=True)
    player = make_mcts_player(net, device, max_score=max_score,
                               num_simulations=400, top_k=30, batch_size=16)

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
        if turns % 100 == 0:
            elapsed = time.time() - t0
            print(f"  turn {turns}, score={game2.score}, "
                  f"{elapsed:.0f}s ({elapsed/turns*1000:.0f}ms/turn)",
                  flush=True)
    elapsed = time.time() - t0
    print(f"\n  Final: score={game2.score}, turns={turns}, "
          f"{elapsed:.0f}s ({elapsed/turns*1000:.0f}ms/turn)", flush=True)

    # CPU single-thread benchmark
    print(f"\n=== CPU single-thread benchmark ===\n", flush=True)
    torch.set_num_threads(1)
    net_cpu, _ = load_model('alphatrain/data/alphatrain_td_best.pt',
                            torch.device('cpu'))
    mcts_cpu = MCTS(net_cpu, torch.device('cpu'), max_score=max_score,
                     num_simulations=400, top_k=30, batch_size=8)
    # Warmup
    game3 = ColorLinesGame(seed=42)
    game3.reset()
    mcts_cpu.search(game3)

    t0 = time.perf_counter()
    for _ in range(n_moves):
        mcts_cpu.search(game3)
    per_move_cpu = (time.perf_counter() - t0) / n_moves
    print(f"  CPU per move: {per_move_cpu*1000:.0f}ms (400 sims)", flush=True)
    print(f"  CPU estimated per game: {per_move_cpu * 300 / 60:.1f} min",
          flush=True)
    print(f"  With 18 workers: {per_move_cpu * 300 / 60 / 18:.2f} min effective",
          flush=True)


if __name__ == '__main__':
    main()
