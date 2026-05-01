"""cProfile MCTS at 400 sims, feature-value mode.

Identifies the hottest Python functions to find safe speedup targets.
Run with current code to baseline, then re-run after each optimization.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import cProfile
import pstats
import io
import time
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _legal_priors_jit, NUM_MOVES
from alphatrain.evaluate import load_model


def warm_jit():
    g = ColorLinesGame(seed=42)
    g.reset()
    _build_obs_for_game(g)
    _legal_priors_jit(g.board, np.zeros(NUM_MOVES, dtype=np.float32), 30)


def play_game(mcts, seed, max_turns=200):
    """Play up to max_turns moves to get a sample of search calls."""
    game = ColorLinesGame(seed=seed)
    game.reset()
    while not game.game_over and game.turns < max_turns:
        action = mcts.search(game)
        if action is None:
            break
        r = game.move(action[0], action[1])
        if not r['valid']:
            break
    return game


def main():
    print("Warming JIT...", flush=True)
    warm_jit()

    device = torch.device('mps')
    net, max_score = load_model(
        'alphatrain/data/pillar2w2_epoch_10.pt',
        device, fp16=True, jit_trace=True)

    import sys
    early_stop = '--early-stop' in sys.argv
    print(f"early_stop={early_stop}", flush=True)
    mcts = MCTS(
        net=net, device=device, max_score=max_score,
        num_simulations=400, c_puct=2.5, top_k=30, batch_size=8,
        feature_weights_path='alphatrain/data/feature_value_weights.npz',
        early_stop=early_stop)

    # Warm: 10 search calls to stabilize fp16/MPS caches
    print("Warming model...", flush=True)
    g = ColorLinesGame(seed=0)
    g.reset()
    for _ in range(10):
        mcts.search(g)
        moves = g.get_legal_moves()
        if not moves:
            break
        g.move(moves[0][0], moves[0][1])

    # Profile a fresh game for ~80 turns
    print("Profiling 80 search calls (400 sims, bs=8)...", flush=True)
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    game = play_game(mcts, seed=42, max_turns=80)
    pr.disable()
    elapsed = time.time() - t0
    print(f"Played {game.turns} turns in {elapsed:.1f}s "
          f"({elapsed/max(game.turns,1)*1000:.0f} ms/turn)", flush=True)

    # Top functions by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(40)
    print("\n=== TOP 40 by cumulative time ===")
    print(s.getvalue())

    # Top functions by total (self) time — where the real work happens
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(40)
    print("\n=== TOP 40 by self time ===")
    print(s.getvalue())


if __name__ == '__main__':
    main()
