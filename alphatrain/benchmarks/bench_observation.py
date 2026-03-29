"""Benchmark observation building speed.

Target: <50us per observation (single board), <5ms per batch of 200.
"""

import time
import numpy as np
from game.board import ColorLinesGame
from game.fast_heuristic import get_best_move_fast
from alphatrain.observation import build_observation


def bench_single():
    """Benchmark single observation build."""
    g = ColorLinesGame(seed=42)
    g.reset()
    # Play 30 turns to get a mid-game board
    for _ in range(30):
        m = get_best_move_fast(g)
        if m is None or g.game_over:
            break
        g.move(m[0], m[1])

    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = len(g.next_balls)
    for i, ((r, c), col) in enumerate(g.next_balls):
        if i >= 3:
            break
        nr[i] = r
        nc[i] = c
        ncol[i] = col

    # Warmup
    for _ in range(10):
        build_observation(g.board, nr, nc, ncol, nn)

    # Benchmark
    N = 10000
    t0 = time.perf_counter()
    for _ in range(N):
        build_observation(g.board, nr, nc, ncol, nn)
    t1 = time.perf_counter()
    us_per = (t1 - t0) / N * 1e6
    print(f"Single observation: {us_per:.1f} us/call ({N/(t1-t0):.0f}/s)")
    return us_per


def bench_batch():
    """Benchmark batch observation build (simulating rollout)."""
    g = ColorLinesGame(seed=42)
    g.reset()
    for _ in range(30):
        m = get_best_move_fast(g)
        if m is None or g.game_over:
            break
        g.move(m[0], m[1])

    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = len(g.next_balls)
    for i, ((r, c), col) in enumerate(g.next_balls):
        if i >= 3:
            break
        nr[i] = r
        nc[i] = c
        ncol[i] = col

    # Warmup
    build_observation(g.board, nr, nc, ncol, nn)

    # Batch of 200
    N_BATCH = 200
    N_REPS = 50
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        obs = np.empty((N_BATCH, 18, 9, 9), dtype=np.float32)
        for i in range(N_BATCH):
            obs[i] = build_observation(g.board, nr, nc, ncol, nn)
    t1 = time.perf_counter()
    ms_per_batch = (t1 - t0) / N_REPS * 1000
    print(f"Batch of {N_BATCH}: {ms_per_batch:.1f} ms/batch "
          f"({ms_per_batch/N_BATCH*1000:.1f} us/obs)")
    return ms_per_batch


if __name__ == '__main__':
    print("=== Observation Benchmark ===")
    us = bench_single()
    ms = bench_batch()
    print(f"\nTargets: single <50us {'PASS' if us < 50 else 'FAIL'}, "
          f"batch <5ms {'PASS' if ms < 5 else 'FAIL'}")
