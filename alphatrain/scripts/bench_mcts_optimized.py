"""Benchmark MCTS search speed before/after optimizations.

Measures the actual MCTS.search() throughput with DummyNet to isolate
CPU-side overhead from GPU inference latency.

Usage:
    python -m alphatrain.scripts.bench_mcts_optimized
"""

import time
import numpy as np
import torch
from game.board import ColorLinesGame
from alphatrain.mcts import (
    MCTS, _build_obs_for_game, _get_legal_priors, _get_legal_priors_flat,
    _legal_priors_jit, Node, NUM_MOVES
)
from alphatrain.observation import build_observation


class DummyNet:
    """Fast mock net for CPU-side profiling."""
    def __init__(self, value=500.0, num_value_bins=64):
        self.num_value_bins = num_value_bins
        self._value = value
        self._pol = torch.zeros(1, NUM_MOVES)
        self._val = torch.full((1, num_value_bins), 0.0)

    def __call__(self, obs):
        B = obs.shape[0]
        return self._pol.expand(B, -1), self._val.expand(B, -1)

    def predict_value(self, val_logits, max_val=30000.0):
        B = val_logits.shape[0]
        return torch.full((B,), self._value)

    def parameters(self):
        return iter([self._pol])

    def train(self, mode):
        return self


def bench_legal_priors(game, n=10000):
    """Compare _get_legal_priors (tuple keys) vs _get_legal_priors_flat."""
    pol = np.random.randn(6561).astype(np.float32)

    # Warmup
    _ = _get_legal_priors(game, pol, 30)
    _ = _get_legal_priors_flat(game.board, pol, 30)

    # Old: tuple keys
    t0 = time.perf_counter()
    for _ in range(n):
        _get_legal_priors(game, pol, 30)
    old_us = (time.perf_counter() - t0) / n * 1e6

    # New: flat int keys
    t0 = time.perf_counter()
    for _ in range(n):
        _get_legal_priors_flat(game.board, pol, 30)
    new_us = (time.perf_counter() - t0) / n * 1e6

    print(f"  _get_legal_priors (tuple keys): {old_us:.1f} us", flush=True)
    print(f"  _get_legal_priors_flat (int):   {new_us:.1f} us", flush=True)
    print(f"  Speedup: {old_us / new_us:.2f}x ({old_us - new_us:.1f} us saved)", flush=True)
    print(f"  Per search (400 calls): {(old_us - new_us) * 400 / 1000:.1f} ms saved", flush=True)
    return old_us, new_us


def bench_clone(game, n=50000):
    """Compare clone() with/without shared RNG."""
    shared_rng = np.random.default_rng(42)

    # Old: default clone
    t0 = time.perf_counter()
    for _ in range(n):
        game.clone()
    old_us = (time.perf_counter() - t0) / n * 1e6

    # New: shared rng
    t0 = time.perf_counter()
    for _ in range(n):
        game.clone(rng=shared_rng)
    new_us = (time.perf_counter() - t0) / n * 1e6

    print(f"  clone() (new RNG each):  {old_us:.1f} us", flush=True)
    print(f"  clone(rng=shared):       {new_us:.1f} us", flush=True)
    print(f"  Speedup: {old_us / new_us:.2f}x ({old_us - new_us:.1f} us saved)", flush=True)
    print(f"  Per search (400 calls): {(old_us - new_us) * 400 / 1000:.1f} ms saved", flush=True)
    return old_us, new_us


def bench_full_search(game, n_searches=20, num_sims=400, batch_size=8):
    """Benchmark actual MCTS.search() speed."""
    net = DummyNet()
    device = torch.device('cpu')
    mcts = MCTS(net, device, max_score=30000.0,
                num_simulations=num_sims, batch_size=batch_size,
                top_k=30, c_puct=2.5)

    # Warmup
    mcts.search(game)
    mcts.search(game)

    t0 = time.perf_counter()
    for _ in range(n_searches):
        mcts.search(game)
    elapsed = (time.perf_counter() - t0) / n_searches * 1000

    print(f"  MCTS.search({num_sims} sims, bs={batch_size}): "
          f"{elapsed:.1f} ms/search", flush=True)
    return elapsed


def main():
    print("Warming up...", flush=True)
    game = ColorLinesGame(seed=42)
    game.reset()
    _ = _build_obs_for_game(game)
    _ = _legal_priors_jit(game.board, np.zeros(6561, dtype=np.float32), 30)
    _ = _get_legal_priors(game, np.zeros(6561, dtype=np.float32), 30)
    _ = _get_legal_priors_flat(game.board, np.zeros(6561, dtype=np.float32), 30)

    print("\n=== Legal Priors Benchmark ===", flush=True)
    bench_legal_priors(game)

    print("\n=== Clone Benchmark ===", flush=True)
    bench_clone(game)

    print("\n=== Full MCTS Search Benchmark ===", flush=True)
    for bs in [8, 16]:
        bench_full_search(game, n_searches=30, num_sims=400, batch_size=bs)

    # Also test with return_policy (selfplay mode)
    print("\n=== MCTS Search with return_policy (selfplay mode) ===", flush=True)
    net = DummyNet()
    device = torch.device('cpu')
    mcts = MCTS(net, device, max_score=30000.0,
                num_simulations=400, batch_size=8,
                top_k=30, c_puct=2.5)
    mcts.search(game, temperature=1.0, dirichlet_alpha=0.3,
                dirichlet_weight=0.25, return_policy=True)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        action, pol = mcts.search(game, temperature=1.0,
                                   dirichlet_alpha=0.3,
                                   dirichlet_weight=0.25,
                                   return_policy=True)
    elapsed = (time.perf_counter() - t0) / n * 1000
    print(f"  selfplay search: {elapsed:.1f} ms/search", flush=True)
    print(f"  action={action}, policy sum={pol.sum():.4f}", flush=True)


if __name__ == '__main__':
    main()
