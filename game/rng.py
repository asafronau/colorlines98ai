"""Game RNG backed by numpy PCG64.

Drop-in replacement for the old SplitMix64 SimpleRng. Same interface,
but uses numpy's battle-tested PCG64 internally. Dirichlet sampling
is now native numpy (much faster than hand-rolled Gamma).
"""

import numpy as np


class SimpleRng:
    """Numpy-backed RNG with game-engine and MCTS convenience methods."""

    __slots__ = ('_gen',)

    def __init__(self, seed: int = 0):
        self._gen = np.random.Generator(np.random.PCG64(seed & 0xFFFFFFFFFFFFFFFF))

    def next_u64(self) -> int:
        return int(self._gen.integers(0, 2**64, dtype=np.uint64))

    def next_f64(self) -> float:
        return float(self._gen.random())

    def randint(self, low: int, high: int) -> int:
        return int(self._gen.integers(low, high))

    def choice_no_replace(self, n: int, k: int) -> list[int]:
        return self._gen.choice(n, size=k, replace=False).tolist()

    def integers(self, low: int, high: int, size: int) -> list[int]:
        return self._gen.integers(low, high, size=size).tolist()

    def dirichlet(self, alphas: list[float]) -> list[float]:
        return self._gen.dirichlet(alphas).tolist()
