"""Cross-language deterministic RNG (SplitMix64).

Identical implementation in Python and Rust produces the same sequence
for the same seed. Used by the game engine and MCTS instead of numpy's
PCG64 (which is impossible to replicate exactly in other languages).

SplitMix64 is a well-studied, fast, 64-bit PRNG by Sebastiano Vigna.
Period: 2^64. Passes BigCrush. Single 64-bit state word.

Includes Dirichlet sampling via Gamma(Marsaglia-Tsang) + Box-Muller normal,
all built from the same u64 stream for cross-language reproducibility.
"""

import math

MASK64 = 0xFFFFFFFFFFFFFFFF


class SimpleRng:
    """SplitMix64 PRNG with game-engine and MCTS convenience methods."""

    __slots__ = ('state',)

    def __init__(self, seed: int = 0):
        self.state = seed & MASK64

    def next_u64(self) -> int:
        """Generate next 64-bit unsigned integer."""
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return (z ^ (z >> 31)) & MASK64

    def next_f64(self) -> float:
        """Uniform float in [0, 1). Uses top 53 bits for full double precision."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def randint(self, low: int, high: int) -> int:
        """Random integer in [low, high). Uses rejection-free modulo."""
        r = high - low
        return low + int(self.next_u64() % r)

    def choice_no_replace(self, n: int, k: int) -> list[int]:
        """Choose k items from range(n) without replacement.

        Partial Fisher-Yates shuffle. Consumes exactly k random values.
        """
        arr = list(range(n))
        result = []
        for i in range(k):
            j = i + int(self.next_u64() % (n - i))
            arr[i], arr[j] = arr[j], arr[i]
            result.append(arr[i])
        return result

    def integers(self, low: int, high: int, size: int) -> list[int]:
        """Generate `size` random integers in [low, high)."""
        return [self.randint(low, high) for _ in range(size)]

    # ── Normal and Gamma distributions (for Dirichlet) ──────────────

    def next_normal(self) -> float:
        """Standard normal via Box-Muller transform. Consumes 2 u64 values."""
        u1 = self.next_f64()
        u2 = self.next_f64()
        # Avoid log(0)
        while u1 == 0.0:
            u1 = self.next_f64()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def next_gamma(self, alpha: float) -> float:
        """Gamma(alpha, 1) sample via Marsaglia-Tsang method.

        For alpha >= 1: direct Marsaglia-Tsang.
        For alpha < 1: Gamma(alpha+1, 1) * U^(1/alpha).
        """
        if alpha < 1.0:
            # Boost: Gamma(alpha, 1) = Gamma(alpha+1, 1) * U^(1/alpha)
            u = self.next_f64()
            while u == 0.0:
                u = self.next_f64()
            return self.next_gamma(alpha + 1.0) * (u ** (1.0 / alpha))

        d = alpha - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)

        while True:
            x = self.next_normal()
            v = 1.0 + c * x
            if v <= 0.0:
                continue
            v = v * v * v
            u = self.next_f64()
            # Squeeze test
            if u < 1.0 - 0.0331 * (x * x) * (x * x):
                return d * v
            if math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v)):
                return d * v

    def dirichlet(self, alphas: list[float]) -> list[float]:
        """Dirichlet sample from list of alpha parameters.

        Each component is Gamma(alpha_i, 1), then normalized.
        """
        samples = [self.next_gamma(a) for a in alphas]
        total = sum(samples)
        if total == 0.0:
            # Degenerate case: return uniform
            n = len(samples)
            return [1.0 / n] * n
        return [s / total for s in samples]
