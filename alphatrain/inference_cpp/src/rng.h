// SplitMix64 PRNG — small, fast, good quality. Header-only.
//
// NOTE: the Python game uses numpy PCG64 (see game/rng.py), which we do NOT
// reproduce bit-for-bit (its choice()/integers() internals are fragile to
// match). We use our own RNG: the C++ engine therefore plays *different*
// specific games than Python for the same seed, but the same score
// *distribution* (identical policy + rules). Deterministic game logic is
// golden-tested separately and matches Python exactly.

#ifndef CLINES_RNG_H_
#define CLINES_RNG_H_

#include <cmath>
#include <cstdint>
#include <vector>

namespace clines {

class SimpleRng {
 public:
  explicit SimpleRng(uint64_t seed) : state_(seed) {}

  uint64_t NextU64() {
    state_ += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  // Uniform integer in [low, high).
  int RandInt(int low, int high) {
    return low + static_cast<int>(NextU64() % static_cast<uint64_t>(high - low));
  }

  // Uniform double in [0, 1). Top 53 bits for full double precision.
  double NextF64() {
    return static_cast<double>(NextU64() >> 11) * (1.0 / 9007199254740992.0);
  }

  // Standard normal via Box-Muller (port of rust_engine/src/rng.rs).
  double NextNormal() {
    double u1 = NextF64();
    double u2 = NextF64();
    while (u1 == 0.0) u1 = NextF64();
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(2.0 * 3.14159265358979323846 * u2);
  }

  // Gamma(alpha, 1) via Marsaglia-Tsang (port of rng.rs::next_gamma).
  double NextGamma(double alpha) {
    if (alpha < 1.0) {
      double u = NextF64();
      while (u == 0.0) u = NextF64();
      return NextGamma(alpha + 1.0) * std::pow(u, 1.0 / alpha);
    }
    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / std::sqrt(9.0 * d);
    while (true) {
      double x = NextNormal();
      double v = 1.0 + c * x;
      if (v <= 0.0) continue;
      v = v * v * v;
      double u = NextF64();
      if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v;
    }
  }

  // Symmetric Dirichlet(alpha) over n dims -> out (sums to 1).
  void Dirichlet(double alpha, int n, std::vector<double>& out) {
    out.resize(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
      out[i] = NextGamma(alpha);
      total += out[i];
    }
    if (total == 0.0) {
      for (int i = 0; i < n; ++i) out[i] = 1.0 / n;
    } else {
      for (int i = 0; i < n; ++i) out[i] /= total;
    }
  }

  // Choose k distinct indices from [0, n) (partial Fisher-Yates).
  void ChoiceNoReplace(int n, int k, std::vector<int>& out) {
    scratch_.resize(n);
    for (int i = 0; i < n; ++i) scratch_[i] = i;
    out.clear();
    out.reserve(k);
    for (int i = 0; i < k; ++i) {
      int j = i + static_cast<int>(NextU64() % static_cast<uint64_t>(n - i));
      std::swap(scratch_[i], scratch_[j]);
      out.push_back(scratch_[i]);
    }
  }

 private:
  uint64_t state_;
  std::vector<int> scratch_;
};

}  // namespace clines

#endif  // CLINES_RNG_H_
