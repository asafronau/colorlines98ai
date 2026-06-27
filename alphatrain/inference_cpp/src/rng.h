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
