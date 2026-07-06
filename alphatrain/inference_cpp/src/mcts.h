// Neural MCTS — port of alphatrain/mcts.py (the MCTS class, feature-value
// leaf mode). Open-loop determinized search: every simulation clones the root
// game and replays the tree path with fresh stochastic spawns drawn from one
// shared per-search RNG. Batched virtual-loss selection feeds the policy net
// in batches of `batch_size`.
//
// NOTE on reproducibility: the per-search sim RNG is seeded from a hash of the
// game state using OUR SplitMix64 (Python uses MD5 + numpy PCG64), so C++
// visit counts will NOT bit-match Python's. Validation is by playing STRENGTH
// (score distribution), same principle as the C++ eval.

#ifndef CLINES_MCTS_H_
#define CLINES_MCTS_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "feature_value.h"
#include "game.h"
#include "rng.h"

namespace clines {

// Batched policy forward: `obs` holds n contiguous (18,9,9) fp32 observations;
// writes n*6561 logits to `out`. Provided by the driver (direct LibTorch call
// or a shared inference-server thread).
using PolicyFn = std::function<void(const float* obs, int n, float* out)>;

struct MctsConfig {
  int num_simulations = 100;
  double c_puct = 2.5;
  int top_k = 30;
  int batch_size = 8;      // leaves per batched policy forward (virtual loss)
  double q_weight = 1.0;   // PUCT: q_weight*q_norm + U
  bool early_stop = false; // eval-only: stop when the argmax can't change
};

struct Candidate {
  int action;     // flat = (sr*9+sc)*81 + (tr*9+tc)
  int visits;
  double prior;   // post-noise prior actually used in search
  double q;       // value_sum / visits (0 if unvisited)
};

struct SearchResult {
  int action = -1;                // chosen move; -1 = no legal moves (dead)
  double root_value = 0.0;
  double q_min = 0.0, q_max = 0.0;
  std::vector<Candidate> cands;   // root children, visit-count descending
};

// Top-K legal moves by policy logit, softmax over just those K logits.
// Port of mcts.py::_legal_priors_jit (top-k over ALL legal moves, never
// top-k-then-filter). Returns k; fills out_actions/out_priors (descending
// logit order). out_* must hold at least top_k entries.
int LegalPriors(const int8_t* board, const float* logits, int top_k,
                int* out_actions, double* out_priors);

class MCTS {
 public:
  MCTS(PolicyFn policy, const FeatureEval* fe, const MctsConfig& cfg)
      : policy_(std::move(policy)), fe_(fe), cfg_(cfg) {}

  // One search from `game`. temperature: 0 = argmax visits; >0 = sample
  // visits^(1/T) using move_rng (selfplay exploration).
  SearchResult Search(const Game& game, double temperature, SimpleRng& move_rng);

 private:
  PolicyFn policy_;
  const FeatureEval* fe_;
  MctsConfig cfg_;
};

}  // namespace clines

#endif  // CLINES_MCTS_H_
