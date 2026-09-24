// CPU-only checks of the real search loop, with a synthetic policy/value.
// These establish search mechanics, not playing strength.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>

#include "mcts.h"

using namespace clines;

SearchResult Run(bool mean, int batch, float offset, double floor = 0.0) {
  Game game(17);
  game.Reset();
  PolicyFn policy = [offset](const float* obs, int n, float* logits, float* values) {
    for (int b = 0; b < n; ++b) {
      // Distinct, nonuniform priors and dyadic values avoid precision noise
      // when translating the value function. The 32-simulation budget and
      // shallow batch fixture stay nonterminal.
      for (int a = 0; a < kActions; ++a)
        logits[b * kActions + a] = -static_cast<float>(a % 127) / 16.0f;
      float v = offset;
      for (int cell = 0; cell < kNN; ++cell)
        v += obs[b * 18 * kNN + 7 * kNN + cell] * (cell % 5) / 128.0f;
      values[b] = v;
    }
  };
  MctsConfig cfg;
  cfg.nn_value = true;
  cfg.num_simulations = 32;
  cfg.batch_size = batch;
  cfg.top_k = 10;
  cfg.virtual_mean = mean;
  cfg.q_range_floor = floor;
  SimpleRng rng(1);
  auto result = MCTS(policy, nullptr, cfg).Search(game, 0, rng);
  int visits = 0;
  for (auto c : result.cands) {
    visits += c.visits;
    if (c.visits && (c.q < result.q_min - 1e-8 || c.q > result.q_max + 1e-8))
      throw std::runtime_error("pending value leaked into completed Q");
  }
  if (visits != cfg.num_simulations)
    throw std::runtime_error("simulation count mismatch");
  if (result.q_min <= offset)
    throw std::runtime_error("fixture must remain nonterminal");
  return result;
}

bool SameVisits(const SearchResult& a, const SearchResult& b) {
  std::map<int, int> av, bv;
  for (auto c : a.cands) av[c.action] = c.visits;
  for (auto c : b.cands) bv[c.action] = c.visits;
  return av == bv;
}

int main() {
  // A constant offset to all nonterminal values should not change normalized
  // Q decisions. Historical fixed -1 reservations break that invariance.
  for (int batch : {1, 8}) {
    auto a = Run(true, batch, 4);
    auto b = Run(true, batch, 64);
    if (!SameVisits(a, b)) throw std::runtime_error("virtual mean not translation invariant");
    if (batch == 1 && !SameVisits(a, Run(false, batch, 4)))
      throw std::runtime_error("single-leaf search changed");
  }
  // Also exercise incomplete final batches and a positive normalization floor.
  Run(true, 7, 4, 1.0);
  auto old_a = Run(false, 8, -8);
  auto old_b = Run(false, 8, 64);
  if (SameVisits(old_a, old_b))
    throw std::runtime_error("fixture failed to expose historical offset sensitivity");
  std::printf("PASS: virtual mean is translation invariant; batch=1 unchanged; "
              "pending values cancel; partial batch and Q floor valid.\n");
  std::printf("Historical fixed virtual loss changes visits under value translation: ");
  for (auto c : old_a.cands) std::printf("%d ", c.visits);
  std::printf("-> ");
  for (auto c : old_b.cands) std::printf("%d ", c.visits);
  std::printf("\n");
}
