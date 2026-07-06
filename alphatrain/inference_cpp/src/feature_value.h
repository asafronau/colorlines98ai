// 27-feature linear leaf-value evaluator for MCTS — a port of
// alphatrain/scripts/mine_death_features.py (board_features / _with_next) +
// alphatrain/mcts.py::_evaluate_features_linear. Board-based (no NN), so it
// works with any policy backbone. Validated: static-feature MCTS lifts the
// 128ch policy's median +61% (see the eval_parallel run).

#ifndef CLINES_FEATURE_VALUE_H_
#define CLINES_FEATURE_VALUE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "game.h"  // NextBall, kN, kNN

namespace clines {

// 16 board features -> out16 (indices match board_features()).
void BoardFeatures(const int8_t* board, double* out16);

// 25 features (16 board + 9 next-ball) -> out25 (board_features_with_next()).
void BoardFeaturesWithNext(const int8_t* board, const std::vector<NextBall>& nb,
                           double* out25);

// Standardized linear model over the 25 features + 2 derived (ratio, frag).
class FeatureEval {
 public:
  bool Load(const std::string& path);  // reads data/feature_value.bin (CLFV)
  double Value(const int8_t* board, const std::vector<NextBall>& nb) const;

 private:
  float coefs_[27], means_[27], stds_[27], bias_ = 0.0f;
};

}  // namespace clines

#endif  // CLINES_FEATURE_VALUE_H_
