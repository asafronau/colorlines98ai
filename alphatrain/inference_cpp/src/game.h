// Color Lines 98 game engine in C++ — a faithful port of game/board.py.
//
// Board: 9x9, int8, 0 = empty, 1..7 = colors. Each turn: move a ball along a
// free path; if it completes a line of 5+, clear it (and score); otherwise 3
// new balls spawn. Game ends when the board fills.

#ifndef CLINES_GAME_H_
#define CLINES_GAME_H_

#include <array>
#include <cstdint>
#include <vector>

#include "rng.h"

namespace clines {

constexpr int kN = 9;            // board side
constexpr int kNN = 81;          // cells
constexpr int kColors = 7;
constexpr int kBallsPerTurn = 3;
constexpr int kMinLine = 5;
constexpr int kActions = kNN * kNN;  // 6561 flat (src*81 + tgt)

// Score for clearing n balls: n*(n-4) for n>=5, else 0.  5->5, 6->12, 7->21...
inline int LineScore(int n) { return n < kMinLine ? 0 : n * (n - 4); }

struct NextBall {
  int r, c, color;
};

class Game {
 public:
  explicit Game(uint64_t seed) : rng_(seed) {}

  // Empty board -> spawn 3 balls -> generate the next 3 (preview). Matches
  // ColorLinesGame.reset() with board=None.
  void Reset();

  // Apply a move (greedy-eval semantics, mirrors ColorLinesGame.move). Returns
  // true if the move was legal+applied. Updates board/score/turns/over.
  bool Move(int sr, int sc, int tr, int tc);

  // Legal-move mask over the 6561 flat actions (src*81 + tgt): 1.0 legal else 0.
  void LegalMask(float* out) const;

  // 18x9x9 observation (row-major, channel-major) into out[18*81]. (obs.cc)
  void BuildObs(float* out) const;

  const std::array<int8_t, kNN>& board() const { return board_; }
  const std::vector<NextBall>& next_balls() const { return next_balls_; }
  int score() const { return score_; }
  int turns() const { return turns_; }
  bool over() const { return over_; }
  int CountEmpty() const;

  // Test hooks: drive the engine from a fixed state (no RNG), for golden tests.
  void SetState(const int8_t* board81, const std::vector<NextBall>& nb);

  // Pure kernels exposed for golden tests (operate on a caller's board buffer).
  static int ClearLinesAt(int8_t* board, int r, int c);
  static void LabelEmpty(const int8_t* board, int8_t* labels);  // 0=ball, 1+=id

 private:
  void GenerateNextBalls();
  std::vector<int> SpawnBalls();  // returns landed flat cell indices

  std::array<int8_t, kNN> board_{};
  std::vector<NextBall> next_balls_;
  SimpleRng rng_;
  int score_ = 0;
  int turns_ = 0;
  bool over_ = false;
};

}  // namespace clines

#endif  // CLINES_GAME_H_
