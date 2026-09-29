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

  // Apply a move known to be legal, drawing spawn RNG from an EXTERNAL rng
  // (mirrors ColorLinesGame.trusted_move). Used by MCTS: each simulation clones
  // the root game and replays the tree path against one shared sim-RNG, so the
  // stochastic spawns advance a single stream across the batch (open-loop).
  void TrustedMove(int sr, int sc, int tr, int tc, SimpleRng& rng);

  // Legal-move mask over the 6561 flat actions (src*81 + tgt): 1.0 legal else 0.
  void LegalMask(float* out) const;

  // 18x9x9 observation (row-major, channel-major) into out[18*81]. (obs.cc)
  void BuildObs(float* out) const;

  // Fast path for the batched greedy loops (HISTORY 257): the empty-cell components are computed
  // once per state (Labels) and shared by the observation, the legal mask and the move. Results are
  // bit-identical to BuildObs / LegalMask / Move (eval_cpu_bench checks it on real states).
  void Labels(int8_t* labels) const { LabelEmpty(board_.data(), labels); }
  void BuildObs(float* out, const int8_t* labels) const;
  // uint8 legal mask (1 = legal); returns the number of legal moves.
  int LegalMaskU8(uint8_t* out, const int8_t* labels) const;
  // Move validated against `labels`, which must be Labels() of the current board.
  bool Move(int sr, int sc, int tr, int tc, const int8_t* labels);

  const std::array<int8_t, kNN>& board() const { return board_; }
  const std::vector<NextBall>& next_balls() const { return next_balls_; }
  int score() const { return score_; }
  int turns() const { return turns_; }
  bool over() const { return over_; }
  int CountEmpty() const;

  // Restore an arbitrary state (golden tests; crisis-anchor replay restores
  // score/turns too, mirroring crisis_mining.py's game.reset + score/turns).
  void SetState(const int8_t* board81, const std::vector<NextBall>& nb,
                int score = 0, int turns = 0);

  // Pure kernels exposed for golden tests (operate on a caller's board buffer).
  static int ClearLinesAt(int8_t* board, int r, int c);
  static void LabelEmpty(const int8_t* board, int8_t* labels);  // 0=ball, 1+=id

 private:
  void GenerateNextBalls(SimpleRng& rng);
  // Places the preview balls; writes the landed flat cells to `landed` (<= 3) and returns how many.
  int SpawnBalls(SimpleRng& rng, int* landed);
  void AfterMove(int tr, int tc, SimpleRng& rng);  // clears / spawns / next preview / game over

  std::array<int8_t, kNN> board_{};
  std::vector<NextBall> next_balls_;
  SimpleRng rng_;
  int score_ = 0;
  int turns_ = 0;
  bool over_ = false;
};

}  // namespace clines

#endif  // CLINES_GAME_H_
