#include "game.h"

#include <algorithm>

namespace clines {
namespace {
constexpr int kDr4[4] = {0, 0, 1, -1};   // 4-neighbour offsets (for flood/reach)
constexpr int kDc4[4] = {1, -1, 0, 0};
constexpr int kLineDr[4] = {0, 1, 1, 1};  // line directions: H, V, D1, D2
constexpr int kLineDc[4] = {1, 0, 1, -1};
inline int Idx(int r, int c) { return r * kN + c; }
inline bool InB(int r, int c) { return r >= 0 && r < kN && c >= 0 && c < kN; }
}  // namespace

void Game::LabelEmpty(const int8_t* board, int8_t* labels) {
  std::fill(labels, labels + kNN, int8_t{0});
  int qr[kNN], qc[kNN];
  int8_t cur = 0;
  for (int sr = 0; sr < kN; ++sr) {
    for (int sc = 0; sc < kN; ++sc) {
      if (board[Idx(sr, sc)] != 0 || labels[Idx(sr, sc)] != 0) continue;
      ++cur;
      labels[Idx(sr, sc)] = cur;
      qr[0] = sr; qc[0] = sc;
      int head = 0, tail = 1;
      while (head < tail) {
        int r = qr[head], c = qc[head];
        ++head;
        for (int d = 0; d < 4; ++d) {
          int nr = r + kDr4[d], nc = c + kDc4[d];
          if (InB(nr, nc) && board[Idx(nr, nc)] == 0 && labels[Idx(nr, nc)] == 0) {
            labels[Idx(nr, nc)] = cur;
            qr[tail] = nr; qc[tail] = nc;
            ++tail;
          }
        }
      }
    }
  }
}

int Game::ClearLinesAt(int8_t* board, int row, int col) {
  int8_t color = board[Idx(row, col)];
  if (color == 0) return 0;
  bool mark[kNN] = {false};
  int n_clear = 0;
  for (int di = 0; di < 4; ++di) {
    int dr = kLineDr[di], dc = kLineDc[di];
    int line[kN]; int n = 0;
    line[n++] = Idx(row, col);
    for (int r = row + dr, c = col + dc; InB(r, c) && board[Idx(r, c)] == color;
         r += dr, c += dc)
      line[n++] = Idx(r, c);
    for (int r = row - dr, c = col - dc; InB(r, c) && board[Idx(r, c)] == color;
         r -= dr, c -= dc)
      line[n++] = Idx(r, c);
    if (n >= kMinLine) {
      for (int i = 0; i < n; ++i) {
        if (!mark[line[i]]) { mark[line[i]] = true; ++n_clear; }
      }
    }
  }
  for (int i = 0; i < kNN; ++i)
    if (mark[i]) board[i] = 0;
  return n_clear;
}

int Game::CountEmpty() const {
  int n = 0;
  for (int i = 0; i < kNN; ++i) n += (board_[i] == 0);
  return n;
}

void Game::LegalMask(float* out) const {
  std::fill(out, out + kActions, 0.0f);
  int8_t labels[kNN];
  LabelEmpty(board_.data(), labels);
  for (int sr = 0; sr < kN; ++sr) {
    for (int sc = 0; sc < kN; ++sc) {
      int s = Idx(sr, sc);
      if (board_[s] == 0) continue;
      // distinct empty-component labels adjacent to this source
      int8_t nbr[4]; int nn = 0;
      for (int d = 0; d < 4; ++d) {
        int nr = sr + kDr4[d], nc = sc + kDc4[d];
        if (!InB(nr, nc)) continue;
        int8_t lb = labels[Idx(nr, nc)];
        if (lb <= 0) continue;
        bool seen = false;
        for (int i = 0; i < nn; ++i) if (nbr[i] == lb) { seen = true; break; }
        if (!seen) nbr[nn++] = lb;
      }
      if (nn == 0) continue;
      for (int t = 0; t < kNN; ++t) {
        if (board_[t] != 0) continue;
        int8_t tl = labels[t];
        for (int i = 0; i < nn; ++i) {
          if (nbr[i] == tl) { out[s * kNN + t] = 1.0f; break; }
        }
      }
    }
  }
}

void Game::GenerateNextBalls() {
  next_balls_.clear();
  std::vector<int> empty;
  for (int i = 0; i < kNN; ++i) if (board_[i] == 0) empty.push_back(i);
  int n_empty = static_cast<int>(empty.size());
  if (n_empty == 0) return;
  int n = std::min(kBallsPerTurn, n_empty);
  std::vector<int> idx;
  rng_.ChoiceNoReplace(n_empty, n, idx);
  for (int i = 0; i < n; ++i) {
    int cell = empty[idx[i]];
    int color = rng_.RandInt(1, kColors + 1);
    next_balls_.push_back({cell / kN, cell % kN, color});
  }
}

std::vector<int> Game::SpawnBalls() {
  std::vector<int> landed;
  for (const NextBall& nb : next_balls_) {
    int cell = Idx(nb.r, nb.c);
    if (board_[cell] == 0) {
      board_[cell] = static_cast<int8_t>(nb.color);
      landed.push_back(cell);
    } else {
      std::vector<int> empty;
      for (int i = 0; i < kNN; ++i) if (board_[i] == 0) empty.push_back(i);
      if (!empty.empty()) {
        int j = rng_.RandInt(0, static_cast<int>(empty.size()));
        board_[empty[j]] = static_cast<int8_t>(nb.color);
        landed.push_back(empty[j]);
      }
    }
  }
  return landed;
}

void Game::Reset() {
  board_.fill(0);
  score_ = 0; turns_ = 0; over_ = false;
  GenerateNextBalls();
  SpawnBalls();
  GenerateNextBalls();
}

bool Game::Move(int sr, int sc, int tr, int tc) {
  if (over_) return false;
  int s = Idx(sr, sc), t = Idx(tr, tc);
  if (board_[s] == 0 || board_[t] != 0) return false;

  // reachability: target's empty-component must touch the source
  int8_t labels[kNN];
  LabelEmpty(board_.data(), labels);
  int8_t tl = labels[t];
  if (tl <= 0) return false;
  bool reachable = false;
  for (int d = 0; d < 4; ++d) {
    int nr = sr + kDr4[d], nc = sc + kDc4[d];
    if (InB(nr, nc) && labels[Idx(nr, nc)] == tl) { reachable = true; break; }
  }
  if (!reachable) return false;

  // execute
  int8_t color = board_[s];
  board_[s] = 0;
  board_[t] = color;
  ++turns_;

  int cleared = ClearLinesAt(board_.data(), tr, tc);
  if (cleared > 0) {
    score_ += LineScore(cleared);
  } else {
    std::vector<int> landed = SpawnBalls();
    for (int cell : landed) {
      if (board_[cell] != 0) {
        int sc2 = ClearLinesAt(board_.data(), cell / kN, cell % kN);
        if (sc2 > 0) score_ += LineScore(sc2);
      }
    }
    GenerateNextBalls();
    if (CountEmpty() == 0) over_ = true;
  }
  return true;
}

void Game::SetState(const int8_t* board81, const std::vector<NextBall>& nb) {
  for (int i = 0; i < kNN; ++i) board_[i] = board81[i];
  next_balls_ = nb;
  score_ = 0; turns_ = 0; over_ = false;
}

}  // namespace clines
