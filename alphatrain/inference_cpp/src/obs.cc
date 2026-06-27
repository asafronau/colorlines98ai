// 18-channel observation, a faithful port of observation.py::build_observation.
//   0-6  one-hot color   7 empty   8-10 next-ball color/7   11 next-ball mask
//   12   empty-component size / 81
//   13-16 line length per direction (H,V,D1,D2) / 9   17 max line length / 9

#include <algorithm>

#include "game.h"

namespace clines {
namespace {
constexpr int kLineDr[4] = {0, 1, 1, 1};
constexpr int kLineDc[4] = {1, 0, 1, -1};
inline int Idx(int r, int c) { return r * kN + c; }
inline bool InB(int r, int c) { return r >= 0 && r < kN && c >= 0 && c < kN; }
inline int Ch(int ch, int r, int c) { return ch * kNN + r * kN + c; }

// Same-color run length through (r,c) in direction (dr,dc), forward+backward.
int LineLen(const int8_t* b, int r, int c, int dr, int dc) {
  int8_t color = b[Idx(r, c)];
  if (color == 0) return 0;
  int count = 1;
  for (int nr = r + dr, nc = c + dc; InB(nr, nc) && b[Idx(nr, nc)] == color;
       nr += dr, nc += dc)
    ++count;
  for (int nr = r - dr, nc = c - dc; InB(nr, nc) && b[Idx(nr, nc)] == color;
       nr -= dr, nc -= dc)
    ++count;
  return count;
}
}  // namespace

void Game::BuildObs(float* out) const {
  std::fill(out, out + 18 * kNN, 0.0f);
  const int8_t* b = board_.data();

  // 0-6 colors, 7 empty
  for (int i = 0; i < kNN; ++i) {
    int8_t v = b[i];
    if (v == 0) out[7 * kNN + i] = 1.0f;
    else out[(v - 1) * kNN + i] = 1.0f;
  }

  // 8-10 next-ball color/7, 11 next-ball mask  (at most 3 balls)
  int nn = std::min<int>(static_cast<int>(next_balls_.size()), 3);
  for (int i = 0; i < nn; ++i) {
    const NextBall& nb = next_balls_[i];
    out[Ch(8 + i, nb.r, nb.c)] = nb.color / 7.0f;
    out[Ch(11, nb.r, nb.c)] = 1.0f;
  }

  // 12 component-size heatmap (size of the empty component each cell belongs to)
  int8_t labels[kNN];
  LabelEmpty(b, labels);
  int counts[kNN + 1] = {0};
  for (int i = 0; i < kNN; ++i)
    if (labels[i] > 0) counts[labels[i]]++;
  for (int i = 0; i < kNN; ++i)
    if (labels[i] > 0) out[12 * kNN + i] = counts[labels[i]] / 81.0f;

  // 13-16 directional line potentials, 17 max
  for (int r = 0; r < kN; ++r) {
    for (int c = 0; c < kN; ++c) {
      if (b[Idx(r, c)] == 0) continue;
      int max_len = 0;
      for (int di = 0; di < 4; ++di) {
        int len = LineLen(b, r, c, kLineDr[di], kLineDc[di]);
        out[Ch(13 + di, r, c)] = len / 9.0f;
        if (len > max_len) max_len = len;
      }
      out[Ch(17, r, c)] = max_len / 9.0f;
    }
  }
}

}  // namespace clines
