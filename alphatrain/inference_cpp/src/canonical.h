// Board symmetries (D4) and the canonical form of a position under D4 x color
// relabeling. Mirrors alphatrain/canonical.py exactly (tests compare them).
#ifndef CLINES_CANONICAL_H_
#define CLINES_CANONICAL_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "game.h"

namespace clines {

// cell[v][i] = where original cell i lands in view v; act[v*6561 + a] = the
// action in view v that corresponds to original action a.
struct D4Maps {
  int cell[8][81];
  std::vector<int64_t> act;
  D4Maps() : act(8 * 6561) {
    const int m = 8;
    for (int r = 0; r < 9; ++r)
      for (int c = 0; c < 9; ++c) {
        const int rc[8][2] = {{r, c}, {c, m - r}, {m - r, m - c}, {m - c, r},
                              {r, m - c}, {m - r, c}, {c, r}, {m - c, m - r}};
        for (int v = 0; v < 8; ++v) cell[v][r * 9 + c] = rc[v][0] * 9 + rc[v][1];
      }
    for (int v = 0; v < 8; ++v)
      for (int s = 0; s < 81; ++s)
        for (int d = 0; d < 81; ++d)
          act[v * 6561 + s * 81 + d] = cell[v][s] * 81 + cell[v][d];
  }
};

struct Canonical {
  int view = 0;
  int8_t board[81];
  std::vector<NextBall> next;  // sorted by cell, colors relabeled
};

// For each view: transform board + preview, sort preview by cell, relabel
// colors by first appearance (board cells 0..80, then sorted preview), build
// an 87-entry key; keep the smallest key (ties -> smallest view).
inline Canonical Canonicalize(const int8_t* board, const std::vector<NextBall>& nb,
                              const D4Maps& d4) {
  Canonical best;
  int best_key[87];
  bool have = false;
  const int n = static_cast<int>(nb.size());
  for (int v = 0; v < 8; ++v) {
    int8_t bv[81];
    for (int i = 0; i < 81; ++i) bv[d4.cell[v][i]] = board[i];
    int pc[3], pk[3];
    for (int t = 0; t < n; ++t) { pc[t] = d4.cell[v][nb[t].r * 9 + nb[t].c]; pk[t] = nb[t].color; }
    for (int a = 1; a < n; ++a)
      for (int j = a; j > 0 && pc[j - 1] > pc[j]; --j) { std::swap(pc[j], pc[j - 1]); std::swap(pk[j], pk[j - 1]); }
    int cmap[8] = {0}; int nxt = 1;
    for (int i = 0; i < 81; ++i) if (bv[i] > 0 && cmap[bv[i]] == 0) cmap[bv[i]] = nxt++;
    for (int t = 0; t < n; ++t) if (pk[t] > 0 && cmap[pk[t]] == 0) cmap[pk[t]] = nxt++;
    for (int c = 1; c < 8; ++c) if (cmap[c] == 0) cmap[c] = nxt++;
    int key[87];
    for (int i = 0; i < 81; ++i) key[i] = bv[i] > 0 ? cmap[bv[i]] : 0;
    for (int t = 0; t < 3; ++t) {
      key[81 + 2 * t] = t < n ? pc[t] : 255;
      key[82 + 2 * t] = t < n ? cmap[pk[t]] : 0;
    }
    bool better = !have;
    if (have)
      for (int i = 0; i < 87; ++i)
        if (key[i] != best_key[i]) { better = key[i] < best_key[i]; break; }
    if (better) {
      have = true;
      std::copy(key, key + 87, best_key);
      best.view = v;
      for (int i = 0; i < 81; ++i) best.board[i] = static_cast<int8_t>(key[i]);
      best.next.clear();
      for (int t = 0; t < n; ++t) best.next.push_back({pc[t] / 9, pc[t] % 9, cmap[pk[t]]});
    }
  }
  return best;
}

}  // namespace clines

#endif  // CLINES_CANONICAL_H_
