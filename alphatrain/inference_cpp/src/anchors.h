// Saved start states ("anchors"), e.g. mcts_crisis rewind points, for eval --anchors and
// anchor_search. A game restarts exactly as the crisis replay did: Game(seed) + SetState, so the
// spawn stream is the replay's. One anchor per line:
//   seed start_turn cap b0 .. b80 r0 c0 col0 r1 c1 col1 r2 c2 col2   (unused next balls: -1 -1 -1)
// Written by alphatrain/scripts/crisis_anchors.py export.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "game.h"

namespace clines {

struct Anchor {
  uint64_t seed;
  int turn, cap;
  int8_t board[81];
  std::vector<NextBall> next;
};

inline std::vector<Anchor> ReadAnchors(const std::string& path) {
  std::vector<Anchor> out;
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) { std::fprintf(stderr, "FATAL: cannot open anchors %s\n", path.c_str()); std::exit(2); }
  unsigned long long seed;
  int turn, cap;
  while (std::fscanf(f, "%llu %d %d", &seed, &turn, &cap) == 3) {
    Anchor a{seed, turn, cap, {}, {}};
    for (int i = 0; i < 81; ++i) {
      int v;
      if (std::fscanf(f, "%d", &v) != 1 || v < 0 || v > 7) {
        std::fprintf(stderr, "FATAL: bad board cell in anchor %zu\n", out.size()); std::exit(2);
      }
      a.board[i] = (int8_t)v;
    }
    for (int k = 0; k < 3; ++k) {
      int r, c, col;
      if (std::fscanf(f, "%d %d %d", &r, &c, &col) != 3) {
        std::fprintf(stderr, "FATAL: bad next ball in anchor %zu\n", out.size()); std::exit(2);
      }
      if (col > 0) a.next.push_back({r, c, col});
    }
    out.push_back(std::move(a));
  }
  std::fclose(f);
  return out;
}

inline Game StartFromAnchor(const Anchor& a) {
  Game g(a.seed);  // RNG only: no Reset(), exactly like the mcts_crisis replay
  g.SetState(a.board, a.next, 0, a.turn);
  return g;
}

}  // namespace clines
