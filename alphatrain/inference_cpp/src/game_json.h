// Shared moves-schema JSON writer for selfplay + crisis game recorders.
// Schema consumed by alphatrain/scripts/build_expert_v2_tensor.py: per move
// board (before the move), next_balls, num_next, chosen_move, cand_moves /
// cand_visits / cand_prior (CLEAN pre-Dirichlet prior as LOG-prob) / cand_q,
// root_value, q_min, q_max.

#ifndef CLINES_GAME_JSON_H_
#define CLINES_GAME_JSON_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "game.h"
#include "mcts.h"

namespace clines {

constexpr int kTopKSave = 15;  // selfplay.py / crisis_mining.py top_k_save

struct MoveRec {
  int8_t board[81];
  std::vector<NextBall> next_balls;
  int action;
  std::vector<Candidate> cands;  // top kTopKSave by visits
  double root_value, q_min, q_max;
};

// Snapshot a move about to be played from game state `g` + search result `r`.
inline MoveRec MakeMoveRec(const Game& g, const SearchResult& r) {
  MoveRec mr;
  std::memcpy(mr.board, g.board().data(), 81);
  mr.next_balls = g.next_balls();
  mr.action = r.action;
  int nc = std::min<int>(kTopKSave, static_cast<int>(r.cands.size()));
  mr.cands.assign(r.cands.begin(), r.cands.begin() + nc);
  mr.root_value = r.root_value;
  mr.q_min = r.q_min;
  mr.q_max = r.q_max;
  return mr;
}

// Append a double with enough precision for float32 round-trip.
inline void AppendD(std::string& s, double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.8g", v);
  s += buf;
}

// Append the "moves": [...] array (without the key) to `s`.
// slim (default): only what train_path_b consumes — cand_moves + cand_visits
// (-> pol_indices/pol_values + decisiveness top-share). cand_prior/cand_q/
// root_value/q_min/q_max are Gumbel-only (~35% of the bytes); written only
// when full=true. Precedent: the pillar3k corpus itself had zero Q (HISTORY
// 173: "all-zero-Q warning benign; 3b uses pol_values").
inline void AppendMovesArray(std::string& s, const std::vector<MoveRec>& moves,
                             bool full = false) {
  s += "[";
  for (size_t m = 0; m < moves.size(); ++m) {
    const MoveRec& mr = moves[m];
    if (m) s += ", ";
    s += "{\"board\": [";
    for (int r = 0; r < 9; ++r) {
      if (r) s += ", ";
      s += "[";
      for (int c = 0; c < 9; ++c) {
        if (c) s += ", ";
        s += std::to_string(static_cast<int>(mr.board[r * 9 + c]));
      }
      s += "]";
    }
    s += "], \"next_balls\": [";
    for (size_t i = 0; i < mr.next_balls.size(); ++i) {
      if (i) s += ", ";
      s += "{\"row\": " + std::to_string(mr.next_balls[i].r) +
           ", \"col\": " + std::to_string(mr.next_balls[i].c) +
           ", \"color\": " + std::to_string(mr.next_balls[i].color) + "}";
    }
    s += "], \"num_next\": " + std::to_string(mr.next_balls.size());
    int src = mr.action / 81, tgt = mr.action % 81;
    s += ", \"chosen_move\": {\"sr\": " + std::to_string(src / 9) +
         ", \"sc\": " + std::to_string(src % 9) +
         ", \"tr\": " + std::to_string(tgt / 9) +
         ", \"tc\": " + std::to_string(tgt % 9) + "}";
    s += ", \"cand_moves\": [";
    for (size_t i = 0; i < mr.cands.size(); ++i) {
      if (i) s += ", ";
      s += std::to_string(mr.cands[i].action);
    }
    s += "], \"cand_visits\": [";
    for (size_t i = 0; i < mr.cands.size(); ++i) {
      if (i) s += ", ";
      s += std::to_string(mr.cands[i].visits);
    }
    s += "]";
    if (full) {
      s += ", \"cand_prior\": [";
      for (size_t i = 0; i < mr.cands.size(); ++i) {
        if (i) s += ", ";
        AppendD(s, std::log(std::max(mr.cands[i].prior, 1e-30)));
      }
      s += "], \"cand_q\": [";
      for (size_t i = 0; i < mr.cands.size(); ++i) {
        if (i) s += ", ";
        AppendD(s, mr.cands[i].q);
      }
      s += "], \"root_value\": ";
      AppendD(s, mr.root_value);
      s += ", \"q_min\": ";
      AppendD(s, mr.q_min);
      s += ", \"q_max\": ";
      AppendD(s, mr.q_max);
    }
    s += "}";
  }
  s += "]";
}

// Write string to path ATOMICALLY (tmp + rename) or abort. Atomicity makes
// aborted runs resume-safe: a kill mid-write can't leave a truncated .json.
inline void WriteFileOrDie(const std::string& path, const std::string& body) {
  std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "FATAL: cannot write %s\n", tmp.c_str());
    std::abort();
  }
  std::fwrite(body.data(), 1, body.size(), f);
  std::fclose(f);
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    std::fprintf(stderr, "FATAL: cannot rename %s\n", tmp.c_str());
    std::abort();
  }
}

}  // namespace clines

#endif  // CLINES_GAME_JSON_H_
