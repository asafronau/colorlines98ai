#include "feature_value.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <vector>

namespace clines {
namespace {
constexpr int kDr4[4] = {0, 0, 1, -1};
constexpr int kDc4[4] = {1, -1, 0, 0};
constexpr int kLineDr[4] = {0, 1, 1, 1};   // H, V, D1, D2
constexpr int kLineDc[4] = {1, 0, 1, -1};
inline int Idx(int r, int c) { return r * kN + c; }
inline bool InB(int r, int c) { return r >= 0 && r < kN && c >= 0 && c < kN; }
}  // namespace

void BoardFeatures(const int8_t* board, double* out) {
  // --- empty connected components ---
  int labels[kNN];
  std::fill(labels, labels + kNN, 0);
  int qr[kNN], qc[kNN];
  int current = 0, empty = 0;
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c) {
      if (board[Idx(r, c)] != 0) continue;
      ++empty;
      if (labels[Idx(r, c)] != 0) continue;
      ++current;
      labels[Idx(r, c)] = current;
      qr[0] = r; qc[0] = c;
      int head = 0, tail = 1;
      while (head < tail) {
        int cr = qr[head], cc = qc[head];
        ++head;
        for (int d = 0; d < 4; ++d) {
          int nr = cr + kDr4[d], nc = cc + kDc4[d];
          if (InB(nr, nc) && board[Idx(nr, nc)] == 0 && labels[Idx(nr, nc)] == 0) {
            labels[Idx(nr, nc)] = current;
            qr[tail] = nr; qc[tail] = nc; ++tail;
          }
        }
      }
    }
  int n_components = current;
  std::vector<int> comp_sizes(current + 1, 0);
  for (int i = 0; i < kNN; ++i)
    if (labels[i] > 0) comp_sizes[labels[i]]++;
  int largest = 0, tiny_count = 0;
  for (int i = 1; i <= current; ++i) {
    if (comp_sizes[i] > largest) largest = comp_sizes[i];
    if (comp_sizes[i] <= 3) ++tiny_count;
  }

  // --- mobility (reachable empties per ball, dedup by component) ---
  int mobility = 0, low_mob = 0, n_balls = 0, min_reach = 999;
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c) {
      if (board[Idx(r, c)] <= 0) continue;
      int reachable = 0, seen[4], ns = 0;
      for (int d = 0; d < 4; ++d) {
        int nr = r + kDr4[d], nc = c + kDc4[d];
        if (!InB(nr, nc)) continue;
        int lbl = labels[Idx(nr, nc)];
        if (lbl <= 0) continue;
        bool s = false;
        for (int i = 0; i < ns; ++i) if (seen[i] == lbl) { s = true; break; }
        if (!s) { seen[ns++] = lbl; reachable += comp_sizes[lbl]; }
      }
      mobility += reachable;
      if (reachable < 5) ++low_mob;
      if (reachable < min_reach) min_reach = reachable;
      ++n_balls;
    }
  double avg_reach = static_cast<double>(mobility) / (n_balls > 0 ? n_balls : 1);
  if (n_balls == 0) min_reach = 0;

  // --- colors present ---
  int color_counts[7] = {0};
  for (int i = 0; i < kNN; ++i) {
    int v = board[i];
    if (v > 0) color_counts[v - 1]++;
  }
  int n_colors = 0;
  for (int i = 0; i < 7; ++i) if (color_counts[i] > 0) ++n_colors;

  // --- adjacency (right,down) + line potential (from-start, 4 dirs) ---
  int same_adj = 0, diff_adj = 0, line3 = 0, line4 = 0;
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c) {
      int color = board[Idx(r, c)];
      if (color == 0) continue;
      for (int d = 0; d < 2; ++d) {
        int ar = (d == 0 ? r : r + 1), ac = (d == 0 ? c + 1 : c);
        if (InB(ar, ac)) {
          int nv = board[Idx(ar, ac)];
          if (nv > 0) { if (nv == color) ++same_adj; else ++diff_adj; }
        }
      }
      for (int di = 0; di < 4; ++di) {
        int dr = kLineDr[di], dc = kLineDc[di];
        int pr = r - dr, pc = c - dc;
        if (InB(pr, pc) && board[Idx(pr, pc)] == color) continue;  // not the start
        int length = 1, cr = r + dr, cc = c + dc;
        while (InB(cr, cc) && board[Idx(cr, cc)] == color) { ++length; cr += dr; cc += dc; }
        if (length == 3) ++line3;
        else if (length == 4) ++line4;
      }
    }

  // --- central 3x3 (rows/cols 3-5) ---
  int center_balls = 0, center_colors[7] = {0};
  for (int r = 3; r < 6; ++r)
    for (int c = 3; c < 6; ++c) {
      int v = board[Idx(r, c)];
      if (v > 0) { ++center_balls; center_colors[v - 1] = 1; }
    }
  int center_cc = 0;
  for (int i = 0; i < 7; ++i) if (center_colors[i]) ++center_cc;

  out[0] = empty;       out[1] = n_components; out[2] = largest;   out[3] = tiny_count;
  out[4] = mobility;    out[5] = avg_reach;    out[6] = min_reach; out[7] = low_mob;
  out[8] = n_balls;     out[9] = n_colors;
  out[10] = same_adj;   out[11] = diff_adj;
  out[12] = line3;      out[13] = line4;
  out[14] = center_balls; out[15] = center_cc;
}

void BoardFeaturesWithNext(const int8_t* board, const std::vector<NextBall>& nb,
                           double* out) {
  double before[16];
  BoardFeatures(board, before);

  int8_t after[kNN];
  std::memcpy(after, board, kNN);
  int n_next = std::min<int>(static_cast<int>(nb.size()), 3);
  int n_blocked = 0, n_same = 0;
  for (int i = 0; i < n_next; ++i) {
    int r = nb[i].r, c = nb[i].c, col = nb[i].color;
    if (board[Idx(r, c)] != 0) { ++n_blocked; continue; }
    bool has_same = false;
    if (r > 0 && board[Idx(r - 1, c)] == col) has_same = true;
    else if (r < kN - 1 && board[Idx(r + 1, c)] == col) has_same = true;
    else if (c > 0 && board[Idx(r, c - 1)] == col) has_same = true;
    else if (c < kN - 1 && board[Idx(r, c + 1)] == col) has_same = true;
    if (has_same) ++n_same;
    after[Idx(r, c)] = static_cast<int8_t>(col);
  }
  double afterf[16];
  BoardFeatures(after, afterf);

  int max_next_line = 0;
  for (int i = 0; i < n_next; ++i) {
    int r = nb[i].r, c = nb[i].c, col = nb[i].color;
    if (board[Idx(r, c)] != 0) continue;  // blocked spawn
    for (int di = 0; di < 4; ++di) {
      int dr = kLineDr[di], dc = kLineDc[di];
      int length = 1, cr = r + dr, cc = c + dc;
      while (InB(cr, cc) && after[Idx(cr, cc)] == col) { ++length; cr += dr; cc += dc; }
      cr = r - dr; cc = c - dc;
      while (InB(cr, cc) && after[Idx(cr, cc)] == col) { ++length; cr -= dr; cc -= dc; }
      if (length > max_next_line) max_next_line = length;
    }
  }

  for (int i = 0; i < 16; ++i) out[i] = before[i];
  out[16] = afterf[2] - before[2];   // delta_largest
  out[17] = afterf[1] - before[1];   // delta_components
  out[18] = afterf[7] - before[7];   // delta_low_mob
  out[19] = afterf[5] - before[5];   // delta_avg_reach
  out[20] = n_same;
  out[21] = n_blocked;
  out[22] = max_next_line;
  out[23] = max_next_line >= 4 ? 1 : 0;
  out[24] = max_next_line >= 5 ? 1 : 0;
}

bool FeatureEval::Load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[4];
  f.read(magic, 4);
  if (std::string(magic, 4) != "CLFV") return false;
  f.read(reinterpret_cast<char*>(coefs_), 27 * sizeof(float));
  f.read(reinterpret_cast<char*>(means_), 27 * sizeof(float));
  f.read(reinterpret_cast<char*>(stds_), 27 * sizeof(float));
  f.read(reinterpret_cast<char*>(&bias_), sizeof(float));
  return static_cast<bool>(f);
}

double FeatureEval::Value(const int8_t* board,
                          const std::vector<NextBall>& nb) const {
  double feats[25];
  BoardFeaturesWithNext(board, nb, feats);
  double empty = feats[0], largest = feats[2], n_comp = feats[1];
  double ratio = largest / (empty > 0 ? empty : 1.0);
  double frag = (empty - largest) * n_comp;
  double v = bias_;
  for (int i = 0; i < 25; ++i) v += coefs_[i] * (feats[i] - means_[i]) / stds_[i];
  v += coefs_[25] * (ratio - means_[25]) / stds_[25];
  v += coefs_[26] * (frag - means_[26]) / stds_[26];
  return v;
}

}  // namespace clines
