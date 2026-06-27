// Golden test for the C++ game engine: validates obs, legal mask, and
// line-clear against vectors exported from the authoritative Python engine
// (export_game_golden.py). These kernels are RNG-free, so they must match
// Python bit-for-bit. Run from inference_cpp/ so it finds data/.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "game.h"

namespace {
struct Reader {
  std::ifstream f;
  explicit Reader(const char* path) : f(path, std::ios::binary) {}
  bool ok() const { return f.good(); }
  int32_t I32() { int32_t v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; }
  void F32(float* dst, int n) { f.read(reinterpret_cast<char*>(dst), n * 4); }
};

float MaxDiff(const float* a, const float* b, int n) {
  float m = 0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
}  // namespace

int main() {
  Reader r("data/golden_game.bin");
  if (!r.ok()) { std::printf("cannot open data/golden_game.bin\n"); return 1; }
  char magic[4]; r.f.read(magic, 4);
  if (std::string(magic, 4) != "CLGM") { std::printf("bad magic\n"); return 1; }

  using namespace clines;
  float obs_diff = 0, legal_diff = 0;
  int K = r.I32();
  for (int k = 0; k < K; ++k) {
    float bf[kNN]; r.F32(bf, kNN);
    int8_t board[kNN];
    for (int i = 0; i < kNN; ++i) board[i] = static_cast<int8_t>(bf[i]);
    int nn = r.I32();
    float nbf[9]; r.F32(nbf, 9);
    std::vector<NextBall> nb;
    for (int i = 0; i < nn; ++i)
      nb.push_back({(int)nbf[i*3], (int)nbf[i*3+1], (int)nbf[i*3+2]});
    std::vector<float> obs_g(18 * kNN), legal_g(kActions);
    r.F32(obs_g.data(), 18 * kNN);
    r.F32(legal_g.data(), kActions);

    Game g(0);
    g.SetState(board, nb);
    std::vector<float> obs(18 * kNN), legal(kActions);
    g.BuildObs(obs.data());
    g.LegalMask(legal.data());
    obs_diff = std::max(obs_diff, MaxDiff(obs.data(), obs_g.data(), 18 * kNN));
    legal_diff = std::max(legal_diff, MaxDiff(legal.data(), legal_g.data(), kActions));
  }

  int clear_mismatch = 0, board_mismatch = 0;
  int M = r.I32();
  for (int m = 0; m < M; ++m) {
    float bf[kNN]; r.F32(bf, kNN);
    int8_t board[kNN];
    for (int i = 0; i < kNN; ++i) board[i] = static_cast<int8_t>(bf[i]);
    int row = r.I32(), col = r.I32(), cleared_g = r.I32();
    float bout_f[kNN]; r.F32(bout_f, kNN);

    int cleared = Game::ClearLinesAt(board, row, col);
    if (cleared != cleared_g) ++clear_mismatch;
    for (int i = 0; i < kNN; ++i)
      if (board[i] != static_cast<int8_t>(bout_f[i])) { ++board_mismatch; break; }
  }

  std::printf("obs   max|diff| over %d cases = %.3e  -> %s\n", K, obs_diff,
              obs_diff < 1e-5 ? "PASS" : "FAIL");
  std::printf("legal max|diff| over %d cases = %.3e  -> %s\n", K, legal_diff,
              legal_diff < 1e-6 ? "PASS" : "FAIL");
  std::printf("clear: %d/%d count-mismatch, %d/%d board-mismatch  -> %s\n",
              clear_mismatch, M, board_mismatch, M,
              (clear_mismatch == 0 && board_mismatch == 0) ? "PASS" : "FAIL");
  bool pass = obs_diff < 1e-5 && legal_diff < 1e-6 && clear_mismatch == 0 &&
              board_mismatch == 0;
  std::printf("%s\n", pass ? "ALL PASS \xE2\x9C\x85" : "FAIL \xE2\x9D\x8C");
  return pass ? 0 : 1;
}
