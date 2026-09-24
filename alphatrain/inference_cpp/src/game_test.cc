// Golden test for the C++ game engine: validates obs, legal mask, and
// line-clear against vectors exported from the authoritative Python engine
// (export_game_golden.py). These kernels are RNG-free, so they must match
// Python bit-for-bit. Run from inference_cpp/ so it finds data/.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "game.h"
#include "game_json.h"
#include "mcts.h"  // LegalPriors cross-check

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

bool NeuralTerminalValueIsZero() {
  using namespace clines;

  // One empty target with three adjacent sources.  Every legal move fills the
  // target, then the fixed preview ball fills the vacated source: game over.
  // The 7-color affine pattern has no five-ball lines, so no move can escape
  // terminality by clearing.
  int8_t board[kNN];
  for (int r = 0; r < kN; ++r)
    for (int c = 0; c < kN; ++c)
      board[r * kN + c] = static_cast<int8_t>((3 * r + 5 * c) % 7 + 1);
  board[1] = 0;
  Game g(7);
  g.SetState(board, {{0, 0, 7}});

  int evaluated_states = 0;
  PolicyFn constant_head = [&evaluated_states](const float*, int n, float* logits,
                                                float* values) {
    evaluated_states += n;
    std::fill(logits, logits + static_cast<size_t>(n) * kActions, 0.0f);
    for (int i = 0; i < n; ++i) values[i] = 7.0f;
  };
  MctsConfig cfg;
  cfg.num_simulations = 4;
  cfg.batch_size = 4;
  cfg.top_k = 3;
  cfg.nn_value = true;
  MCTS search(constant_head, nullptr, cfg);
  SimpleRng move_rng(11);
  SearchResult out = search.Search(g, 0.0, move_rng);

  // Only the nonterminal root is sent to the head.  Terminal simulations must
  // lower q_min to zero and must never inherit the head's synthetic 7.0.
  if (evaluated_states != 1 || std::fabs(out.root_value - 7.0) > 1e-9 ||
      std::fabs(out.q_min) > 1e-9 || std::fabs(out.q_max - 7.0) > 1e-9)
    return false;
  for (const Candidate& cand : out.cands)
    if (cand.visits > 0 && std::fabs(cand.q) > 1e-9) return false;
  return true;
}
}  // namespace

int main() {
  Reader r("data/golden_game.bin");
  if (!r.ok()) { std::printf("cannot open data/golden_game.bin\n"); return 1; }
  char magic[4]; r.f.read(magic, 4);
  if (std::string(magic, 4) != "CLGM") { std::printf("bad magic\n"); return 1; }

  using namespace clines;
  float obs_diff = 0, legal_diff = 0;
  int lp_mismatch = 0;  // LegalPriors vs golden legal-mask cross-check
  std::vector<float> zero_logits(kActions, 0.0f);
  std::vector<int> lp_acts(kActions);
  std::vector<double> lp_pris(kActions);
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

    // LegalPriors with uniform logits + top_k=all must return exactly the
    // golden legal set, with priors summing to 1.
    int kk = LegalPriors(board, zero_logits.data(), kActions,
                         lp_acts.data(), lp_pris.data());
    int want = 0;
    for (int a = 0; a < kActions; ++a) want += legal_g[a] > 0.5f;
    double psum = 0;
    bool ok = (kk == want);
    for (int i = 0; i < kk && ok; ++i) {
      if (legal_g[lp_acts[i]] < 0.5f) ok = false;
      psum += lp_pris[i];
    }
    if (kk > 0 && std::fabs(psum - 1.0) > 1e-9) ok = false;
    if (!ok) ++lp_mismatch;
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
  std::printf("LegalPriors vs golden mask: %d/%d mismatch  -> %s\n",
              lp_mismatch, K, lp_mismatch == 0 ? "PASS" : "FAIL");
  // A temperature-sampled behavior action must remain distinct from the
  // visit-winner teacher stored in the moves schema.
  SearchResult sampled;
  sampled.action = 81;
  sampled.cands = {{162, 9, 0.6, 0.2}, {81, 3, 0.4, 0.1}};
  Game label_game(1);
  label_game.Reset();
  MoveRec rec = MakeMoveRec(label_game, sampled);
  bool teacher_ok = rec.action == 81 && rec.teacher_action == 162;
  std::printf("behavior/teacher schema split -> %s\n",
              teacher_ok ? "PASS" : "FAIL");
  bool terminal_value_ok = NeuralTerminalValueIsZero();
  std::printf("NN terminal survival value = 0 -> %s\n",
              terminal_value_ok ? "PASS" : "FAIL");
  bool pass = obs_diff < 1e-5 && legal_diff < 1e-6 && clear_mismatch == 0 &&
              board_mismatch == 0 && lp_mismatch == 0 && teacher_ok &&
              terminal_value_ok;
  std::printf("%s\n", pass ? "ALL PASS \xE2\x9C\x85" : "FAIL \xE2\x9D\x8C");
  return pass ? 0 : 1;
}
