// Golden test for the 27-feature linear leaf-value evaluator: checks the 25
// features and the final V against Python (export_feature_weights.py). No
// LibTorch needed. Run from inference_cpp/ so it finds data/.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "feature_value.h"
#include "game.h"

namespace {
struct Reader {
  std::ifstream f;
  explicit Reader(const char* p) : f(p, std::ios::binary) {}
  bool ok() const { return f.good(); }
  int I32() { int v = 0; f.read(reinterpret_cast<char*>(&v), 4); return v; }
  void F32(float* d, int n) { f.read(reinterpret_cast<char*>(d), n * 4); }
};
}  // namespace

int main() {
  using namespace clines;
  FeatureEval ev;
  if (!ev.Load("data/feature_value.bin")) {
    std::printf("cannot load data/feature_value.bin\n");
    return 1;
  }
  Reader r("data/golden_feature.bin");
  if (!r.ok()) { std::printf("cannot open data/golden_feature.bin\n"); return 1; }
  char magic[4]; r.f.read(magic, 4);
  if (std::string(magic, 4) != "CLFG") { std::printf("bad magic\n"); return 1; }

  int K = r.I32();
  double feat_diff = 0, v_diff = 0;
  for (int k = 0; k < K; ++k) {
    float bf[kNN]; r.F32(bf, kNN);
    int8_t board[kNN];
    for (int i = 0; i < kNN; ++i) board[i] = static_cast<int8_t>(bf[i]);
    int nn = r.I32();
    float nbf[9]; r.F32(nbf, 9);
    std::vector<NextBall> nb;
    for (int i = 0; i < nn; ++i)
      nb.push_back({(int)nbf[i * 3], (int)nbf[i * 3 + 1], (int)nbf[i * 3 + 2]});
    float gfeat[25]; r.F32(gfeat, 25);
    float gv; r.F32(&gv, 1);

    double feats[25];
    BoardFeaturesWithNext(board, nb, feats);
    for (int i = 0; i < 25; ++i)
      feat_diff = std::max(feat_diff, std::fabs(feats[i] - gfeat[i]));
    v_diff = std::max(v_diff, std::fabs(ev.Value(board, nb) - gv));
  }
  std::printf("feats max|diff| over %d cases = %.3e -> %s\n", K, feat_diff,
              feat_diff < 1e-3 ? "PASS" : "FAIL");
  std::printf("V     max|diff| over %d cases = %.3e -> %s\n", K, v_diff,
              v_diff < 1e-3 ? "PASS" : "FAIL");
  bool pass = feat_diff < 1e-3 && v_diff < 1e-3;
  std::printf("%s\n", pass ? "ALL PASS \xE2\x9C\x85" : "FAIL \xE2\x9D\x8C");
  return pass ? 0 : 1;
}
