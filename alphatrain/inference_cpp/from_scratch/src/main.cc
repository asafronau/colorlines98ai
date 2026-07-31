// Milestone harness: load the exported weights + golden, run a piece of the
// forward pass, and check it against PyTorch bit-close.
//
// Milestone 1 (now): the stem conv. Conv2d(obs, stem.0.weight) must match the
// golden 'stem_conv_out' to ~1e-3.
// Later milestones: once you've implemented BatchNorm/ReLU/Forward in net.cc,
// flip RUN_FULL_FORWARD to true to check the final 6561 logits.

#include <cmath>
#include <string>

#include "absl/strings/str_format.h"
#include "net.h"

namespace {
constexpr bool kRunFullForward = false;  // set true after net.cc::Forward is done
constexpr int kNumBlocks = 10;

// Max absolute difference between two equally-shaped tensors.
float MaxAbsDiff(const clines::Tensor& a, const clines::Tensor& b) {
  float m = 0.0f;
  for (int i = 0; i < a.size(); ++i)
    m = std::max(m, std::abs(a.data[i] - b.data[i]));
  return m;
}
}  // namespace

int main() {
  auto weights = clines::LoadBlob("data/weights.bin");
  auto golden = clines::LoadBlob("data/golden.bin");
  if (!weights.ok() || !golden.ok()) {
    absl::FPrintF(stderr, "load failed: %s / %s\n",
                  weights.status().message(), golden.status().message());
    return 1;
  }
  const clines::Tensor& obs = golden->at("obs");
  absl::PrintF("loaded %d weight tensors; obs shape {%d,%d,%d}\n",
               (int)weights->size(), obs.shape[0], obs.shape[1], obs.shape[2]);

  // ---- Milestone 1: stem conv ----
  const clines::Tensor& stem_w = weights->at("stem.0.weight");  // {C,18,3,3}
  clines::Tensor got = clines::Conv2d(obs, stem_w, /*pad=*/1);
  const clines::Tensor& want = golden->at("stem_conv_out");
  float diff = MaxAbsDiff(got, want);
  absl::PrintF("stem conv: max|diff| = %.3e  -> %s\n", diff,
               diff < 1e-3f ? "PASS ✅" : "FAIL ❌");

  // ---- Later: full forward ----
  if (kRunFullForward) {
    clines::Tensor logits = clines::Forward(*weights, obs, kNumBlocks);
    float ld = MaxAbsDiff(logits, golden->at("logits"));
    absl::PrintF("full forward logits: max|diff| = %.3e -> %s\n", ld,
                 ld < 1e-2f ? "PASS ✅" : "FAIL ❌");
  }
  return 0;
}
