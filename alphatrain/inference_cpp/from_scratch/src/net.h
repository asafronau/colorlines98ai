// Color Lines 98 — policy net inference in C++.
//
// Architecture (matches the PyTorch PolicyNet):
//   stem:  Conv2d(18 -> C, 3x3, pad 1, no bias) -> BatchNorm -> ReLU
//   body:  N x ResBlock, each: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3, then + input
//   tail:  backbone_bn -> ReLU
//   head:  Conv2d(C -> 128, 1x1) -> BN -> ReLU -> Conv2d(128 -> 81, 1x1, with bias)
//   out:   reshape (81,9,9) -> 6561 logits
//
// Tensors are stored row-major (C-contiguous), same as numpy/PyTorch, so the
// weights exported by export_weights.py drop straight in.

#ifndef CLINES_NET_H_
#define CLINES_NET_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"

namespace clines {

// A dense float tensor. `shape` is e.g. {C,H,W} for activations or
// {Cout,Cin,kh,kw} for conv weights. `data` is row-major.
struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;

  int size() const {
    int n = 1;
    for (int d : shape) n *= d;
    return n;
  }
  // Index a {C,H,W} tensor. (data[(c*H + h)*W + w])
  float& at(int c, int h, int w) {
    return data[(c * shape[1] + h) * shape[2] + w];
  }
  float at(int c, int h, int w) const {
    return data[(c * shape[1] + h) * shape[2] + w];
  }
};

// name -> tensor. Loaded from the CLNW binary blob.
using Weights = absl::flat_hash_map<std::string, Tensor>;

// Read a CLNW blob (weights.bin or golden.bin). Returns an error Status if the
// file is missing/corrupt — abseil's StatusOr is how you return "value or error".
absl::StatusOr<Weights> LoadBlob(const std::string& path);

// Stride-1 2D convolution. `in` is {Cin,H,W}, `w` is {Cout,Cin,kh,kw}, output is
// {Cout,H,W} (spatial size preserved when pad == (k-1)/2). `bias` may be empty.
Tensor Conv2d(const Tensor& in, const Tensor& w, int pad,
              const std::vector<float>& bias = {});

// ===================== YOUR TURN (see README "Milestones") =====================
// Implement these in net.cc, checking against the golden as you go.
//
//   ReLU in place: x = max(x, 0).
void ReluInPlace(Tensor& x);
//   Inference BatchNorm per channel: y = (x - mean)/sqrt(var+eps) * gamma + beta.
Tensor BatchNorm(const Tensor& in, const Tensor& gamma, const Tensor& beta,
                 const Tensor& mean, const Tensor& var, float eps = 1e-5f);
//   Full forward: obs {18,9,9} -> logits {6561}. Compose the ops above per the
//   architecture comment at the top of this file.
Tensor Forward(const Weights& w, const Tensor& obs, int num_blocks);

}  // namespace clines

#endif  // CLINES_NET_H_
