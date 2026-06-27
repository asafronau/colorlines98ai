#include "net.h"

#include <cmath>
#include <cstdint>
#include <fstream>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"

namespace clines {
namespace {

// Little helper: read a fixed number of bytes or fail. (std::ifstream::read
// doesn't throw by default, so we check gcount.)
absl::Status ReadExact(std::ifstream& f, void* dst, std::streamsize n) {
  f.read(static_cast<char*>(dst), n);
  if (f.gcount() != n) return absl::DataLossError("unexpected EOF");
  return absl::OkStatus();
}

template <typename T>
T ReadPod(std::ifstream& f, absl::Status* st) {
  T v{};
  if (st->ok()) *st = ReadExact(f, &v, sizeof(T));
  return v;
}

}  // namespace

absl::StatusOr<Weights> LoadBlob(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return absl::NotFoundError(absl::StrFormat("cannot open %s", path));

  char magic[4];
  absl::Status st = ReadExact(f, magic, 4);
  if (!st.ok()) return st;
  if (std::string(magic, 4) != "CLNW")
    return absl::DataLossError("bad magic (not a CLNW blob)");

  uint32_t num = ReadPod<uint32_t>(f, &st);
  Weights out;
  for (uint32_t i = 0; i < num && st.ok(); ++i) {
    uint32_t name_len = ReadPod<uint32_t>(f, &st);
    std::string name(name_len, '\0');
    if (st.ok()) st = ReadExact(f, name.data(), name_len);

    uint32_t ndim = ReadPod<uint32_t>(f, &st);
    Tensor t;
    t.shape.resize(ndim);
    int total = 1;
    for (uint32_t d = 0; d < ndim && st.ok(); ++d) {
      int32_t dim = ReadPod<int32_t>(f, &st);
      t.shape[d] = dim;
      total *= dim;
    }
    t.data.resize(total);
    if (st.ok()) st = ReadExact(f, t.data.data(), total * sizeof(float));
    out[name] = std::move(t);
  }
  if (!st.ok()) return st;
  return out;
}

// ---- The one op we implement together for Milestone 1 ----
//
// Naive stride-1 conv. For each output channel `oc` and pixel (oh,ow), sum over
// every input channel `ic` and kernel offset (kh,kw) of w[oc,ic,kh,kw] * in at
// the (padded) input pixel. Padding is implicit: out-of-bounds reads are skipped
// (equivalent to zero-padding). This is the textbook definition — slow but
// obviously correct, which is what we want first. (Speed comes later.)
Tensor Conv2d(const Tensor& in, const Tensor& w, int pad,
              const std::vector<float>& bias) {
  const int Cin = in.shape[0], H = in.shape[1], W = in.shape[2];
  const int Cout = w.shape[0], kh = w.shape[2], kw = w.shape[3];

  Tensor out;
  out.shape = {Cout, H, W};
  out.data.assign(static_cast<size_t>(Cout) * H * W, 0.0f);

  for (int oc = 0; oc < Cout; ++oc) {
    const float b = bias.empty() ? 0.0f : bias[oc];
    for (int oh = 0; oh < H; ++oh) {
      for (int ow = 0; ow < W; ++ow) {
        float acc = b;
        for (int ic = 0; ic < Cin; ++ic) {
          for (int r = 0; r < kh; ++r) {
            const int ih = oh + r - pad;
            if (ih < 0 || ih >= H) continue;
            for (int c = 0; c < kw; ++c) {
              const int iw = ow + c - pad;
              if (iw < 0 || iw >= W) continue;
              // weight index: ((oc*Cin + ic)*kh + r)*kw + c
              const float wv = w.data[((static_cast<size_t>(oc) * Cin + ic) * kh + r) * kw + c];
              acc += wv * in.at(ic, ih, iw);
            }
          }
        }
        out.at(oc, oh, ow) = acc;
      }
    }
  }
  return out;
}

// ===================== YOUR TURN =====================
// Implement these. Each is a few lines; check against the golden after each.
//
// HINT (ReLU): loop over x.data, set negatives to 0.
void ReluInPlace(Tensor& x) {
  // TODO(you): for (float& v : x.data) v = std::max(v, 0.0f);
}

// HINT (BatchNorm): gamma/beta/mean/var are per-channel (shape {C}). For each
// channel c, precompute scale = gamma[c]/sqrt(var[c]+eps) and shift =
// beta[c] - mean[c]*scale, then y = x*scale + shift over that channel's HxW.
Tensor BatchNorm(const Tensor& in, const Tensor& gamma, const Tensor& beta,
                 const Tensor& mean, const Tensor& var, float eps) {
  // TODO(you).
  return in;
}

// HINT (Forward): follow the architecture comment in net.h.
//   stem: Conv2d(obs, w["stem.0.weight"], pad=1) -> BatchNorm(w["stem.1.*"]) -> ReLU
//   blocks i in [0,num_blocks): pre-activation residual (save input, BN1->ReLU->
//     Conv1 (w["blocks.i.conv1.weight"], pad=1) -> BN2->ReLU->Conv2 -> add input)
//   tail: BatchNorm(w["backbone_bn.*"]) -> ReLU
//   head: Conv2d(.,w["policy_conv1.weight"],pad=0) -> BN(w["policy_bn.*"]) -> ReLU
//         -> Conv2d(.,w["policy_conv2.weight"],pad=0, bias=w["policy_conv2.bias"].data)
//   reshape the {81,9,9} result's flat data into a {6561} Tensor.
Tensor Forward(const Weights& w, const Tensor& obs, int num_blocks) {
  // TODO(you).
  return obs;
}

}  // namespace clines
