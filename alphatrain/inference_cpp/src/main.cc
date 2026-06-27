// Color Lines 98 — policy inference using LibTorch (PyTorch's official C++ API).
//
// The entire "engine" is: load the TorchScript module, run forward(), read the
// move. LibTorch already implements conv/batchnorm/relu, so there is no math to
// write here. We also load one real obs + PyTorch's logits and check they match.
//
// Note: the module was traced in inference mode (BatchNorm running-stats baked
// in), so it is already frozen for inference — no mode toggle needed here.

#include <torch/script.h>  // the one LibTorch header we need

#include <fstream>
#include <iostream>
#include <vector>

// Read `count` little-endian float32 values from a raw file into a vector.
static std::vector<float> ReadFloats(const std::string& path, int count) {
  std::vector<float> v(count);
  std::ifstream f(path, std::ios::binary);
  f.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
  if (!f) {
    std::cerr << "failed to read " << path << "\n";
    std::exit(1);
  }
  return v;
}

int main() {
  // 1) Load the network. Weights, BatchNorm stats, architecture — all baked in.
  torch::jit::Module net;
  try {
    net = torch::jit::load("data/policy_ts.pt");
  } catch (const c10::Error& e) {
    std::cerr << "could not load data/policy_ts.pt (run export_ts.py first): "
              << e.what() << "\n";
    return 1;
  }

  // 2) Build the input tensor from the example obs (shape {1, 18, 9, 9}).
  constexpr int kObs = 18 * 9 * 9, kLogits = 6561;
  std::vector<float> obs_data = ReadFloats("data/example_obs.f32", kObs);
  // from_blob wraps our vector's memory without copying; clone() makes the
  // tensor own a copy so it stays valid for the rest of the program.
  torch::Tensor obs = torch::from_blob(obs_data.data(), {1, 18, 9, 9}).clone();

  // 3) Run the forward pass. forward() takes a list of inputs (here, just obs).
  torch::Tensor logits = net.forward({obs}).toTensor();  // shape {1, 6561}

  // 4) The chosen move = argmax over the 6561 outputs (= source*81 + target).
  int move = logits.argmax(1).item<int>();
  std::cout << "predicted move index " << move
            << "  (source cell " << move / 81 << ", target cell " << move % 81
            << ")\n";

  // 5) Sanity check: does C++ match PyTorch numerically?
  std::vector<float> want = ReadFloats("data/example_logits.f32", kLogits);
  torch::Tensor want_t = torch::from_blob(want.data(), {kLogits});
  float diff = (logits.view({kLogits}) - want_t).abs().max().item<float>();
  std::cout << "max|diff| vs PyTorch = " << diff
            << (diff < 1e-3f ? "   PASS \xE2\x9C\x85\n" : "   FAIL \xE2\x9D\x8C\n");
  return 0;
}
