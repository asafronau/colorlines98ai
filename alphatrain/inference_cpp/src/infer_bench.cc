// Where does an InferenceServer round trip spend its time? For one TorchScript module (policy-only or the fused
// policy+value export) and a list of batch sizes, times (ms per call, MPS, fp16 like the servers):
//   forward  : upload + forward + a 1-number readback (so the GPU work is complete)
//   server   : what InferenceServer does today: forward, logits .to(float) on the GPU, .cpu() of all 6561 logits
//              (+ the values when fused)
//   half     : the same but reading the logits back as fp16 (half the bytes; conversion left to the caller)
//
//   ./build/infer_bench --model data/pv_S1_ts.pt --fused 1 --batches 55,145,300,600,1200
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  std::string model;
  bool fused = false;
  std::vector<int> batches = {55, 145, 300, 600, 1200};
  int iters = 60;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--model" && i + 1 < argc) model = argv[++i];
    else if (k == "--fused" && i + 1 < argc) fused = std::stoi(argv[++i]) != 0;
    else if (k == "--iters" && i + 1 < argc) iters = std::stoi(argv[++i]);
    else if (k == "--batches" && i + 1 < argc) {
      batches.clear();
      std::stringstream ss(argv[++i]);
      for (std::string t; std::getline(ss, t, ',');) batches.push_back(std::stoi(t));
    } else {
      std::fprintf(stderr, "FATAL: unknown or incomplete flag %s\n", k.c_str());
      return 2;
    }
  }
  if (model.empty()) { std::fprintf(stderr, "FATAL: --model required\n"); return 2; }
  torch::InferenceMode guard;
  torch::Device dev(torch::kMPS);
  torch::jit::Module m = torch::jit::load(model);
  m.to(dev);
  m.to(torch::kHalf);
  auto logits_of = [&](const torch::IValue& out, torch::Tensor* values) {
    if (!fused) return out.toTensor();
    auto tup = out.toTuple();
    if (values) *values = tup->elements()[1].toTensor();
    return tup->elements()[0].toTensor();
  };
  std::printf("%s (%s)\n batch   forward ms   server ms   half ms   server us/position\n", model.c_str(),
              fused ? "fused policy+value" : "policy");
  for (int b : batches) {
    torch::Tensor obs_cpu = torch::rand({b, 18, 9, 9});
    auto time_ms = [&](int mode) {
      double total = 0;
      for (int it = -5; it < iters; ++it) {
        auto t0 = Clock::now();
        torch::Tensor obs = obs_cpu.to(torch::kHalf).to(dev);
        torch::Tensor values;
        torch::Tensor lg = logits_of(m.forward({obs}), fused ? &values : nullptr);
        if (mode == 0) {
          (void)lg.sum().item<float>();
        } else if (mode == 1) {
          torch::Tensor c = lg.to(torch::kFloat).cpu().contiguous();
          if (fused) (void)values.to(torch::kFloat).cpu().contiguous();
          (void)c.data_ptr<float>()[0];
        } else {
          torch::Tensor c = lg.cpu().contiguous();
          if (fused) (void)values.to(torch::kFloat).cpu().contiguous();
          (void)c.data_ptr<at::Half>()[0];
        }
        if (it >= 0) total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      }
      return total / iters;
    };
    double f = time_ms(0), s = time_ms(1), h = time_ms(2);
    std::printf(" %5d   %10.2f   %9.2f   %7.2f   %10.1f\n", b, f, s, h, 1000.0 * s / b);
    std::fflush(stdout);
  }
  return 0;
}
