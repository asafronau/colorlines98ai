// Color Lines 98 — policy inference using LibTorch (PyTorch's official C++ API).
//
// The "engine" is just: load the TorchScript module and call forward().
// LibTorch implements conv/batchnorm/relu, so there is no math to write here.
//
// This program also (a) checks C++ output == PyTorch on CPU and MPS, and
// (b) benchmarks CPU vs MPS across batch sizes — because the real consumers
// (native_selfplay / native_crisis_mining / native_eval_policy) score MANY
// boards at once, and the GPU only pays off once the batch is big enough.

#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

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

// Average wall-time (ms) of one forward() call, for a given input on a given
// device. Key GPU detail: MPS runs ASYNCHRONOUSLY — forward() just queues work
// and returns. So we synchronize() before reading the clock, otherwise we'd be
// timing how fast we *queued* the work, not how fast the GPU *ran* it.
static double BenchMs(torch::jit::Module& net, torch::Tensor input,
                      torch::Device dev, int iters) {
  net.to(dev);
  input = input.to(dev);
  for (int i = 0; i < 3; ++i) net.forward({input});      // warmup (1-time setup)
  if (dev.is_mps()) torch::mps::synchronize();
  auto t0 = Clock::now();
  for (int i = 0; i < iters; ++i) net.forward({input});
  if (dev.is_mps()) torch::mps::synchronize();           // wait for the GPU
  auto t1 = Clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
  const bool have_mps = torch::mps::is_available();
  torch::Device cpu(torch::kCPU), mps(torch::kMPS);

  // 1) Load the network. Weights, BatchNorm stats, architecture — all baked in.
  torch::jit::Module net;
  try {
    net = torch::jit::load("data/policy_ts.pt");
  } catch (const c10::Error& e) {
    std::cerr << "could not load data/policy_ts.pt (run export_ts.py first): "
              << e.what() << "\n";
    return 1;
  }

  // 2) One real obs (shape {1,18,9,9}) + PyTorch's logits for it = the oracle.
  constexpr int kObs = 18 * 9 * 9, kLogits = 6561;
  std::vector<float> obs_data = ReadFloats("data/example_obs.f32", kObs);
  torch::Tensor obs = torch::from_blob(obs_data.data(), {1, 18, 9, 9}).clone();
  std::vector<float> want = ReadFloats("data/example_logits.f32", kLogits);
  torch::Tensor want_t = torch::from_blob(want.data(), {kLogits}).clone();

  // 3) Correctness on CPU. To compare, move results back to CPU first.
  net.to(cpu);
  torch::Tensor cpu_logits =
      net.forward({obs.to(cpu)}).toTensor().view({kLogits});
  std::cout << "CPU  move=" << cpu_logits.argmax().item<int>()
            << "  max|diff| vs PyTorch = "
            << (cpu_logits - want_t).abs().max().item<float>() << "\n";

  // 4) Correctness on MPS (the GPU). Expect a *tiny* diff (~1e-5): GPU kernels
  //    aren't bit-identical to CPU. Same argmax = same move, which is what counts.
  if (have_mps) {
    net.to(mps);
    torch::Tensor mps_logits =
        net.forward({obs.to(mps)}).toTensor().view({kLogits}).cpu();
    std::cout << "MPS  move=" << mps_logits.argmax().item<int>()
              << "  max|diff| vs PyTorch = "
              << (mps_logits - want_t).abs().max().item<float>()
              << "  (tiny = normal)\n";
  } else {
    std::cout << "MPS not available on this machine.\n";
  }

  // 5) Speed: CPU vs MPS as the batch grows. We report ms PER BOARD so the
  //    columns are directly comparable. Watch for the crossover.
  std::cout << "\n batch | CPU ms/board | MPS ms/board\n"
               "-------+--------------+-------------\n";
  for (int B : {1, 16, 64, 256, 1024}) {
    torch::Tensor batch = obs.repeat({B, 1, 1, 1});  // B copies (fine for timing)
    int iters = std::max(5, 800 / B);                // keep each measurement short
    double cms = BenchMs(net, batch, cpu, iters) / B;
    double mms = have_mps ? BenchMs(net, batch, mps, iters) / B : 0.0;
    std::cout << "  " << B << "\t|   " << cms << "\t|   " << mms << "\n";
  }
  return 0;
}
