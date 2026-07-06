// Shared batched policy-inference server. Game threads submit small obs
// batches (MCTS batch_size=8 leaves) and block; one server thread coalesces
// everything pending into a single forward — bigger GPU batches, one round
// trip. Same architecture as Python's alphatrain/inference_server.py (which
// averaged bs~64 across 16 workers on this workload).

#ifndef CLINES_INFER_SERVER_H_
#define CLINES_INFER_SERVER_H_

#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace clines {

class InferenceServer {
 public:
  // log_every_forwards > 0: print "[GPU] ... evals/s" every N forwards (rare,
  // like Python's inference_server) so C++/Python throughput is comparable.
  InferenceServer(const std::string& model_path, torch::Device dev, bool fp16,
                  int64_t log_every_forwards = 10000)
      : dev_(dev), fp16_(fp16), log_every_(log_every_forwards),
        t0_(std::chrono::steady_clock::now()) {
    module_ = torch::jit::load(model_path);
    module_.to(dev_);
    if (fp16_) module_.to(torch::kHalf);
    worker_ = std::thread([this] { Loop(); });
  }
  ~InferenceServer() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    worker_.join();
  }

  // Blocking. obs = n contiguous (18,9,9) fp32; writes n*6561 logits to out.
  void Eval(const float* obs, int n, float* out) {
    Req r{obs, n, out, {}};
    std::future<void> done = r.prom.get_future();
    {
      std::lock_guard<std::mutex> l(mu_);
      queue_.push_back(&r);
    }
    cv_.notify_one();
    done.get();
  }

  int64_t forwards() const { return fwd_.load(); }
  int64_t evals() const { return evals_.load(); }

 private:
  struct Req {
    const float* obs;
    int n;
    float* out;
    std::promise<void> prom;
  };

  void Loop() {
    torch::InferenceMode guard;
    std::vector<Req*> batch;
    while (true) {
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return stop_ || !queue_.empty(); });
        if (stop_ && queue_.empty()) return;
        batch.assign(queue_.begin(), queue_.end());
        queue_.clear();
      }
      int total = 0;
      for (Req* r : batch) total += r->n;
      // Pack all requests into one contiguous CPU tensor.
      torch::Tensor obs = torch::empty({total, 18, 9, 9});
      float* dst = obs.data_ptr<float>();
      for (Req* r : batch) {
        std::memcpy(dst, r->obs, static_cast<size_t>(r->n) * 18 * 81 * sizeof(float));
        dst += static_cast<size_t>(r->n) * 18 * 81;
      }
      if (fp16_) obs = obs.to(torch::kHalf);  // convert BEFORE upload (Lever 1)
      torch::Tensor logits = module_.forward({obs.to(dev_)})
                                 .toTensor()
                                 .to(torch::kFloat)
                                 .cpu()
                                 .contiguous();
      const float* src = logits.data_ptr<float>();
      for (Req* r : batch) {
        std::memcpy(r->out, src, static_cast<size_t>(r->n) * 6561 * sizeof(float));
        src += static_cast<size_t>(r->n) * 6561;
        r->prom.set_value();
      }
      fwd_ += 1;
      evals_ += total;
      if (log_every_ > 0 && fwd_ % log_every_ == 0) {
        double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0_).count();
        std::printf("  [GPU] %lld evals, %lld fwd (avg bs=%.1f, %.0f evals/s)\n",
                    (long long)evals_.load(), (long long)fwd_.load(),
                    (double)evals_.load() / fwd_.load(), evals_.load() / el);
        std::fflush(stdout);
      }
    }
  }

  torch::jit::Module module_;
  torch::Device dev_;
  bool fp16_;
  int64_t log_every_;
  std::chrono::steady_clock::time_point t0_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Req*> queue_;
  std::atomic<int64_t> fwd_{0}, evals_{0};
  bool stop_ = false;
  std::thread worker_;
};

}  // namespace clines

#endif  // CLINES_INFER_SERVER_H_
