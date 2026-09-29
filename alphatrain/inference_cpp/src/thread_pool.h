// Minimal persistent thread pool for per-game loops (eval's batched greedy step). Games are
// independent, so splitting them across threads changes nothing but the wall time.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace clines {

class ThreadPool {
 public:
  explicit ThreadPool(int n_threads) {
    for (int i = 1; i < n_threads; ++i) workers_.emplace_back([this] { Worker(); });
  }
  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> l(mu_);
      stop_ = true;
      ++generation_;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
  }
  int threads() const { return static_cast<int>(workers_.size()) + 1; }

  // fn(begin, end) over [0, n) in contiguous chunks; the caller's thread works too; returns when done.
  void ParallelFor(int n, const std::function<void(int, int)>& fn) {
    if (n <= 0) return;
    if (workers_.empty() || n < 64) { fn(0, n); return; }
    {
      std::lock_guard<std::mutex> l(mu_);
      fn_ = &fn;
      n_ = n;
      chunk_ = std::max(16, n / (threads() * 4));
      next_.store(0);
      pending_ = static_cast<int>(workers_.size());
      ++generation_;
    }
    cv_.notify_all();
    RunChunks();
    std::unique_lock<std::mutex> l(mu_);
    done_cv_.wait(l, [this] { return pending_ == 0; });
    fn_ = nullptr;
  }

 private:
  void RunChunks() {
    for (;;) {
      int b = next_.fetch_add(chunk_);
      if (b >= n_) return;
      (*fn_)(b, std::min(n_, b + chunk_));
    }
  }
  void Worker() {
    long seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [&] { return generation_ != seen; });
        seen = generation_;
        if (stop_) return;
      }
      RunChunks();
      {
        std::lock_guard<std::mutex> l(mu_);
        if (--pending_ == 0) done_cv_.notify_one();
      }
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int, int)>* fn_ = nullptr;
  int n_ = 0, chunk_ = 16, pending_ = 0;
  long generation_ = 0;
  bool stop_ = false;
  std::atomic<int> next_{0};
};

}  // namespace clines
