// native_eval_policy in C++: greedy policy play, batched.
//
// Mirrors scripts/eval_policy.py: hold B games in flight, do ONE batched
// forward per step (build obs -> forward -> argmax over legal -> move), refill a
// slot when a game dies. Reports the score distribution over a seed range.
//
// Run from inference_cpp/ (so it finds data/). Examples:
//   ./build/eval --seed-start 50000 --seed-end 50300 --batch 256
//   ./build/eval --device mps --batch 512 --seed-start 50000 --seed-end 51000

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "game.h"

using clines::Game;
using Clock = std::chrono::high_resolution_clock;

namespace {
struct Args {
  std::string model = "data/policy_ts.pt";
  std::string device = "cpu";
  uint64_t seed_start = 50000, seed_end = 50300;  // [start, end)
  int batch = 256;
  long max_turns = 1000000;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc - 1; ++i) {
    std::string k = argv[i];
    if (k == "--model") a.model = argv[++i];
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--seed-start") a.seed_start = std::stoull(argv[++i]);
    else if (k == "--seed-end") a.seed_end = std::stoull(argv[++i]);
    else if (k == "--batch") a.batch = std::stoi(argv[++i]);
    else if (k == "--max-turns") a.max_turns = std::stol(argv[++i]);
  }
  return a;
}

void Percentile(std::vector<int>& s, const char* tag) {
  std::sort(s.begin(), s.end());
  auto pct = [&](double p) { return s[std::min((size_t)(p / 100.0 * s.size()), s.size() - 1)]; };
  double mean = 0; for (int v : s) mean += v; mean /= s.size();
  int lt500 = 0, lt1000 = 0, gt5000 = 0, gt10000 = 0;
  for (int v : s) { lt500 += v < 500; lt1000 += v < 1000; gt5000 += v > 5000; gt10000 += v > 10000; }
  std::printf("%s  n=%zu  min=%d max=%d mean=%.0f\n", tag, s.size(), s.front(), s.back(), mean);
  std::printf("  P1=%d P5=%d P10=%d P25=%d P50=%d P75=%d P90=%d P95=%d\n",
              pct(1), pct(5), pct(10), pct(25), pct(50), pct(75), pct(90), pct(95));
  std::printf("  <500: %d (%.1f%%)  <1000: %d (%.1f%%)  >5000: %d (%.0f%%)  >10000: %d (%.0f%%)\n",
              lt500, 100.0 * lt500 / s.size(), lt1000, 100.0 * lt1000 / s.size(),
              gt5000, 100.0 * gt5000 / s.size(), gt10000, 100.0 * gt10000 / s.size());
}
}  // namespace

int main(int argc, char** argv) {
  torch::InferenceMode guard;
  Args args = ParseArgs(argc, argv);
  torch::Device dev(args.device == "mps" ? torch::kMPS : torch::kCPU);
  if (dev.is_mps() && !torch::mps::is_available()) {
    std::printf("MPS requested but unavailable; using CPU\n");
    dev = torch::Device(torch::kCPU);
  }

  torch::jit::Module net;
  try { net = torch::jit::load(args.model); }
  catch (const c10::Error& e) {
    std::printf("could not load %s: %s\n", args.model.c_str(), e.what());
    return 1;
  }
  net.to(dev);

  // Seed queue.
  std::vector<uint64_t> todo;
  for (uint64_t s = args.seed_start; s < args.seed_end; ++s) todo.push_back(s);
  size_t next = 0;
  const int B = std::min<int>(args.batch, (int)todo.size());

  struct Slot { uint64_t seed; Game game; };
  std::vector<Slot> slots;
  auto make_slot = [&](Slot& dst) -> bool {
    if (next >= todo.size()) return false;
    dst.seed = todo[next++];
    dst.game = Game(dst.seed);
    dst.game.Reset();
    return true;
  };
  slots.reserve(B);
  for (int i = 0; i < B; ++i) {
    Slot s{0, Game(0)};
    if (make_slot(s)) slots.push_back(std::move(s));
  }

  std::vector<int> scores;
  scores.reserve(todo.size());
  std::vector<float> obs_buf, legal_buf;
  long fwd = 0;
  auto t0 = Clock::now();
  size_t done = 0, log_next = 5000;

  while (!slots.empty()) {
    int n = (int)slots.size();
    obs_buf.resize((size_t)n * 18 * clines::kNN);
    legal_buf.resize((size_t)n * clines::kActions);
    // Build each game's obs+legal. Single-threaded on purpose: profiling showed
    // this eval is forward-bound (the heavy-tail long games run solo at tiny
    // batch and dominate wall-clock), so parallelizing this loop gave ~0 gain.
    for (int i = 0; i < n; ++i) {
      slots[i].game.BuildObs(obs_buf.data() + (size_t)i * 18 * clines::kNN);
      slots[i].game.LegalMask(legal_buf.data() + (size_t)i * clines::kActions);
    }
    torch::Tensor obs = torch::from_blob(obs_buf.data(), {n, 18, 9, 9}).to(dev);
    torch::Tensor legal = torch::from_blob(legal_buf.data(), {n, clines::kActions}).to(dev);
    torch::Tensor logits = net.forward({obs}).toTensor();
    float ninf = -std::numeric_limits<float>::infinity();
    torch::Tensor moves = logits.masked_fill(legal < 0.5f, ninf).argmax(1).to(torch::kCPU);
    auto mv = moves.accessor<int64_t, 1>();
    ++fwd;

    std::vector<Slot> survivors;
    survivors.reserve(n);
    for (int i = 0; i < n; ++i) {
      // legal-move count for this game (no legal moves => dead)
      const float* lg = legal_buf.data() + (size_t)i * clines::kActions;
      bool any_legal = false;
      for (int a = 0; a < clines::kActions; ++a) if (lg[a] > 0.5f) { any_legal = true; break; }
      bool dead = !any_legal;
      if (!dead) {
        int64_t m = mv[i];
        int s = (int)(m / 81), t = (int)(m % 81);
        bool ok = slots[i].game.Move(s / 9, s % 9, t / 9, t % 9);
        dead = !ok || slots[i].game.over() || slots[i].game.turns() >= args.max_turns;
      }
      if (dead) {
        scores.push_back(slots[i].game.score());
        ++done;
        Slot repl{0, Game(0)};
        if (make_slot(repl)) survivors.push_back(std::move(repl));
      } else {
        survivors.push_back(std::move(slots[i]));
      }
    }
    slots.swap(survivors);

    if (done >= log_next) {
      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::printf("  %zu/%zu games  %ld fwd  %.0f games/s  %.0f fwd/s\n",
                  done, todo.size(), fwd, done / el, fwd / el);
      std::fflush(stdout);
      log_next += 5000;
    }
  }

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %zu games in %.1fs (%.0f games/s, %ld forwards, batch=%d, %s)\n",
              scores.size(), el, scores.size() / el, fwd, B, args.device.c_str());
  char tag[64];
  std::snprintf(tag, sizeof(tag), "scores [%llu,%llu):",
                (unsigned long long)args.seed_start, (unsigned long long)args.seed_end);
  Percentile(scores, tag);
  return 0;
}
