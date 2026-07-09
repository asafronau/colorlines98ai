// native MCTS eval: play a seed range with feature-value MCTS, report the
// score distribution. C++ port of the eval_parallel MCTS path (validation
// target: Python @100 sims, q=1.0, early-stop, 48 seeds 775000.. gave
// median 15,072 / mean 14,114). Compare DISTRIBUTIONS, never per-seed.
//
//   ./build/mcts_eval --model data/policy_ts.pt --device mps \
//       --seed-start 775000 --seed-end 775048 --sims 100 --q-weight 1.0 \
//       --early-stop --max-turns 12000 --threads 14

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "feature_value.h"
#include "game.h"
#include "infer_server.h"
#include "mcts.h"

using Clock = std::chrono::high_resolution_clock;

namespace {

struct Args {
  std::string model = "data/policy_ts.pt";
  std::string device = "mps";
  std::string value_module;  // fused policy+value TS -> NN leaf value
  uint64_t seed_start = 775000, seed_end = 775048;  // [start, end)
  int sims = 100;
  int batch_size = 8;
  int top_k = 30;
  double c_puct = 2.5;
  double q_weight = 1.0;
  long max_turns = 12000;
  int threads = 14;
  bool early_stop = false;
  bool fp32 = false;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--early-stop") { a.early_stop = true; continue; }
    if (k == "--fp32") { a.fp32 = true; continue; }
    if (i + 1 >= argc) break;
    if (k == "--model") a.model = argv[++i];
    else if (k == "--value-module") a.value_module = argv[++i];
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--seed-start") a.seed_start = std::stoull(argv[++i]);
    else if (k == "--seed-end") a.seed_end = std::stoull(argv[++i]);
    else if (k == "--sims") a.sims = std::stoi(argv[++i]);
    else if (k == "--batch-size") a.batch_size = std::stoi(argv[++i]);
    else if (k == "--top-k") a.top_k = std::stoi(argv[++i]);
    else if (k == "--c-puct") a.c_puct = std::stod(argv[++i]);
    else if (k == "--q-weight") a.q_weight = std::stod(argv[++i]);
    else if (k == "--max-turns") a.max_turns = std::stol(argv[++i]);
    else if (k == "--threads") a.threads = std::stoi(argv[++i]);
  }
  return a;
}

void Percentile(std::vector<int> s, const char* tag) {
  std::sort(s.begin(), s.end());
  auto pct = [&](double p) {
    return s[std::min(static_cast<size_t>(p / 100.0 * s.size()), s.size() - 1)];
  };
  double mean = 0;
  for (int v : s) mean += v;
  mean /= s.size();
  int lt1000 = 0, gt5000 = 0, gt10000 = 0;
  for (int v : s) { lt1000 += v < 1000; gt5000 += v > 5000; gt10000 += v > 10000; }
  std::printf("%s  n=%zu  min=%d max=%d mean=%.0f\n", tag, s.size(), s.front(),
              s.back(), mean);
  std::printf("  P1=%d P5=%d P10=%d P25=%d P50=%d P75=%d P90=%d P95=%d\n",
              pct(1), pct(5), pct(10), pct(25), pct(50), pct(75), pct(90), pct(95));
  std::printf("  <1000: %d (%.1f%%)  >5000: %d (%.0f%%)  >10000: %d (%.0f%%)\n",
              lt1000, 100.0 * lt1000 / s.size(), gt5000, 100.0 * gt5000 / s.size(),
              gt10000, 100.0 * gt10000 / s.size());
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  torch::Device dev(args.device == "mps" ? torch::kMPS : torch::kCPU);
  if (dev.is_mps() && !torch::mps::is_available()) {
    std::printf("MPS unavailable; using CPU\n");
    dev = torch::Device(torch::kCPU);
  }
  const bool fp16 = dev.is_mps() && !args.fp32;

  clines::FeatureEval fe;
  if (!fe.Load("data/feature_value.bin")) {
    std::printf("cannot load data/feature_value.bin (run export_feature_weights.py)\n");
    return 1;
  }
  const bool nn_value = !args.value_module.empty();
  clines::InferenceServer server(nn_value ? args.value_module : args.model,
                                 dev, fp16, 10000, nn_value);
  if (nn_value) std::printf("NN value head: %s\n", args.value_module.c_str());

  std::vector<uint64_t> seeds;
  for (uint64_t s = args.seed_start; s < args.seed_end; ++s) seeds.push_back(s);
  std::vector<int> scores(seeds.size(), 0);
  std::vector<int> turns_out(seeds.size(), 0);
  std::atomic<size_t> next_idx{0};
  std::atomic<int> done{0};
  std::mutex print_mu;
  auto t0 = Clock::now();

  clines::MctsConfig cfg;
  cfg.num_simulations = args.sims;
  cfg.c_puct = args.c_puct;
  cfg.top_k = args.top_k;
  cfg.batch_size = args.batch_size;
  cfg.q_weight = args.q_weight;
  cfg.early_stop = args.early_stop;
  cfg.nn_value = nn_value;

  std::printf("mcts_eval: %zu seeds [%llu,%llu)  sims=%d q=%.2f batch=%d "
              "top_k=%d early_stop=%d  %s %s  threads=%d\n",
              seeds.size(), (unsigned long long)args.seed_start,
              (unsigned long long)args.seed_end, args.sims, args.q_weight,
              args.batch_size, args.top_k, (int)args.early_stop,
              args.device.c_str(), fp16 ? "fp16" : "fp32", args.threads);
  std::fflush(stdout);

  auto worker = [&](int tid) {
    clines::MCTS mcts(
        [&server](const float* o, int n, float* out, float* out_v) { server.Eval(o, n, out, out_v); },
        &fe, cfg);
    while (true) {
      size_t i = next_idx.fetch_add(1);
      if (i >= seeds.size()) return;
      uint64_t seed = seeds[i];
      clines::Game g(seed);
      g.Reset();
      clines::SimpleRng move_rng(seed * 2654435761ULL + 0x9E3779B97F4A7C15ULL);

      while (!g.over() && g.turns() < args.max_turns) {
        clines::SearchResult r = mcts.Search(g, /*temperature=*/0.0, move_rng);
        if (r.action < 0) break;  // no legal moves
        int src = r.action / 81, tgt = r.action % 81;
        if (!g.Move(src / 9, src % 9, tgt / 9, tgt % 9)) {
          std::fprintf(stderr, "FATAL: MCTS chose illegal move %d (seed=%llu turn=%d)\n",
                       r.action, (unsigned long long)seed, g.turns());
          std::abort();  // crash on invalid state, never skip
        }
        if (g.turns() % 500 == 0) {
          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          std::lock_guard<std::mutex> l(print_mu);
          std::printf("    [t%d] seed=%llu turn=%d score=%d (%.0fs)\n", tid,
                      (unsigned long long)seed, g.turns(), g.score(), el);
          std::fflush(stdout);
        }
      }
      scores[i] = g.score();
      turns_out[i] = g.turns();
      int d = done.fetch_add(1) + 1;
      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::lock_guard<std::mutex> l(print_mu);
      std::printf("  [%d/%zu] seed=%llu score=%d turns=%d  (%.0fs elapsed, "
                  "%.1fs/game, ETA %.0fs)\n",
                  d, seeds.size(), (unsigned long long)seed, g.score(), g.turns(),
                  el, el / d, el / d * (seeds.size() - d));
      std::fflush(stdout);
    }
  };

  int T = std::min<int>(args.threads, (int)seeds.size());
  std::vector<std::thread> pool;
  for (int t = 0; t < T; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %zu games in %.0fs (%.1fs/game)  %lld forwards, %lld leaf evals "
              "(avg bs=%.1f, %.0f evals/s)\n",
              seeds.size(), el, el / seeds.size(), (long long)server.forwards(),
              (long long)server.evals(),
              server.forwards() ? (double)server.evals() / server.forwards() : 0.0,
              server.evals() / el);
  char tag[80];
  std::snprintf(tag, sizeof(tag), "MCTS scores [%llu,%llu):",
                (unsigned long long)args.seed_start,
                (unsigned long long)args.seed_end);
  Percentile(scores, tag);
  return 0;
}
