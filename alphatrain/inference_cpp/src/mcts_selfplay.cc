// native selfplay: MCTS games with visit-distribution recording, in the
// moves-schema JSON that alphatrain/scripts/build_expert_v2_tensor.py consumes
// (port of alphatrain/scripts/selfplay.py, feature-value leaf mode).
//
// Per move: board + next_balls BEFORE the move, chosen_move, and the root
// record — cand_moves (flat), cand_visits, cand_prior (CLEAN pre-Dirichlet
// prior as log-prob), cand_q, root_value, q_min, q_max (top 15 by visits).
// Per game: game_seed{seed}_score{score}.json in --out-dir.
//
//   ./build/mcts_selfplay --model data/policy_ts.pt --device mps \
//       --seed-start 900000 --seed-end 900040 --sims 1600 --threads 14 \
//       --out-dir ../../data/selfplay_cpp_v1

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "feature_value.h"
#include "game.h"
#include "game_json.h"
#include "infer_server.h"
#include "mcts.h"

using Clock = std::chrono::high_resolution_clock;

namespace {

struct Args {
  std::string model = "data/policy_ts.pt";
  std::string device = "mps";
  std::string out_dir = "selfplay_out";
  uint64_t seed_start = 900000, seed_end = 900010;  // [start, end)
  int sims = 1600;
  int batch_size = 8;
  int top_k = 30;
  double c_puct = 2.5;
  double q_weight = 1.0;          // validated operating point for the
                                  // feature-value leaf (eval_parallel +61%)
  int temperature_moves = 15;     // temp=1.0 for the first N moves, then 0
  double dirichlet_alpha = 0.3;
  double dirichlet_weight = 0.25;
  long max_turns = 0;             // 0 = play to natural death
  int threads = 14;
  bool fp32 = false;
  bool full_record = false;  // also write cand_prior/cand_q/root_value/q_min/
                             // q_max (Gumbel-only; train_path_b ignores them)
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--fp32") { a.fp32 = true; continue; }
    if (k == "--full-record") { a.full_record = true; continue; }
    if (i + 1 >= argc) break;
    if (k == "--model") a.model = argv[++i];
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--out-dir") a.out_dir = argv[++i];
    else if (k == "--seed-start") a.seed_start = std::stoull(argv[++i]);
    else if (k == "--seed-end") a.seed_end = std::stoull(argv[++i]);
    else if (k == "--sims") a.sims = std::stoi(argv[++i]);
    else if (k == "--batch-size") a.batch_size = std::stoi(argv[++i]);
    else if (k == "--top-k") a.top_k = std::stoi(argv[++i]);
    else if (k == "--c-puct") a.c_puct = std::stod(argv[++i]);
    else if (k == "--q-weight") a.q_weight = std::stod(argv[++i]);
    else if (k == "--temperature-moves") a.temperature_moves = std::stoi(argv[++i]);
    else if (k == "--dirichlet-alpha") a.dirichlet_alpha = std::stod(argv[++i]);
    else if (k == "--dirichlet-weight") a.dirichlet_weight = std::stod(argv[++i]);
    else if (k == "--max-turns") a.max_turns = std::stol(argv[++i]);
    else if (k == "--threads") a.threads = std::stoi(argv[++i]);
  }
  return a;
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
  ::mkdir(args.out_dir.c_str(), 0755);
  clines::InferenceServer server(args.model, dev, fp16);

  std::vector<uint64_t> seeds;
  for (uint64_t s = args.seed_start; s < args.seed_end; ++s) seeds.push_back(s);
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
  cfg.early_stop = false;  // selfplay needs the full visit distribution
  cfg.dirichlet_alpha = args.dirichlet_alpha;
  cfg.dirichlet_weight = args.dirichlet_weight;

  std::printf("mcts_selfplay: %zu seeds [%llu,%llu)  sims=%d q=%.2f batch=%d "
              "temp_moves=%d dir=%.2f/%.2f max_turns=%ld  %s %s  threads=%d\n"
              "out: %s\n",
              seeds.size(), (unsigned long long)args.seed_start,
              (unsigned long long)args.seed_end, args.sims, args.q_weight,
              args.batch_size, args.temperature_moves, args.dirichlet_alpha,
              args.dirichlet_weight, args.max_turns, args.device.c_str(),
              fp16 ? "fp16" : "fp32", args.threads, args.out_dir.c_str());
  std::fflush(stdout);

  auto worker = [&](int tid) {
    clines::MCTS mcts(
        [&server](const float* o, int n, float* out) { server.Eval(o, n, out); },
        &fe, cfg);
    while (true) {
      size_t i = next_idx.fetch_add(1);
      if (i >= seeds.size()) return;
      uint64_t seed = seeds[i];
      clines::Game g(seed);
      g.Reset();
      clines::SimpleRng move_rng(seed * 2654435761ULL + 0x9E3779B97F4A7C15ULL);
      std::vector<clines::MoveRec> recs;
      bool capped = false;

      while (!g.over()) {
        if (args.max_turns > 0 && g.turns() >= args.max_turns) {
          capped = true;
          break;
        }
        double temp = g.turns() < args.temperature_moves ? 1.0 : 0.0;
        clines::SearchResult r = mcts.Search(g, temp, move_rng);
        if (r.action < 0) break;  // no legal moves

        recs.push_back(clines::MakeMoveRec(g, r));

        int src = r.action / 81, tgt = r.action % 81;
        if (!g.Move(src / 9, src % 9, tgt / 9, tgt % 9)) {
          std::fprintf(stderr, "FATAL: illegal MCTS move %d (seed=%llu turn=%d)\n",
                       r.action, (unsigned long long)seed, g.turns());
          std::abort();
        }
        if (g.turns() % 500 == 0) {
          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          std::lock_guard<std::mutex> l(print_mu);
          std::printf("    [t%d] seed=%llu turn=%d score=%d (%.0fs)\n", tid,
                      (unsigned long long)seed, g.turns(), g.score(), el);
          std::fflush(stdout);
        }
      }

      std::string json;
      json.reserve(recs.size() * 900 + 256);
      json += "{\"seed\": " + std::to_string(seed) +
              ", \"score\": " + std::to_string(g.score()) +
              ", \"capped\": " + (capped ? std::string("true") : std::string("false")) +
              ", \"moves\": ";
      clines::AppendMovesArray(json, recs, args.full_record);
      json += "}";
      clines::WriteFileOrDie(args.out_dir + "/game_seed" + std::to_string(seed) +
                                 "_score" + std::to_string(g.score()) + ".json",
                             json);

      int d = done.fetch_add(1) + 1;
      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::lock_guard<std::mutex> l(print_mu);
      std::printf("  [%d/%zu] seed=%llu score=%d turns=%d moves=%zu%s  "
                  "(%.0fs, %.1fs/game, ETA %.0fs)\n",
                  d, seeds.size(), (unsigned long long)seed, g.score(), g.turns(),
                  recs.size(), capped ? " CAPPED" : "", el, el / d,
                  el / d * (seeds.size() - d));
      std::fflush(stdout);
    }
  };

  int T = std::min<int>(args.threads, (int)seeds.size());
  std::vector<std::thread> pool;
  for (int t = 0; t < T; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %zu games in %.0fs  %lld forwards, %lld leaf evals "
              "(%.0f evals/s)\n",
              seeds.size(), el, (long long)server.forwards(),
              (long long)server.evals(), server.evals() / el);
  return 0;
}
