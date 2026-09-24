// native selfplay: MCTS games with visit-distribution recording, in the
// moves-schema JSON that alphatrain/scripts/build_expert_v2_tensor.py consumes
// (port of alphatrain/scripts/selfplay.py, feature-value leaf mode).
//
// Per move: board + next_balls BEFORE the move, chosen_move, and the root
// record — cand_moves (flat), cand_visits, cand_prior (CLEAN pre-Dirichlet
// prior as log-prob), cand_q, root_value, q_min, q_max (complete top-30
// searched support by visits).
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
#include <dirent.h>
#include <filesystem>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unordered_set>
#include <vector>

#include "feature_value.h"
#include "game.h"
#include "game_json.h"
#include "infer_server.h"
#include "mcts.h"

using Clock = std::chrono::high_resolution_clock;

namespace {

struct Args {
  std::string run_id;  // enables immutable run_config.json resume guard
  std::string model = "data/policy_ts.pt";
  std::string device = "mps";
  std::string out_dir = "selfplay_out";
  std::string value_module;  // fused policy+value TS -> NN leaf value
  uint64_t seed_start = 900000, seed_end = 900010;  // [start, end)
  int sims = 1600;
  // Optional independent, noise-free search whose candidates are recorded as
  // the learning label.  The first search still chooses the behavior action.
  int clean_label_sims = 0;
  int batch_size = 8;
  int top_k = 30;
  double c_puct = 2.5;
  double q_weight = 1.0;          // validated operating point for the
                                  // feature-value leaf (eval_parallel +61%)
  int temperature_moves = 15;     // temp=1.0 for the first N moves, then 0
  double dirichlet_alpha = 0.3;
  double dirichlet_weight = 0.25;
  // Many independent capped games provide better board diversity than a few
  // enormous continuations once strong play reaches its sustainable regime.
  long max_turns = 1000;
  int threads = 14;
  bool fp32 = false;
  // Search controls (see mcts.h): default reproduces historical search.
  bool virtual_mean = false;
  double q_range_floor = 0.0;
  bool full_record = false;  // also write cand_prior/cand_q/root_value/q_min/
                             // q_max (Gumbel-only; train_path_b ignores them)
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--fp32") { a.fp32 = true; continue; }
    if (k == "--full-record") { a.full_record = true; continue; }
    if (k == "--virtual-mean") { a.virtual_mean = true; continue; }
    if (i + 1 >= argc) {
      std::fprintf(stderr, "FATAL: missing value for %s\n", k.c_str());
      std::exit(2);
    }
    if (k == "--run-id") a.run_id = argv[++i];
    else if (k == "--model") a.model = argv[++i];
    else if (k == "--value-module") a.value_module = argv[++i];
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--out-dir") a.out_dir = argv[++i];
    else if (k == "--seed-start") a.seed_start = std::stoull(argv[++i]);
    else if (k == "--seed-end") a.seed_end = std::stoull(argv[++i]);
    else if (k == "--sims") a.sims = std::stoi(argv[++i]);
    else if (k == "--clean-label-sims") a.clean_label_sims = std::stoi(argv[++i]);
    else if (k == "--batch-size") a.batch_size = std::stoi(argv[++i]);
    else if (k == "--top-k") a.top_k = std::stoi(argv[++i]);
    else if (k == "--c-puct") a.c_puct = std::stod(argv[++i]);
    else if (k == "--q-weight") a.q_weight = std::stod(argv[++i]);
    else if (k == "--q-range-floor") a.q_range_floor = std::stod(argv[++i]);
    else if (k == "--temperature-moves") a.temperature_moves = std::stoi(argv[++i]);
    else if (k == "--dirichlet-alpha") a.dirichlet_alpha = std::stod(argv[++i]);
    else if (k == "--dirichlet-weight") a.dirichlet_weight = std::stod(argv[++i]);
    else if (k == "--max-turns") a.max_turns = std::stol(argv[++i]);
    else if (k == "--threads") a.threads = std::stoi(argv[++i]);
    else {
      std::fprintf(stderr, "FATAL: unknown argument %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (a.seed_end <= a.seed_start || a.sims <= 0 || a.batch_size <= 0 ||
      a.top_k <= 0 || a.max_turns <= 0 || a.threads <= 0) {
    std::fprintf(stderr, "FATAL: invalid seed/search/batch/turn arguments\n");
    std::exit(2);
  }
  if (a.device != "mps" && a.device != "cpu") {
    std::fprintf(stderr, "FATAL: unsupported device %s (expected mps or cpu)\n",
                 a.device.c_str());
    std::exit(2);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  torch::Device dev(args.device == "mps" ? torch::kMPS : torch::kCPU);
  if (dev.is_mps() && !torch::mps::is_available()) {
    std::fprintf(stderr,
                 "FATAL: --device mps was requested, but MPS is unavailable; "
                 "refusing to fall back to CPU\n");
    return 2;
  }
  const bool fp16 = dev.is_mps() && !args.fp32;

  clines::FeatureEval fe;
  if (!fe.Load("data/feature_value.bin")) {
    std::printf("cannot load data/feature_value.bin (run export_feature_weights.py)\n");
    return 1;
  }
  std::error_code mkdir_error;
  std::filesystem::create_directories(args.out_dir, mkdir_error);
  if (mkdir_error) {
    std::fprintf(stderr, "FATAL: cannot create %s: %s\n",
                 args.out_dir.c_str(), mkdir_error.message().c_str());
    return 1;
  }
  if (!args.run_id.empty()) {
    std::string config =
        "{\"schema_version\": 1, \"generator\": \"mcts_selfplay\", "
        "\"run_id\": \"" + args.run_id + "\", \"model\": \"" +
        args.model + "\", \"value_module\": \"" + args.value_module +
        "\", \"device\": \"" + args.device + "\", \"precision\": \"" +
        (fp16 ? std::string("fp16") : std::string("fp32")) +
        "\", \"seed_start\": " + std::to_string(args.seed_start) +
        ", \"seed_end\": " + std::to_string(args.seed_end) +
        ", \"behavior_sims\": " + std::to_string(args.sims) +
        ", \"clean_label_sims\": " +
        std::to_string(args.clean_label_sims) +
        ", \"batch_size\": " + std::to_string(args.batch_size) +
        ", \"top_k\": " + std::to_string(args.top_k) +
        ", \"c_puct\": ";
    clines::AppendD(config, args.c_puct);
    config += ", \"q_weight\": ";
    clines::AppendD(config, args.q_weight);
    config += ", \"virtual_mean\": " + std::string(args.virtual_mean ? "true" : "false") +
              ", \"q_range_floor\": ";
    clines::AppendD(config, args.q_range_floor);
    config += ", \"temperature_moves\": " +
              std::to_string(args.temperature_moves) +
              ", \"dirichlet_alpha\": ";
    clines::AppendD(config, args.dirichlet_alpha);
    config += ", \"dirichlet_weight\": ";
    clines::AppendD(config, args.dirichlet_weight);
    config += ", \"max_turns\": " + std::to_string(args.max_turns) +
              ", \"threads\": " + std::to_string(args.threads) +
              ", \"full_record\": " +
              (args.full_record ? std::string("true") : std::string("false")) +
              "}\n";
    clines::EnsureRunConfigOrDie(args.out_dir, config);
  }
  const bool nn_value = !args.value_module.empty();
  clines::InferenceServer server(nn_value ? args.value_module : args.model,
                                 dev, fp16, 10000, nn_value);
  if (nn_value) std::printf("NN value head: %s\n", args.value_module.c_str());

  // Resume: skip seeds whose game file already exists in out_dir.
  std::unordered_set<uint64_t> done_seeds;
  if (DIR* dp = opendir(args.out_dir.c_str())) {
    while (dirent* e = readdir(dp)) {
      std::string name = e->d_name;
      const std::string pfx = "game_seed";
      if (name.rfind(pfx, 0) != 0) continue;
      if (name.size() < 6 || name.substr(name.size() - 5) != ".json") continue;
      size_t p = pfx.size(), q = p;
      while (q < name.size() && isdigit(name[q])) ++q;
      if (q > p) done_seeds.insert(std::stoull(name.substr(p, q - p)));
    }
    closedir(dp);
  }

  std::vector<uint64_t> seeds;
  for (uint64_t s = args.seed_start; s < args.seed_end; ++s)
    if (!done_seeds.count(s)) seeds.push_back(s);
  if (!done_seeds.empty())
    std::printf("resume: %zu seeds already on disk — %zu remaining\n",
                done_seeds.size(), seeds.size());
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
  cfg.virtual_mean = args.virtual_mean;
  cfg.q_range_floor = args.q_range_floor;
  cfg.early_stop = false;  // selfplay needs the full visit distribution
  cfg.nn_value = nn_value;
  cfg.dirichlet_alpha = args.dirichlet_alpha;
  cfg.dirichlet_weight = args.dirichlet_weight;

  clines::MctsConfig clean_cfg = cfg;
  clean_cfg.num_simulations = std::max(1, args.clean_label_sims);
  clean_cfg.dirichlet_alpha = 0.0;
  clean_cfg.dirichlet_weight = 0.0;

  std::printf("mcts_selfplay: %zu seeds [%llu,%llu)  behavior_sims=%d "
              "clean_label_sims=%d q=%.2f batch=%d temp_moves=%d "
              "dir=%.2f/%.2f max_turns=%ld  %s %s  threads=%d\n"
              "out: %s\n",
              seeds.size(), (unsigned long long)args.seed_start,
              (unsigned long long)args.seed_end, args.sims,
              args.clean_label_sims, args.q_weight, args.batch_size,
              args.temperature_moves, args.dirichlet_alpha,
              args.dirichlet_weight, args.max_turns, args.device.c_str(),
              fp16 ? "fp16" : "fp32", args.threads, args.out_dir.c_str());
  std::fflush(stdout);

  auto worker = [&](int tid) {
    clines::MCTS mcts(
        [&server](const float* o, int n, float* out, float* out_v) { server.Eval(o, n, out, out_v); },
        &fe, cfg);
    clines::MCTS clean_mcts(
        [&server](const float* o, int n, float* out, float* out_v) { server.Eval(o, n, out, out_v); },
        &fe, clean_cfg);
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

        if (args.clean_label_sims > 0) {
          clines::SearchResult label = clean_mcts.Search(
              g, /*temperature=*/0.0, move_rng);
          if (label.action < 0) {
            std::fprintf(stderr,
                         "FATAL: clean label has no move (seed=%llu turn=%d)\n",
                         (unsigned long long)seed, g.turns());
            std::abort();
          }
          clines::MoveRec rec = clines::MakeMoveRec(g, label);
          rec.action = r.action;  // behavior remains what advances the game
          recs.push_back(std::move(rec));
        } else {
          recs.push_back(clines::MakeMoveRec(g, r));
        }

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
      json += "{\"generator_schema_version\": 4" +
              (args.run_id.empty()
                   ? std::string("")
                   : ", \"run_id\": \"" + args.run_id + "\"") +
              ", \"policy_model\": \"" + args.model + "\"" +
              ", \"value_module\": \"" + args.value_module + "\"" +
              ", \"precision\": \"" +
                  (fp16 ? std::string("fp16") : std::string("fp32")) + "\"" +
              ", \"full_record\": " +
                  (args.full_record ? std::string("true") : std::string("false")) +
              ", \"max_turns\": " + std::to_string(args.max_turns) +
              ", \"seed\": " + std::to_string(seed) +
              ", \"score\": " + std::to_string(g.score()) +
              ", \"turns\": " + std::to_string(g.turns()) +
              ", \"capped\": " + (capped ? std::string("true") : std::string("false")) +
              ", \"behavior_sims\": " + std::to_string(args.sims) +
              ", \"clean_label_sims\": " + std::to_string(args.clean_label_sims) +
              ", \"q_weight\": " + std::to_string(args.q_weight) +
              ", \"c_puct\": " + std::to_string(args.c_puct) +
              ", \"virtual_mean\": " + (args.virtual_mean ? std::string("true") : std::string("false")) +
              ", \"q_range_floor\": " + std::to_string(args.q_range_floor) +
              ", \"top_k\": " + std::to_string(args.top_k) +
              ", \"mcts_batch_size\": " +
                  std::to_string(args.batch_size) +
              ", \"temperature_moves\": " +
                  std::to_string(args.temperature_moves) +
              ", \"value_kind\": \"" +
                  (nn_value ? std::string("neural") : std::string("feature")) +
                  "\"" +
              ", \"behavior_dirichlet_alpha\": " +
                  std::to_string(args.dirichlet_alpha) +
              ", \"behavior_dirichlet_weight\": " +
                  std::to_string(args.dirichlet_weight) +
              ", \"label_dirichlet_weight\": 0.0" +
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
  if (!args.run_id.empty()) {
    std::string complete =
        "{\"schema_version\": 1, \"run_id\": \"" + args.run_id +
        "\", \"seed_start\": " + std::to_string(args.seed_start) +
        ", \"seed_end\": " + std::to_string(args.seed_end) +
        ", \"expected_games\": " +
        std::to_string(args.seed_end - args.seed_start) + "}\n";
    clines::WriteFileOrDie(args.out_dir + "/generation_complete.json",
                           complete);
  }
  return 0;
}
