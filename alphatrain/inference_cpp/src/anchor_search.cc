// Teacher-strength probe. From saved start states (anchors.h, e.g. mcts_crisis rewind points),
// play the first --search-turns moves with MCTS, then greedy policy moves until the game dies or
// reaches the anchor's turn cap. Each game starts with Game(seed) + SetState, the spawn stream of
// the original replay, so escape rates compare directly with `eval --anchors` (greedy only) and
// with the replays themselves. --search-turns 0 is the greedy control through this same code path.
//
//   ./build/anchor_search --model data/A1_pair2_orig_e40_ts.pt --value-module data/pv_A1_ts.pt \
//       --device mps --anchors data/gen1_anchors.txt --every 4 --sims 600 --search-turns 30 \
//       --c-puct 1.5 --q-weight 2.0 --virtual-mean --threads 14 --out data/as_600_w30.csv
//
// Output CSV: seed,start_turn,survived,escaped (one row per anchor, streamed).

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "anchors.h"
#include "feature_value.h"
#include "game.h"
#include "infer_server.h"
#include "mcts.h"

using Clock = std::chrono::high_resolution_clock;

namespace {

struct Args {
  std::string model = "data/policy_ts.pt";
  std::string value_module;  // fused policy+value TS -> NN leaf value
  std::string device = "mps";
  std::string anchors, out;
  int every = 1;         // use every k-th anchor (subsample for expensive budgets)
  int sims = 600, search_turns = 30;
  int batch_size = 8, top_k = 30, threads = 14;
  double c_puct = 1.5, q_weight = 2.0;
  bool virtual_mean = false, fp32 = false, early_stop = true;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--virtual-mean") { a.virtual_mean = true; continue; }
    if (k == "--fp32") { a.fp32 = true; continue; }
    if (k == "--no-early-stop") { a.early_stop = false; continue; }
    if (i + 1 >= argc) { std::fprintf(stderr, "FATAL: missing value for %s\n", k.c_str()); std::exit(2); }
    if (k == "--model") a.model = argv[++i];
    else if (k == "--value-module") a.value_module = argv[++i];
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--anchors") a.anchors = argv[++i];
    else if (k == "--out") a.out = argv[++i];
    else if (k == "--every") a.every = std::stoi(argv[++i]);
    else if (k == "--sims") a.sims = std::stoi(argv[++i]);
    else if (k == "--search-turns") a.search_turns = std::stoi(argv[++i]);
    else if (k == "--batch-size") a.batch_size = std::stoi(argv[++i]);
    else if (k == "--top-k") a.top_k = std::stoi(argv[++i]);
    else if (k == "--threads") a.threads = std::stoi(argv[++i]);
    else if (k == "--c-puct") a.c_puct = std::stod(argv[++i]);
    else if (k == "--q-weight") a.q_weight = std::stod(argv[++i]);
    else { std::fprintf(stderr, "FATAL: unknown argument %s\n", k.c_str()); std::exit(2); }
  }
  if (a.anchors.empty() || a.out.empty() || a.every <= 0 || a.sims <= 0 || a.search_turns < 0 ||
      a.threads <= 0 || (a.device != "mps" && a.device != "cpu")) {
    std::fprintf(stderr, "FATAL: need --anchors, --out and valid --every/--sims/--search-turns/"
                         "--threads/--device\n");
    std::exit(2);
  }
  return a;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  torch::Device dev(args.device == "mps" ? torch::kMPS : torch::kCPU);
  if (dev.is_mps() && !torch::mps::is_available()) {
    std::fprintf(stderr, "FATAL: MPS unavailable; refusing to fall back to CPU\n");
    return 2;
  }
  const bool fp16 = dev.is_mps() && !args.fp32;
  clines::FeatureEval fe;
  if (!fe.Load("data/feature_value.bin")) {
    std::printf("cannot load data/feature_value.bin (run export_feature_weights.py)\n");
    return 1;
  }
  const bool nn_value = !args.value_module.empty();
  clines::InferenceServer server(nn_value ? args.value_module : args.model, dev, fp16, 10000,
                                 nn_value);

  std::vector<clines::Anchor> all = clines::ReadAnchors(args.anchors);
  std::vector<const clines::Anchor*> todo;
  for (size_t i = 0; i < all.size(); i += args.every) todo.push_back(&all[i]);
  std::printf("anchor_search: %zu of %zu anchors  search %d turns @%d sims  leaf=%s  c=%.2f q=%.2f "
              "virtual_mean=%d early_stop=%d  %s %s threads=%d\n",
              todo.size(), all.size(), args.search_turns, args.sims,
              nn_value ? "neural" : "feature", args.c_puct, args.q_weight, args.virtual_mean,
              args.early_stop, args.device.c_str(), fp16 ? "fp16" : "fp32", args.threads);
  std::fflush(stdout);

  FILE* out = std::fopen(args.out.c_str(), "w");
  if (!out) { std::printf("cannot open %s\n", args.out.c_str()); return 1; }
  std::fprintf(out, "seed,start_turn,survived,escaped\n");
  std::mutex mu;
  std::atomic<size_t> next{0};
  std::atomic<int> done{0}, escaped{0};
  auto t0 = Clock::now();

  auto worker = [&]() {
    clines::MctsConfig cfg;
    cfg.num_simulations = args.sims;
    cfg.c_puct = args.c_puct;
    cfg.top_k = args.top_k;
    cfg.batch_size = args.batch_size;
    cfg.q_weight = args.q_weight;
    cfg.virtual_mean = args.virtual_mean;
    cfg.nn_value = nn_value;
    cfg.early_stop = args.early_stop;  // same argmax as the full search, fewer simulations
    clines::MCTS mcts(
        [&server](const float* o, int n, float* lg, float* v) { server.Eval(o, n, lg, v); }, &fe,
        cfg);
    std::vector<float> obs(18 * clines::kNN), logits(clines::kActions);
    int act = -1;
    double pri = 0.0;
    while (true) {
      size_t ti = next.fetch_add(1);
      if (ti >= todo.size()) return;
      const clines::Anchor& a = *todo[ti];
      clines::Game g = clines::StartFromAnchor(a);
      clines::SimpleRng move_rng(a.seed * 2654435761ULL + 1);  // as in mcts_crisis
      int played = 0;
      bool died = false;
      while (played < a.cap) {
        int action = -1;
        if (played < args.search_turns) {
          action = mcts.Search(g, /*temperature=*/0.0, move_rng).action;
        } else {
          g.BuildObs(obs.data());
          server.Eval(obs.data(), 1, logits.data());
          if (clines::LegalPriors(g.board().data(), logits.data(), 1, &act, &pri) > 0) action = act;
        }
        if (action < 0) { died = true; break; }
        const int s = action / 81, t = action % 81;
        if (!g.Move(s / 9, s % 9, t / 9, t % 9)) {
          std::fprintf(stderr, "FATAL: illegal move (seed=%llu)\n", (unsigned long long)a.seed);
          std::abort();
        }
        ++played;
        if (g.over()) { died = true; break; }
      }
      const bool esc = !died && played >= a.cap;
      const int d = done.fetch_add(1) + 1;
      const int e = escaped.fetch_add(esc) + esc;
      std::lock_guard<std::mutex> l(mu);
      std::fprintf(out, "%llu,%d,%d,%d\n", (unsigned long long)a.seed, a.turn, played, esc ? 1 : 0);
      std::fflush(out);
      if (d % 100 == 0 || d == (int)todo.size()) {
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("  %d/%zu anchors  escaped %.1f%%  (%.0fs, ETA %.0fs)\n", d, todo.size(),
                    100.0 * e / d, el, el / d * (todo.size() - d));
        std::fflush(stdout);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < args.threads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
  std::fclose(out);
  std::printf("done: %d anchors, escaped %d (%.2f%%)\n", done.load(), escaped.load(),
              100.0 * escaped.load() / std::max(1, done.load()));
  return 0;
}
