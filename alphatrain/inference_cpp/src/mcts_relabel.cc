// mcts_relabel: re-run the root search on STORED states (no game play).
//
// Reads the rollout-judge state file (data/judge_states.bin format, 'CLRJ'),
// runs one MCTS search per state under the given search controls, and writes
// one CSV row per state with the prior argmax, the visit argmax, and the
// top candidates.  This is how a teacher change is tested at the LABEL level:
// minutes per configuration instead of a day of self-play.
#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "feature_value.h"
#include "game.h"
#include "infer_server.h"
#include "mcts.h"

namespace {

using Clock = std::chrono::steady_clock;

struct State {
  int8_t board[81];
  std::vector<clines::NextBall> nb;
  int teacher_move, base_move;  // as recorded in the file (informational)
  float top_share;
};

struct Args {
  std::string model = "data/policy_ts.pt";
  std::string value_module;
  std::string device = "mps";
  std::string states = "data/judge_states.bin";
  std::string out = "data/relabel.csv";
  int sims = 400;
  int batch_size = 8;
  int top_k = 30;
  double c_puct = 2.5;
  double q_weight = 1.0;
  bool virtual_mean = false;
  double q_range_floor = 0.0;
  int threads = 14;
  int keep = 8;      // candidates written per row
  bool fp32 = false;
  uint64_t seed_salt = 0;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--fp32") { a.fp32 = true; continue; }
    if (k == "--virtual-mean") { a.virtual_mean = true; continue; }
    if (i + 1 >= argc) break;
    if (k == "--model") a.model = argv[++i];
    else if (k == "--value-module") a.value_module = argv[++i];
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--states") a.states = argv[++i];
    else if (k == "--out") a.out = argv[++i];
    else if (k == "--sims") a.sims = std::stoi(argv[++i]);
    else if (k == "--batch-size") a.batch_size = std::stoi(argv[++i]);
    else if (k == "--top-k") a.top_k = std::stoi(argv[++i]);
    else if (k == "--c-puct") a.c_puct = std::stod(argv[++i]);
    else if (k == "--q-weight") a.q_weight = std::stod(argv[++i]);
    else if (k == "--q-range-floor") a.q_range_floor = std::stod(argv[++i]);
    else if (k == "--threads") a.threads = std::stoi(argv[++i]);
    else if (k == "--keep") a.keep = std::stoi(argv[++i]);
    else if (k == "--seed-salt") a.seed_salt = std::stoull(argv[++i]);
    else { std::fprintf(stderr, "FATAL: unknown arg %s\n", k.c_str()); std::exit(2); }
  }
  if (a.sims <= 0 || a.batch_size <= 0 || a.top_k <= 0 || a.threads <= 0 ||
      a.keep <= 0 || !std::isfinite(a.q_range_floor) || a.q_range_floor < 0) {
    std::fprintf(stderr, "FATAL: invalid search settings\n");
    std::exit(2);
  }
  return a;
}

std::vector<State> LoadStates(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<State> out;
  if (!f) return out;
  char magic[4];
  f.read(magic, 4);
  if (std::string(magic, 4) != "CLRJ") return out;
  int32_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 4);
  for (int i = 0; i < n; ++i) {
    State s;
    f.read(reinterpret_cast<char*>(s.board), 81);
    int32_t nn = 0;
    f.read(reinterpret_cast<char*>(&nn), 4);
    for (int t = 0; t < 3; ++t) {
      int32_t r, c, col;
      f.read(reinterpret_cast<char*>(&r), 4);
      f.read(reinterpret_cast<char*>(&c), 4);
      f.read(reinterpret_cast<char*>(&col), 4);
      if (t < nn) s.nb.push_back({(int)r, (int)c, (int)col});
    }
    int32_t tm, bm; float ts;
    f.read(reinterpret_cast<char*>(&tm), 4);
    f.read(reinterpret_cast<char*>(&bm), 4);
    f.read(reinterpret_cast<char*>(&ts), 4);
    s.teacher_move = tm; s.base_move = bm; s.top_share = ts;
    out.push_back(std::move(s));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args args = ParseArgs(argc, argv);
  torch::Device dev(args.device == "mps" ? torch::kMPS : torch::kCPU);
  if (dev.is_mps() && !torch::mps::is_available()) {
    std::fprintf(stderr, "FATAL: --device mps requested but MPS unavailable\n");
    return 2;
  }
  const bool fp16 = dev.is_mps() && !args.fp32;

  clines::FeatureEval fe;
  if (!fe.Load("data/feature_value.bin")) {
    std::printf("cannot load data/feature_value.bin (run export_feature_weights.py)\n");
    return 1;
  }
  std::vector<State> states = LoadStates(args.states);
  if (states.empty()) { std::printf("cannot load %s\n", args.states.c_str()); return 1; }

  const bool nn_value = !args.value_module.empty();
  clines::InferenceServer server(nn_value ? args.value_module : args.model,
                                 dev, fp16, 10000, nn_value);

  clines::MctsConfig cfg;
  cfg.num_simulations = args.sims;
  cfg.c_puct = args.c_puct;
  cfg.top_k = args.top_k;
  cfg.batch_size = args.batch_size;
  cfg.q_weight = args.q_weight;
  cfg.virtual_mean = args.virtual_mean;
  cfg.q_range_floor = args.q_range_floor;
  cfg.nn_value = nn_value;
  cfg.seed_salt = args.seed_salt;

  std::printf("mcts_relabel: %zu states  sims=%d batch=%d top_k=%d c_puct=%g "
              "q_weight=%g virtual_mean=%d q_floor=%g  %s %s threads=%d\n",
              states.size(), args.sims, args.batch_size, args.top_k, args.c_puct,
              args.q_weight, (int)args.virtual_mean, args.q_range_floor,
              args.device.c_str(), fp16 ? "fp16" : "fp32", args.threads);
  std::fflush(stdout);

  // One result line per state, assembled by the worker, written in order.
  std::vector<std::string> rows(states.size());
  std::atomic<size_t> next_idx{0};
  std::atomic<int> done{0};
  std::mutex print_mu;
  auto t0 = Clock::now();

  auto worker = [&](int) {
    clines::MCTS mcts(
        [&server](const float* o, int n, float* out, float* out_v) {
          server.Eval(o, n, out, out_v);
        },
        &fe, cfg);
    std::vector<clines::Candidate> cands;
    while (true) {
      size_t i = next_idx.fetch_add(1);
      if (i >= states.size()) return;
      const State& st = states[i];
      // The game RNG only matters for spawns inside the search; seed it from
      // the state index so a configuration is reproducible run to run.
      clines::Game g(0x9E3779B97F4A7C15ULL ^ (i * 2654435761ULL));
      g.SetState(st.board, st.nb, 0, 0);
      clines::SimpleRng move_rng(i + 1);
      clines::SearchResult r = mcts.Search(g, 0.0, move_rng);

      // Prior argmax and visit total from the root candidates.
      int prior_arg = -1; double best_p = -1; int total = 0, top_v = 0;
      for (const auto& c : r.cands) {
        total += c.visits;
        top_v = std::max(top_v, c.visits);
        if (c.prior > best_p) { best_p = c.prior; prior_arg = c.action; }
      }
      cands = r.cands;  // already visit-descending
      char buf[256];
      std::string row;
      std::snprintf(buf, sizeof(buf), "%zu,%d,%d,%d,%d,%.4f,%.4f,%.5f,%.5f,%.5f,%d",
                    i, st.teacher_move, st.base_move, prior_arg, r.action,
                    total > 0 ? (double)top_v / total : 0.0, st.top_share,
                    r.root_value, r.q_min, r.q_max, (int)cands.size());
      row = buf;
      for (int k = 0; k < args.keep; ++k) {
        if (k < (int)cands.size())
          std::snprintf(buf, sizeof(buf), ",%d:%d:%.5f:%.4f", cands[k].action,
                        cands[k].visits, cands[k].prior, cands[k].q);
        else
          std::snprintf(buf, sizeof(buf), ",-1:0:0:0");
        row += buf;
      }
      rows[i] = row;
      int d = ++done;
      if (d % 250 == 0 || d == (int)states.size()) {
        std::lock_guard<std::mutex> l(print_mu);
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("  [%d/%zu]  %.0fs elapsed, %.2fs/state, ETA %.0fs\n", d,
                    states.size(), el, el / d, el / d * (states.size() - d));
        std::fflush(stdout);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int t = 0; t < args.threads; ++t) pool.emplace_back(worker, t);
  for (auto& t : pool) t.join();

  std::ofstream out(args.out);
  if (!out) { std::fprintf(stderr, "FATAL: cannot write %s\n", args.out.c_str()); return 2; }
  out << "idx,file_teacher,file_base,prior_argmax,visit_argmax,top_share,file_top_share,"
         "root_value,q_min,q_max,n_cands";
  for (int k = 0; k < args.keep; ++k) out << ",cand" << k;
  out << '\n';
  for (const auto& r : rows) out << r << '\n';
  out.close();
  if (!out) { std::fprintf(stderr, "FATAL: failed writing %s\n", args.out.c_str()); return 2; }

  int flips_vs_prior = 0, same_as_file = 0;
  for (size_t i = 0; i < states.size(); ++i) {
    // cheap re-parse of the two ints we care about
    int fteach, fbase, parg, varg;
    std::sscanf(rows[i].c_str(), "%*zu,%d,%d,%d,%d", &fteach, &fbase, &parg, &varg);
    flips_vs_prior += (varg != parg);
    same_as_file += (varg == fteach);
  }
  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("done: %zu states in %.0fs (%.2fs/state)  visit_argmax != prior_argmax: %d "
              "(%.2f%%)  visit_argmax == file_teacher: %d (%.2f%%)\nSaved %s\n",
              states.size(), el, el / states.size(), flips_vs_prior,
              100.0 * flips_vs_prior / states.size(), same_as_file,
              100.0 * same_as_file / states.size(), args.out.c_str());
  return 0;
}
