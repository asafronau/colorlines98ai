// native crisis mining: port of alphatrain/scripts/crisis_mining.py.
//
// TWO PHASES (each at its optimal batching):
//   Phase 1 — BULK policy-only probes: all probe seeds in flight at once,
//     ONE large batched forward per step (the eval.cc slot pattern; no
//     per-game round-trips). Each game keeps a ring of the last
//     prevention_turns+1 snapshots; on death, the two anchor checkpoints
//     (recovery: death-15, prevention: death-75) go to the replay work list.
//   Phase 2 — deep-MCTS replays from the saved checkpoints: worker threads
//     run one replay each (temp=0 + Dirichlet, continue_turns cap), leaf
//     batches coalesced by the shared inference server.
// Output: slim moves-schema JSON per replay (escape-or-die games — the
// decisive share of the pillar3k corpus, HISTORY 173-174).
//
//   ./build/mcts_crisis --model data/policy_ts.pt --device mps \
//       --seed-start 940100 --seed-end 940400 --threads 14 \
//       --out-dir ../../data/crisis_cpp128_v1

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
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
  std::string out_dir = "crisis_out";
  uint64_t seed_start = 940100, seed_end = 940400;  // [start, end)
  int recovery_turns = 15, recovery_sims = 1600;
  int prevention_turns = 75, prevention_sims = 2400;
  int continue_turns = 500;
  long policy_max_turns = 40000;
  int probe_batch = 256;  // games in flight during the bulk probe phase
  int batch_size = 8;     // MCTS leaves per forward (phase 2)
  int top_k = 30;
  double c_puct = 2.5;
  double q_weight = 1.0;
  double dirichlet_alpha = 0.3, dirichlet_weight = 0.25;
  int threads = 14;
  bool fp32 = false;
  bool full_record = false;
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
    else if (k == "--recovery-turns") a.recovery_turns = std::stoi(argv[++i]);
    else if (k == "--recovery-sims") a.recovery_sims = std::stoi(argv[++i]);
    else if (k == "--prevention-turns") a.prevention_turns = std::stoi(argv[++i]);
    else if (k == "--prevention-sims") a.prevention_sims = std::stoi(argv[++i]);
    else if (k == "--continue-turns") a.continue_turns = std::stoi(argv[++i]);
    else if (k == "--policy-max-turns") a.policy_max_turns = std::stol(argv[++i]);
    else if (k == "--probe-batch") a.probe_batch = std::stoi(argv[++i]);
    else if (k == "--batch-size") a.batch_size = std::stoi(argv[++i]);
    else if (k == "--top-k") a.top_k = std::stoi(argv[++i]);
    else if (k == "--c-puct") a.c_puct = std::stod(argv[++i]);
    else if (k == "--q-weight") a.q_weight = std::stod(argv[++i]);
    else if (k == "--dirichlet-alpha") a.dirichlet_alpha = std::stod(argv[++i]);
    else if (k == "--dirichlet-weight") a.dirichlet_weight = std::stod(argv[++i]);
    else if (k == "--threads") a.threads = std::stoi(argv[++i]);
  }
  return a;
}

struct Snapshot {
  int8_t board[81];
  std::vector<clines::NextBall> next_balls;
  int score, turn;
};

// One deep-MCTS replay task = one saved checkpoint (anchor).
struct ReplayTask {
  uint64_t original_seed;
  Snapshot anchor;
  const char* label;  // "recovery" | "prevention"
  int rewind, sims;
};

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
  auto t0 = Clock::now();

  std::printf("mcts_crisis: %zu probe seeds [%llu,%llu)  recovery=%d@%d "
              "prevention=%d@%d continue=%d probe_cap=%ld probe_batch=%d  "
              "q=%.2f mcts_batch=%d %s %s threads=%d\nout: %s\n",
              seeds.size(), (unsigned long long)args.seed_start,
              (unsigned long long)args.seed_end, args.recovery_turns,
              args.recovery_sims, args.prevention_turns, args.prevention_sims,
              args.continue_turns, args.policy_max_turns, args.probe_batch,
              args.q_weight, args.batch_size, args.device.c_str(),
              fp16 ? "fp16" : "fp32", args.threads, args.out_dir.c_str());
  std::fflush(stdout);

  // ============ Phase 1: BULK policy-only probes ============
  struct Slot {
    uint64_t seed;
    clines::Game game;
    std::deque<Snapshot> ring;
  };
  const size_t ring_cap = args.prevention_turns + 1;
  size_t next_seed = 0;
  std::vector<Slot> slots;
  auto fill_slot = [&](Slot& s) -> bool {
    if (next_seed >= seeds.size()) return false;
    s.seed = seeds[next_seed++];
    s.game = clines::Game(s.seed);
    s.game.Reset();
    s.ring.clear();
    return true;
  };
  int B = std::min<int>(args.probe_batch, (int)seeds.size());
  slots.reserve(B);
  for (int i = 0; i < B; ++i) {
    Slot s{0, clines::Game(0), {}};
    if (fill_slot(s)) slots.push_back(std::move(s));
  }

  std::vector<ReplayTask> tasks;
  std::vector<float> obs_buf, logits_buf;
  std::vector<int> lp_acts(1);
  std::vector<double> lp_pris(1);
  int probes_done = 0, probe_deaths = 0;

  while (!slots.empty()) {
    int n = (int)slots.size();
    obs_buf.resize((size_t)n * 18 * clines::kNN);
    logits_buf.resize((size_t)n * clines::kActions);
    for (int i = 0; i < n; ++i) {
      Slot& s = slots[i];
      // snapshot BEFORE the move (crisis_mining.py per-turn snapshot)
      Snapshot snap;
      std::memcpy(snap.board, s.game.board().data(), 81);
      snap.next_balls = s.game.next_balls();
      snap.score = s.game.score();
      snap.turn = s.game.turns();
      s.ring.push_back(std::move(snap));
      if (s.ring.size() > ring_cap) s.ring.pop_front();
      s.game.BuildObs(obs_buf.data() + (size_t)i * 18 * clines::kNN);
    }
    server.Eval(obs_buf.data(), n, logits_buf.data());  // ONE bulk forward

    std::vector<Slot> survivors;
    survivors.reserve(n);
    for (int i = 0; i < n; ++i) {
      Slot& s = slots[i];
      int k = clines::LegalPriors(
          s.game.board().data(),
          logits_buf.data() + (size_t)i * clines::kActions, 1,
          lp_acts.data(), lp_pris.data());
      bool dead = (k == 0);
      if (!dead) {
        int src = lp_acts[0] / 81, tgt = lp_acts[0] % 81;
        if (!s.game.Move(src / 9, src % 9, tgt / 9, tgt % 9)) {
          std::fprintf(stderr, "FATAL: illegal greedy move (seed=%llu turn=%d)\n",
                       (unsigned long long)s.seed, s.game.turns());
          std::abort();
        }
        dead = s.game.over();
      }
      bool capped_out = s.game.turns() >= args.policy_max_turns;
      if (dead || capped_out) {
        ++probes_done;
        if (dead) {
          ++probe_deaths;
          int death_turn = s.game.turns();
          struct { const char* label; int rewind, sims; } bands[2] = {
              {"recovery", args.recovery_turns, args.recovery_sims},
              {"prevention", args.prevention_turns, args.prevention_sims}};
          for (auto& band : bands) {
            int want = std::max(0, death_turn - band.rewind);
            for (const Snapshot& sn : s.ring) {
              if (sn.turn == want) {
                tasks.push_back({s.seed, sn, band.label, band.rewind, band.sims});
                break;
              }
            }
          }
        }
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("  [probe %d/%zu] seed=%llu %s turn=%d score=%d | "
                    "deaths=%d tasks=%zu (%.0fs)\n",
                    probes_done, seeds.size(), (unsigned long long)s.seed,
                    dead ? "DIED" : "capped", s.game.turns(), s.game.score(),
                    probe_deaths, tasks.size(), el);
        std::fflush(stdout);
        Slot repl{0, clines::Game(0), {}};
        if (fill_slot(repl)) survivors.push_back(std::move(repl));
      } else {
        survivors.push_back(std::move(s));
      }
    }
    slots.swap(survivors);
  }
  double el1 = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("phase 1 done: %d probes, %d deaths, %zu replay checkpoints in "
              "%.0fs\n\n",
              probes_done, probe_deaths, tasks.size(), el1);
  std::fflush(stdout);

  // ============ Phase 2: deep-MCTS replays from the checkpoints ============
  std::atomic<size_t> next_task{0};
  std::atomic<int> games_written{0};
  std::mutex print_mu;

  auto replay_worker = [&](int tid) {
    while (true) {
      size_t ti = next_task.fetch_add(1);
      if (ti >= tasks.size()) return;
      const ReplayTask& task = tasks[ti];
      auto tg0 = Clock::now();

      uint64_t replay_seed = task.original_seed * 37 + task.rewind;
      clines::Game rg(replay_seed);
      rg.SetState(task.anchor.board, task.anchor.next_balls, task.anchor.score,
                  task.anchor.turn);

      clines::MctsConfig cfg;
      cfg.num_simulations = task.sims;
      cfg.c_puct = args.c_puct;
      cfg.top_k = args.top_k;
      cfg.batch_size = args.batch_size;
      cfg.q_weight = args.q_weight;
      cfg.early_stop = false;  // full visit distribution for targets
      cfg.dirichlet_alpha = args.dirichlet_alpha;
      cfg.dirichlet_weight = args.dirichlet_weight;
      clines::MCTS mcts(
          [&server](const float* o, int n, float* out) { server.Eval(o, n, out); },
          &fe, cfg);
      clines::SimpleRng move_rng(replay_seed * 2654435761ULL + 1);

      std::vector<clines::MoveRec> recs;
      int replayed = 0;
      while (!rg.over() && replayed < args.continue_turns) {
        clines::SearchResult r = mcts.Search(rg, /*temperature=*/0.0, move_rng);
        if (r.action < 0) break;
        recs.push_back(clines::MakeMoveRec(rg, r));
        int src = r.action / 81, tgt = r.action % 81;
        if (!rg.Move(src / 9, src % 9, tgt / 9, tgt % 9)) {
          std::fprintf(stderr, "FATAL: illegal replay move (seed=%llu)\n",
                       (unsigned long long)replay_seed);
          std::abort();
        }
        ++replayed;
      }
      bool capped = !rg.over() && replayed >= args.continue_turns;
      double game_time = std::chrono::duration<double>(Clock::now() - tg0).count();

      std::string json;
      json.reserve(recs.size() * 700 + 512);
      json += "{\"seed\": " + std::to_string(replay_seed) +
              ", \"original_seed\": " + std::to_string(task.original_seed) +
              ", \"score\": " + std::to_string(rg.score()) +
              ", \"turns\": " + std::to_string(rg.turns()) +
              ", \"replay_from_turn\": " + std::to_string(task.anchor.turn) +
              ", \"replay_sims\": " + std::to_string(task.sims) +
              ", \"capped\": " + (capped ? std::string("true") : std::string("false")) +
              ", \"bootstrap_value\": 0.0" +
              ", \"time\": ";
      clines::AppendD(json, game_time);
      json += ", \"label\": \"" + std::string(task.label) + "\"" +
              ", \"rewind_turns\": " + std::to_string(task.rewind) +
              ", \"continue_turns\": " + std::to_string(args.continue_turns) +
              ", \"policy_max_turns\": " + std::to_string(args.policy_max_turns) +
              ", \"moves\": ";
      clines::AppendMovesArray(json, recs, args.full_record);
      json += "}";
      clines::WriteFileOrDie(
          args.out_dir + "/game_seed" + std::to_string(task.original_seed) +
              "_" + task.label + "_score" + std::to_string(rg.score()) + ".json",
          json);
      int g = games_written.fetch_add(1) + 1;

      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::lock_guard<std::mutex> l(print_mu);
      std::printf("  [replay %d/%zu t%d] seed=%llu %s@%d anchor_t=%d "
                  "replayed=%d final=%d%s (%.0fs, ETA %.0fs)\n",
                  g, tasks.size(), tid, (unsigned long long)task.original_seed,
                  task.label, task.sims, task.anchor.turn, replayed, rg.score(),
                  capped ? " CAPPED" : " died", el,
                  (el - el1) / g * (tasks.size() - g));
      std::fflush(stdout);
    }
  };

  int T = std::min<int>(args.threads, std::max<int>(1, (int)tasks.size()));
  std::vector<std::thread> pool;
  for (int t = 0; t < T; ++t) pool.emplace_back(replay_worker, t);
  for (auto& th : pool) th.join();

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %d probes, %d deaths, %d crisis games in %.0fs  "
              "(%lld forwards, %lld evals, %.0f evals/s)\n",
              probes_done, probe_deaths, games_written.load(), el,
              (long long)server.forwards(), (long long)server.evals(),
              server.evals() / el);
  return 0;
}
