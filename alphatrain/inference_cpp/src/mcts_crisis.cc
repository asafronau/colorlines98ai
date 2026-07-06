// native crisis mining: port of alphatrain/scripts/crisis_mining.py.
//
// Per probe seed: play the policy GREEDILY to death (the model's own failure),
// keeping a ring buffer of the last prevention_turns+1 snapshots. On death,
// rewind to two anchors and replay each with deep MCTS (temp=0 + Dirichlet):
//   recovery:   death_turn - recovery_turns   @ recovery_sims   (near-death save)
//   prevention: death_turn - prevention_turns @ prevention_sims (avoid the spiral)
// Each replay runs continue_turns moves (or to death) and is recorded in the
// same moves-schema JSON as selfplay + crisis extras (original_seed, label,
// replay_from_turn, replay_sims, rewind_turns, bootstrap_value=0, ...).
// These escape-or-die games are the DECISIVE training states — the 70% share
// of the pillar3k corpus that cracked the de-peak wall (HISTORY 173-174).
//
//   ./build/mcts_crisis --model data/policy_ts.pt --device mps \
//       --seed-start 940000 --seed-end 940100 --threads 14 \
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
  uint64_t seed_start = 940000, seed_end = 940100;  // [start, end)
  int recovery_turns = 15, recovery_sims = 1600;
  int prevention_turns = 75, prevention_sims = 2400;
  int continue_turns = 500;
  long policy_max_turns = 40000;
  int batch_size = 8;
  int top_k = 30;
  double c_puct = 2.5;
  double q_weight = 1.0;
  double dirichlet_alpha = 0.3, dirichlet_weight = 0.25;
  int threads = 14;
  bool fp32 = false;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--fp32") { a.fp32 = true; continue; }
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
  std::atomic<int> done{0}, deaths{0}, games_written{0};
  std::mutex print_mu;
  auto t0 = Clock::now();

  std::printf("mcts_crisis: %zu probe seeds [%llu,%llu)  recovery=%d@%d "
              "prevention=%d@%d continue=%d probe_cap=%ld  q=%.2f batch=%d "
              "%s %s threads=%d\nout: %s\n",
              seeds.size(), (unsigned long long)args.seed_start,
              (unsigned long long)args.seed_end, args.recovery_turns,
              args.recovery_sims, args.prevention_turns, args.prevention_sims,
              args.continue_turns, args.policy_max_turns, args.q_weight,
              args.batch_size, args.device.c_str(), fp16 ? "fp16" : "fp32",
              args.threads, args.out_dir.c_str());
  std::fflush(stdout);

  auto worker = [&](int tid) {
    std::vector<float> obs(18 * clines::kNN), logits(clines::kActions);
    std::vector<int> acts(clines::kActions);
    std::vector<double> pris(clines::kActions);

    while (true) {
      size_t i = next_idx.fetch_add(1);
      if (i >= seeds.size()) return;
      uint64_t seed = seeds[i];
      auto tg0 = Clock::now();

      // === Phase 1: greedy probe to death, ring of last prevention+1 turns ===
      clines::Game g(seed);
      g.Reset();
      std::deque<Snapshot> ring;
      const size_t ring_cap = args.prevention_turns + 1;
      while (!g.over() && g.turns() < args.policy_max_turns) {
        Snapshot snap;
        std::memcpy(snap.board, g.board().data(), 81);
        snap.next_balls = g.next_balls();
        snap.score = g.score();
        snap.turn = g.turns();
        ring.push_back(std::move(snap));
        if (ring.size() > ring_cap) ring.pop_front();

        g.BuildObs(obs.data());
        server.Eval(obs.data(), 1, logits.data());
        int k = clines::LegalPriors(g.board().data(), logits.data(), 1,
                                    acts.data(), pris.data());
        if (k == 0) break;  // dead: no legal moves
        int src = acts[0] / 81, tgt = acts[0] % 81;
        if (!g.Move(src / 9, src % 9, tgt / 9, tgt % 9)) {
          std::fprintf(stderr, "FATAL: illegal greedy move (seed=%llu turn=%d)\n",
                       (unsigned long long)seed, g.turns());
          std::abort();
        }
        if (g.turns() % 2000 == 0) {
          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          std::lock_guard<std::mutex> l(print_mu);
          std::printf("    [t%d probe] seed=%llu turn=%d score=%d (%.0fs)\n",
                      tid, (unsigned long long)seed, g.turns(), g.score(), el);
          std::fflush(stdout);
        }
      }
      bool died = g.over() && g.turns() < args.policy_max_turns;
      int death_turn = g.turns();

      if (died) {
        deaths.fetch_add(1);
        // === Phase 2: two anchor bands, deep-MCTS replay ===
        struct Band { const char* label; int rewind, sims; };
        Band bands[2] = {{"recovery", args.recovery_turns, args.recovery_sims},
                         {"prevention", args.prevention_turns, args.prevention_sims}};
        for (const Band& band : bands) {
          int want = std::max(0, death_turn - band.rewind);
          const Snapshot* anchor = nullptr;
          for (const Snapshot& s : ring)
            if (s.turn == want) { anchor = &s; break; }
          if (!anchor) continue;  // game shorter than the ring window covers

          uint64_t replay_seed = seed * 37 + band.rewind;
          clines::Game rg(replay_seed);
          rg.SetState(anchor->board, anchor->next_balls, anchor->score,
                      anchor->turn);

          clines::MctsConfig cfg;
          cfg.num_simulations = band.sims;
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
          double game_time =
              std::chrono::duration<double>(Clock::now() - tg0).count();

          std::string json;
          json.reserve(recs.size() * 900 + 512);
          json += "{\"seed\": " + std::to_string(replay_seed) +
                  ", \"original_seed\": " + std::to_string(seed) +
                  ", \"score\": " + std::to_string(rg.score()) +
                  ", \"turns\": " + std::to_string(rg.turns()) +
                  ", \"replay_from_turn\": " + std::to_string(anchor->turn) +
                  ", \"replay_sims\": " + std::to_string(band.sims) +
                  ", \"capped\": " + (capped ? std::string("true") : std::string("false")) +
                  ", \"bootstrap_value\": 0.0" +
                  ", \"time\": ";
          clines::AppendD(json, game_time);
          json += ", \"label\": \"" + std::string(band.label) + "\"" +
                  ", \"rewind_turns\": " + std::to_string(band.rewind) +
                  ", \"continue_turns\": " + std::to_string(args.continue_turns) +
                  ", \"policy_max_turns\": " + std::to_string(args.policy_max_turns) +
                  ", \"moves\": ";
          clines::AppendMovesArray(json, recs);
          json += "}";
          clines::WriteFileOrDie(
              args.out_dir + "/game_seed" + std::to_string(seed) + "_" +
                  band.label + "_score" + std::to_string(rg.score()) + ".json",
              json);
          games_written.fetch_add(1);

          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          std::lock_guard<std::mutex> l(print_mu);
          std::printf("    [t%d %s] seed=%llu anchor_t=%d replayed=%d "
                      "final=%d%s (%.0fs)\n",
                      tid, band.label, (unsigned long long)seed, anchor->turn,
                      replayed, rg.score(), capped ? " CAPPED" : " died", el);
          std::fflush(stdout);
        }
      }

      int d = done.fetch_add(1) + 1;
      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::lock_guard<std::mutex> l(print_mu);
      std::printf("  [%d/%zu] seed=%llu probe: %s at turn=%d score=%d | "
                  "deaths=%d games=%d (%.0fs, ETA %.0fs)\n",
                  d, seeds.size(), (unsigned long long)seed,
                  died ? "DIED" : "capped", death_turn, g.score(),
                  deaths.load(), games_written.load(), el,
                  el / d * (seeds.size() - d));
      std::fflush(stdout);
    }
  };

  int T = std::min<int>(args.threads, (int)seeds.size());
  std::vector<std::thread> pool;
  for (int t = 0; t < T; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %zu probes, %d deaths, %d crisis games in %.0fs  "
              "(%lld forwards, %lld evals, %.0f evals/s)\n",
              seeds.size(), deaths.load(), games_written.load(), el,
              (long long)server.forwards(), (long long)server.evals(),
              server.evals() / el);
  return 0;
}
