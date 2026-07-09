// Rollout judge for correction states: from the same board, play the TEACHER's
// move vs the BASE's move, then R greedy-policy rollouts each (common seed list
// across arms), and compare died-within-H rates. Decides whether the corpus
// corrections are genuine improvements for GREEDY play or phantoms
// (FV-visit-favored moves worse than the base's — HISTORY 172 hypothesis 2).
//
//   ./build/rollout_judge --model data/policy_ts.pt --device mps \
//       --reps 64 --horizon 300 --batch 512
// Reads data/judge_states.bin (export_judge_states.py). Writes
// data/judge_results.csv + prints the aggregate verdict.

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "game.h"
#include "infer_server.h"
#include "mcts.h"  // LegalPriors

using Clock = std::chrono::high_resolution_clock;

namespace {

struct JudgeState {
  int8_t board[81];
  std::vector<clines::NextBall> nb;
  int teacher_move, base_move;
  float top_share;
};

struct Args {
  std::string model = "data/policy_ts.pt";
  std::string device = "mps";
  std::string states = "data/judge_states.bin";
  std::string out = "data/judge_results.csv";
  int reps = 64;
  int horizon = 300;
  int batch = 512;
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
    else if (k == "--states") a.states = argv[++i];
    else if (k == "--out") a.out = argv[++i];
    else if (k == "--reps") a.reps = std::stoi(argv[++i]);
    else if (k == "--horizon") a.horizon = std::stoi(argv[++i]);
    else if (k == "--batch") a.batch = std::stoi(argv[++i]);
  }
  return a;
}

std::vector<JudgeState> LoadStates(const char* path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<JudgeState> out;
  if (!f) return out;
  char magic[4];
  f.read(magic, 4);
  if (std::string(magic, 4) != "CLRJ") return out;
  int32_t n = 0;
  f.read(reinterpret_cast<char*>(&n), 4);
  for (int i = 0; i < n; ++i) {
    JudgeState s;
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
  if (dev.is_mps() && !torch::mps::is_available()) dev = torch::Device(torch::kCPU);
  const bool fp16 = dev.is_mps() && !args.fp32;

  std::vector<JudgeState> states = LoadStates(args.states.c_str());
  if (states.empty()) {
    std::printf("cannot load %s\n", args.states.c_str());
    return 1;
  }
  clines::InferenceServer server(args.model, dev, fp16);
  const int N = (int)states.size(), R = args.reps;
  std::printf("rollout_judge: %d correction states x 2 arms x %d reps, "
              "horizon %d, %s %s\n",
              N, R, args.horizon, args.device.c_str(), fp16 ? "fp16" : "fp32");
  std::fflush(stdout);

  // outcome[state][arm][rep] = turns survived (capped at horizon); died flag
  std::vector<int> turns_out((size_t)N * 2 * R, 0);
  std::vector<char> died_out((size_t)N * 2 * R, 0);

  struct Job { int si, arm, rep; };
  std::vector<Job> jobs;
  jobs.reserve((size_t)N * 2 * R);
  for (int si = 0; si < N; ++si)
    for (int arm = 0; arm < 2; ++arm)
      for (int rep = 0; rep < R; ++rep) jobs.push_back({si, arm, rep});

  struct Slot { Job job; clines::Game game; int start_turns; };
  size_t next_job = 0;
  std::vector<Slot> slots;
  auto fill = [&](Slot& s) -> bool {
    while (next_job < jobs.size()) {
      Job j = jobs[next_job++];
      const JudgeState& st = states[j.si];
      // Common seed list across arms: seed depends on (state, rep) only.
      uint64_t seed = 777000000ULL + (uint64_t)j.si * 1000 + j.rep;
      clines::Game g(seed);
      g.SetState(st.board, st.nb, 0, 0);
      int mv = j.arm == 0 ? st.teacher_move : st.base_move;
      int src = mv / 81, tgt = mv % 81;
      if (!g.Move(src / 9, src % 9, tgt / 9, tgt % 9)) {
        std::fprintf(stderr, "FATAL: judge move illegal (state %d arm %d)\n",
                     j.si, j.arm);
        std::abort();
      }
      s.job = j;
      s.game = std::move(g);
      s.start_turns = 0;  // SetState reset turns to 0; first move made turns=1
      return true;
    }
    return false;
  };
  int B = std::min<int>(args.batch, (int)jobs.size());
  slots.reserve(B);
  for (int i = 0; i < B; ++i) {
    Slot s{{0,0,0}, clines::Game(0), 0};
    if (fill(s)) slots.push_back(std::move(s));
  }

  std::vector<float> obs_buf, logits_buf;
  std::vector<int> acts(1);
  std::vector<double> pris(1);
  size_t done = 0;
  auto t0 = Clock::now();

  while (!slots.empty()) {
    int n = (int)slots.size();
    obs_buf.resize((size_t)n * 18 * clines::kNN);
    logits_buf.resize((size_t)n * clines::kActions);
    for (int i = 0; i < n; ++i)
      slots[i].game.BuildObs(obs_buf.data() + (size_t)i * 18 * clines::kNN);
    server.Eval(obs_buf.data(), n, logits_buf.data());

    std::vector<Slot> live;
    live.reserve(n);
    for (int i = 0; i < n; ++i) {
      Slot& s = slots[i];
      bool dead = s.game.over();
      if (!dead) {
        int k = clines::LegalPriors(
            s.game.board().data(),
            logits_buf.data() + (size_t)i * clines::kActions, 1,
            acts.data(), pris.data());
        if (k == 0) dead = true;
        else {
          int src = acts[0] / 81, tgt = acts[0] % 81;
          s.game.Move(src / 9, src % 9, tgt / 9, tgt % 9);
          dead = s.game.over();
        }
      }
      bool horizon_hit = s.game.turns() >= args.horizon;
      if (dead || horizon_hit) {
        size_t oi = ((size_t)s.job.si * 2 + s.job.arm) * R + s.job.rep;
        turns_out[oi] = s.game.turns();
        died_out[oi] = dead ? 1 : 0;
        ++done;
        if (done % 5000 == 0) {
          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          std::printf("  %zu/%zu rollouts (%.0f/s, %.0fs)\n", done, jobs.size(),
                      done / el, el);
          std::fflush(stdout);
        }
        Slot repl{{0,0,0}, clines::Game(0), 0};
        if (fill(repl)) live.push_back(std::move(repl));
      } else {
        live.push_back(std::move(s));
      }
    }
    slots.swap(live);
  }

  // ---- aggregate ----
  FILE* csv = std::fopen(args.out.c_str(), "w");
  std::fprintf(csv, "state,top_share,teacher_died,base_died,teacher_turns,base_turns\n");
  int genuine = 0, phantom = 0, tie = 0;
  double sum_td = 0, sum_bd = 0;
  const double margin = 0.08;  // ~1.5 SE at R=64
  for (int si = 0; si < N; ++si) {
    double td = 0, bd = 0, tt = 0, bt = 0;
    for (int r = 0; r < R; ++r) {
      size_t ti = ((size_t)si * 2 + 0) * R + r, bi = ((size_t)si * 2 + 1) * R + r;
      td += died_out[ti]; bd += died_out[bi];
      tt += turns_out[ti]; bt += turns_out[bi];
    }
    td /= R; bd /= R; tt /= R; bt /= R;
    sum_td += td; sum_bd += bd;
    if (td < bd - margin) ++genuine;
    else if (td > bd + margin) ++phantom;
    else ++tie;
    std::fprintf(csv, "%d,%.3f,%.4f,%.4f,%.1f,%.1f\n",
                 si, states[si].top_share, td, bd, tt, bt);
  }
  std::fclose(csv);

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %zu rollouts in %.0fs (%lld evals, %.0f evals/s)\n",
              jobs.size(), el, (long long)server.evals(), server.evals() / el);
  std::printf("mean died-within-%d rate: TEACHER move %.3f vs BASE move %.3f\n",
              args.horizon, sum_td / N, sum_bd / N);
  std::printf("per-state verdicts (|gap| > %.2f): GENUINE %d (%.0f%%)  TIE %d (%.0f%%)  "
              "PHANTOM %d (%.0f%%)\n",
              margin, genuine, 100.0 * genuine / N, tie, 100.0 * tie / N,
              phantom, 100.0 * phantom / N);
  std::printf("results: %s\n", args.out.c_str());
  return 0;
}
