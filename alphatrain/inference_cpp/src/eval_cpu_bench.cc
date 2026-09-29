// CPU cost of eval's per-step game work on real states (no GPU), old path vs fast path (HISTORY 257).
//   old:  BuildObs + LegalMask (6561 floats) + fp32->fp16 and float->uint8 conversions + Move
//   fast: Labels once -> BuildObs(labels) + LegalMaskU8 + fp32->fp16 + Move(labels), optionally on a
//         thread pool. Verifies the observations, the legal masks and the post-move states (board,
//         score, turns, preview, and the next move after it) are bit-identical before timing.
//
//   ./build/eval_cpu_bench data/bench_states.txt [threads=8] [reps=20]

#include <torch/torch.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "anchors.h"
#include "game.h"
#include "thread_pool.h"

using clines::Game;
using clines::kActions;
using clines::kNN;
using Clock = std::chrono::high_resolution_clock;

namespace {

int FirstLegal(const uint8_t* m) {
  for (int a = 0; a < kActions; ++a) if (m[a]) return a;
  return -1;
}

bool SameState(const Game& a, const Game& b) {
  if (a.board() != b.board() || a.score() != b.score() || a.turns() != b.turns() || a.over() != b.over())
    return false;
  const auto& x = a.next_balls();
  const auto& y = b.next_balls();
  if (x.size() != y.size()) return false;
  for (size_t i = 0; i < x.size(); ++i)
    if (x[i].r != y[i].r || x[i].c != y[i].c || x[i].color != y[i].color) return false;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: eval_cpu_bench <anchors> [threads] [reps]\n"); return 2; }
  const int threads = argc > 2 ? std::stoi(argv[2]) : 8;
  const int reps = argc > 3 ? std::stoi(argv[3]) : 20;
  std::vector<clines::Anchor> anchors = clines::ReadAnchors(argv[1]);
  const int N = static_cast<int>(anchors.size());
  std::vector<Game> base;
  base.reserve(N);
  for (const auto& a : anchors) base.push_back(clines::StartFromAnchor(a));
  std::printf("%d real states, %d threads, %d reps\n", N, threads, reps);

  // ---- equivalence on every state (two consecutive moves, so the RNG stream is covered) ----
  std::vector<float> obs_old(18 * kNN), obs_new(18 * kNN), mask_f(kActions);
  std::vector<uint8_t> mask_u8(kActions), mask_old_u8(kActions);
  int8_t labels[kNN];
  int mismatches = 0;
  for (int i = 0; i < N; ++i) {
    Game g_old = base[i], g_new = base[i];
    for (int step = 0; step < 2 && !g_old.over(); ++step) {
      g_old.BuildObs(obs_old.data());
      g_old.LegalMask(mask_f.data());
      for (int a = 0; a < kActions; ++a) mask_old_u8[a] = mask_f[a] > 0.5f;
      g_new.Labels(labels);
      g_new.BuildObs(obs_new.data(), labels);
      int n_legal = g_new.LegalMaskU8(mask_u8.data(), labels);
      int n_old = 0;
      for (int a = 0; a < kActions; ++a) n_old += mask_old_u8[a];
      if (std::memcmp(obs_old.data(), obs_new.data(), obs_old.size() * sizeof(float)) != 0 ||
          std::memcmp(mask_old_u8.data(), mask_u8.data(), kActions) != 0 || n_legal != n_old) {
        ++mismatches;
        break;
      }
      int m = FirstLegal(mask_u8.data());
      if (m < 0) break;
      int s = m / kNN, t = m % kNN;
      bool ok_old = g_old.Move(s / 9, s % 9, t / 9, t % 9);
      bool ok_new = g_new.Move(s / 9, s % 9, t / 9, t % 9, labels);
      if (ok_old != ok_new || !SameState(g_old, g_new)) { ++mismatches; break; }
    }
  }
  std::printf("equivalence: %d mismatching states of %d -> %s\n", mismatches, N,
              mismatches ? "FAIL" : "PASS");
  if (mismatches) return 1;

  clines::ThreadPool pool(threads);
  for (int n : {500, 1000, 2000, 4000}) {
    if (n > N) break;
    std::vector<float> obs(static_cast<size_t>(n) * 18 * kNN), legal_f(static_cast<size_t>(n) * kActions);
    std::vector<uint8_t> legal_u8(static_cast<size_t>(n) * kActions);
    std::vector<int8_t> lab(static_cast<size_t>(n) * kNN);
    std::vector<int> count(n);
    double t_old = 0, t_fast1 = 0, t_fastT = 0;
    for (int rep = 0; rep < reps; ++rep) {
      // old path
      std::vector<Game> g(base.begin(), base.begin() + n);
      auto t0 = Clock::now();
      for (int i = 0; i < n; ++i) {
        g[i].BuildObs(obs.data() + static_cast<size_t>(i) * 18 * kNN);
        g[i].LegalMask(legal_f.data() + static_cast<size_t>(i) * kActions);
      }
      torch::Tensor o = torch::from_blob(obs.data(), {n, 18, 9, 9}).to(torch::kHalf);
      torch::Tensor l = torch::from_blob(legal_f.data(), {n, kActions}).to(torch::kByte);
      for (int i = 0; i < n; ++i) {
        const float* lg = legal_f.data() + static_cast<size_t>(i) * kActions;
        int m = -1;
        for (int a = 0; a < kActions; ++a) if (lg[a] > 0.5f) { m = a; break; }
        if (m >= 0) g[i].Move(m / kNN / 9, m / kNN % 9, m % kNN / 9, m % kNN % 9);
      }
      t_old += std::chrono::duration<double>(Clock::now() - t0).count();

      // fast path, single thread then pooled
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<Game> h(base.begin(), base.begin() + n);
        auto body = [&](int b, int e) {
          for (int i = b; i < e; ++i) {
            int8_t* li = lab.data() + static_cast<size_t>(i) * kNN;
            h[i].Labels(li);
            h[i].BuildObs(obs.data() + static_cast<size_t>(i) * 18 * kNN, li);
            count[i] = h[i].LegalMaskU8(legal_u8.data() + static_cast<size_t>(i) * kActions, li);
          }
        };
        auto moves = [&](int b, int e) {
          for (int i = b; i < e; ++i) {
            if (count[i] == 0) continue;
            int m = FirstLegal(legal_u8.data() + static_cast<size_t>(i) * kActions);
            h[i].Move(m / kNN / 9, m / kNN % 9, m % kNN / 9, m % kNN % 9,
                      lab.data() + static_cast<size_t>(i) * kNN);
          }
        };
        auto t1 = Clock::now();
        if (pass == 0) body(0, n); else pool.ParallelFor(n, body);
        torch::Tensor o2 = torch::from_blob(obs.data(), {n, 18, 9, 9}).to(torch::kHalf);
        torch::Tensor l2 = torch::from_blob(legal_u8.data(), {n, kActions}, torch::kUInt8);
        if (pass == 0) moves(0, n); else pool.ParallelFor(n, moves);
        (pass == 0 ? t_fast1 : t_fastT) += std::chrono::duration<double>(Clock::now() - t1).count();
      }
    }
    std::printf("batch %4d: old %7.2f ms/step   fast 1 thread %6.2f ms (%.1fx)   fast %d threads %6.2f ms (%.1fx)\n",
                n, 1e3 * t_old / reps, 1e3 * t_fast1 / reps, t_old / t_fast1, threads, 1e3 * t_fastT / reps,
                t_old / t_fastT);
  }
  return 0;
}
