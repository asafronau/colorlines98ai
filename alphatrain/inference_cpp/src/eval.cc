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
#include "game_json.h"
#include "canonical.h"
#include "anchors.h"

using clines::Game;
using Clock = std::chrono::high_resolution_clock;

namespace {
struct Args {
  std::string model = "data/policy_ts.pt";
  std::string device = "cpu";
  uint64_t seed_start = 50000, seed_end = 50300;  // [start, end)
  int batch = 256;
  long max_turns = 1000000;
  bool fp32 = false;  // default: fp16 on MPS (like eval_policy.py)
  int tta = 1;  // 1 = plain greedy; 8 = average logits over the 8 exact board symmetries (D4)
  bool canon = false;  // feed the net the canonical form (D4 x color relabel), map logits back
  bool verbose_games = false;  // opt-in per-game + every-500-turn traces
  int progress_every = 100;    // compact aggregate progress by default
  long progress_forwards = 50000;  // heartbeat for the long tail (games in flight, longest game); 0 = off
  std::string scores_out;  // optional streamed CSV for distribution statistics
  // On-policy trajectory recording (DAgger harvest + value-head fuel):
  // keep the FULL last `record_tail` turns (death band) + every
  // `record_every`-th turn before that (broad coverage). Empty dir = off.
  std::string record_dir;
  int record_every = 8;
  int record_tail = 160;
  // Anchor mode: start each game from a saved state (anchors.h) and play greedily for at most its
  // `cap` more turns, with the same spawn stream as the mcts_crisis replay from that state.
  std::string anchors;
};

// One recorded pre-move snapshot: state + the move the policy chose there.
struct TurnRec {
  int8_t board[81];
  int8_t nb[9];  // (r, c, color) x up to 3
  int8_t n_next;
  int32_t move;
  int32_t turn;  // 0-based move index within the game
};

TurnRec SnapTurn(const Game& g, int move) {
  TurnRec t;
  std::memcpy(t.board, g.board().data(), 81);
  const auto& nb = g.next_balls();
  t.n_next = (int8_t)std::min<size_t>(nb.size(), 3);
  std::memset(t.nb, 0, sizeof(t.nb));
  for (int i = 0; i < t.n_next; ++i) {
    t.nb[i * 3 + 0] = (int8_t)nb[i].r;
    t.nb[i * 3 + 1] = (int8_t)nb[i].c;
    t.nb[i * 3 + 2] = (int8_t)nb[i].color;
  }
  t.move = move;
  t.turn = (int)g.turns();
  return t;
}

void AppendTurnRec(std::string& s, const TurnRec& t) {
  s += "{\"turn\": " + std::to_string(t.turn) + ", \"board\": [";
  for (int r = 0; r < 9; ++r) {
    if (r) s += ", ";
    s += "[";
    for (int c = 0; c < 9; ++c) {
      if (c) s += ", ";
      s += std::to_string((int)t.board[r * 9 + c]);
    }
    s += "]";
  }
  s += "], \"next_balls\": [";
  for (int i = 0; i < t.n_next; ++i) {
    if (i) s += ", ";
    s += "{\"row\": " + std::to_string((int)t.nb[i * 3]) +
         ", \"col\": " + std::to_string((int)t.nb[i * 3 + 1]) +
         ", \"color\": " + std::to_string((int)t.nb[i * 3 + 2]) + "}";
  }
  s += "], \"num_next\": " + std::to_string((int)t.n_next);
  s += ", \"move\": " + std::to_string(t.move) + "}";
}

void WriteGameRecord(const Args& args, uint64_t seed, int score, long turns,
                     bool died, const std::vector<TurnRec>& broad,
                     const std::vector<TurnRec>& tail_ring, size_t ring_head) {
  std::string s = "{\"seed\": " + std::to_string(seed) +
                  ", \"final_score\": " + std::to_string(score) +
                  ", \"final_turns\": " + std::to_string(turns) +
                  ", \"died\": " + (died ? "true" : "false") +
                  ", \"record_every\": " + std::to_string(args.record_every) +
                  ", \"record_tail\": " + std::to_string(args.record_tail) +
                  ", \"states\": [";
  // Tail window start (turn index): everything at or after this is in the ring.
  long tail_start = std::max<long>(0, turns - (long)tail_ring.size());
  bool first = true;
  for (const TurnRec& t : broad) {
    if (t.turn >= tail_start) break;  // avoid duplicating tail states
    if (!first) s += ", ";
    AppendTurnRec(s, t);
    first = false;
  }
  for (size_t i = 0; i < tail_ring.size(); ++i) {
    const TurnRec& t = tail_ring[(ring_head + i) % tail_ring.size()];
    if (!first) s += ", ";
    AppendTurnRec(s, t);
    first = false;
  }
  s += "]}";
  clines::WriteFileOrDie(
      args.record_dir + "/game_seed" + std::to_string(seed) + ".json", s);
}

using clines::D4Maps;

// Observation of view v of `g`, built from the transformed board + preview so
// every derived channel (components, line potentials) is computed natively.
void BuildViewObs(const Game& g, const D4Maps& d4, int v, float* out) {
  int8_t b[81];
  for (int i = 0; i < 81; ++i) b[d4.cell[v][i]] = g.board()[i];
  std::vector<clines::NextBall> nb;
  for (const auto& x : g.next_balls()) {
    const int t = d4.cell[v][x.r * 9 + x.c];
    nb.push_back({t / 9, t % 9, x.color});
  }
  Game tmp(0);
  tmp.SetState(b, nb, g.score(), g.turns());
  tmp.BuildObs(out);
}

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--fp32") { a.fp32 = true; continue; }
    if (k == "--verbose-games") { a.verbose_games = true; continue; }
    if (k == "--canon") { a.canon = true; continue; }
    if (i + 1 >= argc) break;  // remaining flags take a value
    if (k == "--model") a.model = argv[++i];
    else if (k == "--tta") a.tta = std::stoi(argv[++i]);
    else if (k == "--device") a.device = argv[++i];
    else if (k == "--seed-start") a.seed_start = std::stoull(argv[++i]);
    else if (k == "--seed-end") a.seed_end = std::stoull(argv[++i]);
    else if (k == "--batch") a.batch = std::stoi(argv[++i]);
    else if (k == "--max-turns") a.max_turns = std::stol(argv[++i]);
    else if (k == "--progress-every") a.progress_every = std::stoi(argv[++i]);
    else if (k == "--progress-forwards") a.progress_forwards = std::stol(argv[++i]);
    else if (k == "--scores-out") a.scores_out = argv[++i];
    else if (k == "--record-dir") a.record_dir = argv[++i];
    else if (k == "--record-every") a.record_every = std::stoi(argv[++i]);
    else if (k == "--record-tail") a.record_tail = std::stoi(argv[++i]);
    else if (k == "--anchors") a.anchors = argv[++i];
  }
  if (a.device != "mps" && a.device != "cpu") {
    std::fprintf(stderr, "FATAL: unsupported device %s (expected mps or cpu)\n",
                 a.device.c_str());
    std::exit(2);
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
    std::fprintf(stderr,
                 "FATAL: --device mps was requested, but MPS is unavailable; "
                 "refusing to fall back to CPU\n");
    return 2;
  }

  // fp16 only on the GPU (CPU fp16 ops are slow/unsupported); fp32 on CPU.
  const bool use_half = dev.is_mps() && !args.fp32;

  torch::jit::Module net;
  try { net = torch::jit::load(args.model); }
  catch (const c10::Error& e) {
    std::printf("could not load %s: %s\n", args.model.c_str(), e.what());
    return 1;
  }
  net.to(dev);
  if (use_half) net.to(torch::kHalf);  // convert weights + BN buffers to fp16

  if (args.tta != 1 && args.tta != 8) {
    std::printf("--tta must be 1 or 8\n");
    return 2;
  }
  const D4Maps d4;
  torch::Tensor act_idx = torch::from_blob(const_cast<int64_t*>(d4.act.data()),
                                           {8, clines::kActions}, torch::kLong)
                              .slice(0, 0, args.tta).clone().to(dev);
  if (args.tta > 1) std::printf("TTA: averaging logits over %d board symmetries\n", args.tta);
  if (args.canon && args.tta != 1) { std::printf("--canon and --tta are exclusive\n"); return 2; }
  if (args.canon) std::printf("CANON: net sees the canonical form (D4 x color relabel) of every position\n");
  torch::Tensor act_all = torch::from_blob(const_cast<int64_t*>(d4.act.data()),
                                           {8, clines::kActions}, torch::kLong).clone().to(dev);
  std::vector<int64_t> canon_view;

  // Seed queue (anchor mode: one entry per anchor; `todo` holds anchor indices).
  const bool anchor_mode = !args.anchors.empty();
  std::vector<clines::Anchor> anchors;
  std::vector<uint64_t> todo;
  if (anchor_mode) {
    anchors = clines::ReadAnchors(args.anchors);
    for (uint64_t i = 0; i < anchors.size(); ++i) todo.push_back(i);
    std::printf("ANCHORS: %zu start states from %s\n", anchors.size(), args.anchors.c_str());
  } else {
    for (uint64_t s = args.seed_start; s < args.seed_end; ++s) todo.push_back(s);
  }
  if (todo.empty()) { std::printf("nothing to play\n"); return 2; }
  size_t next = 0;
  const int B = std::min<int>(args.batch, (int)todo.size());

  struct Slot {
    uint64_t seed; Game game;
    std::vector<TurnRec> broad, ring;  // recording only
    size_t ring_head = 0;
    long next_progress_turn = 500;
    long turn_limit = 0;   // absolute turn at which the game counts as capped
    int start_turn = 0;    // anchor mode: turn of the start state
  };
  const bool recording = !args.record_dir.empty();
  std::vector<Slot> slots;
  auto make_slot = [&](Slot& dst) -> bool {
    if (next >= todo.size()) return false;
    if (anchor_mode) {
      const clines::Anchor& a = anchors[todo[next++]];
      dst.seed = a.seed;
      dst.game = clines::StartFromAnchor(a);  // same spawn stream as the mcts_crisis replay
      dst.start_turn = a.turn;
      dst.turn_limit = (long)a.turn + a.cap;
    } else {
      dst.seed = todo[next++];
      dst.game = Game(dst.seed);
      dst.game.Reset();
      dst.start_turn = 0;
      dst.turn_limit = args.max_turns;
    }
    dst.broad.clear();
    dst.ring.clear();
    dst.ring_head = 0;
    dst.next_progress_turn = 500;
    return true;
  };
  slots.reserve(B);
  for (int i = 0; i < B; ++i) {
    Slot s{0, Game(0)};
    if (make_slot(s)) slots.push_back(std::move(s));
  }

  std::vector<int> scores;
  scores.reserve(todo.size());
  size_t capped_games = 0;
  long double turn_sum = 0.0;
  // Stream score rows as games finish.  Besides making a long evaluation
  // observable/recoverable, this preserves all completed work if the process
  // or machine is interrupted before the final summary.
  FILE* scores_file = nullptr;
  if (!args.scores_out.empty()) {
    scores_file = std::fopen(args.scores_out.c_str(), "w");
    if (!scores_file) {
      std::printf("could not open scores output: %s\n", args.scores_out.c_str());
      return 1;
    }
    std::fprintf(scores_file, "seed,score,turns,capped\n");
    std::fflush(scores_file);
  }
  std::vector<float> obs_buf, legal_buf;
  long fwd = 0;
  auto t0 = Clock::now();
  size_t done = 0;
  size_t log_next = std::max(1, args.progress_every);
  double score_sum = 0.0;

  while (!slots.empty()) {
    int n = (int)slots.size();
    const int V = args.tta;
    canon_view.assign(n, 0);
    obs_buf.resize((size_t)n * V * 18 * clines::kNN);
    legal_buf.resize((size_t)n * clines::kActions);
    // Build each game's obs+legal. Single-threaded on purpose: profiling showed
    // this eval is forward-bound (the heavy-tail long games run solo at tiny
    // batch and dominate wall-clock), so parallelizing this loop gave ~0 gain.
    for (int i = 0; i < n; ++i) {
      if (V == 1 && args.canon) {
        const clines::Canonical cf = clines::Canonicalize(slots[i].game.board().data(),
                                                          slots[i].game.next_balls(), d4);
        canon_view[i] = cf.view;
        Game tmp(0);
        tmp.SetState(cf.board, cf.next, slots[i].game.score(), slots[i].game.turns());
        tmp.BuildObs(obs_buf.data() + (size_t)i * 18 * clines::kNN);
      } else if (V == 1) {
        slots[i].game.BuildObs(obs_buf.data() + (size_t)i * 18 * clines::kNN);
      } else {
        for (int v = 0; v < V; ++v)
          BuildViewObs(slots[i].game, d4, v,
                       obs_buf.data() + ((size_t)i * V + v) * 18 * clines::kNN);
      }
      slots[i].game.LegalMask(legal_buf.data() + (size_t)i * clines::kActions);
    }
    // --- Shrink the CPU->GPU transfer (the profiled copy_and_sync bottleneck) ---
    // Obs: convert fp32->fp16 on the CPU *before* uploading, so we ship half the
    // bytes. (Uploading fp32 then .to(kHalf) on the GPU pays the full fp32
    // transfer plus an extra GPU kernel.)
    torch::Tensor obs = torch::from_blob(obs_buf.data(), {(int64_t)n * V, 18, 9, 9});
    if (use_half) obs = obs.to(torch::kHalf);
    obs = obs.to(dev);
    // Legal mask: it's just 0/1, so upload it as uint8 (1 byte) instead of fp32
    // (4 bytes) -> 4x less, and it's the single biggest per-step transfer.
    // `legal == 0` is then the bool "illegal" mask for masked_fill (works on the
    // fp16 logits regardless of the mask's own dtype).
    torch::Tensor legal = torch::from_blob(legal_buf.data(), {n, clines::kActions})
                              .to(torch::kByte)
                              .to(dev);
    torch::Tensor logits = net.forward({obs}).toTensor();
    if (V == 1 && args.canon) {
      // logits are in each game's canonical frame; bring them back to the original frame.
      torch::Tensor views = torch::from_blob(canon_view.data(), {n}, torch::kLong).to(dev);
      logits = torch::gather(logits.to(torch::kFloat), 1, act_all.index_select(0, views));
    }
    if (V > 1) {
      // (n*V, 6561) -> (n, V, 6561); gather each view's logit for every ORIGINAL
      // action, average over views in fp32 (== geometric mean of legal probs).
      logits = logits.to(torch::kFloat).view({n, V, clines::kActions});
      logits = torch::gather(logits, 2, act_idx.unsqueeze(0).expand({n, V, clines::kActions}))
                   .mean(1);
    }
    float ninf = -std::numeric_limits<float>::infinity();
    torch::Tensor moves = logits.masked_fill(legal == 0, ninf).argmax(1).to(torch::kCPU);
    auto mv = moves.accessor<int64_t, 1>();
    ++fwd;
    if (args.progress_forwards > 0 && fwd % args.progress_forwards == 0) {
      size_t longest = 0;
      for (size_t i = 1; i < slots.size(); ++i)
        if (slots[i].game.turns() > slots[longest].game.turns()) longest = i;
      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      std::printf("  [%ld forwards] %zu/%zu done, %d in flight, longest seed=%llu turn=%d "
                  "score=%d  elapsed=%.0fs\n",
                  fwd, done, todo.size(), n, (unsigned long long)slots[longest].seed,
                  slots[longest].game.turns(), slots[longest].game.score(), el);
      std::fflush(stdout);
    }

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
        if (recording) {
          TurnRec tr = SnapTurn(slots[i].game, (int)m);
          if (tr.turn % args.record_every == 0) slots[i].broad.push_back(tr);
          if ((int)slots[i].ring.size() < args.record_tail) {
            slots[i].ring.push_back(tr);
          } else {
            slots[i].ring[slots[i].ring_head] = tr;
            slots[i].ring_head = (slots[i].ring_head + 1) % slots[i].ring.size();
          }
        }
        int s = (int)(m / 81), t = (int)(m % 81);
        bool ok = slots[i].game.Move(s / 9, s % 9, t / 9, t % 9);
        dead = !ok || slots[i].game.over() || slots[i].game.turns() >= slots[i].turn_limit;
        if (args.verbose_games && !dead &&
            slots[i].game.turns() >= slots[i].next_progress_turn) {
          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          std::printf("    seed=%llu turn=%d score=%d elapsed=%.0fs\n",
                      (unsigned long long)slots[i].seed,
                      slots[i].game.turns(), slots[i].game.score(), el);
          std::fflush(stdout);
          while (slots[i].next_progress_turn <= slots[i].game.turns())
            slots[i].next_progress_turn += 500;
        }
      }
      if (dead) {
        const bool capped = (!slots[i].game.over()
                             && slots[i].game.turns() >= slots[i].turn_limit);
        if (recording)
          WriteGameRecord(args, slots[i].seed, slots[i].game.score(),
                          slots[i].game.turns(), slots[i].game.over(),
                          slots[i].broad, slots[i].ring, slots[i].ring_head);
        scores.push_back(slots[i].game.score());
        score_sum += slots[i].game.score();
        turn_sum += slots[i].game.turns();
        capped_games += capped;
        ++done;
        if (scores_file) {
          // Anchor mode: `score` is gained since the anchor (SetState starts it at 0) and
          // `turns` is the absolute turn; survived = turns - start_turn.
          std::fprintf(scores_file, "%llu,%d,%d,%d\n",
                       (unsigned long long)slots[i].seed,
                       slots[i].game.score(), slots[i].game.turns(),
                       capped ? 1 : 0);
          std::fflush(scores_file);
        }
        if (args.verbose_games) {
          double el = std::chrono::duration<double>(Clock::now() - t0).count();
          double rate = done / std::max(el, 1e-9);
          double eta = (todo.size() - done) / std::max(rate, 1e-9);
          std::printf("  game %zu/%zu seed=%llu score=%d turns=%d "
                      "elapsed=%.0fs ETA=%.0fs\n",
                      done, todo.size(), (unsigned long long)slots[i].seed,
                      slots[i].game.score(), slots[i].game.turns(), el, eta);
          std::fflush(stdout);
        }
        Slot repl{0, Game(0)};
        if (make_slot(repl)) survivors.push_back(std::move(repl));
      } else {
        survivors.push_back(std::move(slots[i]));
      }
    }
    slots.swap(survivors);

    if (done >= log_next) {
      double el = std::chrono::duration<double>(Clock::now() - t0).count();
      double rate = done / std::max(el, 1e-9);
      double eta = (todo.size() - done) / std::max(rate, 1e-9);
      // Completion order is length-biased while a batched run is in flight;
      // this is an observability number, not an interim policy estimate.
      std::printf("  %zu/%zu games  completed_mean=%.0f  elapsed=%.0fs "
                  "ETA=%.0fs  %ld forwards\n",
                  done, todo.size(), score_sum / done, el, eta, fwd);
      std::fflush(stdout);
      while (log_next <= done)
        log_next += std::max(1, args.progress_every);
    }

  }

  double el = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("\ndone: %zu games in %.1fs (%.0f games/s, %ld forwards, batch=%d, %s %s)\n",
              scores.size(), el, scores.size() / el, fwd, B, args.device.c_str(),
              use_half ? "fp16" : "fp32");
  char tag[64];
  std::snprintf(tag, sizeof(tag), "scores [%llu,%llu):",
                (unsigned long long)args.seed_start, (unsigned long long)args.seed_end);
  Percentile(scores, tag);
  if (anchor_mode)
    std::printf("  anchors reaching their turn cap (escaped): %zu (%.2f%%)\n", capped_games,
                100.0 * capped_games / std::max<size_t>(scores.size(), 1));
  else
    std::printf("  mean turns=%.1Lf  capped@%ld: %zu (%.3f%%)\n",
                turn_sum / std::max<size_t>(scores.size(), 1), args.max_turns,
                capped_games, 100.0 * capped_games
                / std::max<size_t>(scores.size(), 1));
  if (scores_file) {
    std::fclose(scores_file);
    std::printf("score-distribution CSV: %s\n", args.scores_out.c_str());
  }
  return 0;
}
