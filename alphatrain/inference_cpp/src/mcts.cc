#include "mcts.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <utility>

namespace clines {
namespace {

constexpr double kVirtualLoss = 1.0;  // mcts.py VIRTUAL_LOSS
constexpr int kDr4[4] = {0, 0, 1, -1};
constexpr int kDc4[4] = {1, -1, 0, 0};

struct Node {
  std::vector<std::pair<int, Node*>> children;  // (flat action, child)
  int n = 0;       // visit_count
  double w = 0.0;  // value_sum
  double p = 0.0;  // prior
};

// Deterministic per-state seed for the sim RNG (Python: MD5 of state -> PCG64;
// we use FNV-1a -> SplitMix64. Different stream by design, same role).
uint64_t StateSeed(const Game& g) {
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&h](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
  for (int i = 0; i < kNN; ++i) mix(static_cast<uint8_t>(g.board()[i]));
  for (const NextBall& nb : g.next_balls()) {
    mix(nb.r); mix(nb.c); mix(nb.color);
  }
  mix(static_cast<uint64_t>(g.score()));
  mix(static_cast<uint64_t>(g.turns()));
  return h;
}

// Is `tgt`'s empty-component adjacent to `src`? (mcts.py::_is_reachable)
bool ReachableFrom(const int8_t* labels, int src, int tgt) {
  int8_t tl = labels[tgt];
  if (tl <= 0) return false;
  int sr = src / kN, sc = src % kN;
  for (int d = 0; d < 4; ++d) {
    int nr = sr + kDr4[d], nc = sc + kDc4[d];
    if (nr >= 0 && nr < kN && nc >= 0 && nc < kN && labels[nr * kN + nc] == tl)
      return true;
  }
  return false;
}

}  // namespace

int LegalPriors(const int8_t* board, const float* logits, int top_k,
                int* out_actions, double* out_priors) {
  int8_t labels[kNN];
  Game::LabelEmpty(board, labels);

  // Collect every legal (logit, action). Legality = src occupied, tgt empty,
  // tgt's empty-component adjacent to src (same rule as Game::LegalMask).
  std::vector<std::pair<float, int>> legal;
  legal.reserve(256);
  for (int s = 0; s < kNN; ++s) {
    if (board[s] == 0) continue;
    int sr = s / kN, sc = s % kN;
    int8_t nbr[4];
    int nn = 0;
    for (int d = 0; d < 4; ++d) {
      int nr = sr + kDr4[d], nc = sc + kDc4[d];
      if (nr < 0 || nr >= kN || nc < 0 || nc >= kN) continue;
      int8_t lb = labels[nr * kN + nc];
      if (lb <= 0) continue;
      bool seen = false;
      for (int i = 0; i < nn; ++i) if (nbr[i] == lb) { seen = true; break; }
      if (!seen) nbr[nn++] = lb;
    }
    if (nn == 0) continue;
    for (int t = 0; t < kNN; ++t) {
      if (board[t] != 0) continue;
      int8_t tl = labels[t];
      for (int i = 0; i < nn; ++i) {
        if (nbr[i] == tl) {
          int a = s * kNN + t;
          legal.push_back({logits[a], a});
          break;
        }
      }
    }
  }
  if (legal.empty()) return 0;

  int k = std::min<int>(top_k, static_cast<int>(legal.size()));
  std::partial_sort(legal.begin(), legal.begin() + k, legal.end(),
                    [](const std::pair<float, int>& a,
                       const std::pair<float, int>& b) { return a.first > b.first; });
  // Softmax over just the k selected logits (max-subtract; legal[0] is max).
  double vmax = legal[0].first, sum = 0.0;
  for (int i = 0; i < k; ++i) {
    double e = std::exp(static_cast<double>(legal[i].first) - vmax);
    out_actions[i] = legal[i].second;
    out_priors[i] = e;
    sum += e;
  }
  for (int i = 0; i < k; ++i) out_priors[i] /= sum;
  return k;
}

SearchResult MCTS::Search(const Game& game, double temperature,
                          SimpleRng& move_rng) {
  SearchResult res;
  std::deque<Node> pool;  // per-search node arena (deque: stable pointers)
  pool.emplace_back();
  Node* root = &pool.back();

  SimpleRng sim_rng(StateSeed(game));

  // --- Root: policy priors + feature value ---
  std::vector<float> obs(18 * kNN);
  game.BuildObs(obs.data());
  std::vector<float> root_logits(kActions);
  policy_(obs.data(), 1, root_logits.data());

  std::vector<int> acts(cfg_.top_k);
  std::vector<double> pris(cfg_.top_k);
  int k = LegalPriors(game.board().data(), root_logits.data(), cfg_.top_k,
                      acts.data(), pris.data());
  if (k == 0) return res;  // no legal moves: action stays -1

  double root_value = fe_->Value(game.board().data(), game.next_balls());
  root->children.reserve(k);
  for (int i = 0; i < k; ++i) {
    pool.emplace_back();
    pool.back().p = pris[i];
    root->children.push_back({acts[i], &pool.back()});
  }
  root->n = 1;          // mcts.py: root.visit_count = 1
  root->w = root_value;
  double min_q = root_value, max_q = root_value;

  // --- Batched virtual-loss simulation loop ---
  const int B = cfg_.batch_size;
  std::vector<float> obs_buf(static_cast<size_t>(B) * 18 * kNN);
  std::vector<float> logits_buf(static_cast<size_t>(B) * kActions);
  std::vector<std::vector<Node*>> paths(B);
  std::vector<Node*> leaves(B);
  std::vector<Game> leaf_games;
  leaf_games.reserve(B);
  std::vector<char> over_flags(B);
  std::vector<int> nn_slot(B);

  int sims_done = 0;
  const int num_sims = cfg_.num_simulations;
  while (sims_done < num_sims) {
    int bs = std::min(B, num_sims - sims_done);
    leaf_games.clear();
    int obs_count = 0;

    // === SELECT bs leaves with virtual loss ===
    for (int b = 0; b < bs; ++b) {
      Node* node = root;
      Game sim = game;  // clone; spawns come from the shared sim_rng below
      auto& path = paths[b];
      path.clear();
      path.push_back(root);
      int depth = 0;

      while (!node->children.empty() && !sim.over()) {
        double sqrt_parent = std::sqrt(static_cast<double>(node->n));
        double q_range = max_q - min_q;
        const int8_t* board = sim.board().data();
        bool need_filter = depth > 0;  // root children always valid at root
        std::vector<int> banned;
        int8_t cc[kNN];
        bool have_cc = false;
        Node* best_child = nullptr;
        int best_action = 0;

        // Open-loop PUCT with lazy reachability: cheap occupancy filter on all
        // children, full reachability check only on the argmax; ban + retry.
        while (true) {
          double best_score = -1e30;
          best_child = nullptr;
          for (const auto& [act, ch] : node->children) {
            if (need_filter) {
              if (!banned.empty() &&
                  std::find(banned.begin(), banned.end(), act) != banned.end())
                continue;
              if (board[act / kNN] == 0 || board[act % kNN] != 0) continue;
            }
            double qn;
            if (ch->n > 0) {
              double q = ch->w / ch->n;
              qn = q_range > 0 ? (q - min_q) / q_range : 0.5;
            } else {
              qn = 0.5;
            }
            double sc = cfg_.q_weight * qn +
                        cfg_.c_puct * ch->p * sqrt_parent / (1.0 + ch->n);
            if (sc > best_score) {
              best_score = sc;
              best_child = ch;
              best_action = act;
            }
          }
          if (best_child == nullptr) break;   // no legal children on this board
          if (!need_filter) break;            // root: always valid
          if (!have_cc) { Game::LabelEmpty(board, cc); have_cc = true; }
          if (ReachableFrom(cc, best_action / kNN, best_action % kNN)) break;
          banned.push_back(best_action);      // unreachable: ban and retry
        }
        if (best_child == nullptr) break;

        int src = best_action / kNN, tgt = best_action % kNN;
        sim.TrustedMove(src / kN, src % kN, tgt / kN, tgt % kN, sim_rng);
        path.push_back(best_child);
        node = best_child;
        ++depth;
      }

      // Virtual loss on the whole path (canceled at backup).
      for (Node* pn : path) { pn->n += 1; pn->w -= kVirtualLoss; }

      leaves[b] = node;
      over_flags[b] = sim.over() ? 1 : 0;
      if (!sim.over()) {
        sim.BuildObs(obs_buf.data() + static_cast<size_t>(obs_count) * 18 * kNN);
        nn_slot[b] = obs_count++;
      } else {
        nn_slot[b] = -1;
      }
      leaf_games.push_back(std::move(sim));
    }

    // === BATCH EVALUATE (policy only; value is the feature evaluator) ===
    if (obs_count > 0) policy_(obs_buf.data(), obs_count, logits_buf.data());

    // === EXPAND + BACKUP ===
    for (int b = 0; b < bs; ++b) {
      const Game& lg = leaf_games[b];
      if (!over_flags[b]) {
        Node* node = leaves[b];
        int kk = LegalPriors(
            lg.board().data(),
            logits_buf.data() + static_cast<size_t>(nn_slot[b]) * kActions,
            cfg_.top_k, acts.data(), pris.data());
        for (int i = 0; i < kk; ++i) {
          bool exists = false;
          for (const auto& pr : node->children)
            if (pr.first == acts[i]) { exists = true; break; }
          if (!exists) {
            pool.emplace_back();
            pool.back().p = pris[i];
            node->children.push_back({acts[i], &pool.back()});
          }
        }
      }
      // Leaf value = feature evaluator on the leaf board (terminal included —
      // matches mcts.py's feature_coefs branch for both cases).
      double value = fe_->Value(lg.board().data(), lg.next_balls());
      if (value < min_q) min_q = value;
      if (value > max_q) max_q = value;
      for (Node* pn : paths[b]) pn->w += kVirtualLoss + value;
    }
    sims_done += bs;

    // Exact greedy-action early stop (eval only): if the most-visited root
    // child can't be overtaken by the runner-up with the remaining sims, the
    // final argmax is fixed. (mcts.py — eval-only; selfplay needs full visits.)
    if (cfg_.early_stop && temperature == 0.0) {
      int v1 = -1, v2 = -1;
      for (const auto& [act, ch] : root->children) {
        (void)act;
        if (ch->n > v1) { v2 = v1; v1 = ch->n; }
        else if (ch->n > v2) v2 = ch->n;
      }
      if (v1 - v2 > num_sims - sims_done) break;
    }
  }

  // --- Result: action + recorded root stats ---
  // Argmax by visits in insertion order (first max wins, like np.argmax).
  int best_visits = -1, best_action = -1;
  for (const auto& [act, ch] : root->children) {
    if (ch->n > best_visits) { best_visits = ch->n; best_action = act; }
  }
  res.cands.reserve(root->children.size());
  for (const auto& [act, ch] : root->children)
    res.cands.push_back({act, ch->n, ch->p, ch->n > 0 ? ch->w / ch->n : 0.0});
  std::stable_sort(res.cands.begin(), res.cands.end(),
                   [](const Candidate& a, const Candidate& b) {
                     return a.visits > b.visits;
                   });
  res.root_value = root_value;
  res.q_min = min_q;
  res.q_max = max_q;

  if (temperature <= 0.0) {
    res.action = best_action;
  } else {
    // Sample proportional to visits^(1/T).
    double inv_t = 1.0 / temperature, sum = 0.0;
    std::vector<double> w(res.cands.size());
    for (size_t i = 0; i < res.cands.size(); ++i) {
      w[i] = std::pow(static_cast<double>(res.cands[i].visits), inv_t);
      sum += w[i];
    }
    if (sum <= 0.0) {
      res.action = best_action;
    } else {
      double r = move_rng.NextF64() * sum, acc = 0.0;
      res.action = res.cands.back().action;
      for (size_t i = 0; i < res.cands.size(); ++i) {
        acc += w[i];
        if (r <= acc) { res.action = res.cands[i].action; break; }
      }
    }
  }
  return res;
}

}  // namespace clines
