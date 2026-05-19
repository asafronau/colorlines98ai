"""Profile MCTS search — aggregate timing approach.

Instead of measuring each perf_counter in the inner loop (which adds 27ms
overhead from ~3000 timer calls), measure by *disabling* one component at a
time and comparing total search time.

Also measures with the actual real model on MPS to get end-to-end numbers.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import time
import math
import numpy as np
import torch
from game.board import ColorLinesGame, _label_empty_components, _clear_lines_at, _count_empty, _get_empty_array
from game.rng import SimpleRng
from alphatrain.mcts import (
    MCTS, Node, _build_obs_for_game, _get_legal_priors_flat,
    _legal_priors_jit, VIRTUAL_LOSS, NUM_MOVES
)
from alphatrain.observation import build_observation

BOARD_SIZE = 9

# ─── Warm up JIT ──────────────────────────────────────────────────────
print("Warming up JIT...", flush=True)
g = ColorLinesGame(seed=42)
g.reset()
_build_obs_for_game(g)
_legal_priors_jit(g.board, np.zeros(NUM_MOVES, dtype=np.float32), 30)
build_observation(g.board, np.zeros(3, dtype=np.intp), np.zeros(3, dtype=np.intp),
                  np.zeros(3, dtype=np.intp), 0)
print("JIT warm-up done.\n", flush=True)

# ─── Find a mid-game board state ─────────────────────────────────────
game = ColorLinesGame(seed=7)
game.reset()
for i in range(20):
    if game.game_over:
        break
    moves = game.get_legal_moves()
    if not moves:
        break
    game.move(moves[i % len(moves)][0], moves[i % len(moves)][1])

n_empty = int(np.sum(game.board == 0))
n_balls = 81 - n_empty
legal_moves = game.get_legal_moves()
print(f"Board: turn={game.turns}, score={game.score}, "
      f"empty={n_empty}, balls={n_balls}, legal_moves={len(legal_moves)}", flush=True)
if game.game_over:
    print("ERROR: game is over, need a live game!", flush=True)
    exit(1)


# ─── DummyNet ─────────────────────────────────────────────────────────
class DummyNet:
    def __init__(self):
        self.pol = torch.randn(1, NUM_MOVES)
        self.val = torch.zeros(1, 201)
        self.num_value_bins = 201

    def __call__(self, x):
        bs = x.shape[0]
        return self.pol.expand(bs, -1), self.val.expand(bs, -1)

    def predict_value(self, val_logits, max_val=30000.0):
        bs = val_logits.shape[0]
        return torch.full((bs,), 500.0)

    def parameters(self):
        return iter([self.pol])


# ─── Measure DummyNet search times ────────────────────────────────────
print("\n=== DummyNet Search Timing (CPU-only costs) ===", flush=True)

for sims in [400, 800]:
    for bs in [8, 16]:
        dummy_net = DummyNet()
        mcts = MCTS(dummy_net, torch.device('cpu'), max_score=30000.0,
                     num_simulations=sims, batch_size=bs, top_k=30, c_puct=2.5)

        # Warmup
        for _ in range(3):
            g = game.clone()
            mcts.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)

        n_runs = 30
        times = []
        for _ in range(n_runs):
            g = game.clone()
            t0 = time.perf_counter()
            mcts.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)
            times.append(time.perf_counter() - t0)

        mean_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        per_sim_us = mean_ms * 1000 / sims
        print(f"  {sims} sims, bs={bs}: {mean_ms:.1f} +/- {std_ms:.1f} ms "
              f"({per_sim_us:.1f} us/sim)", flush=True)


# ─── Decomposition: measure cost of removing each component ──────────
print("\n\n=== Component Cost Estimation (substitution method) ===", flush=True)
print("Measure full search, then with each component made cheaper.\n", flush=True)

sims = 800
bs = 8

# Baseline: full search
dummy_net = DummyNet()
mcts = MCTS(dummy_net, torch.device('cpu'), max_score=30000.0,
            num_simulations=sims, batch_size=bs, top_k=30, c_puct=2.5)

for _ in range(3):
    g = game.clone()
    mcts.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)

n_runs = 30
baseline_times = []
for _ in range(n_runs):
    g = game.clone()
    t0 = time.perf_counter()
    mcts.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)
    baseline_times.append(time.perf_counter() - t0)
baseline_ms = np.mean(baseline_times) * 1000
print(f"Baseline: {baseline_ms:.1f} ms", flush=True)


# ─── Measure individual component costs with batch timing ────────────
print("\n\n=== Batch Timing of Individual Components ===", flush=True)

# 1. Measure clone + trusted_move cost directly
rng = SimpleRng(12345)
n_iter = 100000
t0 = time.perf_counter()
for _ in range(n_iter):
    g2 = game.clone(rng=rng)
t_clone = (time.perf_counter() - t0) / n_iter * 1e6
print(f"  clone(rng=shared): {t_clone:.2f} us", flush=True)

# 2. trusted_move (need legal move)
s, t_mv = legal_moves[0]
t0 = time.perf_counter()
for _ in range(n_iter):
    g2 = game.clone(rng=rng)
    g2.trusted_move(s[0], s[1], t_mv[0], t_mv[1])
t_clone_plus_move = (time.perf_counter() - t0) / n_iter * 1e6
t_trusted_move = t_clone_plus_move - t_clone
print(f"  trusted_move: {t_trusted_move:.2f} us "
      f"(clone+move={t_clone_plus_move:.2f} us)", flush=True)

# 3. trusted_move that clears lines (different cost)
# Find a move that clears a line
clearing_move = None
for mv in legal_moves:
    g2 = game.clone(rng=rng)
    before = g2.score
    g2.trusted_move(mv[0][0], mv[0][1], mv[1][0], mv[1][1])
    if g2.score > before:
        clearing_move = mv
        break
if clearing_move:
    s_c, t_c = clearing_move
    t0 = time.perf_counter()
    for _ in range(n_iter):
        g2 = game.clone(rng=rng)
        g2.trusted_move(s_c[0], s_c[1], t_c[0], t_c[1])
    t_clear_move = (time.perf_counter() - t0) / n_iter * 1e6 - t_clone
    print(f"  trusted_move (clearing): {t_clear_move:.2f} us", flush=True)
else:
    print(f"  (no clearing move found)", flush=True)

# 4. Non-clearing trusted_move breakdown:
# The move calls: _clear_lines_at, _spawn_balls, _generate_next_balls, _count_empty
# _spawn_balls does: iterate next_balls, set board[r,c] = color, invalidate cc
# _generate_next_balls does: _get_empty_array, rng.choice_no_replace, rng.integers, list comp

# 5. build_observation
n_obs_iter = 50000
t0 = time.perf_counter()
for _ in range(n_obs_iter):
    _build_obs_for_game(game)
t_obs = (time.perf_counter() - t0) / n_obs_iter * 1e6
print(f"  _build_obs_for_game: {t_obs:.2f} us", flush=True)

# 6. _legal_priors_jit
fake_pol = np.random.randn(NUM_MOVES).astype(np.float32)
n_jit_iter = 50000
t0 = time.perf_counter()
for _ in range(n_jit_iter):
    _legal_priors_jit(game.board, fake_pol, 30)
t_jit = (time.perf_counter() - t0) / n_jit_iter * 1e6
print(f"  _legal_priors_jit: {t_jit:.2f} us", flush=True)

# 7. _get_legal_priors_flat (JIT + Python dict)
t0 = time.perf_counter()
for _ in range(n_jit_iter):
    _get_legal_priors_flat(game.board, fake_pol, 30)
t_flat = (time.perf_counter() - t0) / n_jit_iter * 1e6
t_dict_overhead = t_flat - t_jit
print(f"  _get_legal_priors_flat: {t_flat:.2f} us "
      f"(dict overhead: {t_dict_overhead:.2f} us)", flush=True)

# 8. Node expansion: 30 Nodes with int/float conversion
k_arr = np.arange(30, dtype=np.int32)
p_arr = np.full(30, 1.0/30, dtype=np.float32)
n_exp_iter = 50000
t0 = time.perf_counter()
for _ in range(n_exp_iter):
    ch = {}
    for i in range(30):
        ch[int(k_arr[i])] = Node(prior=float(p_arr[i]))
t_node_expand = (time.perf_counter() - t0) / n_exp_iter * 1e6
print(f"  Node expand (30 children): {t_node_expand:.2f} us", flush=True)

# 9. PUCT selection
parent_node = Node()
parent_node.visit_count = 200
for i in range(30):
    c = Node(prior=1.0/30)
    c.visit_count = max(1, i)
    c.value_sum = float(i * 50.0)
    parent_node.children[i] = c

n_puct_iter = 100000
t0 = time.perf_counter()
for _ in range(n_puct_iter):
    best_score = -1e30
    best_action = 0
    best_child = None
    sqrt_parent = math.sqrt(parent_node.visit_count)
    q_range = 500.0
    for act_i, child in parent_node.children.items():
        vc = child.visit_count
        if vc > 0:
            q = child.value_sum / vc
            q_norm = (q + 100.0) / q_range
        else:
            q_norm = 0.5
        u = 2.5 * child.prior * sqrt_parent / (1 + vc)
        score = q_norm + u
        if score > best_score:
            best_score = score
            best_action = act_i
            best_child = child
t_puct = (time.perf_counter() - t0) / n_puct_iter * 1e6
print(f"  PUCT select (30 children): {t_puct:.2f} us", flush=True)

# 10. SimpleRng methods
rng_test = SimpleRng(42)
n_rng_iter = 500000
t0 = time.perf_counter()
for _ in range(n_rng_iter):
    rng_test.next_u64()
t_u64 = (time.perf_counter() - t0) / n_rng_iter * 1e6
print(f"  rng.next_u64: {t_u64:.3f} us", flush=True)

t0 = time.perf_counter()
for _ in range(n_rng_iter):
    rng_test.randint(0, 60)
t_randint = (time.perf_counter() - t0) / n_rng_iter * 1e6
print(f"  rng.randint: {t_randint:.3f} us", flush=True)


# ─── Estimated per-search breakdown (800 sims, bs=8) ─────────────────
print("\n\n=== Estimated Per-Search Breakdown (800 sims, bs=8) ===", flush=True)

# From prior profiling we know tree depth averages ~2.2 moves per sim
# That means per simulation:
#   - 1 clone
#   - ~2.2 PUCT selections
#   - ~2.2 trusted_move calls (but only ~1.2 non-root, since first is always root->child)
#   - 1 obs build
#   - 1 legal_priors_jit call
#   - 1 Node expansion
#   - ~3.2 backup steps

avg_depth = 2.2
total_moves = sims * avg_depth
total_puct = sims * avg_depth

est_clone = sims * t_clone / 1000
est_puct = total_puct * t_puct / 1000
est_move = total_moves * t_trusted_move / 1000
est_obs = sims * t_obs / 1000
est_priors = sims * t_jit / 1000
est_expand = sims * t_node_expand / 1000
est_total = est_clone + est_puct + est_move + est_obs + est_priors + est_expand

print(f"Estimated CPU time: {est_total:.1f} ms", flush=True)
print(f"  clone:           {est_clone:6.1f} ms ({est_clone/est_total*100:5.1f}%)", flush=True)
print(f"  PUCT ({total_puct:.0f}x):   {est_puct:6.1f} ms ({est_puct/est_total*100:5.1f}%)", flush=True)
print(f"  trusted_move:    {est_move:6.1f} ms ({est_move/est_total*100:5.1f}%)", flush=True)
print(f"  obs build:       {est_obs:6.1f} ms ({est_obs/est_total*100:5.1f}%)", flush=True)
print(f"  legal priors:    {est_priors:6.1f} ms ({est_priors/est_total*100:5.1f}%)", flush=True)
print(f"  Node expand:     {est_expand:6.1f} ms ({est_expand/est_total*100:5.1f}%)", flush=True)
print(f"\nMeasured baseline: {baseline_ms:.1f} ms", flush=True)
gap = baseline_ms - est_total
print(f"Gap (overhead): {gap:.1f} ms ({gap/baseline_ms*100:.0f}%)", flush=True)


# ─── SimpleRng pure-Python hotspot analysis ───────────────────────────
print("\n\n=== SimpleRng in trusted_move: RNG call count ===", flush=True)

# A non-clearing trusted_move calls:
#   _clear_lines_at: 0 RNG calls
#   _spawn_balls: 0-3 RNG calls (only if ball position occupied, calls randint)
#   _generate_next_balls: calls _get_empty_array (0 RNG),
#       choice_no_replace(n, 3): 3 RNG calls
#       integers(1, 8, 3): 3 RNG calls
# Total: 6 RNG calls minimum per non-clearing move
# Each RNG call: next_u64 + Python arithmetic = ~0.3 us
# 6 calls = ~1.8 us of the ~5 us trusted_move cost

# With 800 sims * 2.2 moves = 1760 moves * ~6 RNG calls = 10,560 RNG calls
rng_calls_per_sim = avg_depth * 6
total_rng = sims * rng_calls_per_sim * t_u64 / 1000
print(f"  RNG calls per move: ~6", flush=True)
print(f"  Total RNG calls per search: ~{sims * rng_calls_per_sim:.0f}", flush=True)
print(f"  Total RNG time: ~{total_rng:.1f} ms ({total_rng/baseline_ms*100:.0f}% of search)", flush=True)


# ─── What if we made SimpleRng a Numba jitclass? ─────────────────────
print("\n\n=== Potential: Numba jitclass RNG ===", flush=True)
from numba import njit
from numba.experimental import jitclass
from numba import types

MASK64_NB = np.uint64(0xFFFFFFFFFFFFFFFF)

@njit(cache=True)
def splitmix64_next(state):
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & MASK64_NB
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64_NB
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64_NB
    return state, (z ^ (z >> np.uint64(31))) & MASK64_NB

# Warmup
_ = splitmix64_next(np.uint64(42))

n_jit_rng = 1000000
t0 = time.perf_counter()
state = np.uint64(42)
for _ in range(n_jit_rng):
    state, val = splitmix64_next(state)
t_jit_u64 = (time.perf_counter() - t0) / n_jit_rng * 1e6
print(f"  JIT splitmix64: {t_jit_u64:.3f} us (vs Python {t_u64:.3f} us, "
      f"{t_u64/t_jit_u64:.1f}x faster)", flush=True)

# But the real overhead is Python method dispatch, not the math
# A jitclass or Cython RNG would eliminate the method call overhead


# ─── What about a fully JIT trusted_move? ─────────────────────────────
print("\n\n=== Potential: Full JIT trusted_move ===", flush=True)
print("trusted_move is ~5 us because:", flush=True)
print("  1. Python method call overhead (~0.3 us)", flush=True)
print("  2. _clear_lines_at JIT call (~0.5 us)", flush=True)
print("  3. _spawn_balls Python loop (~0.5 us)", flush=True)
print("  4. _generate_next_balls Python (~2.5 us)", flush=True)
print("     - _get_empty_array JIT: 0.25 us", flush=True)
print("     - rng.choice_no_replace: 1.16 us (Python list + RNG)", flush=True)
print("     - rng.integers: 0.97 us (Python list + RNG)", flush=True)
print("     - list comprehension: ~0.3 us", flush=True)
print("  5. _count_empty JIT: ~0.1 us", flush=True)
print(f"\n  Making RNG pure-JIT would save ~2 us per move = "
      f"~{sims * avg_depth * 2 / 1000:.1f} ms per search", flush=True)


# ─── MPS model test (if available) ───────────────────────────────────
if torch.backends.mps.is_available():
    print("\n\n=== Real Model on MPS ===", flush=True)
    device = torch.device('mps')
    model_path = 'alphatrain/data/pillar2u_epoch_9.pt'
    if os.path.exists(model_path):
        from alphatrain.evaluate import load_model
        net, max_score = load_model(model_path, device, fp16=True, jit_trace=True)

        for sims_test in [400, 800]:
            mcts_real = MCTS(net, device, max_score=max_score,
                             num_simulations=sims_test, batch_size=8,
                             top_k=30, c_puct=2.5)

            # Warmup
            for _ in range(2):
                g = game.clone()
                mcts_real.search(g)

            n_runs = 10
            times = []
            for _ in range(n_runs):
                g = game.clone()
                t0 = time.perf_counter()
                mcts_real.search(g, temperature=1.0,
                                 dirichlet_alpha=0.3, dirichlet_weight=0.25)
                times.append(time.perf_counter() - t0)

            mean_ms = np.mean(times) * 1000
            std_ms = np.std(times) * 1000
            print(f"  {sims_test} sims, bs=8: {mean_ms:.1f} +/- {std_ms:.1f} ms "
                  f"({mean_ms/sims_test*1000:.1f} us/sim)", flush=True)
    else:
        print(f"  Model not found: {model_path}", flush=True)
else:
    print("\n  (MPS not available)", flush=True)


print("\n\nDone.", flush=True)
