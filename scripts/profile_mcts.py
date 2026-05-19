"""Profile MCTS search CPU-side costs in detail.

Measures per-simulation time breakdown: clone, trusted_move, obs build,
legal_priors_jit, Node creation, PUCT selection, IPC overhead.

Uses a DummyNet to isolate CPU costs from GPU/server latency.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import time
import math
import numpy as np
import torch
from game.board import ColorLinesGame
from game.rng import SimpleRng
from alphatrain.mcts import (
    MCTS, Node, _build_obs_for_game, _get_legal_priors_flat,
    _legal_priors_jit, _flat_to_action, VIRTUAL_LOSS, NUM_MOVES
)
from alphatrain.observation import build_observation

BOARD_SIZE = 9


# ─── Warm up JIT ──────────────────────────────────────────────────────
print("Warming up JIT...", flush=True)
g = ColorLinesGame(seed=42)
g.reset()
_build_obs_for_game(g)
_legal_priors_jit(g.board, np.zeros(NUM_MOVES, dtype=np.float32), 30)
print("JIT warm-up done.\n", flush=True)


# ─── Microbenchmark individual operations ─────────────────────────────
def bench(fn, n=10000, label=""):
    """Benchmark fn() over n iterations, return mean us."""
    # Warmup
    for _ in range(min(n, 100)):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - t0
    us = elapsed / n * 1e6
    print(f"  {label}: {us:.2f} us/call ({n} iters)", flush=True)
    return us


print("=== Microbenchmarks (per-call costs) ===", flush=True)

# Setup: find a seed that gives a mid-game board with decent density
# We need the game to be alive (not game_over) with ~30-50 balls
game = None
for test_seed in range(100):
    g = ColorLinesGame(seed=test_seed)
    g.reset()
    # Play some turns using random legal moves
    for _ in range(30):
        if g.game_over:
            break
        moves = g.get_legal_moves()
        if not moves:
            break
        # Pick a random move
        idx = test_seed % len(moves)
        g.move(moves[idx][0], moves[idx][1])
    if not g.game_over:
        n_empty = int(np.sum(g.board == 0))
        if 20 <= n_empty <= 55:
            game = g
            print(f"Using seed={test_seed}", flush=True)
            break
if game is None:
    # Fallback: just use a fresh game
    game = ColorLinesGame(seed=42)
    game.reset()

print(f"Board state: turn={game.turns}, score={game.score}, "
      f"empty={np.sum(game.board == 0)}", flush=True)

# 1. game.clone()
rng = SimpleRng(12345)
bench(lambda: game.clone(rng=rng), n=50000, label="game.clone(rng=shared)")

# 2. game.clone() without shared RNG
bench(lambda: game.clone(), n=50000, label="game.clone(rng=new)")

# 3. SimpleRng creation
bench(lambda: SimpleRng(0), n=100000, label="SimpleRng(0)")

# 4. board.copy()
bench(lambda: game.board.copy(), n=100000, label="board.copy()")

# 5. list(next_balls)
bench(lambda: list(game.next_balls), n=100000, label="list(next_balls)")

# 6. _build_obs_for_game
bench(lambda: _build_obs_for_game(game), n=20000, label="_build_obs_for_game")

# 7. build_observation directly
nr = np.zeros(3, dtype=np.intp)
nc = np.zeros(3, dtype=np.intp)
ncol = np.zeros(3, dtype=np.intp)
for i, ((r, c), col) in enumerate(game.next_balls[:3]):
    nr[i], nc[i], ncol[i] = r, c, col
bench(lambda: build_observation(game.board, nr, nc, ncol, len(game.next_balls)),
      n=20000, label="build_observation JIT")

# 8. _legal_priors_jit
fake_pol = np.random.randn(NUM_MOVES).astype(np.float32)
bench(lambda: _legal_priors_jit(game.board, fake_pol, 30),
      n=20000, label="_legal_priors_jit")

# 9. _get_legal_priors_flat (includes Python dict construction)
bench(lambda: _get_legal_priors_flat(game.board, fake_pol, 30),
      n=20000, label="_get_legal_priors_flat")

# 10. trusted_move on a clone
def bench_trusted_move():
    g2 = game.clone(rng=rng)
    moves = game.get_legal_moves()
    if moves:
        s, t = moves[0]
        g2.trusted_move(s[0], s[1], t[0], t[1])
precomputed_moves = game.get_legal_moves()
if precomputed_moves:
    s0, t0_mv = precomputed_moves[0]
    def bench_trusted_only():
        g2 = game.clone(rng=rng)
        g2.trusted_move(s0[0], s0[1], t0_mv[0], t0_mv[1])
    bench(bench_trusted_only, n=20000, label="clone+trusted_move")

# 11. Node creation
bench(lambda: Node(prior=0.5), n=100000, label="Node(prior=0.5)")

# 12. dict operations (simulating child expansion)
def bench_dict_expand():
    ch = {}
    for i in range(30):
        ch[i] = Node(prior=0.03)
bench(bench_dict_expand, n=10000, label="dict expand 30 children")

# 13. PUCT selection over 30 children
parent = Node()
parent.visit_count = 200
for i in range(30):
    c = Node(prior=1.0/30)
    c.visit_count = int(np.random.randint(1, 20))
    c.value_sum = float(np.random.randn() * 100)
    parent.children[i] = c

def bench_puct():
    best_score = -1e30
    best_action = 0
    best_child = None
    sqrt_parent = math.sqrt(parent.visit_count)
    min_q, max_q = -100.0, 500.0
    q_range = max_q - min_q
    c_puct = 2.5
    for act_i, child in parent.children.items():
        vc = child.visit_count
        if vc > 0:
            q = child.value_sum / vc
            q_norm = (q - min_q) / q_range if q_range > 0 else 0.5
        else:
            q_norm = 0.5
        u = c_puct * child.prior * sqrt_parent / (1 + vc)
        score = q_norm + u
        if score > best_score:
            best_score = score
            best_action = act_i
            best_child = child

bench(bench_puct, n=50000, label="PUCT select (30 children)")

# 14. path backup
path = [parent] + list(parent.children.values())[:5]
def bench_backup():
    for n in path:
        n.value_sum += VIRTUAL_LOSS + 100.0
bench(bench_backup, n=50000, label="backup (5 nodes)")

# 15. Dirichlet sampling
def bench_dirichlet():
    rng2 = SimpleRng(99)
    rng2.dirichlet([0.3] * 30)
bench(bench_dirichlet, n=5000, label="dirichlet(0.3, k=30)")

# 16. tobytes + state_seed computation
def bench_seed():
    board_bytes = game.board.tobytes()
    state_seed = int.from_bytes(board_bytes[:8], 'little')
    state_seed = (state_seed ^ (game.score * 31) ^ (game.turns * 7)) & 0xFFFFFFFF
bench(bench_seed, n=100000, label="state_seed computation")


print("\n\n=== Full Search Profile (DummyNet, 800 sims, bs=8) ===", flush=True)

# Create a DummyNet that returns realistic-shaped outputs
class DummyNet:
    def __init__(self):
        self.pol = torch.randn(1, NUM_MOVES)
        self.val = torch.zeros(1, 201)  # 201 bins
        self.num_value_bins = 201

    def __call__(self, x):
        bs = x.shape[0]
        return self.pol.expand(bs, -1), self.val.expand(bs, -1)

    def predict_value(self, val_logits, max_val=30000.0):
        bs = val_logits.shape[0]
        return torch.full((bs,), 500.0)

    def parameters(self):
        return iter([self.pol])


dummy_net = DummyNet()
mcts = MCTS(dummy_net, torch.device('cpu'), max_score=30000.0,
            num_simulations=800, batch_size=8, top_k=30, c_puct=2.5)

# Run search multiple times to get stable timing
n_searches = 20
times = []
for i in range(n_searches):
    game2 = game.clone()

    t0 = time.perf_counter()
    action = mcts.search(game2, temperature=1.0,
                         dirichlet_alpha=0.3, dirichlet_weight=0.25)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    if (i + 1) % 5 == 0:
        print(f"  search {i+1}/{n_searches}: {elapsed*1000:.1f} ms", flush=True)

mean_ms = np.mean(times) * 1000
std_ms = np.std(times) * 1000
print(f"\nDummyNet search: {mean_ms:.1f} +/- {std_ms:.1f} ms "
      f"(min={np.min(times)*1000:.1f}, max={np.max(times)*1000:.1f})", flush=True)


# ─── Instrumented search to measure time in each phase ─────────────
print("\n\n=== Instrumented Search Breakdown (800 sims, bs=8) ===", flush=True)

# Reuse same game state from microbenchmarks
game3 = game.clone()

# Manually instrument the search
t_clone = 0
t_puct = 0
t_move = 0
t_obs = 0
t_nn = 0
t_priors = 0
t_expand = 0
t_backup = 0
t_other = 0
n_clones = 0
n_obs = 0
n_moves_total = 0
n_expansions = 0

root = Node()
board_bytes = game3.board.tobytes()
state_seed = int.from_bytes(board_bytes[:8], 'little')
state_seed = (state_seed ^ (game3.score * 31) ^ (game3.turns * 7)) & 0xFFFFFFFF
sim_rng = SimpleRng(state_seed)

# Expand root
t0 = time.perf_counter()
obs_np = _build_obs_for_game(game3)
t_obs += time.perf_counter() - t0
n_obs += 1

t0 = time.perf_counter()
obs_t = torch.from_numpy(obs_np).unsqueeze(0)
pol, val = dummy_net(obs_t)
root_value = dummy_net.predict_value(val, max_val=30000.0).item()
pol_np = pol[0].numpy()
t_nn += time.perf_counter() - t0

t0 = time.perf_counter()
priors = _get_legal_priors_flat(game3.board, pol_np, 30)
t_priors += time.perf_counter() - t0

for action, prior in priors.items():
    root.children[action] = Node(prior=prior)
root.visit_count = 1
root.value_sum = root_value

# Dirichlet noise
noise = sim_rng.dirichlet([0.3] * len(root.children))
for i, child in enumerate(root.children.values()):
    child.prior = (0.75 * child.prior + 0.25 * noise[i])

min_q = root_value
max_q = root_value
c_puct = 2.5
top_k = 30
batch_size = 8
num_sims = 800
obs_buf = torch.empty(batch_size, 18, 9, 9)

sims_done = 0
while sims_done < num_sims:
    bs = min(batch_size, num_sims - sims_done)
    batch_paths = []
    batch_games = []
    batch_leaf_nodes = []
    batch_game_over = []
    obs_count = 0

    for _ in range(bs):
        # Clone
        t0 = time.perf_counter()
        sim_game = game3.clone(rng=sim_rng)
        t_clone += time.perf_counter() - t0
        n_clones += 1

        node = root
        path = [node]

        while node.children and not sim_game.game_over:
            # PUCT selection
            t0 = time.perf_counter()
            best_score = -1e30
            best_action = 0
            best_child = None
            sqrt_parent = math.sqrt(node.visit_count)
            q_range = max_q - min_q
            for act_i, child in node.children.items():
                vc = child.visit_count
                if vc > 0:
                    q = child.value_sum / vc
                    q_norm = (q - min_q) / q_range if q_range > 0 else 0.5
                else:
                    q_norm = 0.5
                u = c_puct * child.prior * sqrt_parent / (1 + vc)
                score = q_norm + u
                if score > best_score:
                    best_score = score
                    best_action = act_i
                    best_child = child
            t_puct += time.perf_counter() - t0

            # Execute trusted move
            t0 = time.perf_counter()
            src_flat = best_action // 81
            tgt_flat = best_action % 81
            sim_game.trusted_move(
                src_flat // 9, src_flat % 9,
                tgt_flat // 9, tgt_flat % 9)
            t_move += time.perf_counter() - t0
            n_moves_total += 1

            path.append(best_child)
            node = best_child

        # Virtual loss
        t0 = time.perf_counter()
        for n in path:
            n.visit_count += 1
            n.value_sum -= VIRTUAL_LOSS
        t_backup += time.perf_counter() - t0

        batch_paths.append(path)
        batch_leaf_nodes.append(node)
        batch_games.append(sim_game)

        if sim_game.game_over:
            batch_game_over.append(True)
        else:
            batch_game_over.append(False)
            t0 = time.perf_counter()
            obs = _build_obs_for_game(sim_game)
            t_obs += time.perf_counter() - t0
            n_obs += 1
            obs_buf[obs_count] = torch.from_numpy(obs)
            obs_count += 1

    # Batch NN eval
    if obs_count > 0:
        t0 = time.perf_counter()
        pol_logits, val_logits = dummy_net(obs_buf[:obs_count])
        values_t = dummy_net.predict_value(val_logits, max_val=30000.0)
        pol_np_batch = pol_logits.numpy()
        val_np_batch = values_t.numpy()
        t_nn += time.perf_counter() - t0

    # Expand + backup
    nn_idx = 0
    for b in range(bs):
        path = batch_paths[b]

        if batch_game_over[b]:
            value = float(batch_games[b].score)
        else:
            value = float(val_np_batch[nn_idx])
            node = batch_leaf_nodes[b]

            t0 = time.perf_counter()
            k, flat_idx, pri = _legal_priors_jit(
                batch_games[b].board, pol_np_batch[nn_idx], top_k)
            t_priors += time.perf_counter() - t0

            t0 = time.perf_counter()
            ch = node.children
            for i in range(k):
                ch[int(flat_idx[i])] = Node(prior=float(pri[i]))
            t_expand += time.perf_counter() - t0
            n_expansions += 1
            nn_idx += 1

        min_q = min(min_q, value)
        max_q = max(max_q, value)

        t0 = time.perf_counter()
        for n in path:
            n.value_sum += VIRTUAL_LOSS + value
        t_backup += time.perf_counter() - t0

    sims_done += bs

total = t_clone + t_puct + t_move + t_obs + t_nn + t_priors + t_expand + t_backup

print(f"\nTotal instrumented time: {total*1000:.1f} ms", flush=True)
print(f"  clone:         {t_clone*1000:7.2f} ms ({t_clone/total*100:5.1f}%) "
      f"  [{n_clones} calls, {t_clone/n_clones*1e6:.1f} us/call]", flush=True)
print(f"  PUCT select:   {t_puct*1000:7.2f} ms ({t_puct/total*100:5.1f}%) "
      f"  [{n_moves_total} selections]", flush=True)
print(f"  trusted_move:  {t_move*1000:7.2f} ms ({t_move/total*100:5.1f}%) "
      f"  [{n_moves_total} moves, {t_move/max(n_moves_total,1)*1e6:.1f} us/move]", flush=True)
print(f"  obs build:     {t_obs*1000:7.2f} ms ({t_obs/total*100:5.1f}%) "
      f"  [{n_obs} calls, {t_obs/n_obs*1e6:.1f} us/call]", flush=True)
print(f"  NN forward:    {t_nn*1000:7.2f} ms ({t_nn/total*100:5.1f}%)", flush=True)
print(f"  legal priors:  {t_priors*1000:7.2f} ms ({t_priors/total*100:5.1f}%) "
      f"  [{n_expansions} calls, {t_priors/max(n_expansions,1)*1e6:.1f} us/call]", flush=True)
print(f"  Node expand:   {t_expand*1000:7.2f} ms ({t_expand/total*100:5.1f}%) "
      f"  [{n_expansions} expansions]", flush=True)
print(f"  backup:        {t_backup*1000:7.2f} ms ({t_backup/total*100:5.1f}%)", flush=True)

# Compute per-simulation budget
per_sim_us = total / num_sims * 1e6
print(f"\nPer-simulation budget: {per_sim_us:.1f} us", flush=True)
print(f"  clone:        {t_clone/num_sims*1e6:.1f} us", flush=True)
print(f"  PUCT:         {t_puct/num_sims*1e6:.1f} us", flush=True)
print(f"  trusted_move: {t_move/num_sims*1e6:.1f} us", flush=True)
print(f"  obs build:    {t_obs/num_sims*1e6:.1f} us", flush=True)
print(f"  NN forward:   {t_nn/num_sims*1e6:.1f} us", flush=True)
print(f"  legal priors: {t_priors/num_sims*1e6:.1f} us", flush=True)
print(f"  Node expand:  {t_expand/num_sims*1e6:.1f} us", flush=True)
print(f"  backup:       {t_backup/num_sims*1e6:.1f} us", flush=True)

# Tree depth statistics
def tree_depth(node, depth=0):
    if not node.children:
        return [depth]
    depths = []
    for child in node.children.values():
        if child.visit_count > 0:
            depths.extend(tree_depth(child, depth + 1))
    return depths if depths else [depth]

depths = tree_depth(root)
print(f"\nTree statistics:", flush=True)
print(f"  Mean depth: {np.mean(depths):.1f}, max: {max(depths)}", flush=True)
print(f"  Root children: {len(root.children)}", flush=True)
print(f"  Total moves in tree traversals: {n_moves_total}", flush=True)
print(f"  Moves per simulation: {n_moves_total/num_sims:.1f}", flush=True)
print(f"  Total Node objects: ~{n_expansions * 30}", flush=True)


# ─── Additional: measure _generate_next_balls cost ──────────────────
print("\n\n=== _generate_next_balls breakdown ===", flush=True)

g_test = ColorLinesGame(seed=42)
g_test.reset()
for _ in range(30):
    if g_test.game_over:
        break
    moves = g_test.get_legal_moves()
    if moves:
        g_test.move(moves[0][0], moves[0][1])

def bench_gen_next():
    g_test._generate_next_balls()
bench(bench_gen_next, n=20000, label="_generate_next_balls")

# choice_no_replace
def bench_choice():
    rng3 = SimpleRng(42)
    rng3.choice_no_replace(60, 3)
bench(bench_choice, n=50000, label="rng.choice_no_replace(60,3)")

# integers
def bench_integers():
    rng3 = SimpleRng(42)
    rng3.integers(1, 8, 3)
bench(bench_integers, n=50000, label="rng.integers(1,8,3)")

# _get_empty_array
from game.board import _get_empty_array
bench(lambda: _get_empty_array(g_test.board), n=50000, label="_get_empty_array")

# list comprehension in _generate_next_balls
empty = _get_empty_array(g_test.board)
rng4 = SimpleRng(42)
def bench_list_comp():
    indices = rng4.choice_no_replace(len(empty), 3)
    colors = rng4.integers(1, 8, 3)
    nb = [
        ((int(empty[indices[i], 0]), int(empty[indices[i], 1])), int(colors[i]))
        for i in range(3)
    ]
bench(bench_list_comp, n=20000, label="next_balls list comprehension")


# ─── RNG method costs ───────────────────────────────────────────────
print("\n\n=== SimpleRng method costs ===", flush=True)
rng5 = SimpleRng(42)
bench(lambda: rng5.next_u64(), n=200000, label="next_u64")
bench(lambda: rng5.next_f64(), n=200000, label="next_f64")
bench(lambda: rng5.randint(0, 60), n=200000, label="randint(0, 60)")

print("\nDone.", flush=True)
