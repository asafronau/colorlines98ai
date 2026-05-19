"""Profile end-to-end time split between CPU and GPU for real model.

Measures:
1. Total search time with real MPS model
2. CPU-only time with DummyNet (same game state)
3. Difference = GPU time (inference + data transfer)
4. GPU evals/s at different batch sizes
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

import time
import numpy as np
import torch
from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _legal_priors_jit, NUM_MOVES
from alphatrain.observation import build_observation

# Warm up JIT
print("Warming up JIT...", flush=True)
g = ColorLinesGame(seed=42)
g.reset()
_build_obs_for_game(g)
_legal_priors_jit(g.board, np.zeros(NUM_MOVES, dtype=np.float32), 30)
print("JIT warm done.\n", flush=True)

# Game state: early-game (lots of empty space, many legal moves)
game_early = ColorLinesGame(seed=7)
game_early.reset()
for i in range(5):
    if game_early.game_over:
        break
    moves = game_early.get_legal_moves()
    if moves:
        game_early.move(moves[0][0], moves[0][1])
n_empty_early = int(np.sum(game_early.board == 0))
n_legal_early = len(game_early.get_legal_moves())
print(f"Early game: empty={n_empty_early}, legal_moves={n_legal_early}", flush=True)

# Game state: mid-game (congested board)
game_mid = ColorLinesGame(seed=7)
game_mid.reset()
for i in range(20):
    if game_mid.game_over:
        break
    moves = game_mid.get_legal_moves()
    if moves:
        game_mid.move(moves[i % len(moves)][0], moves[i % len(moves)][1])
n_empty_mid = int(np.sum(game_mid.board == 0))
n_legal_mid = len(game_mid.get_legal_moves()) if not game_mid.game_over else 0
print(f"Mid game: empty={n_empty_mid}, legal_moves={n_legal_mid}", flush=True)

if game_mid.game_over:
    print("Mid game is over, using early game only", flush=True)
    game_mid = game_early

# ─── MPS GPU profiling ──────────────────────────────────────────────
device = torch.device('mps')
model_path = 'alphatrain/data/pillar2u_epoch_9.pt'
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}", flush=True)
    exit(1)

from alphatrain.evaluate import load_model
net, max_score = load_model(model_path, device, fp16=True, jit_trace=True)

# ─── GPU forward pass timing at different batch sizes ────────────────
print("\n=== GPU Forward Pass Timing (fp16+jit on MPS) ===", flush=True)
obs = torch.randn(1, 18, 9, 9, device=device, dtype=torch.float16)

for bs_test in [1, 4, 8, 16, 32, 64, 128]:
    obs_batch = torch.randn(bs_test, 18, 9, 9, device=device, dtype=torch.float16)
    # Warmup
    for _ in range(10):
        with torch.inference_mode():
            net(obs_batch)
    # Time
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        with torch.inference_mode():
            pol, val = net(obs_batch)
            # Force sync for accurate timing
            _ = pol[0, 0].item()
    elapsed = time.perf_counter() - t0
    ms_per = elapsed / n_iter * 1000
    evals_per_s = bs_test / (elapsed / n_iter)
    us_per_eval = ms_per * 1000 / bs_test
    print(f"  bs={bs_test:3d}: {ms_per:.2f} ms/fwd, "
          f"{evals_per_s:.0f} evals/s, {us_per_eval:.1f} us/eval", flush=True)


# ─── GPU data transfer timing ────────────────────────────────────────
print("\n=== GPU Data Transfer Timing ===", flush=True)
for bs_test in [8, 16, 32]:
    obs_np = np.random.randn(bs_test, 18, 9, 9).astype(np.float32)
    gpu_buf = torch.empty(bs_test, 18, 9, 9, device=device, dtype=torch.float16)

    # CPU -> GPU (numpy -> torch -> device)
    n_iter = 500
    t0 = time.perf_counter()
    for _ in range(n_iter):
        gpu_buf[:] = torch.from_numpy(obs_np).half()
    t_to_gpu = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  bs={bs_test} CPU->GPU: {t_to_gpu:.3f} ms", flush=True)

    # GPU -> CPU (.float().cpu().numpy())
    pol_gpu = torch.randn(bs_test, 6561, device=device, dtype=torch.float16)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = pol_gpu.float().cpu().numpy()
    t_from_gpu = (time.perf_counter() - t0) / n_iter * 1000
    print(f"  bs={bs_test} GPU->CPU (pol {bs_test}x6561): {t_from_gpu:.3f} ms", flush=True)


# ─── End-to-end search: CPU vs GPU split ─────────────────────────────
print("\n=== End-to-End Search: CPU/GPU Split ===", flush=True)

class DummyNet:
    def __init__(self):
        self.pol = torch.randn(1, NUM_MOVES)
        self.val = torch.zeros(1, 201)
        self.num_value_bins = 201
    def __call__(self, x):
        bs = x.shape[0]
        return self.pol.expand(bs, -1), self.val.expand(bs, -1)
    def predict_value(self, val_logits, max_val=30000.0):
        return torch.full((val_logits.shape[0],), 500.0)
    def parameters(self):
        return iter([self.pol])

for game_label, game_state in [("early", game_early), ("mid", game_mid)]:
    if game_state.game_over:
        continue
    print(f"\n  --- {game_label} game ---", flush=True)

    for sims in [400, 800]:
        # CPU only (DummyNet)
        dummy = DummyNet()
        mcts_cpu = MCTS(dummy, torch.device('cpu'), max_score=30000.0,
                        num_simulations=sims, batch_size=8, top_k=30)
        for _ in range(3):
            g = game_state.clone()
            mcts_cpu.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)
        n_runs = 15
        cpu_times = []
        for _ in range(n_runs):
            g = game_state.clone()
            t0 = time.perf_counter()
            mcts_cpu.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)
            cpu_times.append(time.perf_counter() - t0)
        cpu_ms = np.mean(cpu_times) * 1000

        # Real model (GPU)
        mcts_gpu = MCTS(net, device, max_score=max_score,
                        num_simulations=sims, batch_size=8, top_k=30)
        for _ in range(2):
            g = game_state.clone()
            mcts_gpu.search(g)
        gpu_times = []
        for _ in range(n_runs):
            g = game_state.clone()
            t0 = time.perf_counter()
            mcts_gpu.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)
            gpu_times.append(time.perf_counter() - t0)
        gpu_ms = np.mean(gpu_times) * 1000
        nn_ms = gpu_ms - cpu_ms

        print(f"  {sims} sims: total={gpu_ms:.0f} ms, "
              f"CPU={cpu_ms:.0f} ms ({cpu_ms/gpu_ms*100:.0f}%), "
              f"GPU+xfer={nn_ms:.0f} ms ({nn_ms/gpu_ms*100:.0f}%)", flush=True)
        print(f"    Per-sim: total={gpu_ms/sims*1000:.0f} us, "
              f"CPU={cpu_ms/sims*1000:.0f} us, GPU={nn_ms/sims*1000:.0f} us", flush=True)

        # With larger batch size
        mcts_gpu16 = MCTS(net, device, max_score=max_score,
                          num_simulations=sims, batch_size=16, top_k=30)
        for _ in range(2):
            g = game_state.clone()
            mcts_gpu16.search(g)
        gpu16_times = []
        for _ in range(n_runs):
            g = game_state.clone()
            t0 = time.perf_counter()
            mcts_gpu16.search(g, temperature=1.0, dirichlet_alpha=0.3, dirichlet_weight=0.25)
            gpu16_times.append(time.perf_counter() - t0)
        gpu16_ms = np.mean(gpu16_times) * 1000
        print(f"    bs=16: {gpu16_ms:.0f} ms ({gpu16_ms/gpu_ms*100:.0f}% of bs=8)", flush=True)


# ─── Server mode simulation ──────────────────────────────────────────
print("\n\n=== Server Mode Analysis ===", flush=True)
print("With inference server, CPU costs are what limit per-worker throughput.", flush=True)
print("Each worker does: clone + PUCT*depth + trusted_move*depth + obs + IPC", flush=True)
print("GPU processes batches from all workers in parallel.\n", flush=True)

# In server mode with 16 workers, 800 sims, bs=8:
# Each worker submits 800/8 = 100 batches of 8 obs
# CPU work per batch of 8 sims:
# - 8 * (clone + depth*PUCT + depth*move + obs_build + expand)
# - IPC roundtrip (queue put/get + memcpy)

# Measure IPC costs
from multiprocessing import Queue
q = Queue()
n_ipc = 10000
t0 = time.perf_counter()
for _ in range(n_ipc):
    q.put(42)
    q.get()
t_ipc = (time.perf_counter() - t0) / n_ipc * 1e6
print(f"  Queue roundtrip: {t_ipc:.1f} us", flush=True)

# Shared memory write cost (simulating obs_buf write)
obs_np = np.random.randn(8, 18, 9, 9).astype(np.float32)
dst = np.empty_like(obs_np)
n_memcpy = 50000
t0 = time.perf_counter()
for _ in range(n_memcpy):
    np.copyto(dst[:8], obs_np[:8])
t_memcpy = (time.perf_counter() - t0) / n_memcpy * 1e6
print(f"  SHM memcpy (8 obs): {t_memcpy:.1f} us", flush=True)

# Per-worker CPU time per batch of 8 sims (no GPU)
cpu_per_batch_ms = cpu_ms / (800 / 8)  # 100 batches for 800 sims
print(f"\n  CPU work per batch of 8 sims: {cpu_per_batch_ms:.2f} ms", flush=True)
print(f"  GPU forward per batch of 8: ~2.0 ms (measured above)", flush=True)
print(f"  IPC roundtrip: ~{t_ipc/1000:.2f} ms", flush=True)

# Worker throughput limit (CPU-bound):
worker_batch_time = cpu_per_batch_ms + t_ipc/1000  # ms per batch
worker_evals_per_s = 8 / (worker_batch_time / 1000)
print(f"\n  Worker CPU throughput: {worker_evals_per_s:.0f} evals/s (per worker)", flush=True)
print(f"  16 workers: {16 * worker_evals_per_s:.0f} evals/s", flush=True)
print(f"  GPU throughput at avg bs=128: ~16,000 evals/s (measured previously)", flush=True)
print(f"  Bottleneck: {'CPU' if 16 * worker_evals_per_s < 16000 else 'GPU'}", flush=True)


print("\n\nDone.", flush=True)
