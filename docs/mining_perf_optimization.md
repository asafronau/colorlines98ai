# Mining performance optimization — measured analysis

(Autonomous session 2026-06-02. Everything here is measured on the M5 Max, not
estimated. Scripts: profile_rollout.py, bench_sync.py, bench_forward.py,
parallel_rollout.py.)

## TL;DR

- The rollout (the mining hot path) is **GPU-forward-bound**, not CPU- or
  transfer-bound, as I first guessed.
- **The M5 GPU caps at ~19,700 board-forwards/s** for the pillar3b 10×256 model.
  The single-process rollout already runs at ~10,600 turns/s = **54% of that
  ceiling**, so **the hard ceiling for any batching/parallelism on the M5 is
  ~1.9×.** 4× on the M5 is physically impossible with this model.
- Multi-process parallelism gives only **1.40×** on MPS (verified correct) —
  MPS serializes GPU work across processes, so extra workers only overlap the
  CPU/sync slack, they can't add GPU throughput.
- **True 4× comes from Colab's much faster GPU** — but only if the rollout is
  GPU-bound there. Today it isn't (Colab's 2 weak vCPUs bottleneck the per-step
  CPU game logic). Unblocking that needs the game step on the GPU (vectorization)
  and/or torch.compile on cuda (the inference server already supports it).

## Measurements

### Where the time goes (profile_rollout.py, batch 128, single process)
```
WALL 15.9s, 10,341 turns/s
  obs build       3%
  GPU fwd+sync   81%   (.cpu() 8.2s, .to() 1.3s, dispatch 2.7s)
  game logic     16%   (_get_legal_priors_flat 1.6s + game.move 1.5s)
```

### Is .cpu() payload or sync? (bench_sync.py)
A full `(128,6561)` `.float().cpu()` in isolation is **0.20 ms**. In the rollout
it measured **5.5 ms/call**. So the 5.5 ms is the **GPU forward compute** (the
`.cpu()` just blocks waiting for the dispatched forward) — NOT transfer, NOT
fixed sync overhead. Reducing the transfer payload (top-k / GPU argmax) would
not help. Confirmed: argmax-on-GPU vs full-transfer were both ~0.2 ms in
isolation.

### Forward throughput vs batch (bench_forward.py)
```
batch    boards/s   vs b128
  128     16,218     1.00x
  256     18,438     1.14x
  512     19,255     1.19x
 1024     19,711     1.22x   ← saturates
```
The GPU is near its compute limit already at batch 128; bigger batches buy only
~1.2×. So the forward (the 10×256 ResNet on the M5 GPU) is the wall.

### Parallelism (parallel_rollout.py, 2000 jobs)
```
SINGLE  : 52.1s  10,592 turns/s
PARALLEL: 37.3s  14,787 turns/s   (6 workers)   →  1.40x   (died 253/253, correct)
```
MPS serializes the 6 GPU processes; the 1.4× is just CPU/sync overlap. Ceiling
(perfect overlap) ≈ GPU forward rate ≈ 1.9×.

## Why Colab was no faster than the M5 (and how to fix it)

Colab's GPU is far faster, but the rollout there is bound by the **per-step CPU
game logic** (`game.move`, `_get_legal_priors_flat`, obs build) on only 2 weak
vCPUs — the fast GPU sits idle waiting for the CPU. So Colab ≈ M5 despite a
better GPU. To get 4× on Colab we must make the rollout **GPU-bound**:
1. **torch.compile on cuda** — the inference server already supports
   `use_compile=True` (1.3–2× forward on cuda; MPS unsupported).
2. **Vectorize the game step on GPU** — batched move/line-clear/spawn/reachability
   so there is no per-step CPU↔GPU round-trip. This is the big structural win and
   the only path to Colab's GPU being the limit. It is correctness-critical (must
   reproduce SimpleRng spawns + line logic bit-for-bit) — a careful, golden-tested
   rewrite, not a quick change. **Deferred** (too risky to land unsupervised in
   one session); scoped below.

## What ships from this session

- `parallel_rollout.RolloutPool` (verified 1.40× on MPS; expected to scale better
  on cuda, where processes aren't serialized as hard) — wired into
  `mine_crisis_sweep --workers` (persistent pool across all 3 phases) and
  `overnight_systematic --workers`/`--compile`.
- **End-to-end verified** on a real Phase-B game (`death_50001`, 7699 turns):
  `--workers 1` → 505s mining, `--workers 6` → 367s = **1.38×**, with **identical**
  band (29 depths), per-move labels (576), and confirmed forks. Correctness holds.
- All benchmarks/profilers as reusable scripts (profile_rollout, bench_sync,
  bench_forward, parallel_rollout).

### Operating guidance
- **M5:** `--workers 6` (verified 1.38×; MPS serializes the GPU so more workers
  give diminishing returns — the ~1.9× forward ceiling is the wall).
- **Colab (cuda):** `--workers` is UNVERIFIED (I can't run cuda locally). It may
  scale better (cuda overlaps processes) but is bounded by the 2 vCPUs for the
  per-step CPU game logic, and pays a per-game model-load × N (overnight is
  subprocess-per-game). Try `--workers 2-4 --compile`; measure before trusting.
- **Fleet:** the real near-term multiplier you already use — 3 Colab + M5 = 4
  instances on disjoint seeds = ~4× games/hour, independent of per-instance speed.

## Recommended path to 4× (for review)

1. **Run mining on Colab with the inference server + `--compile` (cuda).** Biggest
   near-term lever; no game rewrite. The server centralizes GPU work and compiles
   the forward.
2. **Vectorize the game step on GPU** (the real 4×+, esp. on A100). Plan:
   board as `(B,9,9)` int8 tensor; batched move (scatter), batched line detection
   (4-direction conv/scan), batched clear, batched SimpleRng spawn (replicate the
   PRNG on-device), batched reachability (empty-component labels — already done on
   GPU in `dataset._build_obs_core`). Verify against the CPU engine with golden
   games before trusting any mined fork.
3. If staying on CPU game logic: the M5 ceiling is ~1.9×; combine with the
   `--r-screen` knob (algorithmic, ~2× fewer rollouts) for ~3–4× more games/hour.
