# AlphaTrain — Experiment History

## Goal

Build a self-improving AI for Color Lines 98 using proper reinforcement learning.
Target: exceed 15,000 points mean score. Ultimate goal: learn ML deeply.

## Current Best: Heuristic Tournament (baseline to beat)

**Player:** Tournament bracket with successive halving + ML oracle (linear, 30 features)
- 2-ply heuristic evaluates all ~300 legal moves → top 30
- Quarter-finals: 10 rollouts × 30 → top 10
- Semi-finals: 40 rollouts × 10 → top 3
- Finals: 150 rollouts × 3 → pick best
- Rollouts use JIT-compiled 5-weight CMA-ES heuristic + softmax sampling
- ML oracle (30-feature pairwise-trained linear model) blends at root

**Score:** ~5,700 mean across 500 games (seed 10000-10499). Seed 42 = 1,205.
**Speed:** ~1.0 mv/s on M5 Max (18 CPU cores)

## What We Tried in Phase 18 (Neural Approaches)

### 1. AlphaZero-style ResNet (12.1M params, 10 blocks × 256ch)

**Architecture iterations:**
- v1: Factored policy (source 81 + target 81) — FAILED (source-target coupling broken)
- v2: Flat joint policy (6561) — better but still can't replace 2-ply
- v3: Triple-head (policy + value + Q-head) — Q-head loss too sparse

**Training:**
- 500 expert games, 1.3M states, 8x dihedral augmentation
- Trained on H100 (Colab) at 41K samples/s with AMP
- Policy loss converged to 1.63, Value MAE to 2,361

**Results (standalone policy):** 288-377 points (~5x over heuristic's 67)
**Results (hybrid, NN replaces 2-ply):** 529 points (NN top-K misses good moves)

### 2. Afterstate Value Network (v2, contrastive training)

**Data:** 46M afterstates (13M expert + 26M heuristic traps + 6.6M random)
**Training:** Balanced 33/33/33 tier sampling, categorical CE, 20 epochs on H100
**Result:** Val loss 1.24, MAE 0.57 log1p — but standalone scores only ~20

**Why it failed:** Value head noise (MAE 0.57) is larger than per-move signal (~0.1).
The model learned "what a board generally looks like" but not "which specific move is better."

### 3. Neural Rollouts (NN policy replaces heuristic in rollout simulation)

**Architecture:** Batched parallel rollouts — all N boards evaluated in one GPU forward pass
**Optimizations:** Gumbel-max sampling, pre-allocated GPU buffers, native fp16, channels_last

| Config | Score (seed 42) | Speed |
|---|---|---|
| Pure NN rollout (50×10) | 241 | 0.89 mv/s |
| NN + heuristic blend=3 (50×10) | 516 | 0.70 mv/s |
| NN + heuristic blend=3 bracket (200×20) | 451 | 0.35 mv/s |
| **Heuristic tournament (200×20)** | **1,205** | **1.04 mv/s** |

**Why it failed:** The NN makes occasional tactical micro-errors (missing a line, blocking a path).
One mistake per 20-step rollout corrupts the entire simulation. The heuristic makes zero tactical
errors because it counts lines deterministically. Game survival (turns) matters most — heuristic
survived 592 turns, neural blend only 258.

### 4. Knowledge Distillation (ResNet → linear oracle)

Attempted to distill ResNet strategic knowledge into the 30 spatial features via linear regression.
Result: correlation 0.26 — the 30 features can't capture what the ResNet learned.

---

## Pillar 1: 18-Channel Input Representation (DONE)

### What we did
Added tactical features as input channels to fix "CNN blindness":
- Channels 0-6: one-hot color planes (7 colors)
- Channel 7: empty cells
- Channels 8-10: next ball positions (color/7.0)
- Channel 11: next ball mask
- Channel 12: component area heatmap (empty cell = component_size / 81)
- Channels 13-16: line potentials (H, V, D1, D2) — same-color count per direction
- Channel 17: max line length at each cell

### Training
- Data: 1.31M states from 302 games, 8x dihedral augmentation = 10.5M effective
- Model: 10 blocks × 256ch ResNet, 12.1M params
- Trained on Colab A100: 20 epochs, batch=4096, lr=1e-3, AMP
- Throughput: 31K s/s (A100), 44K s/s collate benchmark (MPS)

### Results
| Metric | Value |
|---|---|
| Policy loss (val) | 1.88 |
| Value loss (val) | 2.00 |
| Value MAE | 2,035 |
| Best val_loss | 3.80 (epoch 14) |
| Standalone policy score | mean=265 (20 games) |

### Performance fixes
- **150x collate speedup**: Python triple loop calling `_line_length_at` per cell → single
  `build_line_potentials_batch()` JIT call for entire batch
- **GPU-native dataset**: all data on GPU, on-the-fly observation building in collate
- **Dihedral augmentation via precomputed LUTs**: 8x data with zero overhead

---

## Pillar 1.5: Neural MCTS Attempt (BLOCKED)

### What we did
Built AlphaZero-style MCTS (`alphatrain/mcts.py`):
- PUCT selection with MuZero-style Q normalization
- Policy priors from top-K legal moves
- Value head for leaf evaluation (no rollouts)
- Determinized: game.clone() gives fresh RNG per simulation

### Results
| Config | Score (seed 42) | Notes |
|---|---|---|
| Pure policy (greedy argmax) | 265 mean | 0.35s/game |
| MCTS 50 sims | 274 mean | 23s/game |
| MCTS 200 sims | 222 (1 game) | WORSE — more sims hurts |
| Value-based move ranking | 8 mean | Catastrophic — worse than random |

### Why it failed

**1. Value head doesn't rank moves.**
Diagnosed with `alphatrain/scripts/debug_value_head.py`:
- Policy vs value rank correlation: **rho=0.133** (nearly uncorrelated)
- Value head's top pick has policy rank ~120/234
- Root cause: all states in a game share the same `game_score` target, so the value
  head learned "which game pattern this is" not "which move leads to better outcomes"

**2. Determinized MCTS breaks in stochastic games.**
- After move execution, 3 random balls spawn (different per simulation)
- Child nodes conflate evaluations from completely different board states
- More simulations = more noise mixed into Q-values = worse moves

### Lessons learned
11. **game_score is a game-level label, not a position-level signal.** Training value head on
    raw game_score teaches it to recognize high-scoring game patterns, not to predict future
    potential from a specific position. Need TD-learning with per-step rewards.
12. **Determinized MCTS doesn't work for stochastic games with large chance branching.**
    3 balls × remaining positions × 7 colors = too many outcomes per step. Need either
    root-only evaluation, afterstate approach, or proper chance nodes with sampling.
13. **Always validate value head with rank-correlation diagnostic before building search.**
    `debug_value_head.py` would have saved us the MCTS implementation time.

---

## Pillar 2a: TD Value Targets with γ=0.99 (Colab A100, 5.2h)

### What we changed
Replaced raw `game_score` value targets with discounted remaining score:
- `V(t) = Σ γ^k * reward(t+k)` where reward = score delta per turn, γ=0.99
- Each position gets a unique target based on its future trajectory
- Value range: mean=210, std=35, max=325 (64 bins over [0, 500])

**Why γ=0.99 not γ=1.0 (MC return):** Gemini peer review caught this. MC return (γ=1.0) had
std=4461 — identical boards from different games get wildly different targets due to distant RNG.
γ=0.99 gives ~200-turn effective horizon, zeroing out the RNG-driven future. Variance dropped 60x.

**Also critical: max_score=500 not 30000.** With γ=0.99, values max at 325. Old 64 bins over
[0, 30000] = 468 pts/bin (zero resolution). New [0, 500] = 7.8 pts/bin (60x better resolution).

### Training
- Colab A100, 30 epochs, batch=4096, lr=1e-3, 3-epoch warmup + cosine decay, AMP
- Throughput: 16K s/s (slower than Pillar 1's 31K due to different A100 allocation)
- Best epoch: 18 (val_loss=2.86), train loss still decreasing at epoch 30 (2.09)
- Added per-epoch checkpoint save to Google Drive (critical for Colab disconnects)

### Results
| Metric | Pillar 1 | Pillar 2a (TD γ=0.99) |
|---|---|---|
| Policy loss (val) | 1.88 | 1.68 |
| Value MAE | 2035 | 5 (on [0,500] scale) |
| **Policy player mean** | **265** | **494** |
| Policy player max | 592 | 1028 |
| Turns survived | 150 | 255 |

**Policy improved 1.9x** despite only changing value targets. The shared backbone learned
better features from the more meaningful value signal.

### Value head still can't rank moves
- Rank correlation: **rho=-0.083** (was 0.13, now slightly negative)
- Value spread across 234 legal moves: only 25 points (187-213)
- The value head correctly predicts "~200 discounted future points" for all positions
- But per-move differences (2-10 points) are below the MAE noise floor (5)

### Root cause analysis (Gemini peer review)
**The loss function is the problem, not the architecture.**
- Policy head succeeded because it learns **relative preferences** (soft cross-entropy)
- Value head failed because it learns **absolute regression** (categorical CE on exact score)
- Proof: the backbone features ARE good enough (policy improved to 494). The value head
  just isn't being forced to use them for ranking.
- Fix: **pairwise margin ranking loss** — train V(good_afterstate) > V(bad_afterstate)
  by the tournament score margin. The ~200pt "macro-value" cancels out.

### Lessons learned
14. **MC return (γ=1.0) has catastrophic variance in stochastic games.** Two identical boards
    from different games get targets differing by 15,000. Use γ<1.0 to zero out distant RNG.
15. **Bin range must match value range.** 64 bins over [0, 30000] when values max at 325
    wastes 99.5% of resolution. Always check actual value range before training.
16. **Better value targets improve policy even without changing policy loss.** The shared backbone
    benefits from more meaningful gradients through the value head.
17. **Absolute regression can't rank moves when per-move signal < MAE.** Need pairwise/ranking
    loss to cancel out the position's "macro-value" and focus on move-level differences.
18. **Always save checkpoints to Drive every epoch on Colab.** Runtime disconnects without warning.

---

## Pillar 2b: Pairwise Ranked Value Head (FAILED)

### What we built
Added **pairwise margin ranking loss** alongside existing policy + value losses:
1. Afterstate computation: board after move + line clears, before ball spawning (deterministic)
2. For each state with 2+ top moves, pair best vs worse afterstate
3. Loss: `F.relu(margin_scaled - (V(good) - V(bad)))` with tournament score margins
4. Total: `pol_CE + 0.5 * val_CE + rank_loss`
5. Warm start from Pillar 2a checkpoint (policy=494)
6. 1.31M afterstate pairs, mean margin=14.3 tournament score points

### Training attempts

**Attempt 1: From scratch, margin=0 (dead ranking loss)**
- Colab A100, 30 epochs at 20K s/s (later cancelled at epoch 8)
- rank_loss collapsed to 0.0000 by epoch 6 — model satisfied V(good) > V(bad) trivially
- Value head overfitting: val_loss 4.43 (epoch 3) → 11.34 (epoch 7)
- Policy regressed: 494 → 158 (only 3 epochs, not converged)
- Root cause: margin=0 requires only correct direction, not meaningful separation

**Attempt 2: Warm start from 2a, proportional margins**
- Fixed: margin scaled so mean=5 on value head's [0,500] range
- Fixed: warm start preserves 494-scoring policy backbone
- Colab A100, 20 epochs, lr=3e-4, val_weight=0.5
- rank_loss: 11.2 → 0.45 → 0.13 (declining as expected, NOT collapsing to 0)
- pol_loss stable: ~1.71 train, ~2.0 val (backbone not corrupted)
- BUT val_loss increasing from epoch 1: 2.54 → 2.62 → 2.66 (overfitting immediately)
- Best checkpoint: epoch 1 only

### Results (epoch 1 best)
| Metric | Pillar 2a | Pillar 2b (1 epoch pairwise) |
|---|---|---|
| Policy mean | 494 | 325 (regressed — 1 epoch not enough) |
| Value spread | 2 pts | 18 pts (9x better) |
| **Rank correlation** | **-0.083** | **-0.081 (no improvement)** |

### Why it failed

**1. Afterstate observations lack discriminative signal.**
Two afterstates from the same position differ by one ball moved. The 18-channel observation
(mostly color one-hot + empty) looks nearly identical for both. Without next_balls (channels 8-11
are zero for afterstates), the model has even less context. The backbone can't see why one
afterstate is better — the difference is in downstream consequences (pathfinding, future line
setups) that aren't visible in the immediate board state.

**2. The ranking signal doesn't generalize.**
The model learned to spread training pair values (spread went 2 → 18) but in the wrong
direction (correlation still -0.08). It memorized training-set-specific patterns rather
than learning general move quality features.

**3. Overfitting from epoch 1.**
With 1.31M pairs but 12.1M model parameters, the value head can memorize training pairs
without learning generalizable features. Lower lr and val_weight didn't help.

### Performance optimizations (valuable regardless of outcome)
- GPU line potentials via shift operations: eliminated CPU↔GPU sync
- GPU component area via shift-based label propagation: 4x faster
- Fused good+bad afterstate obs build: single call
- torch.compile: 64% faster model forward/backward on H100
- Total collate speedup: 20K → 52K s/s MPS, 11K → 20K s/s CUDA

### Lessons learned
19. **Pairwise ranking on afterstates fails when afterstate obs are nearly identical.**
    Moving one ball on a 9×9 board barely changes the 18-channel representation. The signal
    is in downstream consequences, not the immediate board state.
20. **Warm start is essential for fine-tuning but 1 epoch isn't enough.** The policy regressed
    because the new loss landscape differs from the original. Need many epochs to reconverge,
    but the value head overfits before the policy can recover.
21. **The value head overfitting problem may be fundamental.** With 12.1M shared params and
    the value head trying to predict small differences between similar boards, overfitting
    starts immediately regardless of lr, weight decay, or loss weighting.
22. **Performance optimization pays off even when experiments fail.** The GPU line potentials
    and component area optimizations benefit all future training runs.

---

## Pillar 2c-2e: Scalar Value Head Journey

### 2c: Pure Ranking (unbounded scalar, no categorical CE)
Dropped categorical CE entirely. Scalar value head (bins=1) with ranking loss only.
- **rho=0.158 (p=0.016) at epoch 8** — first statistically significant ranking!
- But: values drifted to 594-650 (unbounded), memorized training pairs
- rank_loss collapsed to 0.0025 while generalization stalled
- Lost epoch 8 checkpoint — overwritten by later epochs with lower val_loss but worse rho

### 2d: Sigmoid Clamp
Added `sigmoid(x) * max_score` to bound output to [0, 500].
- Prevented drift, but: **sigmoid saturation** — all values compressed to [483, 498]
- rho=0.125 (p=0.057) — borderline, gradient vanished near ceiling
- Root cause: no "gravity" pulling values toward true mean, network pushed everything up

### 2e: Sigmoid + Anchor MSE Loss
Added small MSE loss (`weight=0.001`) comparing sigmoid output to true TD targets.
- **anchor_weight=0.1**: anchor overwhelmed everything (537→1.9 total loss). Way too strong.
- **anchor_weight=0.001**: balanced losses (pol≈1.7, rank≈0.3, anchor≈0.3). MAE: 187→18!
- **rho≈0** — within-position move ranking near random
- BUT: excellent absolute state evaluation (MAE=18 on [0,500] scale)

### MCTS Test (Pillar 2e anchor checkpoint)
**The breakthrough: value head works for MCTS despite zero within-position ranking.**

| Seed | Policy (greedy) | MCTS (400 sims) | Change |
|------|----------------|-----------------|--------|
| 42 | 389 | **835** | +115% |
| 43 | 176 | **1767** | +904% |
| 44 | 174 | **2250** | +1193% |
| 45 | 489 | 390 | -20% |
| 46 | 342 | **1636** | +378% |
| **Mean** | **314** | **1376** | **+338%** |

**Caveat:** Each result is a single game (not averaged). High variance from random ball spawns.

**Key insight:** MCTS uses policy to pick moves, value to evaluate resulting states. The value
head doesn't need within-position ranking — it needs cross-state discrimination. MAE=18 means
"state worth 300" vs "state worth 100" is clearly distinguished. The debug_value_head rho metric
was measuring the wrong thing.

### Lessons learned
23. **Within-position rho is misleading for MCTS evaluation.** A value head that can't rank moves
    from one board state can still dramatically improve tree search by accurately evaluating
    different game states reached across the tree.
24. **Sigmoid prevents drift but needs an anchor to avoid saturation.** Without absolute target
    information, sigmoid outputs cluster at the boundary where gradients vanish.
25. **Anchor MSE weight must be calibrated carefully.** MSE scale (~2500) means weight=0.1
    contributes 250 to loss, dwarfing policy (1.7) and rank (0.3). Weight=0.001 gives balance.
26. **Per-epoch checkpoint saving is essential.** Peak quality (rho, correlation) doesn't always
    coincide with best val_loss. Save every epoch and evaluate each one.

---

## MCTS Performance Engineering

### Optimization journey (3500ms/turn → 265ms/turn)

| Optimization | ms/turn | Speedup |
|---|---|---|
| CPU sequential (baseline) | 3500ms | 1x |
| Virtual loss batching (bs=16, MPS) | 218ms | 16x |
| + Vectorized legal priors (numpy) | 195ms | 18x |
| GPU inference server (shared memory) | 120ms (1 worker) | 29x |
| + FP16 + JIT trace | 99ms (1 worker) | 35x |
| Final config: local MPS, fp16+jit, bs=8 | 265ms | 13x |

Note: bs=8 is slower than bs=32 but **critical for quality** (see below).

### Key finding: batch_size controls quality/speed tradeoff

| Batch size | Mean score | ms/turn |
|---|---|---|
| 8 | **863** | 265ms |
| 16 | 512 | 150ms |
| 32 | 475 | 120ms |

Virtual loss at bs=32 degrades PUCT selection — with ~30 root children, 32 simultaneous
virtual losses make selection near-random. bs=8 preserves search quality.

### Simulation count matters

| Sims | Games | Mean | Median | Max | ms/turn |
|---|---|---|---|---|---|
| 400 | 50 | **863** | 812 | 2789 | 265ms |
| 800 | 28 | **1268** | 1160 | 2818 | 550ms |

800 sims gives +47% over 400, but 2x slower. Both beat policy (mean=314) by 3-4x.

### Infrastructure built
- **GPU inference server** (`inference_server.py`): shared memory IPC, centralized MPS
  inference, cross-worker batching. 12,500 evals/s with fp16+jit.
- **Parallel eval** (`eval_parallel.py`): local MPS mode (quality) or server mode (throughput)
- **Profiling tools**: `profile_mcts.py`, `profile_server_mcts.py`, `bench_worker_scaling.py`
- **JIT legal priors**: numba-compiled connected components + softmax + top-K

### Lessons learned
27. **Virtual loss batch size controls quality.** Large batches (32+) degrade PUCT selection
    when the tree has few children at the root. bs=8 is the sweet spot for 30-child trees.
28. **FP16 inference is free quality.** No measurable score difference, 2x GPU throughput.
29. **GPU inference server needs batch caps.** Without caps, 18 workers create batch=256
    which is past MPS's efficient range. GPU_BATCH_CAP=128 keeps per-eval latency low.
30. **torch.set_num_threads(1) is mandatory for CPU multiprocessing.** Without it, 18 workers
    × 8 threads = 144 threads on 18 cores, causing 5x slowdown from cache contention.
31. **Profile before optimizing.** NN forward was 97% of CPU time — all other optimizations
    combined saved <3%. Moving inference to GPU was the only meaningful improvement.

---

## Key Lessons Learned

1. **Tactical precision > strategic vision in rollout quality.** The heuristic wins because it never
   makes a counting error. The NN occasionally misses lines, which is fatal in long rollouts.

2. **CNNs struggle with graph connectivity.** Pathfinding and line counting are non-trivial for
   convolutions. Need to inject these as input features (line potentials, distance maps).

3. **Absolute game score is a bad training target.** High variance (~4,800 std) causes mean
   collapse. Need discounted rewards (TD-learning) for stable gradients.

4. **Imitation learning has a hard ceiling.** The network can only match its teacher. Self-play
   is required to exceed the teacher's level.

5. **Always profile before running experiments.** We wasted hours on slow code that could have
   been 3x faster with proper optimization.

6. **All scripts must be standalone python modules.** Never use `python3 -c` inline. Put analysis
   in `alphatrain/scripts/` and run with `python -m`.

7. **Validate value head rank-correlation before building search on top of it.**

---

## Phase 4: Self-Play Infrastructure & Training Iterations

### Self-Play Data Generation (493 games, 800 sims)
- Built GPU server mode: N CPU workers share one GPU via InferenceServer
- 16 workers on M5 Max: ~7900 evals/s, avg IBS=63
- Generated 493 games (seeds 0-499): 277K states, mean score 1161, max 6896
- Resume-safe: each game saved individually, skips completed seeds

### CPU Threading Bug Fix
CPU self-play scored 60 (vs MPS 1216) for same seed. Root cause: hidden BLAS
threads (OpenBLAS/Accelerate) not controlled by `torch.set_num_threads(1)`.
Fix: set OMP/MKL/OPENBLAS/VECLIB/NUMBA_NUM_THREADS=1 at module top, before
importing numpy/torch. Verified: CPU scores 2580+ at turn 1200.

### Evaluation Baselines (50 games each, seeds 42-46 × 10)
| Player              | Sims | Mean | Median | Min | Max  |
|---------------------|------|------|--------|-----|------|
| Policy (greedy)     | —    |  314 |    342 | 174 |  489 |
| MCTS (400 sims)     | 400  |  911 |    795 | 311 | 2023 |
| MCTS (800 sims)     | 800  | 1053 |    918 | 269 | 3337 |

### Self-Play Training Iteration 1a: Pure Self-Play (FAILED)
- Data: 277K self-play states only, raw MCTS visit distributions (T=1.0)
- Config: lr=3e-4, 10 epochs, warm start
- **Result: Policy 314 → 118 (-62%)**
- Policy loss flat at 3.82 (matched target entropy, never improved)
- Diagnosis: soft targets + no expert anchoring = catastrophic forgetting

### Iteration 1b: 50/50 Mixed + Sharpened (FAILED)
- Data: 277K expert + 277K self-play, T=0.1 sharpening (entropy 3.82 → 0.28)
- Config: lr=1e-4, 10 epochs, warm start, val_weight=1.0
- **Result: Policy 314 → 245 (-22%), MCTS 911 → 362 (-60%)**
- Root cause: value loss (860) was 600x policy loss (1.4)
- Value gradients destroyed backbone features through shared ResNet

### Iteration 1c: Rebalanced + From Scratch (FAILED)
- Data: 200K expert + 200K elite self-play (score>1000), T=0.1
- Config: lr=1e-3, 20 epochs, **from scratch**, val_weight=0.002
- **Result: Policy 111 (worst yet)**
- Train/val gap: pol 1.18/2.20 — massive overfitting
- Training from scratch threw away valuable backbone features

### Frozen Backbone Experiment (FAILED)
- Froze backbone + policy head, trained value head only on expert pairwise data
- **Result: Value head couldn't learn (rank=0.0012 flat, MAE=18→19)**
- Backbone features optimized for policy don't carry sufficient value signal
- Confirms: value head NEEDS its own adapted features

### Key Insight: The Backbone Conflict
The shared ResNet backbone is the root cause of all failures:
- Value head needs backbone adaptation → but that destroys policy features
- Frozen backbone prevents policy damage → but value can't learn
- Loss imbalance (value 600x policy) makes the conflict worse

**Decision: Decouple into separate PolicyNet and ValueNet.**
- PolicyNet: current best model (policy=314), frozen during value training
- ValueNet: trained from scratch on self-play MCTS Q-values
- No shared backbone → no gradient conflict
- Self-improvement loop: MCTS (policy priors + value eval) → self-play → train value → repeat

### Pillar 2f: Asymmetric Joint Training (SUCCESS)
First successful training iteration. Shared backbone with val_weight=0.001.
- Data: 1.3M expert pairwise states (same as original training)
- Config: lr=1e-4, 10 epochs, warm start from epoch 6, val_weight=0.001
- Rank loss + anchor MSE for value, policy CE drives backbone
- Training: anchor MAE 296→236, policy val loss 1.7869→1.7762
- **Result: Policy 315 (preserved), MCTS-400 = 992 (+9% over 911 baseline)**
- Max score jumped 2023→3135
- Converged by epoch 9-10 (val loss flat at 1.7762)

### Standalone ValueNet Experiment (FAILED)
Tested decoupled architecture: separate PolicyNet (10b×256ch) + ValueNet (6b×128ch).
- ValueNet trained from scratch on 277K self-play states
- Training: MAE=6.0 (overfitting: train MSE=16, val MSE=66)
- **Result: MCTS 400 (vs baseline 911) — -56% regression**
- The value head needs the policy backbone's "tactical eyes"
- Decoupled architecture can't learn vision from 277K states alone
- Key lesson: jointly-learned features are essential for value prediction

### Pillar 2g: Hybrid Interleaved Training (FAILED)
Two dataloaders interleaved per step: expert (ranking) + self-play (MSE value).
- Self-play: 1000 games (seeds 500-1500), 400 sims, mean score 688
- Policy sharpened T=0.3, selfplay data contributes both policy CE + value MSE
- Config: val_weight=0.001, lr=5e-5, 15 epochs, bs=2048, warm start from 2f
- **Result: Policy 410 (+8%), MCTS 539 (-39%)**
- Root cause: "Distribution shift" — self-play data (mean 688) taught value head
  what weak play looks like, overwriting expert calibration. Val overfitting:
  train s_val=584, val s_val=3440. 4x cycling of selfplay amplified overfitting.

### Pillar 2h: Elite Filter + Expert-Only Policy (FAILED)
Three fixes from Gemini post-mortem:
1. Elite filter: only selfplay games scoring ≥1000 (220 games, 163K states, mean 1547)
2. Policy from expert only: selfplay contributes value MSE only, no policy CE
3. No cycling: selfplay drives epoch, expert restarts when exhausted
- **Result: epoch 1 closest to 2f (MCTS 825), more training = worse (ep8: 536, ep10: 482)**
- Root cause: even elite selfplay data hurts value head. "Dumber teacher" problem confirmed.
- The self-play loop doesn't work until MCTS matches the heuristic player (~5700).

### Technical Challenges During Training
- **OOM on H100-80GB**: Both datasets GPU-resident + torch.compile = 78GB used.
  Root cause: collate functions lacked @torch.no_grad(), causing autograd to track
  ~150 intermediate tensors per batch. Also: itertools.cycle() caches all yielded
  GPU tensors in memory (leaked 80+ GB). Fixes: @torch.no_grad() on all collate
  methods, manual iterator restart instead of cycle, selfplay data CPU-resident.
- **@torch.no_grad() on collate**: Critical fix identified by Gemini peer review.
  Without it, the GPU observation building (BFS shifts, line scans) accumulated
  autograd graph entries that were never freed.

### Strategic Reset: Expert Data Generation

Self-play training failed because the NN MCTS (mean 891) is too weak to teach
itself — "dumber teacher" problem. Pivoted to generating more expert data from
the heuristic tournament player.

**Rust Engine Built (TDD, 44 tests):**
- Complete game engine rewrite in Rust: 131x faster than Python (1.5 vs 196 µs/turn)
- Custom SplitMix64 RNG: identical output in Python and Rust (cross-language parity)
- Tournament bracket player with heuristic + ML oracle features
- PyO3 bindings: 86x speedup through Python FFI
- Golden tests: exact score verification for seeds 0-9 (50 rollouts)
- Parity verified: old engine (xorshift64) exact match for seeds 0, 10, 12

**Expert V2 Data Generated:**
- 5,310 games, 200 rollouts, 18 workers on M5 Max + 176 workers on GCP
- Mean score: 5,255 (climbing to ~5,500+ as more long games finish)
- Total: 12.8M states with pairwise pairs from top-5 tournament candidates

### Pillar 2i: Expert V2 with Scalar Value Head (POLICY IMPROVED, MCTS STAGNANT)
Training with 10x more data, position-specific TD returns, scalar value head:
- **Data**: 12.8M states from 5,310 expert games (200 rollouts, mean score 5,255)
- **TD returns (γ=0.99)**: mean=210, max=340, max_score=2000 (31 pts/bin)
- **Value head**: scalar sigmoid (num_value_bins=1), output = sigmoid(logit) × 2000
- **Losses**: pol_CE + 0.001×anchor_MSE + 1.0×rank_hinge (val_weight unused for scalar)
- Warm start from Pillar 2f, lr=1e-4 cosine→1e-6, bs=8192, 10 epochs, H100

**Training metrics (all improved steadily):**
| Metric | Ep1 | Ep2 | Ep8 |
|---|---|---|---|
| pol CE | 1.779 | 1.710 | 1.638 |
| rank loss | 0.653 | 0.382 | 0.096 |
| anchor MAE | 105 | 68 | 30 |

**MCTS eval (50 seeds, 400 sims) — did NOT improve:**
| | Pillar 2f | 2i Ep1 | 2i Ep2 | 2i Ep8 |
|---|---|---|---|---|
| MCTS mean | 891 | 527 | 559 | 505 |
| Policy mean | ~315 | 299 | 344 | 435 |
| MCTS over policy | +183% | +76% | +63% | **+16%** |

**Post-mortem — "The Mid-Game Blob":**
Verified data pipeline is correct (no scale mismatch). Root cause identified through
distribution analysis: **84.3% of training positions have TD returns in [190, 240]** — a
50-point band. The value head trains on data where almost everything looks the same.
- Only 29K positions (0.23%) have TD return = 0 (last ~5 moves of each game)
- Value head learns "every board ≈ 210" — can't distinguish healthy from dying
- Result: MCTS search gets no useful signal from value backup (+16% vs +183%)
- Gemini peer review identified γ=0.99 (half-life 69 turns) as the culprit: averages
  everything into an indistinguishable blob. Also recommended categorical head over sigmoid.

### Pillar 2j: High-Contrast Value Head (MCTS 891 → 1,323, BREAKTHROUGH)
Fixing the Mid-Game Blob with shorter horizon, categorical head, and endgame oversampling.

**Changes from 2i:**
| Setting | Pillar 2i | Pillar 2j |
|---|---|---|
| γ | 0.99 (half-life 69 turns) | **0.95 (half-life 14 turns)** |
| Value head | Scalar sigmoid × 2000 | **Categorical 64 bins** |
| max_score | 2000 (31 pts/bin) | **100 (1.59 pts/bin)** |
| val_weight | 0.001 (unused) | **0.01 (categorical CE)** |
| Endgame | No special handling | **30% of batch from last 100 turns** |

**New distribution (γ=0.95):** mean=43, median=43, range 0-155, max_score=100
- P25=39, P75=47 (still concentrated but bin resolution is 20x better)
- 531K endgame positions (4.1%) oversampled to 30% of each batch
- Categorical head + higher val_weight = meaningful value gradient through backbone
- Warm start from Pillar 2i (keeps improved policy at 1.638, reinits value_fc2)

**Training:** H100, 10 epochs, bs=8192, lr=1e-4 cosine, 2-epoch warmup
- val_loss now ACTIVE: started 4.50, dropped to 2.78 (was literally 0.0 in 2i)
- Best checkpoint: **epoch 2** (overfits after — train val_CE 2.73 vs val 2.81 at epoch 3)
- Policy preserved: pol CE 1.67 (vs 2i's 1.64, slight increase from backbone sharing)

**Results (epoch 2 best, 50 seeds, 400 sims):**
| | Pillar 2f | 2i Ep8 | **2j Ep2** |
|---|---|---|---|
| MCTS mean | 891 | 505 | **1,323** |
| Policy mean | ~315 | 435 | 448 |
| MCTS boost | +183% | +16% | **+195%** |
| Max | ~3,135 | 1,303 | **3,784** |

12 seeds broke 1,500+, 6 seeds broke 2,000+. Seed 46 hit 3,784.

### "Value Hallucination" Discovery (1,600 Sims Test)
Tested 7 strongest seeds with 4x more simulations (1,600 vs 400).
**5 out of 7 seeds got WORSE with more search:**

| Seed | 400 sims | 1,600 sims |
|---|---|---|
| 0 | 2,203 | 232 (-89%) |
| 6 | 2,457 | **5,767 (+135%)** |
| 17 | 3,207 | 815 (-75%) |
| 23 | 3,510 | 1,973 (-44%) |
| 46 | 3,784 | 1,061 (-72%) |

**Diagnosis:** The value head is overconfident and wrong on novel positions. At 400 sims,
the policy prior (which is good) still dominates. At 1,600 sims, deeper search relies
more on value backup — and when the value estimates are confidently wrong, more search
converges harder on the wrong answer.

**Seed 6 = existence proof:** hit 5,767 (near heuristic level!) — backbone features ARE
capable, the value head just can't use them consistently. Need better value architecture.

### Pillar 2k-alpha: Heavy Value Head + Adversarial Ranking (HALLUCINATION PERSISTS)
First attempt to fix "Value Hallucination" with architecture changes.

**Changes from 2j:**
| Component | Pillar 2j | Pillar 2k-alpha |
|---|---|---|
| value_conv | 8 channels | **32 channels** |
| value_fc1 | 648→256 | **2,592→512** |
| Dropout | None | **0.3** |
| Value head params | 183K | **1.37M (7.5x)** |
| Ranking pairs | top-1 vs top-5 | **top-1 vs random move** |
| Total model | 12.1M | **13.3M** |

**Results (epoch 2 best, 50 seeds, 400 sims):**
| | 2j Ep2 | 2k-alpha Ep2 |
|---|---|---|
| MCTS mean | 1,323 | 1,134 |
| Policy mean | 448 | 513 (best yet) |
| MCTS boost | +195% | +121% |

**1,600-sim test: STILL FAILS.** 7/7 seeds regressed, mean dropped -66% (worse than 2j's -27%).
The bigger value head, adversarial ranking, and dropout did NOT fix the fundamental problem.
Architecture changes alone can't solve distribution shift.

### The Forensic Diagnosis Breakthrough

**Pivotal moment:** The user insisted on diagnosis before more architecture experiments.
*"I want to understand why this is happening and how to have more confidence that the new
architecture is better. Let's diagnose the problem."*

Gemini suggested a forensic audit: feed the model expert boards, blunder boards (expert + 1
random move), and chaos boards (random ball positions), then compare value predictions.

**Built `diagnose_value_head.py` with three tests:**

**Test 1 — Trap Test (can the model distinguish board types?):**
| Board type | Value prediction |
|---|---|
| Expert mid-game | 43.8 |
| Expert + 1 random move | 43.2 |
| Random chaos | 32.2 |

**SMOKING GUN #1:** Blunder boards indistinguishable from expert (gap: 0.6 out of 44).
The value head literally cannot tell a master move from a random move.

**Test 2 — Target correlation with board health (empty squares):**
| Metric | Correlation |
|---|---|
| TD returns (γ=0.95) | **r = -0.036 (ZERO!)** |
| Remaining turns | r = 0.121 (3x better) |

**SMOKING GUN #2:** TD returns have NO correlation with board health. A dying board (20
empty) gets TD=27, a pristine board (70 empty) gets TD=31. Only 4 points of contrast!
This is because TD returns measure "are points scored nearby?" not "is this board healthy?"
A dying board frantically clearing lines scores similarly to a quiet healthy board.

**Test 3 — Death prediction accuracy:**
| Game stage | NN prediction | Truth |
|---|---|---|
| Early/Mid | 43.9 | 44.1 (accurate) |
| Endgame (<100 turns) | 34.4 | 31.0 (overconfident +3.4) |
| **Death (<20 turns)** | **23.0** | **7.9 (3x too high!)** |

**SMOKING GUN #3:** The value head tells MCTS "this position is fine" (23.0) when the board
is 10 turns from death (truth: 7.9). This is exactly why deeper search hallucinates —
it follows branches to dying positions and the value head says "keep going."

**Root cause: TD returns (γ=0.95) are a terrible value target.** They encode "how many
points are scored in the next ~14 turns" — which depends on luck and line-clearing
frequency, NOT board health. The value head can't learn geometric health from this signal.

### Pillar 2k-survival: The Survival Clock (MCTS 1,791, FIRST 9K+ SCORE)

**The user's key insight about endgames:** *"Regardless of whether a game scores 300 or
30,000, the endgame follows the same pattern. Once the board reaches a certain configuration,
it spirals into death. The 30,000-point game just delayed this longer."*

This led to the idea: predict SURVIVAL TIME, not score. The user also pointed out a concern
with pure survival: *"I don't want the AI to panic and rush to clear every 5-ball line."*

**Solution: hybrid survival + scoring reward.**
`r(t) = 1.0 + score_delta / 10.0` — each turn you survive gives base reward 1.0, plus a
scoring bonus. With γ=0.95, healthy positions get V≈25 (20 turns × 1.25 avg reward),
dying positions get V≈8 (few turns left).

Gemini reviewed the formula, approved C=10, recommended linear bins (no sqrt since γ=0.95
already compresses to [0,35]), and suggested max_score=30 with 128 bins (0.24 pts/bin)
for high resolution in the critical range.

**Changes from 2k-alpha:**
| | 2k-alpha | 2k-survival |
|---|---|---|
| Value target | TD returns (score only) | **Survival hybrid r=1+pts/10** |
| Bins | 64, max_score=100 | **128, max_score=30** |
| Head architecture | Same (32ch, 512h, dropout 0.3) | Same |

**Training:** H100, best at **epoch 7** (no overfitting cliff — dropout working!)

| Metric | 2k-alpha Ep2 | **2k-surv Ep7** |
|---|---|---|
| pol CE | 1.621 | **1.498** (best ever) |
| val CE | 2.79 | 2.72 |
| val MAE | 6 | **3** |

**Results (50 seeds, 400 sims):**
| | 2f | 2j | 2k-alpha | **2k-surv** |
|---|---|---|---|---|
| MCTS mean | 891 | 1,323 | 1,134 | **1,791** |
| Policy mean | ~315 | 448 | 513 | **825** |
| MCTS boost | +183% | +195% | +121% | +117% |
| Max | ~3,135 | 3,784 | 2,715 | **5,326** |

Policy at 825 (no search) outperforms Pillar 2f's MCTS (891). 5 seeds broke 3,000+.

**1,600-sim hallucination test (7 strongest seeds):**
| Seed | @400 | @1600 |
|---|---|---|
| 7 | 3,494 | **9,277 (+165%)** |
| 10 | 5,326 | 4,411 (-17%) |
| 43 | 4,619 | 982 (-79%) |
| **Mean (7 seeds)** | **4,090** | **2,998 (-27%)** |

**Seed 7 at 9,277 = existence proof.** First NN MCTS score above the heuristic mean (5,700).
Hallucination reduced (-27% vs -66%) but not eliminated: 5/7 seeds still regress.

### Post-Survival Forensic Re-Diagnosis

Re-ran `diagnose_value_head.py` on the survival model to verify fixes:

**Death prediction: FIXED**
| Game stage | Old pred / truth | New pred / truth |
|---|---|---|
| Death (<20 turns) | 23.0 / 7.9 (3x over) | **8.0 / 8.1 (perfect!)** |
| Endgame (<100 turns) | 34.4 / 31.0 (+3.4) | 15.0 / 19.4 (-4.4, conservative=safe) |

**Board health correlation: 7x better**
| | Old | New |
|---|---|---|
| r(value, empty_squares) | -0.036 | **0.264** |

**Still unresolved: single-move discrimination**
Expert board vs expert+1 random move: gap of 0.1 (was 0.6). But this may be CORRECT —
one random move genuinely costs only 0.5-2.5% of survival time on a healthy board.

**The remaining hallucination mechanism: compounding errors at depth.**
At 1,600 sims, MCTS explores 5-10 moves deep. Each move is slightly suboptimal (the 0.1
gap is invisible). After 5-10 suboptimal moves, cumulative damage degrades the board
geometry. The value head evaluates the degraded board and says "looks fine" because it
has never seen what happens after consecutive NN mistakes — only expert trajectories.

**This is distribution shift, not a reward formula problem.** The fix requires the model
to see positions from its own play distribution → self-play data generation.

### Next: Self-Play Data (Pillar 2L)

The model is now strong enough for viable self-play:
- Policy 825 (was 315 at the failed 2g/2h attempts)
- MCTS 1,791 (was 891 at the failed attempts)
- The "dumber teacher" problem (lesson 15) may not apply at this level

Plan: generate 1,000-2,000 games from current model (MCTS 400 sims), mix with expert
data for training. Self-play data provides value targets calibrated to the model's own
play level, addressing the distribution shift that architecture changes cannot fix.

### Pillar 2L: First Self-Play Loop (HALLUCINATION FIXED — 5/7 seeds now improve with depth)

**Self-play data generation:** 2,000 games from 2k-surv model (MCTS 400 sims, τ=1.0 for
15 moves, Dirichlet α=0.3). Mean score 1,516, 1.44M states. Generated on M5 Max (16
workers) + Colab L4 (8 workers).

**Training:** 70% expert (12.8M states) + 30% self-play (1.44M states) mixed in each batch.
Self-play positions randomly replace 30% of expert positions during collate. Same survival
hybrid target, same architecture. H100, epoch 5 best.

**Results (50 seeds, 400 sims):**
| | 2k-surv | **2L** |
|---|---|---|
| MCTS mean | 1,791 | 1,657 (-7%) |
| Policy mean | 825 | **903 (+9%)** |
| MCTS boost | +117% | +84% |
| Max | 5,326 | **6,552** |

400-sim mean dipped 7% — "tactical dilution" from mixing weaker self-play data.
Policy improved to 903 (best ever). Max score improved to 6,552.

**THE CRITICAL RESULT — 1,600-sim hallucination test (same 7 seeds):**

| Seed | 2k-surv @400 | 2k-surv @1600 | 2L @400 | 2L @1600 |
|---|---|---|---|---|
| 1 | 4,131 | 1,355 (-67%) | 471 | **1,246 (+165%)** |
| 7 | 3,494 | 9,277 (+165%) | 4,823 | 2,320 (-52%) |
| 10 | 5,326 | 4,411 (-17%) | 1,779 | **6,078 (+242%)** |
| 18 | 4,054 | 2,098 (-48%) | 1,573 | 941 (-40%) |
| 26 | 3,580 | 1,200 (-66%) | 1,455 | **1,907 (+31%)** |
| 35 | 3,423 | 1,661 (-51%) | 1,209 | **2,916 (+141%)** |
| 43 | 4,619 | 982 (-79%) | 1,685 | **6,545 (+288%)** |
| **Mean** | **4,090** | **2,998 (-27%)** | **1,856** | **3,136 (+69%)** |

**Complete reversal:** 5/7 seeds now IMPROVE with 4x more search (was 1/7 before self-play).
Seeds 43 (6,545) and 10 (6,078) exceed the heuristic player (5,700 mean) with deep search.

| Model | Seeds improved @1600 | Seeds regressed |
|---|---|---|
| 2j (TD returns, no self-play) | 1/7 | 5/7 |
| 2k-alpha (bigger head, no self-play) | 0/7 | 7/7 |
| 2k-surv (survival, no self-play) | 1/7 | 5/7 |
| **2L (survival + self-play)** | **5/7** | **2/7** |

**Why it works:** The value head now evaluates positions from its own play distribution
correctly. Self-play data showed the model "when I play from HERE, I survive X more turns"
instead of only "when the expert plays from HERE." The compounding error at depth is
greatly reduced because the value head recognizes its own failure modes.

**Remaining issues:**
- 400-sim mean dipped 7% (tactical dilution from 30% weaker data)
- 2/7 seeds still regress (value head still pessimistic on some high-quality positions)
- Gemini analysis: "Pessimistic Judge" — self-play data at 1,500 mean anchors the value
  head to expect mediocre outcomes, causing it to veto brilliant moves on some seeds

### Pillar 2M: Self-Play Iteration 2 (800-sim teacher, 80/20 mix)

1,500 games at 800 sims from 2L model (mean 1,754). 80/20 expert/self-play.
Epoch 2 best. MCTS@400=1,559, MCTS@1600=3,908 (6/7 improved). Best 1600-sim result.

### Pillar 2N: Self-Play Iteration 3 (800-sim teacher, 80/20 mix, v3 data)

1,760 games at 800 sims from 2M model (mean 1,818). 80/20 mix. Epoch 1 best.
MCTS@400=1,683 (trend reversed upward), MCTS@1600=2,829 (5/7 improved, high variance).
Attempted lr=3e-5 first — model barely learned, wasted run. Reverted to lr=1e-4.
Also wasted a run when --resume silently skipped missing file (fixed: now errors).

### Self-Play Plateau Discovery

After 3 iterations, the self-play loop plateaued:
- Self-play scores: v1=1,516 → v2=1,754 → v3=1,818 (+15% per iteration)
- 1,600-sim eval: 3,136 → 3,908 → 2,829 (no clear upward trend)
- Root cause (Gemini): student caught the teacher. Model scores 1,791 at 400 sims,
  self-play teacher scores 1,818 at 800 sims. Only 1.5% search advantage → no gradient.

### Pillar 2P: Strategic Escalation (1,600-sim teacher, 60/40 mix, IN PROGRESS)

Scaled search to 1,600 sims. 965 games completed so far (mean 2,528, +39% over v3).
- 11.1% of games exceed heuristic level (5,700)
- Best game: 16,623 (7,740 turns)
- Score/turn nearly constant across all tiers (2.0-2.2) — confirms game is about SURVIVAL

### Expert vs Self-Play Quality Analysis

Compared 500 expert heuristic games vs 965 NN self-play games:

| Metric | Expert | NN Self-Play |
|---|---|---|
| Score/turn | 2.14 | 2.07 |
| Mean survival | 2,430 turns | 1,190 turns |
| CV (luck sensitivity) | 0.92 | 0.91 |
| Lucky/unlucky ratio | 9.9x | 8.9x |

**Key finding:** Both players score at nearly the same rate (2.14 vs 2.07). The ONLY
difference is survival time. Both are equally luck-dependent (CV≈0.92). The expert
data teaches nothing the NN hasn't already learned — scoring efficiency is matched.
Expert survival comes from brute-force 200-rollout search, not learnable strategy.

**Conclusion:** Expert data is dead weight. The NN needs to learn survival from its own
experience (self-play), not from imitating a heuristic that "survives" via brute-force.
Plan: drop expert data entirely after Pillar 2P.

### Pillar 2P: Pure Self-Play (ALL-TIME RECORDS — MCTS@400=1,933, @1600=4,357)

**First pure self-play training.** No expert data — 100% self-play v4 (1,595 games at
1,600 sims from 2N model, mean 2,625, 1.97M states). 15 epochs, warm start from 2N.

**Multi-epoch training WORKS with pure self-play:**
Train pol CE dropped steadily: 2.159 → 2.123 → 2.105 → ... → 2.041 over 15 epochs.
No 1-epoch saturation (was the chronic problem with expert data).
Val loss was NOT predictive — best val at epoch 1, but best MCTS eval at epoch 6.

**MCTS eval by epoch (50 seeds, 400 sims):**
| Epoch | MCTS mean | Policy mean | Max |
|---|---|---|---|
| 1 | 1,667 | 894 | 4,869 |
| **6** | **1,933** | **1,044** | **10,115** |
| 10 | 1,685 | 1,118 | 5,666 |
| 15 | 1,543 | 1,036 | 5,150 |

Epoch 6 is the sweet spot. Policy broke 1,000 for the first time. Seed 49 hit 10,115
at just 400 sims. After epoch 6, model overfits to the self-play distribution.

**1,600-sim hallucination test (epoch 6):**
| Seed | @400 | @1600 | Change |
|---|---|---|---|
| 7 | 1,487 | 5,438 | +266% |
| 10 | 898 | 2,009 | +124% |
| 18 | 709 | 5,019 | +608% |
| 43 | 1,069 | **12,223** | +1044% |
| **Mean** | **1,324** | **4,357** | **+229%** |

5/7 seeds improved. Mean @1600 = **4,357** (new record). Seed 43 hit **12,223** —
more than 2x the heuristic player (5,700). Highest NN MCTS score ever.

| Model | MCTS@400 | MCTS@1600 (7 seeds) | Best score |
|---|---|---|---|
| 2k-surv (pre-selfplay) | 1,791 | 2,998 (1/7) | 9,277 |
| 2M (best mixed) | 1,559 | 3,908 (6/7) | 8,386 |
| **2P ep6 (pure self-play)** | **1,933** | **4,357 (5/7)** | **12,223** |

**Tactical diagnostic — no catastrophic forgetting:**
| | Before (2N) | After (2P ep6) | Drop |
|---|---|---|---|
| Top-1 match | 50.3% | 45.8% | -4.5% |
| Top-5 match | 87.8% | 84.6% | -3.2% |

Small decline (~4%) is expected — the model now makes its own moves from self-play
learning, not just imitating the heuristic. Still agrees with expert 85% of the time.

**Key insight:** Val loss is meaningless for pure self-play training. Best val-loss
checkpoint (epoch 1) was NOT the best model. Only MCTS eval reveals true strength.
Future iterations should test multiple epochs by MCTS eval, not val loss.

### Lessons Learned (Phase 4 continued)
8. **Loss magnitude imbalance is deadly.** MSE value loss (860) vs CE policy loss (1.4) means
   value gradients steamroll policy features in shared backbone. Always check loss magnitudes.

9. **Training from scratch requires massive data.** 400K states with 8x augmentation is not enough
   for a 12M param model. Fine-tuning preserves valuable features.

10. **The "Dumber Teacher" problem.** Self-play MCTS (1100 mean) is weaker than the heuristic
    teacher (5700 mean). Training on weaker data regresses the model.

11. **Shared backbones create gradient conflicts.** Policy and value heads need different features.
    When one dominates training, it corrupts features needed by the other.

12. **Separate networks lose "tactical eyes."** Decoupled ValueNet (6b×128ch) scored MCTS 400
    vs baseline 911 despite MAE=6. Value head needs jointly-learned backbone features.

13. **Asymmetric joint training works.** val_weight=0.001 lets the value head learn as a
    "passive observer" without corrupting the policy backbone. First successful iteration.

14. **Don't over-train on converged data.** Pillar 2f plateaued by epoch 9. More epochs on
    the same data won't help — need better data (new self-play from improved model).

15. **Self-play data hurts when teacher is too weak.** Pillars 2g/2h proved that ANY amount of
    selfplay value data (even elite, even value-only) degrades MCTS when the self-play teacher
    (mean 891) is weaker than the expert teacher (mean 5700).

16. **@torch.no_grad() on collate is mandatory.** GPU observation building (BFS, line scans)
    in collate creates 150+ intermediate tensors per batch. Without no_grad, autograd tracks
    them all, leaking memory until OOM.

17. **itertools.cycle() caches all GPU tensors.** Using it to cycle a GPU dataloader leaks
    the entire dataset into memory. Use manual iterator restart instead.

18. **max_score must match actual value range.** TD returns (γ=0.99) peak at ~286. Using
    max_score=30000 wastes 99% of sigmoid resolution. max_score=2000 gives 31-point bins.

19. **game_score is a game-level label — use TD returns instead.** Turn 1 and turn 2000 of a
    5000-point game should NOT get the same value target. TD returns give position-specific
    future potential (mean ~196, max ~286).

20. **Cross-language RNG for deterministic Rust engine.** SplitMix64 in both Python and Rust.
    Custom RNG avoids reverse-engineering numpy's PCG64+SeedSequence.

21. **Golden tests guard performance optimizations.** Any optimization that changes game scores
    has altered the algorithm. Verify exact score match before/after.

22. **Scalar sigmoid compresses value signal.** TD returns [0-286] in sigmoid×2000 range means
    logits cluster around -2.2. Categorical heads (64 bins) express fine-grained differences better.

23. **The "Mid-Game Blob": γ too high → no contrast.** γ=0.99 averages 84% of positions into
    [190-240] TD range. Value head can't distinguish healthy from dying. γ=0.95 (half-life 14)
    focuses on tactical horizon, combined with max_score=100 gives 1.59 pts/bin resolution.

24. **Endgame oversampling is essential.** Death spiral positions (last 100 turns) are only 4.1%
    of expert data but carry the critical "this board is dying" signal. 30% oversampling ensures
    the value head sees enough contrast between healthy and terminal board states.

25. **Training losses improving ≠ inference improving.** Pillar 2i had perfect rank loss (0.096)
    and low anchor MAE (30) but MCTS went from +76% to +16% boost. Always eval MCTS, not just
    training metrics.

26. **Test with MORE sims to detect value hallucination.** If MCTS score drops when increasing
    from 400 to 1,600 sims, the value head is overconfident and wrong. Deeper search amplifies
    value errors. The value head must be trustworthy before increasing sim count.

27. **Value head bottleneck kills search.** 8-channel conv (256→8) throws away 97% of backbone
    information before the FC layers see it. The value head can't form strategic judgments
    from 8 features per cell. Increase to 32+ channels.

28. **Pairwise ranking of two good moves doesn't teach danger.** top-1 vs top-5 are both
    expert-quality moves. The value head learns fine distinctions between "good" and "slightly
    less good" but never sees "bad." Adversarial pairs (top-1 vs random) teach the strategic
    floor — what catastrophically wrong looks like.

29. **Dropout prevents value head memorization.** With 12.8M states, the 183K-param value head
    overfits in 2 epochs. The head memorizes board→value shortcuts instead of learning
    generalizable geometric features. Dropout forces redundant representations.

30. **Seed-level analysis reveals hidden failures.** Pillar 2j's 1,323 mean looked great but
    hid the fact that 5/7 top seeds regressed with more search. Mean scores mask per-seed
    disasters. Always check whether more sims helps or hurts.

31. **Diagnose before fixing.** We wasted an H100 run on architecture changes (2k-alpha) that
    didn't help because we hadn't identified the root cause. The forensic diagnostic
    (diagnose_value_head.py) took 5 minutes and revealed TD returns have r=-0.036 correlation
    with board health. Always measure the failure before proposing solutions.

32. **TD returns are a terrible value target for survival games.** They measure "are points
    scored nearby?" not "is this board healthy?" A dying board frantically clearing lines
    gets similar TD to a quiet healthy board (27 vs 31). The "Inverted-U" trap: the metric
    goes UP as the board gets MORE desperate because clearing is more frequent.

33. **Survival time is the true strategic signal.** In Color Lines, score is an OUTPUT of
    survival — if you stay alive longer, you score more. Predicting remaining turns gives
    monotonic, high-contrast signal (471 vs 2,414 for dying vs healthy). 485x stronger SNR
    than TD returns.

34. **Hybrid reward balances survival with quality.** r(t) = 1 + score/10 prevents pure
    survival from ignoring scoring opportunities. Survival dominates (base 1.0/turn) but
    scoring provides tiebreaking between equal-survival moves.

35. **The 1,600-sim test is the true benchmark.** If more search HELPS, the value head is
    trustworthy. If it HURTS, the value head hallucinates. Architecture changes that improve
    400-sim scores but fail at 1,600 sims haven't solved the real problem.

36. **Distribution shift is the final boss.** Training on expert positions but evaluating on
    NN-explored positions creates a gap no reward formula can close. The model needs to see
    its own failure modes. Self-play is the only fix once the reward signal is correct.

37. **Always run tests before declaring code ready for Colab.** Changed model defaults broke
    2 tests that weren't caught until Colab. Run pytest as part of pre-flight verification.

38. **Self-play fixes distribution shift when the model is strong enough.** At MCTS 891 (Pillars
    2g/2h), self-play data degraded the model. At MCTS 1,791 (Pillar 2L), self-play data
    fixed the hallucination. The threshold is roughly when MCTS > policy × 2.

39. **The 1,600-sim test flipped from 1/7 to 5/7 with self-play.** This proves the value head
    went from "hallucinates on novel positions" to "correctly evaluates its own play."
    Architecture changes alone (2k-alpha: 0/7) couldn't do this — only seeing its own data works.

40. **Self-play data dilutes expert policy quality.** 30% self-play (1,500 mean) mixed with
    expert (5,700 mean) caused a 7% dip in 400-sim MCTS. Use 20% self-play for next
    iteration to protect the tactical foundation.

41. **Increase search depth for self-play generation as the model improves.** Since more search
    now helps (5/7 improved @1600), use 800 sims for the next iteration's self-play to
    generate higher-quality training data (~2,500 mean vs ~1,500 at 400 sims).

42. **The self-play loop plateaus when student ≈ teacher.** Model at 1,791 vs teacher at 1,818
    = only 1.5% search advantage. No learning gradient. Fix: increase teacher sims (800→1600)
    to restore the search advantage gap.

43. **Expert data becomes dead weight once scoring efficiency is matched.** Expert scores
    2.14 pts/turn, NN scores 2.07 — nearly identical. The expert's advantage (2x survival)
    comes from brute-force rollout search, not learnable patterns. Self-play is the only
    path to learning survival.

44. **Color Lines is 100% a survival game.** Score/turn is constant at ~2.1 across all skill
    levels (500-point games to 16,000-point games). The ONLY difference between good and bad
    games is how long you survive. This validates the survival hybrid value target.

45. **--resume must error on missing file.** Silent skip caused an entire H100 training run
    to start from scratch (random init). Always fail loud on missing inputs.

46. **Lower learning rate doesn't fix epoch saturation.** lr=3e-5 made the model learn nothing
    in 2 epochs (pol barely moved). The saturation comes from stale data (12.8M expert states
    seen every iteration), not from learning too fast. Fix: more fresh data, not slower learning.

47. **Pure self-play breaks the 1-epoch saturation.** With 100% fresh self-play data, training
    improved steadily for 6+ epochs (pol CE: 2.16→2.07). The saturation was caused by stale
    expert data, not by the training setup.

48. **Val loss is meaningless for pure self-play.** Best val-loss (epoch 1) was NOT the best
    model (epoch 6). Val loss measures fit to a held-out split of self-play data, not game
    strength. Use MCTS eval as the only metric. Test multiple epoch checkpoints.

49. **Epoch 6 was the sweet spot for 2M-state pure self-play.** Early epochs underfit, late
    epochs overfit to the 1.97M self-play distribution. The optimal epoch scales with dataset
    size — more data → more useful epochs before overfitting.

### Pillar 2Q: Escape Velocity Attempts (VALUE BLOB PARADOX DISCOVERED)

**Attempt 1 (v4+v5 combined):** Mixed v4 (1.97M stale states from 2N) with v5 (2.38M
fresh states from 2P). Peaked at epoch 4 MCTS@400=1,801 — below 2P's 1,933.
Diagnosis: v4 data was stale (already trained on in 2P). 45% of batches had zero
learning gradient. Same failure mode as expert data staleness.

**Attempt 2 (pure v5):** Trained on 4.68M pure v5 states (1,953 games, mean 5,117).
Policy improved to 1,244 (+19%) but MCTS@400 declined from 1,933 to 1,871.
MCTS boost collapsed: +85% (2P) → +59% (ep3) → +39% (ep6).

**1,600-sim test (2Q ep3):** Mean 3,347 (7 seeds). Seed 26 hit 10,720. The model IS
learning but needs deeper search to show it. Not a total failure — but not the jump
we expected from 5,117-mean data.

**Root cause: The Value Blob Paradox.**
With γ=0.95 and games averaging 2,397 turns, **97.4% of v5 positions have V > 23**
(the blob). Only 1.3% have V < 20 (meaningful endgame signal). Only 4.2% are in the
last 100 turns (vs 8.1% for v4's weaker games).

The better the model plays → longer games → fewer death examples → less value contrast.
The value head STARVES for training signal as the model improves. In a typical batch:
- 70% non-endgame: V ≈ 24.3 ± 0.6 (zero gradient)
- 30% endgame oversampled: V ≈ 19.9 ± 5.9 (the only useful signal)

**The fundamental paradox:** Improving play quality HURTS value training quality.
The policy keeps improving (learns from 1,600-sim search targets regardless of value
distribution), but the value head degrades (no contrast in the survival target).

**Fix plan for 2Q-v3:**
1. γ=0.98 (half-life 34 turns): expands useful zone from 100→300 turns, pushes
   saturation to V≈62.5. Positions 100-200 turns from death become distinguishable.
2. Endgame oversampling 50% (from 30%): if 70% of batch has zero gradient, give more
   to the endgame pool.
3. rank_weight=2.0 (from 1.0): stronger mid-game ranking signal since categorical CE
   can't discriminate within the blob.
4. Same v5 data, just re-encoded with new gamma. No new generation needed.

50. **ALWAYS sanity-check data distribution before training.** The v5 blob (97.4% of data
    at V≈24.3) should have been caught before wasting H100 compute. Check value target
    histograms, endgame fractions, and contrast metrics before every training run.

51. **Stale data from previous iterations provides zero gradient.** v4 data in 2Q-attempt-1
    was already learned in 2P. Mixing old+new self-play has the same failure mode as
    mixing expert+self-play. Use ONLY the latest generation's data.

52. **The Value Blob Paradox: better play → worse value training.** With γ=0.95, stronger
    models produce longer games with fewer deaths. 97.4% of positions compress to V≈24.3.
    The value head can't learn mid-game discrimination. Fix: raise γ (wider horizon),
    increase endgame oversampling, strengthen ranking loss.

53. **γ must scale with game length.** γ=0.95 was perfect when games lasted 700 turns
    (v1, 2,625 mean). At 2,400 turns (v5, 5,117 mean), it compresses everything.
    γ=0.98 (half-life 34 turns) matches the longer horizon. As games get even longer,
    γ may need to increase further.

### Pillar 2R: Value Head SNR Crisis (DIAGNOSED)

**Diagnosis: Value head has 0.03 SNR for move discrimination.**
- Expert mid-game boards: V = 64.5 ± 13.3
- Blunder boards (expert + 1 random move): V = 63.6 ± 13.7
- Signal (gap): 0.4 points. Noise (std): 13.3. SNR = 0.03.
- The value head CAN distinguish game stages (early=65, death=14) but CANNOT
  distinguish good moves from bad moves on general boards.

**Afterstate ranking was "dead" — trivially satisfied.**
Adversarial ranking (top-1 vs random) had rank_loss=0.05. ~99% of pairs already
satisfied the 5.0 margin. The model trivially distinguished "best move" from
"random nonsense" with a 22-point gap. Zero gradient for fine-grained discrimination.

**2R-v1 (hard ranking pairs): Confirmed afterstate gap ≠ board gap.**
Switched to pre-computed top-1 vs top-5 MCTS pairs with margin_target=10.
Gap immediately 22 points with only 25% violations. Gap stayed flat at 21.3 across
4,000+ batches — the model already had afterstate discrimination from 2P training.
Afterstate ranking doesn't transfer to board-level evaluation because MCTS evaluates
leaf nodes (general boards post-spawn), not clean afterstates.

**2R-v2 (val_weight=0.1): Give the value head a voice.**
Root cause: val_weight=0.01 gave value CE only 1% of total gradient. The backbone
had zero incentive to learn value-discriminating features. Increased to 0.1 (11% of
gradient). Early results: MAE dropped 38→27 in 2 epochs, policy stable at 2.07.

**Game autopsy: The 8-square cliff.**
Analyzed worst V5 game (seed 50726: 62 points, 56 turns) vs good game (seed 50004:
10,807 points, 5,000 turns). Heuristic tournament player scores 7,800+ on same seed.
- Both games had identical temperature damage (12 random moves, 6 outside top-5)
- Difference: good game cleared a line during temperature → 58 empty at turn 13 vs 50
- 8 extra squares → 32% clear rate vs 20% → positive feedback loop vs death spiral
- Model treats 50-empty and 58-empty as identical (0.03 SNR) — can't trigger recovery
- Density reward encodes 50 vs 58 as ~4 bins (3.0 value points) — learnable with
  sufficient val_weight

**Self-play data stored log-visit-fractions, NOT Q-values.**
top_scores in JSON = log(visit_count/total_visits), not MCTS Q-values. The build
script recovers the visit distribution via softmax. Value predictions are not stored
in the self-play data.

54. **Value head SNR = 0.03 makes MCTS blind.** The value head predicts 64.5±13.3 for
    all mid-game boards. With Q normalization, MCTS gets random value signals. Policy
    prior drives all search decisions. The "MCTS boost" comes from policy refinement
    through visit counts, not value guidance.

55. **Afterstate ranking ≠ board-level discrimination.** The model separates top-1 vs
    top-5 afterstates by 22 points (trivially easy) but general boards by only 0.4.
    Afterstates differ in line potentials and connectivity; general boards (post-spawn)
    have 3 random balls that destroy the clean afterstate signal.

56. **Gradient starvation kills value learning.** val_weight=0.01 gave value CE only
    1% of total loss. The backbone optimized 42x more for policy. The value head was
    a passive observer that never shaped backbone features. Fix: val_weight=0.1.

57. **8 empty squares compound into 10,000+ point difference.** In Color Lines, a
    small early advantage (58 vs 50 empty at turn 13) cascades: more room → more lines
    → more room → survival. The model can't see this because the value head treats both
    as identical. The density reward encodes this as ~4 bins — learnable once the
    backbone is forced to pay attention (val_weight=0.1).

58. **Temperature exploration is NOT the cause of catastrophic games.** Good and bad
    games have identical temperature damage (~12 random moves, ~6 outside top-5).
    The difference is RNG-driven clear rate during temperature. Reduce to 5 moves
    for next self-play to avoid unnecessary variance without losing exploration.

### Pillar 2S-2T: The Value Head Plateau (PROVEN DEAD END)

**2S (density TD, max_score=200):** MCTS@400 = 2,271 on CUDA. Better data (mean 6,971
vs 5,117) gave NO improvement over 2R (2,389 on MPS). The "victory blob" persists:
41% of capped games at V≈90, IQR=3.9 within capped games. max_score=200 removed
ceiling clamp (0% clipped) but didn't fix the compression.

**2T (sqrt(remaining_turns) value target):** The topology pivot. sqrt(turns) has
IQR=29.1 (6.6x more contrast than density TD's 4.4). But MAE stuck at 20-22 for
8 epochs despite val_weight=1.0. The value head cannot learn to predict remaining
turns — the backbone can't extract structural features (connectivity, partitions)
while serving the policy. MCTS@400 = 2,266 (ep3), collapsed to 1,449 (ep6).

**Five value approaches, same result (~2,200 MCTS@400):**
| Target | val_weight | MCTS@400 |
| TD returns | 0.01 | ~2,100 |
| Density TD | 0.1 | ~2,400 |
| Density TD ms=200 | 0.1 | 2,271 |
| sqrt(turns) | 0.1 | 1,830 |
| sqrt(turns) | 1.0 | 2,266 |

**Root cause: the shared backbone conflict.** The policy needs local features (line
potential, path availability). The value head needs global features (connectivity,
partition detection, cluster analysis). A single 10-block ResNet can't learn both.
Every val_weight increase steals backbone capacity from policy.

### Pillar 2U: Pure Policy Distillation (BREAKTHROUGH)

**The fix: drop the value head entirely.** val_weight=0, rank_weight=0. All 13M
parameters focused on one job: learn the 1,600-sim search distribution.

Warm start from 2P ep6 (best policy checkpoint, 50.5% top-1 accuracy, before
val_weight increases destroyed it). Trained on V6 data (1,654 games, mean 6,971).

**Results — best standalone policy EVER:**
| Epoch | Policy Mean | Policy Median | Max |
| 3 | 1,394 | 1,036 | 3,719 |
| 5 | 1,410 | 1,184 | 5,158 |
| 8 | **1,763** | 1,176 | **9,061** |

Previous best: 954 (CUDA), 1,244 (MPS). Epoch 8 is +85% over CUDA best.
Five seeds scored 3,000+ with ZERO search. Seed 12 scored 9,061 — expert level
from a single forward pass.

**The backbone conflict was the entire problem.** Removing the value head unlocked
+85% policy performance. The val_weight increases across 2R→2S→2T were actively
DESTROYING the policy while the value head never learned.

59. **Human domain expertise cracked the value head mystery.** The project owner's
    insights — "danger is multi-colored clusters," "I never score below 1,000,"
    "the game might be infinite for a perfect player" — directly led to: (a) the
    tipping point analysis proving structural features matter more than density,
    (b) diagnosing that val_weight increases destroyed policy (the regression from
    1,244 to 943 was OUR fault), (c) the realization that Color Lines may not need
    a value head at all, leading to the Pillar 2U breakthrough.

61. **The shared backbone is a zero-sum game.** val_weight=0.01→0.1→1.0 progressively
    destroyed policy (1,244→954→943) while never improving value MAE. The backbone
    can serve policy OR value, not both. For Color Lines, policy wins.

62. **Drop the value head for pure policy distillation.** val_weight=0, rank_weight=0.
    The policy improved from 954 to 1,763 (+85%). The backbone concentrated 100% on
    learning move selection, resulting in expert-level play (9,061 max) with no search.

63. **Color Lines may not need a value head at all.** A perfect player survives
    infinitely — the "value" of every healthy board is the same (∞). Value prediction
    is only useful in the endgame. The policy can learn survival tactics directly from
    the search distribution without needing an explicit value function.

64. **The 400/1600 convergence ratio tracks value head quality.** Matched-seed comparison
    (89 seeds): 400-sim mean / 1600-sim mean = 0.34. Zero correlation (r=0.014) between
    scores at different sim counts — search depth dominates seed difficulty.

65. **V6 self-play: temperature_moves=5 eliminated catastrophic games.** V5 had 11.2%
    under 1,000 (min 62). V6 has 6.3% under 1,000 (min 282). Mean improved 5,117→6,971
    (+36%). 39% of games hit the 5,000-turn cap.

66. **The tipping point is at 41 empty squares.** Boards 50 turns before death look
    identical to healthy boards (43.3 vs 42.8 empty). The difference is structural:
    connectivity, partition risk, multi-color cluster density. A human sees this
    instantly; the density reward encodes the same score for both.

67. **"Easy" positions have FEWER empty squares than "hard" positions.** Search entropy
    analysis: uncertain decisions (top-1 < 30%) happen at mean 44.7 empty, while
    confident decisions (top-1 > 70%) happen at mean 39.1. Crowded boards have obvious
    forced moves; open boards have subtle strategic choices.

68. **Skip pairwise collate when rank_weight=0.** Saves ~30% per batch by avoiding
    pair observation building and extra forward pass when ranking loss is disabled.

### Pillar 2U MCTS Evaluation: THE BREAKTHROUGH

**MCTS@400 (epoch 8, 50 seeds):** mean=**5,436**, median=3,940, max=**30,544**.
This is **+139% over previous best** (2,271) and **95% of the heuristic** (5,700).
Seed 48 scored 30,544 — first 30K+ score at 400 sims.

**MCTS@1600 (epoch 3, 50 seeds):** mean=**8,552**, median=6,249, max=**30,330**.
This is **+23% over previous** (6,971) and **150% of the heuristic** (5,700).
Five seeds scored 20K+. Three seeds scored 25K+.

**400/1600 convergence ratio improved from 0.32 to ~0.64** (2x improvement).
Despite the value head being garbage (MAE=67), the stronger policy guides
MCTS so effectively that search quality doubled.

**The value head was HURTING search.** With a broken value head removed from the
loss (val_weight=0), MCTS@400 jumped from 2,271 to 5,436. The previous value
head actively misled search on ~24% of seeds. Now only 16% of seeds regress.

69. **A strong policy compensates for a broken value head.** MCTS@400 with no value
    learning (MAE=67) scored 5,436 — vs 2,271 with active value training. The policy
    prior alone drives +208% MCTS boost through visit count refinement. The value
    head was a net negative.

70. **MCTS@400 nearly matches the heuristic tournament player.** 5,436 vs 5,700 (95%).
    At 1,600 sims, the NN exceeds the heuristic by 50%. The AlphaZero approach works
    — but only when the backbone is fully dedicated to policy.

71. **30K+ scores are achievable.** Seed 48 scored 30,544 at 400 sims. Multiple seeds
    exceeded 20K at 1,600 sims. The 15-20K target range is within reach. The game IS
    "infinite" for a sufficiently strong player — some seeds survive 15,000+ turns.

### Dynamic Sims: Adaptive Search for Self-Play

**The problem:** With 2U's strong policy (1,763 standalone), games last 3,000-5,000
turns. At 800+ sims per move, a single game takes ~80 minutes. Generating 1,000
games for training takes days.

**The insight (from human player analysis + Gemini):** When the policy is confident
(P_max > 0.5), the visit distribution from 50 sims is nearly identical to 1600 sims
— the search just confirms what the policy already knows. Only when the policy is
uncertain (P_max < 0.3) does deep search add information.

**Implementation:** Check raw policy max prior (before Dirichlet noise) after root
expansion. P_max > 0.5 → 50 sims. P_max > 0.3 → sims/4. P_max < 0.3 → full sims.

**A/B test results (50 games each, same seeds 80000-80049):**
| Setting          | Mean  | Median | Min | Max    | Time | Capped |
|------------------|-------|--------|-----|--------|------|--------|
| 400 static       | 4,308 | 3,315  | 466 | 10,840 | 53m  | 14%    |
| 400 dynamic      | 4,018 | 3,158  | 248 | 10,865 | 21m  | 6%     |
| 800 dynamic      | 6,232 | 6,306  | 427 | 10,910 | 63m  | 22%    |

**Key finding:** 400 dynamic is 2.5x faster with only -7% score drop. 800 dynamic
gives +45% score over 400 static in the same compute budget. Dynamic sims saves
compute on "no-brainer" moves and spends it on the strategic crossroads.

**V7 generation plan:** 1,000 games with 1200 dynamic sims (estimated mean ~7,500).
100 existing games at 800 static + 900 new at 1200 dynamic. Same seeds dir, mixed
sim counts are fine — policy visit distributions are valid regardless.

72. **Dynamic sims: adapt search depth to policy confidence.** When P_max > 0.5,
    50 sims produces the same visit distribution as full search. Saves 2.5x compute
    on confident moves. The training targets remain consistent because the visit
    distribution for confident positions is determined by the policy prior, not
    the search depth.

73. **Check policy confidence BEFORE Dirichlet noise.** Dirichlet (weight=0.25) +
    30-way softmax caps P_max at ~0.6, making confidence thresholds unreachable.
    Raw priors reflect true policy certainty.

74. **Color Lines may be "infinite" for a perfect player.** Human insight: score is
    purely a function of survival time (~2.1 pts/turn). A player that maintains
    board connectivity indefinitely scores indefinitely. The value of every healthy
    board is ∞ — only endgame boards have finite value. This is why the value head
    failed: it tried to predict a number that doesn't meaningfully vary mid-game.

### Static vs Dynamic Sims: The Quality Gap

**Static 1600 sims (2U ep8) vs V6 baseline (2R ep3, 1600 static):**
| Metric | V6 (2R ep3) | V7 static 1600 (2U ep8) |
| Median | 7,629 | **10,618** (+39%) |
| Capped | 39% | **67%** |
| <1000 | 6.3% | **2.7%** |
| Min | 282 | **489** |

**Dynamic sims produces inferior data.** V7 dynamic 2000 (avg ~870 effective sims)
scored median 10,130 — close to static, but 4.6% games under 1000 vs 2.7% static.
The "Confidence Trap": when the model is confidently wrong (P_max > 0.3), dynamic
sims reduces to 100-500 sims, too shallow to correct the error. Static 1600 corrects
every mistake. The floor jumped from 282→489 with static.

75. **Dynamic sims is a Confidence Trap.** On "confident" moves (72% of turns), search
    drops to 100-500 sims. If the model is confidently WRONG, the shallow search
    confirms the error. Static 1600 sims corrects every mistake — floor improved from
    min=282 to min=489, <1000 dropped from 6.3% to 2.7%.

76. **Static search is non-negotiable for quality.** For production self-play, use full
    static sims on every move. Solve the speed problem through smaller surrogate
    models (5b×128ch = 4x faster inference), not through search shortcuts.

### Crisis Mining (Human Insight: "solve actual difficult positions")

**The idea (from project owner):** Use cheap policy-only play (0 sims) to instantly
find positions where the model dies, then replay from those positions with 1600+
sims to generate high-quality recovery training data.

**The flow:**
1. Policy plays instantly → dies at turn T (~1 second per game)
2. Recovery: rewind to T-25, replay with 2000 sims → "emergency survival"
3. Prevention: rewind to T-75, replay with 1600 sims → "avoid the crisis"

**Why it matters:** Full static games are 90% cruise control (healthy boards the model
already handles). Crisis mining concentrates compute on the 10% of positions where
the model actually fails. Each crisis scenario costs ~450K evals vs 8M for a full
game — 18x cheaper for the data that matters most.

**Early results:** Seed 100000 policy died at turn 185 (318 pts). The rw75 replay
at 400 sims was at turn 1000+ scoring 2,064 — the deep search SAVED the game from
a board where the policy gave up.

77. **Crisis mining: find failures with policy, solve with search.** Policy-only games
    are instant (~1 second). Use them to probe thousands of seeds for crisis positions.
    Then spend expensive search (1600+ sims) only on the boards where the model
    actually needs help. 18x cheaper per crisis scenario than full static games.
    Compatible with build_expert_v2_tensor.py for seamless training data.

78. **The self-play loop plateaued on same-quality data.** 2V training on V7 dynamic
    data showed zero progress (pol 1.854→1.846 in 4 epochs). The search didn't
    disagree with the policy enough. Fix: static sims + crisis mining provides
    positions where search finds genuinely different moves than the policy predicts.

### Pillar 2V: Static 1600 + Crisis Mining (BREAKTHROUGH)

**Data:** 892 static 1600 games (3.7M states) + 5,956 crisis replays (2.8M states)
= 6.5M total states. 57% full-game, 43% crisis (recovery rw20/25 + prevention rw35/50).

**Training:** Pure policy (val_weight=0), warm start from 2U ep8, 8 epochs on H100.
Policy loss dropped from 1.858 to 1.843 — the static data broke the plateau.

**Results (300-seed policy-only eval):**
| Epoch | Mean  | Median | Min | Max    |
| 3     | 2,452 | 1,786  | 271 | 12,986 |
| 4     | 2,318 | 1,651  | 248 | 11,096 |
| 5     | 2,489 | 1,854  | 223 | 25,411 |
| **6** | **2,680** | **2,088** | 219 | 13,582 |
| 7     | 2,441 | 1,788  | 47  | 12,163 |

**Best epoch: 6.** Policy mean 2,680 (+52% over 2U's 1,763), median 2,088 (+77%).

**MCTS@400 (epoch 6, 50 seeds):** mean=6,016, median=4,018, max=27,607.
+11% over 2U's 5,436. MCTS improvement is smaller because value head is still
garbage — the boost comes purely from stronger policy prior.

79. **Static sims + crisis mining breaks the self-play plateau.** The combination of
    full-depth search (every move gets 1600 sims) and targeted crisis replays
    (positions where the model fails) provides enough "correction signal" to keep
    the policy improving. Dynamic sims failed because it confirmed wrong priors.

80. **Crisis data is first-class training data.** Per-state, crisis replays are as
    valuable as full static games — both use 1600+ sims. Crisis data directly
    targets the model's failure modes, providing correction signal that full games
    (90% cruise control) cannot.

81. **Epoch 6 sweet spot for 6.5M states.** Epoch 7 overfits (min drops to 47).
    More data volume → more useful epochs before saturation.

### The Value Head Removal Disaster

Attempted to remove the value head entirely from the model architecture — deleted
ValueNet, DualNetWrapper, value_sum, Q-tracking, virtual loss through value_sum.
886 lines deleted. Clean code.

**MCTS broke completely.** MCTS@400 went from +50% boost to -49% (actively worse
than policy-only). Games scoring 8,000+ dropped to 176-517.

**Root cause:** Virtual loss through value_sum is load-bearing for batched MCTS.
Without it, multiple leaves in a batch of 64 all go down the same branch, wasting
90%+ of search budget on redundant simulations. Even the "garbage" value head
(trained with val_weight=0) provided essential Q-diversity for exploration.

**Fix:** Reverted MCTS stack to last known-good commit (bfb5fd9). The value head
stays in the model for MCTS inference (provides Q-diversity) but gets zero training
gradient (val_weight=0 in training).

82. **Virtual loss through value_sum is load-bearing.** Even a garbage value head
    provides Q-diversity that guides MCTS exploration. Without it, PUCT with only
    policy priors flattens the visit distribution — MCTS becomes worse than
    policy-only. The value head is dead for training but essential for search.

83. **Don't remove infrastructure you don't understand.** The value_sum virtual loss
    mechanism looked like dead code (value always ~0) but was providing critical
    exploration diversity. Test MCTS after any refactor, not just training.

### Pillar 2W: Openings + Crisis + Mid-game

**Data (V8):** 500 full s1600 (1.8M) + 3K openings at 2000+ sims (650K) +
8K crisis replays (456K) = 2.9M states. Targeted: 62% mid-game, 22% openings,
16% crisis.

**Training:** Pure policy, warm start from 2V ep6, 8 epochs on G4.
Policy loss: 1.809 → 1.792. Val loss improved every epoch — no overfitting.
The diverse data mix (openings + crisis) provided enough signal for 8 full epochs.

**Results (1000-seed policy-only eval, GPU fp16):**
| Model | Mean | Median | Min | Max |
| 2V ep6 | 2,584 | 1,899 | 121 | 16,410 |
| 2W ep5 | **2,972** | 2,163 | 209 | 17,354 |
| 2W ep8 | 2,935 | **2,203** | 110 | 18,913 |

**+14% mean, +16% median over 2V.** Steady improvement from the self-play loop.
The floor remains low (min ~110-209) — needs more work.

84. **Diverse data prevents overfitting.** 2.9M states with 22% openings + 16% crisis
    supported 8 epochs without overfitting. Previous iterations (6.5M homogeneous
    mid-game states) overfit by epoch 6-7. Quality × diversity > volume.

85. **GPU policy eval via InferenceServer.** Route policy-only games through the
    shared-memory GPU server for 2x speedup. fp16 gives different per-seed results
    than fp32 CPU (MPS non-determinism) but means converge at 1000+ seeds.

86. **Opening data at 2200 sims improves mean but not floor.** The first 200 turns
    at deeper search teach better board setup. But catastrophic games (min ~150)
    still occur — structural blind spots in the policy persist.

### Pillar 2W2: Double Data from Same Teacher

Doubled V8 data from 2.9M to 5.8M states (6K openings at 2000+ sims, 21K crisis
replays, 500 full s1600 games). Trained from 2V ep6 (fresh start on full data).

**Training:** Loss 1.810 → 1.784 over 10 epochs. Lower than 2W's 1.792.
No overfitting through all 10 epochs on 5.8M diverse states.

**Results (1000-seed policy-only eval, GPU fp16):**
| Model | Mean | Median | <500 | <1000 | >5000 |
| 2V ep6 | 2,584 | 1,899 | 6.2% | 22.5% | 11% |
| 2W ep8 (2.9M) | 2,935 | 2,203 | 4.4% | 19.2% | 16% |
| **2W2 ep7 (5.8M)** | **3,175** | **2,458** | 4.4% | 18.7% | **18%** |
| 2W2 ep10 | 3,140 | 2,340 | **4.2%** | **16.7%** | 18% |

**Total progression from single teacher (2V ep6):**
Mean +22%, median +29%, <1000 dropped 22.5%→16.7%, >5000 grew 11%→18%.
Floor (P1~300, <500~4.4%) stubbornly flat — needs new teacher for V9.

87. **More diverse data from same teacher still helps.** Doubling data from 2.9M to
    5.8M with more openings (2200 sims) and crisis replays gave +8% mean over 2W.
    The model hadn't exhausted the teacher's signal — it needed more DIVERSITY of
    positions, not just volume.

88. **Fresh start beats warm start when data doubles.** Starting from 2V ep6 on the
    full 5.8M outperformed warm-starting from 2W ep8 (which had already trained on
    half the data). The stale half provided zero gradient in the warm-start run.

### Surrogate Model: FAILED

5b × 128ch model (2.9M params, 4x faster inference). Trained 15 epochs on V8
data (5.8M states). Policy-only scored 390 mean (vs 10-block's 3,200). The model
is too small to learn the policy — 128ch width can't represent the move patterns.

CoreML ANE backend also attempted (4.4x faster at bs=1). Queue overhead between
workers and inference server ate the advantage. Direct worker mode caused ANE
contention with multiple models. Net speedup: only 28%. Abandoned.

89. **Surrogate model needs sufficient width.** 5b×128ch (2.9M params) scored
    390 standalone — 8x worse than 10b×256ch. The 128-channel bottleneck
    loses too much feature capacity. Width matters more than depth for policy.

90. **CoreML ANE: fast inference, slow system.** 4.4x faster per-eval but
    queue overhead (0.3ms) between workers and server dominates. The speedup
    only helps when inference is the bottleneck; in our architecture,
    communication is.

91. **fp16 vs fp32 policy eval gives different scores.** GPU fp16
    (InferenceServer) inflates policy-only scores ~31% vs CPU fp32 (Pool
    workers). 2W2 ep10: GPU fp16 policy mean=4,489, CPU fp32 policy
    mean=3,425. Same model, same seeds — the difference is purely numerical
    precision in softmax tails. Always compare policy baselines and MCTS
    scores using the same precision path.

92. **MCTS boost declines naturally with stronger policy.** The apparent
    "MCTS collapse" from +124% (2U) to +26% (2W2) is not regression — it's
    policy convergence. A stronger standalone policy already makes good
    moves, so search has less room to improve. 2U: policy 1,763 → MCTS
    3,953 (+124%). 2W2: policy 3,425 → MCTS 4,310 (+26%). The MCTS
    absolute score still improved.

93. **Random Q-values > systematically wrong Q-values for MCTS.** Calibrating
    the value head to predict empty_squares/81 (frozen backbone, 2 epochs)
    produced systematic bias that hurt exploration. The calibrated model's
    MCTS boost dropped to -2% vs +26% with the untrained (random) value
    head. Random noise provides unbiased Q-diversity for virtual loss;
    a weakly-trained value head introduces correlated errors that
    consistently mislead the search.

### The V9 Selfplay Collapse and Value Head Investigation

V9 selfplay (2W2, 1600 sims) collapsed to mean=4,299 (9% capped) from V8's
7,709 (50% capped) despite 2W2 having a stronger standalone policy. Extensive
diagnosis revealed the root cause: the untrained value head provides garbage
Q-values that override the policy on 12% of moves, and this is net-harmful
for strong policies.

94. **Ghost moves in MCTS simulations.** The shared RNG (`sim_rng`) across
    batched simulations caused board state divergence at depth 2+. Cached
    children at a tree node could be illegal on a different simulation's
    board (different ball spawns). `trusted_move()` executed these invalid
    moves, corrupting simulation boards. Fixed with open-loop MCTS: filter
    children to those legal on each simulation's actual board during PUCT
    selection. The saved game data was always valid (real game uses `move()`
    with BFS pathfinding), only MCTS simulations were affected.

95. **Open-loop MCTS for stochastic games.** Standard AlphaZero MCTS assumes
    deterministic games (same action = same outcome). Color Lines has random
    ball spawns, so the same tree node can represent different board states
    across simulations. Open-loop MCTS fixes this: during PUCT selection,
    check each cached child's legality against the current simulation's
    board (`board[src] != 0 and board[tgt] == 0`). Skip illegal children.
    Break if no children are legal. Nodes represent action sequences, not
    specific board states.

96. **Prior peakedness is NOT the cause of MCTS degradation.** Diagnosis
    across models showed nearly identical prior distributions:
    2U P_max mean=0.411, 2W2 P_max mean=0.435. Both have P_max > 0.9 on
    only 1% of turns. Gemini's theory of "violently peaked priors (0.99)"
    was empirically disproven.

97. **The garbage value head is the Goldilocks zone — accidentally.**
    Comprehensive testing of value alternatives on seed 36, 2W2 @400 sims:
    - Garbage neural value (c_puct=2.5): 3,782 (12% disagreement) — best
    - Zero value (amputated): 2,788 (0% disagreement) — pure policy
    - High c_puct=25.0: 2,247 (2% disagreement) — suppressed exploration
    - Heuristic board eval: 742 (30% disagreement) — confidently wrong
    The garbage value head provides just enough random exploration (12%
    override rate) without being confident enough to consistently mislead.
    Removing it or replacing it with a "real" signal both made things worse.

98. **MCTS batch_size critically affects value network quality.** With a
    trained value net, batch_size=64 gives MCTS mean=1,830 (-39% vs policy),
    while batch_size=8 gives MCTS mean=7,294 (+218%). The first batch runs
    blind (q_range=0, Q-values invisible). batch_size=64 wastes 16% of sims
    blindly; batch_size=8 wastes only 2%. With a real value signal, more
    informed iterations = dramatically better search. With garbage values,
    batch_size doesn't matter (more iterations of noise doesn't help).

99. **Separate value network: premature victory.** The initial 4-seed test
    (+218%) was misleading. At 50 seeds, the trained ValueNet HURT both
    models: 2U -27%, 2W2 +2%. The "breakthrough" was a fluke from small
    sample size + multiple confounding bugs discovered later (see 100-103).

100. **fp16/fp32 policy eval mismatch invalidated per-seed comparisons.**
     Policy eval used CPU fp32 pool, MCTS eval used GPU fp16. Same seed
     produced different games → per-seed "MCTS destroyed this game" was
     comparing completely different games. Diagnosis of seed 19 confirmed:
     0% disagreement between policy and MCTS, same score (388). The
     "6,243 → 388 (-94%)" was entirely a precision artifact. Fixed by
     routing policy eval through GPU InferenceServer (fp16) to match MCTS.

101. **Root value bug: value_net never got a fair test.** `_nn_evaluate_single`
     always uses the policy model's garbage value head, even when value_net
     is set. The root gets garbage value (~100 for 2W2), which anchors
     min_q = max_q = 100. First value_net leaf returns ~0.95 → q_range = 99.
     All subsequent value_net Q-values (0.95-1.0) compress to q_norm ≈ 0.0.
     The garbage root value poisons Q-normalization for the entire search.
     Discovered by ChatGPT code review.

102. **Garbage value head outperforms trained value net.** With valid fp16
     comparison and 50 seeds at 400 sims:
     - 2U + garbage head: MCTS mean=5,257 (+130% over policy 2,282)
     - 2U + trained ValueNet: MCTS mean=1,666 (-27%)
     - 2W2 + garbage head: MCTS mean=4,188 (+21% over policy 3,465)
     - 2W2 + trained ValueNet: MCTS mean=3,530 (+2%)
     But: the ValueNet comparison is invalidated by bug #101. The garbage
     head's Q-diversity (2U: std=14, 2W2: std=7) drives exploration.
     2U's wider spread explains its bigger MCTS boost.

103. **2W2 MCTS is worse than 2U MCTS despite stronger policy.** 2W2 MCTS
     absolute score (4,188) is LOWER than 2U MCTS (5,257) even though
     2W2 policy (3,465) is higher than 2U policy (2,282). The garbage
     value head produces less Q-diversity for 2W2 (std=7 vs std=14),
     giving MCTS less exploration signal. This needs further investigation.

109. **Frozen spatial head: 62.5% offline → +24% MCTS.** A frozen random
     spatial mixer (conv+BN+FC, kaiming_normal_ init, only fc2 trainable
     with 513 params) gave 62.5% offline accuracy but +24% MCTS boost.
     Compared to the linear GAP head (74.4% offline, +3% MCTS). Spatial
     structure matters for search even when it hurts offline ranking.

110. **Decomposing the garbage head: no special part.** Swapped projector
     and fc2 independently across original and fresh random weights:
     - Original garbage head: mean=4,084
     - orig_proj + rand_fc2: 3,241 / 4,389 / 3,782 (3 seeds)
     - rand_proj + orig_fc2: 5,208 / 4,234 / 3,933 (3 seeds)
     - rand_proj + rand_fc2: 4,098
     Neither the projector nor fc2 is special. A random projector with
     original fc2 (5,208) beat the original (4,084). All combinations
     land in ~3,500-5,000. The median (2,936) is invariant across all
     variants — 14/20 seeds are unaffected by the value head entirely.
     The mechanism is ANY random nonlinear spatial projection of backbone
     features, not specific learned weights.

111. **Offline accuracy is anti-correlated with MCTS performance.**
     Survival ValueNet (96.3% acc) → -17% MCTS. Linear exact (74.4%)
     → +3%. Frozen spatial (62.5%) → +24%. Garbage head (untrained)
     → +27%. Training on MCTS visit preferences makes things worse
     because the objective is misaligned with what search actually needs.

### Known Bugs To Fix (discovered via ChatGPT + Gemini code review)

- **Root value bug**: `_nn_evaluate_single` ignores `self.value_net`.
  Root always gets garbage value, poisoning Q-normalization for value_net.
- **ValueNet accuracy inflated**: 96.3% from position-level split (not
  game-level), adjacent states leak between train/val. Binary threshold
  on saturated target is easy to pass without learning move ranking.
- **Diagnosis q_range wrong**: `mcts_deep_diagnosis.py` recomputes min/max
  from root children only, but MCTS uses global running min/max from all
  backed-up leaves.
- **ValueNet fp32 in workers**: per-worker value_net loaded without .half(),
  runs fp32 while policy runs fp16.

### Value Ablation Study (clean terminal handling)

All tests: 2W2, 400 sims, 20 seeds (0-19), terminal_value=0.0 for all
synthetic modes. Policy mean=3,465 on these seeds.

104. **The garbage value head provides structured state-dependent signal.**
     Clean ablation with normalized terminals:
     - Garbage head (terminal=0): mean=4,407 (+27%)
     - IID noise std=7: mean=3,335 (-4%)
     - IID noise std=14: mean=3,217 (-7%)
     - IID noise std=50: mean=2,756 (-20%)
     - Deterministic hash: mean=2,069 (-40%)
     - Zero value: ≈ policy (no search benefit)
     The garbage head beats all forms of random noise. It's not random —
     it's a frozen random projection of trained backbone features that
     accidentally provides useful state-dependent value ordering.

105. **IID noise helps via visit-dependent uncertainty, but it's weak.**
     Fresh IID noise per evaluation means less-visited nodes keep more
     Q-variance, creating an exploration bonus similar to Thompson sampling.
     But it lacks state consistency (same board → different values each
     time), so the signal is much weaker than structured projections.
     Smaller std is better (7 > 14 > 50) — less noise = more policy trust.

106. **Terminal value contamination was a confound.** Raw game.score on
     terminal leaves (hundreds/thousands) mixed with synthetic values
     (0-200 or 0-1) broke Q-normalization. Previous IID results (+130%
     for 2U) were inflated by terminal leakage. Fixed via terminal_value
     parameter that normalizes terminals to 0.0 for synthetic modes.

107. **A frozen policy backbone contains search-useful latent structure.**
     Testing 5 freshly randomized value heads (different random seeds,
     same frozen 2W2 backbone) on 20 seeds, terminal=0.0:
     - Original garbage head: mean=4,407
     - Random head seed=0: mean=3,534
     - Random head seed=1: mean=3,835
     - Random head seed=2: mean=3,896
     - Random head seed=3: mean=4,084
     - IID noise std=14: mean=3,217
     All random projections of the backbone work (3,534-4,084).
     The original head isn't special — it's one sample from a family
     of useful readouts. The backbone features encode information
     valuable for search, and many random readouts can expose it.

108. **Value head only matters on ~30% of seeds.** 14 out of 20 seeds
     produced identical MCTS scores regardless of which value head was
     used (original, 5 random heads). On these seeds, the policy
     dominates search completely. The value head only influences the
     6 seeds where MCTS faces genuine decision uncertainty. This means
     value head improvements have a bounded ceiling: even a perfect
     value function can only help on the minority of positions where
     the policy is uncertain.

### Phase 19 — Feature-Based Value Evaluator (the breakthrough)

112. **The pillar2w2 NN value head has near-zero predictive power for
     survival.** Trained val_weight=0, so the head is an untrained
     linear projection of backbone features. Tested on 27,900 sampled
     positions from selfplay_v8_combined + crisis_v2 against label
     log(1+remaining_turns):
     - NN value head: Pearson r=+0.086, val R²=0.0043
     - Output range: [54.3, 122.0] on max_score=200 (mean=99.1)
     The head is essentially constant + tiny noise. It carries effectively
     no causal signal for position quality. MCTS@1600 (mean=4,550) was
     achieving its lift from policy priors alone, in *spite* of the
     value head, not because of it.

113. **Linear regression on 18 board features beats NN value 29×.**
     Same dataset, same label. Features = mine_death_features.board_features
     (16 raw: empty, components, largest, tiny_comp, mobility, avg_reach,
     min_reach, low_mob_balls, balls, colors_present, same_adj, diff_adj,
     line3, line4, center_balls, center_colors) + 2 derived (ratio,
     frag_score). Ridge regression with standardization:
     - Features alone: val R²=0.1259
     - NN value alone: val R²=0.0043
     - Combined: val R²=0.1260 (NN coef collapses to -0.003)
     The combined model gives features all the weight and ignores the
     NN value entirely — the head has zero unique information beyond
     what the cheap features provide. R² ratio: 29×.

114. **Feature-value MCTS lifts mean from 3,465 → 8,625 (+149%).**
     pillar2w2_epoch_10 + 18-feature linear evaluator as MCTS leaf
     value, replacing the NN value head:
     - Policy-only (50 seeds, 0-49): mean=3,465
     - NN-value MCTS@400 bs=8: 4,115 (+19%)
     - NN-value MCTS@1600 bs=8: 4,550 (+31%)
     - Feature-value MCTS@400 bs=8: **6,312 (+82%)**
     - Feature-value MCTS@1600 bs=8: **8,625 (+149%), median=10,456**
     The catastrophe seeds where MCTS *destroyed* good policy games
     largely recovered: seed 41 (policy 8,850 → NN-MCTS 916 → feat-MCTS
     10,629), seed 14 (2,112 → 622 → 7,316), seed 12 (2,061 → 408 →
     6,824). Feature-value@400 bs=8 already beats NN-value@1600 bs=8 by
     39% with 4× less compute. At 1600 sims, ~62% of games hit the
     5K turn cap (V7-territory survival rate). Confirmed policy-agnostic
     by testing on pillar2u_epoch_9: policy 2,282 → feature-MCTS 4,295
     (+88%) — same proportional lift. ChatGPT's framing was exact:
     "MCTS becomes a policy improvement operator only when the value
     signal is better than the policy's own implicit judgment." R²=0.13
     features clear that bar; R²=0.004 NN value doesn't.

115. **Batch size > 8 destroys MCTS quality (virtual-loss starvation).**
     Same model, same value source, same sim count, only batch_size
     changed:
     - 400 sims, bs=8: mean=6,312 (+82% vs policy)
     - 400 sims, bs=64: mean=2,889 (-17% vs policy)
     Mechanism: virtual loss is a *placeholder* applied to a path while
     a leaf is queued for batched NN eval. It pushes sibling sims to
     different parts of the tree before the real Q values arrive. With
     bs=8/400 sims, MCTS gets 50 batches of real feedback; with bs=64,
     only 6. The tree never converges on Q-good paths because most sims
     descend with stale virtual losses dominating real Q. Don't trade
     batch size for throughput — scale workers (parallel games) instead.

### Implementation (mcts.py + eval_parallel.py)

- `_evaluate_features_linear(board, coefs, means, stds, bias)` JIT
  function in `alphatrain/mcts.py`. Calls board_features() (already
  @njit), adds the 2 derived features, applies standardization and
  linear combination. ~0.65us per call — zero MCTS slowdown.
- `MCTS(feature_weights_path=...)` loads weights from .npz and uses
  `_evaluate_features_linear` for every leaf value (including game-over
  terminals; the feature function naturally returns low values for
  jammed boards). Root value is also overridden to keep min_q/max_q
  consistent with subsequent leaves.
- `eval_parallel.py --feature-value-weights PATH` wires through both
  server-mode (`_eval_mcts_worker`) and local-mode (`make_mcts_player`).
- Weights produced by `alphatrain/scripts/fit_feature_value.py` from
  ~28K positions sampled across V8+crisis_v2. Output: 18-element
  ridge regression with feature standardization stats.

### Diagnostic scripts (used to discover this)

- `alphatrain/scripts/feature_value_ceiling.py` — Pearson r per feature
  + linear R² ceiling on remaining_turns. Found R²=0.13 ceiling.
- `alphatrain/scripts/feature_vs_nn_value.py` — head-to-head: NN value
  vs features on identical boards. Found 29× R² advantage for features.

### Phase 20 — Architecture decision: drop the value head

116. **The NN value head is permanently abandoned.** A series of
     value-training attempts (Pillar 2b ranking head, 2g/2h self-play
     value training, 2j categorical+TD, 2r SNR=0.03 diagnosis, 2u/2v
     drop-and-restore experiments) plus the recent V9 collapse
     investigation converged on the same answer: **for this game, the
     value head is dead weight**. With val_weight=0 (the policy
     distillation setup that produced our strongest models — 2U, 2V,
     2W, 2W2), the value head outputs are a near-constant random
     projection of backbone features (R²=0.0043 on survival
     prediction). The 18-feature linear evaluator beats it by 29× and
     is faster. Given the variants we've tried, there's no obvious
     path where retraining the value head improves search quality.

117. **V10 self-play data has bootstrap_value=0 for capped games.**
     This is the explicit signal that "we are not going to consume the
     value-head signal at training time." Anyone in the future who
     enables value training on V10 data will find capped boards labeled
     as "future return = 0", which would teach the value head that
     turn-6000 boards are deaths — exactly the wrong lesson. The
     bootstrap=0 is intentional and load-bearing for the policy-only
     training path.

118. **V10 quality validates the decision.** First 660 regular games:
     mean 8,102, median 8,949, 47% > 10K score, 39% capped at 6000
     turns, 2.5M states. V7-territory at 800 sims (V7 was 8,858 at
     1600 sims). The policy is strong, MCTS is helping, and the value
     head plays no role beyond consuming compute and checkpoint bytes.

### Architectural implications (planned for V10 training onward)

The value head's role in inference has already been short-circuited
in feature mode (`mcts.py` skips `predict_value` and `val_logits.cpu()`
when `feature_coefs is not None`). The next step is to make the value
head **optional at training time** via a flag — not to delete the code
path, since we still need to load and test 2U/2V/2W/2W2 (which all
have value heads).

**Both modes are supported in parallel:**

- `policy_only=False` (default) → existing dual-head architecture. All
  pre-V10 checkpoints continue to load and run identically to today,
  including their NN value head if anyone wants to compare to feature
  value or use it for some other experiment.
- `policy_only=True` → smaller architecture, no value head. New
  V10+ training uses this. Forward returns `policy_logits` only.

Concrete changes:

- **Model class** (`alphatrain/model.py`): add `policy_only=False` flag
  to `AlphaTrainNet`. When True, skip building `value_conv`, `value_bn`,
  `value_fc1`, `value_fc2`. `forward()` returns `policy_logits` only;
  otherwise it returns the existing tuple.
- **Training** (`alphatrain/train.py`): when `policy_only=True`, set
  `val_weight=0` automatically, skip value-loss computation and value
  target loading. Reject `val_weight > 0` at config time as a misuse.
- **Inference** (`mcts.py`, `evaluate.py`, `inference_server.py`):
  callers branch on `getattr(net, 'policy_only', False)`. Both shapes
  are handled. When `policy_only=True`, MCTS asserts that a value
  source is provided (feature evaluator or external value net) — the
  policy net alone has no value to back up.
- **Checkpoints**: write `policy_only` (True or False) in metadata.
  `load_model` reads the flag and instantiates the matching
  architecture. Old checkpoints without the field default to False.

Expected savings on `policy_only=True` models: ~5-10% NN forward
time (value head FLOPs eliminated), slightly smaller checkpoints
(~1-2MB). For old models the architecture is unchanged.

When to actually delete the dual-mode code: when we have 2-3
generations of strong policy-only models (2X, 2Y, ...) and no longer
need to compare against 2W2-era checkpoints. Probably around 2Z.
Until then, both modes coexist.

### Phase 21 — V10 results (pillar2x family)

Policy-only evaluation, 500 seeds (0..499), 400 sims, batch_size=8,
deterministic mode. These are the canonical baseline numbers for
comparing to future iterations.

| Model | Train run | Mean | Median | <500 | <1000 | >5000 | >10000 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pillar2v_endgame ep6 | V8/V9 era | 2,452 | 1,746 | 6.4% | 27% | 12% | 1% | 11,454 |
| pillar2w2 ep10 | V8/V9 era | 2,934 | 2,248 | 4.2% | 20% | 17% | 1% | 16,833 |
| **pillar2x ep6** | V10, lr=1e-4 | **3,450** | 2,492 | 3.8% | 16% | 23% | 4% | 24,636 |
| **pillar2x2 ep10** | V10, lr=3e-4 | **4,110** | **3,314** | **1.6%** | **12%** | **32%** | **6%** | 23,314 |

119. **Iteration cadence: each V-step adds ~18-20% to policy-only mean.**
     2V → 2W2: +20% (2,452 → 2,934). 2W2 → 2X: +18% (2,934 → 3,450).
     2X → 2X2: +19% (3,450 → 4,110). Stronger lr (3e-4 vs 1e-4) was
     the difference between 2X and 2X2 — same V10 corpus, same warm-
     start, different optimizer. The "stubborn loss" we observed at
     lr=1e-4 (val plateau ~2.09) was a local minimum, not a global
     floor. lr=3e-4 dislodged it (val=2.07) and the resulting policy
     scored 19% higher. Future iterations: prefer 3e-4 with no decay
     for fine-tune-from-warmstart.

120. **Floor came up dramatically with V10 → pillar2x2.** Sub-500-score
     games dropped from 4.2% (2W2) → 1.6% (2X2). Sub-1000 from 20% → 12%.
     Tail came up too: >5000 from 17% → 32%. >10000 from 1% → 6%.
     This is the V9 collapse fully reversed plus genuine forward
     progress. Each iteration produces both a stronger floor (fewer
     catastrophic deaths) and a heavier tail (more capped/long games)
     simultaneously — the right shape for survival-game improvement.

121. **The feature evaluator improves with stronger self-play data.**
     Re-fitting feature_value_weights on V10 (selfplay_v10_s800 +
     crisis_v10, ~60K positions) jumped val R² from 0.1259 (V8 fit)
     to 0.2060 — +64%. Coefficient pattern shifted: V8 fit was
     dominated by `mobility` (-0.55); V10 fit by `largest` (+0.85)
     and `low_mob_balls` (-0.58). Several features sign-flipped due
     to multicollinearity redistribution (avg_reach, ratio). The
     combined V(board) prediction is what matters for MCTS leaf
     eval; evaluator should be re-fitted each V-iteration as the
     state distribution shifts toward longer/healthier trajectories.
     `feature_value_weights_2x.npz` is the V10-fitted version.

### Roadmap to 30K median / 3K floor target

At ~20%/iteration on policy-only mean:
- Current: 2X2 = 4,110 mean / 3,314 median / 867 P10
- After 5 more iterations (~5 weeks if 1 week each): mean ~10K
- After 10 iterations: mean ~25K
- Target hit: probably 12-15 iterations from here

Floor is the harder problem. P10 doubled 666 → 867 across 2W2 → 2X2,
which is faster than mean. Same compounding rate gets P10 to 3K in
~7 iterations. Tail is easier; floor is the bottleneck.

### Phase 21 addendum — feature-value-weights re-fit on V10 (`_2x.npz`)

Re-fit feature_value_weights on V10 self-play (selfplay_v10_s800 +
crisis_v10, ~60K positions). Val R² jumped 0.1259 (V8 fit) → 0.2060
(+64%). Top coefficients: `largest`=+0.85 (new dominant), `low_mob_balls`
=-0.58, `avg_reach`=-0.48 (sign-flipped vs V8 fit due to multicollinearity
with largest), `diff_adj`=-0.40. Saved as feature_value_weights_2x.npz.

A/B at MCTS@400 sims, pillar2x2_ep10, 50 seeds:

| Weights | Min | P10 | Median | P90 | Max |
|---|---:|---:|---:|---:|---:|
| V8 fit (old) | 541 | 1,300 | 7,916 | 10,505 | 10,566 |
| **V10 fit (new)** | **579** | **2,862** | **10,230** | 10,486 | 10,566 |

The headline impact is **P10 doubled** (1,300 → 2,862, +120%) and
**median +29%** (7,916 → 10,230). Cap-clipped tail unchanged (everyone
hits the cap floor anyway). Mean was misleading at +11% — the real
win is in floor + median, which mean compresses with capped data.

122. **Lesson: focus on min/median/P10 in capped-game regimes.** With
     56-74% of MCTS games hitting the 5K-6K cap, mean is dominated by
     cap value × cap rate, not by quality differences in the body of
     the distribution. Median and P10 separate iterations cleanly;
     mean does not.

123. **The feature-value evaluator should be re-fitted each V-iteration.**
     The R² jump from V8 fit to V10 fit (0.13 → 0.21) plus the +29%
     median lift in MCTS suggests the evaluator needs fresh data as
     the policy distribution shifts. Cheap (5 min, ~60K positions)
     and high-leverage. Rule: refit before each major self-play
     campaign.

### Phase 21 addendum 2 — sims sensitivity test (600 vs 400)

Goal: check if 600 sims gives meaningful quality improvement over 400
to inform V11 sim count.

pillar2x2_ep10 + V10 weights, 50 seeds, batch_size=8. Cap is 5,000
turns ≈ 10,000-10,500 points, so games > 10K are "hit the cap":

| Sims | Min | P10 | Median | %<1K | %>10K (cap) | Wall/game |
|---|---:|---:|---:|---:|---:|---:|
| 400 | 579 | 2,862 | 10,230 | 2% | 56% | ~110s |
| **600** | **1,145** | 2,363 | **10,354** | **0%** | **64%** | 256s |

600 sims is **genuinely better on most metrics**: min nearly doubled
(579 → 1,145), %<1K dropped to zero, cap-hit rate +8pp (56→64%),
median +1%. Only P10 dropped (2,862 → 2,363) — likely sample noise
at n=50; per-seed inspection shows 600-sim **rescued** 400-sim
catastrophes (seed 19: 579 → 10,547) but **broke** a few good 400-sim
games (seed 35: 10,402 → 1,345). Net: fewer extreme failures, slight
shift toward cap-hits at the cost of a few mid-bucket games.

Wall comparison is subtler than raw per-game time suggests:
- Per-game wall: ~1.5× more for 600 sims (110s → 256s)
- BUT a chunk of that increase is because **600-sim games last longer**
  (more turns played → more moves → more wall) since the policy keeps
  them alive longer. Per-move cost is actually ~1.5× higher, not 2.3×.
- For 5K-turn-cap-hit games specifically, both configurations play
  out similarly long; the difference is in dying games where 600
  sims survives longer.

124. **Lesson: more sims meaningfully improves data quality, but at
     real wall cost.** 400 → 600 sims: min +98%, %>10K +8pp, median
     +1%. Wall +50% per move (more for full games because they last
     longer). For self-play data generation, 600 sims is worth the
     compute when targeting quality — better visit distributions,
     fewer truncated dying games in the corpus.

125. **Don't conflate per-game wall with per-move compute cost.** When
     comparing sim counts, longer games at higher sims are partly
     evidence of *quality* (the search keeps the game alive), not
     pure compute overhead. Decompose wall into (turns × per-move
     cost) before declaring "X is too expensive."

126. **Don't lean on a single percentile in capped regimes.** P10 dropped
     17% from 400 → 600 sims while min, median, P25, mean, %<1K, and
     cap rate all improved. With 50-seed samples, single-percentile
     swings are noisy; weight them less than the full-distribution
     trend.

### Phase 22 — q_weight calibration unlocks 2Y2 MCTS (May 2026)

After pillar2y2 (V11, 40 epochs) showed standalone strength (5,586 mean)
without matching MCTS gain — actually a regression vs 2X2 at 400 sims —
ChatGPT review flagged the diagnosis: PUCT's `q_norm + U` formula
overweights the noisy feature evaluator (Pearson r ≈ 0.5 with truth) on
distilled-policy generations. q_norm rescales any signal to [0,1],
giving a noisy r=0.5 estimator the same dynamic range as a perfect Q.

Added `q_weight` parameter to PUCT: `score = q_weight * q_norm + U`.

Sweep on 2Y2 + `_2y.npz` @ 400 sims, 50 seeds:
| q_weight | Mean | Median | P10 | %≥10K |
|---:|---:|---:|---:|---:|
| 0.00 | 6,104 | 6,860 | 924 | 24% |
| 0.25 | 6,755 | 7,387 | 1,475 | 44% |
| **0.50** | **7,517** | **9,248** | **1,821** | **42%** |
| 0.75 | 5,434 | 5,016 | 954 | 24% |
| 1.00 | 6,263 | 6,014 | 958 | 34% |

127. **q_weight calibration matters as policies get sharper.** A
     distilled student inherits the teacher's search through its
     policy, producing peaked priors. With q_weight=1.0, q_norm gives
     full PUCT swing to a low-r leaf evaluator; the noise pulls visits
     away from the policy's good moves more than truth-correlated
     signal pulls them toward better ones. Halving q_weight (0.5)
     proportions Q to its actual confidence and recovers +20% mean /
     +54% median on 2Y2 vs the q=1.0 baseline. q_weight is now a
     first-class hyperparameter — track it per generation.

128. **q=0 is also bad. The evaluator IS doing real work — at the right
     weight.** Pure-prior search (q=0) underperformed q=0.5 by 23%
     mean, 35% median. Bottom-line: don't drop the leaf evaluator,
     just don't overweight it.

### Phase 22 addendum — next-ball features didn't move the needle

Per-iteration ChatGPT review suggested adding next-ball-aware features
to the feature evaluator (delta_largest, delta_components, etc.) since
the policy network sees next_balls via observation channels 8-11 but
the 18-feature evaluator was board-only.

Implementation: extended `board_features` to `board_features_with_next`
emitting 6 new features (4 deltas after applying the 3 known spawns,
plus n_next_same_color_adj and n_next_blocked). Total feature count
went 18 → 24. Refit on V11 corpus (10K games × 10 positions = 80K
samples).

Result: **Val R² = 0.3427 vs the old 0.3424** — essentially zero gain.
All 6 new features got near-zero coefficients (max abs 0.031,
min 0.000).

129. **Linear scalar features can't represent next-ball interactions.**
     The deltas are mechanically small (3 balls perturb a 36-empty
     board's `largest` by 0-3), and that perturbation is largely
     redundant with the existing board features. Where next-ball signal
     would matter — chokepoint landings, chain reactions, specific
     tactical clears — those are non-linear in any reasonable scalar
     feature space. The next bottleneck for Q quality is *non-linearity*,
     not *missing inputs*. The infra is committed for future feature
     experiments, but the path forward for stronger Q is either: (a) NN
     value head distilled from the policy backbone, (b) survival-horizon
     classification that handles cap censoring properly, or (c) accept
     the linear ceiling and lean on stronger teacher search instead.

### Phase 22 addendum 2 — threshold-encoded line completion (still negative)

After the continuous `max_next_line` came back at +0.020 (noise floor),
we added two binary indicators alongside it: `next_makes_4plus` and
`next_makes_5plus`. The idea was the linear model can't represent a
step function with a single continuous coefficient, but binary
thresholds bypass that limitation.

Refit on V11 corpus (16K games × 10 positions = 128,290 samples):

```
max_next_line       +0.0184  (continuous)
next_makes_4plus    +0.0096  (>=4 threshold)
next_makes_5plus    -0.0085  (>=5 threshold, NEGATIVE)
```

R² unchanged at 0.343.

130. **Confounding sinks the threshold features.** `next_makes_5plus=1`
     should *causally* help survival (the spawn completes a clearable
     line, gain points, free space). But the regression assigns it a
     *negative* coefficient. Reason: the event is rare (~2-3% of
     positions) and correlated with crisis-phase positions where
     same-color clusters are dense from accumulated damage. The
     regression learns the correlation ("these are dying positions")
     instead of the local tactic ("this clear buys time"). With
     confounded features and no causal adjustment, linear regression
     locks onto the confound rather than the tactic. To capture
     "spawn enables clear → +survival" causally, you need either an
     interventional setup (compare to counterfactual spawn) or a
     model rich enough to factor the confound out (NN learns
     "high-cluster-density board, plus this spawn-clear" as a 2D
     interaction; linear regression cannot).

### Phase 22 addendum 3 — 800-sim Colab eval validates search headroom

After q_weight=0.5 calibration unlocked 2Y2 MCTS at 400 sims (mean
7,517 / median 9,248 on M5 seeds 0-49), ran an 800-sim Colab L4 eval
on a different 50-seed sample to test scaling.

| Sims | Mean | Median | P10 | %≥10K | Δ vs Pol |
|---:|---:|---:|---:|---:|---:|
| 400 | 6,323 | 7,188 | 1,373 | 36% | +21% |
| 800 | 7,530 | 9,966 | 1,660 | 50% | +45% |

Doubling sims: mean +19%, median +39%, %≥10K +14pp. The recalibrated
PUCT scales cleanly with search depth.

131. **The distillation ceiling tightened by miscalibrated PUCT, not
     by capacity exhaustion.** Earlier diagnosis ("2Y2 standalone
     improved but MCTS plateau'd → student saturated the teacher")
     was incomplete. With q_weight=1.0 the search was overweighting
     a noisy Q and *more sims compounded the miscalibration* — that's
     why MCTS looked plateaued. With q_weight=0.5, doubling sims
     400→800 lifts median by +39%. The student has NOT fully embodied
     the teacher's search; deeper search at inference still extracts
     real signal. Matters for V12 self-play strategy: more sims is
     still a productive lever before the NN-value-head escape.

132. **2Y2 needs ~2× the sims to match 2X2's old MCTS result.** 2X2
     + V10 weights @ q=1.0 @ 400 sims: 7,825 mean (HISTORY lesson 124).
     2Y2 + V11 weights @ q=0.5 @ 800 sims: 7,530 mean (different
     seed set; Pol baselines differ so direct comparison is noisy).
     Sim-count parity is roughly 2:1 — 2Y2's stronger policy means
     each sim adds less marginal information. This is the
     distillation-ceiling-as-search-cost: NOT "MCTS plateaus at this
     model" but "MCTS at this model needs more sims to break out
     past the previous-iteration's MCTS."

### Phase 3R — NN ValueHead beats linear evaluator (May 2026)

Built a tiny `ValueHead` (~8K params) on the **frozen** pillar2y2 backbone:
multi-horizon survival classifier predicting `P(survive ≥ H)` for H ∈
{25, 50, 100, 200}, combined into a scalar `V = 1.0·p25 + 0.8·p50 +
0.5·p100 + 0.25·p200`. Trained on V11 corpus (7.78M states, 5 epochs,
1.5h on M5). Calibration on K=64 rollout val set: r=0.91-0.95 across
horizons (epoch 3 best on both trajectory loss and calibration).

**Wired into the inference server in server-mode.** `_PolicyValueWrapper`
fuses `forward_with_features` + head into one traced GPU pass, writes
scalar V to `val_buf` per batch. MCTS in server mode reads V directly;
no per-leaf head re-run. 16-worker eval at full speed.

**Phase 1 sweep (200 sims × 50 seeds, max-turns 8000):**
- q=1.0: mean 9,152, %<1K 4%, %≥10K 34%
- q=1.5: mean 8,322, %<1K 2%, %≥10K 32%
- q=2.0: mean 9,527, %<1K 6%, %≥10K 50% ← winner

**Phase 2 (400 sims × 50 seeds, q=2.0, max-turns 5000):**

| metric | linear @ 400 | NN q=2.0 @ 400 | delta |
|---|---|---|---|
| mean | 7,517 | **9,051** | **+20%** |
| %<1K | 4% | **2%** | −50% |
| %≥10K | 42% | **74%** | **+76%** |
| %>5K | — | **88%** | — |

133. **Always sweep `q_weight` when introducing a new value source.**
     Initial 50×400 attempt at q_weight=0.5 (the linear-evaluator-tuned
     default) trended near 2X2 quality — looked like the head was
     broken. ChatGPT review pointed out: head's V ∈ [0, 2.55] (sum of
     weighted survival probs, max=1+0.8+0.5+0.25), totally different
     scalar distribution from the linear evaluator. q_weight tuned for
     one source mis-blends the other. 100-sim sweep across q ∈ {0.1,
     0.25, 0.5, 1.0, 2.0} showed monotone improvement past 0.5 — q=1.0
     and q=2.0 *each matched the linear baseline at 4× less sim
     budget*. The "head is bad" hypothesis was wrong; only the blend
     was wrong. Lesson: a new value source is a new scalar distribution;
     PUCT q_weight must be re-tuned, not inherited.

134. **Server-mode fused inference is the right architecture for
     auxiliary heads.** Local-mode head eval (per-worker MCTS owns the
     head) was 4-8× slower than server mode. The fused
     `_PolicyValueWrapper` keeps the head's compute inside the GPU loop
     where backbone features are already computed for the policy
     forward, so the marginal cost of the value head is essentially the
     8K-param head conv + GAP + linear (sub-millisecond on a 256ch
     featuremap batch). MCTS in server mode just reads `val_buf` — no
     per-leaf head invocation. Pattern generalizes: any small auxiliary
     head trained on the policy backbone should fuse into the server's
     forward, not run separately in MCTS.

135. **Cap-clip is a distribution skewer; track it explicitly.** The
     400-sim Phase 2 mean (9,051) was *lower* than 200-sim Phase 1 mean
     (9,527), violating the "more compute → better games" expectation.
     Reason: max-turns 5000 cap clipped many of the 74% reaching 10K+
     at ~10,300 (the empirical cap-hit ceiling, per P95=10,398 in
     earlier 100-sim runs). When a large fraction of games hit the cap,
     the cap becomes the dominant bound on the mean — extra search
     compute can't push past it. Add explicit cap-hit logging
     (`turns == max_turns` count) before drawing conclusions about
     mean differences across runs with different caps. The TRUE
     ceiling for q=2.0/400-sim requires re-running with max-turns 8000+.

136. **The "is the head broken?" diagnostic ladder works.**
     ChatGPT's recommended ladder before retraining the head was:
     (1) q_weight sweep (cheapest, no retrain), (2) fp16 parity smoke,
     (3) saturation check on visited leaves, (4) correlation with
     linear evaluator on the same leaves, (5) only then consider
     retraining target. Level 1 alone solved it here. Lesson: when a
     new component disappoints, try the cheapest alternative
     explanation first — scale-mismatch, blend-tuning, BN calibration,
     etc. — before assuming the component itself is broken. The "head
     doesn't work" hypothesis would have led to retraining or
     architecture changes; the actual fix was a CLI flag.

### Phase 4 — Iteration 1: pillar2z + value_head_v12 (May 2026)

First iteration of the proper AlphaZero loop with NN-driven self-play.

**Pipeline:**
1. Generator: pillar2y2 + value_head_v11 + q=2.0 (Phase 3R winner, mean 13,476).
2. V12 corpus: 9.77M states from 450 self-play games (400 sims, 8K cap)
   + 9,712 crisis recovery replays + 9,668 prevention replays
   (recovery 15t/600 sims, prevention 35t/400 sims, continue 500t).
   Crisis-heavy by design: 95% of corpus from failure-recovery
   trajectories per Pillar 2u's "perfect player survives infinitely"
   framing — only deaths matter.
3. Pillar 2z training: warm-start from pillar2y2_epoch_40, batch 32768,
   lr 3e-4, warmup 2 epochs, AMP + compile, `--policy-only`. Same
   recipe as 2y A/B winner. Stopped at epoch 19 when 1000-game
   policy-only evals plateaued.
4. value_head_v12 retrained on **pillar2z's frozen backbone** using same
   V11 survival targets, same 5-epoch recipe. Calibration r=0.89-0.94
   across horizons (vs v11's 0.91-0.95).

**Results (50 seeds × 400 sims × q=2.0 × 10K-turn cap):**

| metric | pillar2y2 + v11_head (8K cap) | pillar2z + v12_head (10K cap) |
|---|---|---|
| Policy-only mean | 5,586 | **7,460** (+33%) |
| MCTS mean | 13,964 | **15,465** (+11%) |
| MCTS P10 | 5,397 | **5,992** (+11%) |
| MCTS P50 | 16,440 (cap) | 20,004 (cap) |
| MCTS %≥10K | 78% | **82%** |
| MCTS %<1K | 1% | **0%** |

The MCTS comparison isn't perfectly apples-to-apples because pillar2y2's
8K-cap eval clips at ~16,400 while pillar2z's 10K-cap eval clips at
~20,500. P10 and policy-only are the cap-clean comparison: both up
~11-33%.

137. **Iter 1 of NN-driven self-play works.** Policy-only +33% over
     previous iter (5,586 → 7,460), MCTS +11% on cap-clean metrics.
     This is the first time the AlphaZero loop has been demonstrated
     for this project — each iteration's stronger player generates
     stronger data, which produces a stronger next student. Earlier
     attempts (pillar2g/2h era) failed because the generator was too
     weak relative to the model; NN-MCTS at q=2.0 generates target-range
     play that distills cleanly.

138. **Value head MUST be retrained when the backbone moves.** The
     value head is trained on FROZEN backbone features. When the
     backbone changes (e.g., pillar2y2 → pillar2z via policy distillation
     on V12), the head's training distribution shifts. Concretely:
     pillar2z + value_head_v11 + q=2.0 produced a *bimodal* MCTS
     distribution — half of games at P50 ~4,400 (catastrophic), half at
     P75 ~15K (fine). The head was hallucinating value on positions where
     pillar2z's features had drifted. Retraining the same architecture on
     pillar2z's backbone (same V11 targets, same recipe, ~1.5h M5)
     collapsed the bimodal failure into a smooth distribution. Cost is
     small; ship this as a standard step in every iteration.

139. **q_weight is robust across backbone retraining, at least here.**
     Phase 3R established q=2.0 for pillar2y2 + v11_head. After
     retraining the head on pillar2z's backbone, the q sweep
     (200 sims × 50 seeds × 10K cap) showed:
     - q=1.0: mean 9,885
     - q=1.5: mean 9,780
     - q=2.0: mean 12,509 ← clear winner
     Same q wins. Saves a sweep per iteration if this generalizes — but
     keep verifying since calibration metrics did shift slightly
     (v12 r 0.89-0.94 vs v11 0.91-0.95).

140. **At each iter, raise the eval cap.** Pillar2y2 was eval'd at 8K
     cap (most games at cap). Pillar2z at 8K cap looked *worse* (mean
     13,159) because the new policy could play more turns but ran out
     of clock. At 10K cap pillar2z reveals its true mean 15,465 — most
     metrics now at NEW cap (P75 = 20,436 cap-pinned). For Phase 5
     (iter 2 with V13 corpus from pillar2z), plan to eval at 12-15K
     cap. The cap moves with player strength; treat it as a method
     constant per-iter, not project-constant.

141. **The signal that "the iteration converged" is multi-metric, not
     val_loss.** Pillar2z val_loss kept decreasing slowly through
     epoch 19 (2.2237 → 2.2034). But policy-only 1000-game eval went
     5,586 → 7,341 (e11) → 7,229 (e15) → 7,460 (e19) — essentially
     plateaued from e11 onward. Loss said "keep training"; gameplay
     said "stop." For distillation tasks, gameplay eval at 5-epoch
     intervals is the right plateau detector. Don't burn 20 extra
     epochs chasing val_loss reductions that don't translate to score.

142. **Phase 1 oracle mining: 16,897 K=32 rollout-judged crisis anchors.**
     Built `phase1_oracle_fleet.py` (numba JIT, M5 baseline) and
     `phase1_oracle_fleet_gpu.py` (PyTorch + cudagraphs, Colab L4). The
     M5 JIT version reaches 65 rollouts/sec; the L4 GPU version with
     fixed-horizon batched mining + a single concat-compiled
     pre+forward+post step reaches ~70 r/s once cudagraphs stabilize.
     Output: per-anchor top-6 moves × K=32 rollouts × H=300, labeled
     with cap_rate / mean_turns / mean_score from pillar2y2 anchors.

143. **Path B v1: oracle as soft-KL auxiliary loss. FAILED.** Designed
     with ChatGPT input: reliability-weighted conditional KL on top-6
     candidates, β=10, λ ∈ {0.05, 0.10}, warm-start from pillar2y2_ep40,
     same V12 corpus and recipe as pillar2z but with a new pipeline
     (clean train/val split + sample-time random augmentation +
     `--color-augment` + `--warmup-epochs 1`). Four-run ablation
     A/B/C/D × 12 epochs each, ~3-4h Colab L4 per run:

     | run | recipe | ep12 policy mean |
     |---|---|---|
     | pillar2z (reference) | V12 distill, leaky split, warmup=2 | ~7,158 |
     | A | + new split + warmup=1 | **7,752** |
     | B | + color aug | **8,043** ← NEW BEST |
     | C | + oracle λ=0.05 (soft KL) | 7,060 |
     | D | + oracle λ=0.10 (soft KL) | 7,396 |

     **The oracle path is empirically harmful at convergence.** C/D
     peaked at epochs 5-7 (similar to B's mid-training) then regressed
     while B kept improving. By ep12, both oracle arms finished BELOW
     the no-oracle pipeline. λ=0.10 fades less than λ=0.05 but
     neither beats no-oracle.

144. **The new pipeline is the real win, not the oracle.** A_ep12
     beating pillar2z (+8.3%) and B_ep12 beating A (+3.8%) are the
     entire empirical lift. The deltas are dominated by:
     (1) clean train/val split (`make_train_val_split`) — previously
     `random_split` over augmented indices leaked dihedral variants
     from train into val, smoothing the val curve and selecting
     mid-training as "best" via val_loss criterion;
     (2) sample-time random dihedral + color permutation augmentation
     instead of indexed-deterministic 8× expansion;
     (3) `--color-augment` flag (7! color-permutation symmetry of the
     game). The combination probably gets ~5-10% from honest val and
     5-7% from augmentation variety. Color aug alone (B vs A) is
     ~+4% with statistical significance ~2σ at 500 seeds.

145. **Soft KL doesn't preserve discrete argmax flips at convergence.**
     C's oracle metrics through training tell a clean story:
     `KL_weighted` improved monotonically (1.07 → 1.06, beats pillar2z's
     1.06 plateau) AND `top1_all` rose 34% → 37%, BUT the discrete
     `top1≥.15` (model picks oracle's choice on high-margin anchors)
     peaked at 5.8% (ep7) and regressed to 2.9% (ep11). The model
     learns *average* oracle distribution but doesn't *commit* to
     specific corrections. Path B v2 will use hard CE on
     rollout-winner instead.

146. **BN-contamination hypothesis (from ChatGPT) — diagnostic
     DISPROVEN.** Hypothesis: concat-batch training updates BN running
     stats from the 11% crisis-distribution oracle observations, so
     BN drifts from V12 → mixed; deploying with mixed-BN MCTS pruned
     B's good moves. We wrote `recalibrate_bn.py` to forward 500
     V12-only batches through C_ep5/7/11/12 and update BN stats to
     pure-V12 distribution, then re-evaled gameplay:

     | epoch | C non-recal | C_bn recal | Δ |
     |---|---|---|---|
     | 5 | 6,468 | 5,952 | **−8.0%** (~3.8σ) |
     | 7 | 7,345 | 6,963 | **−5.2%** (~2.9σ) |
     | 11 | 7,233 | 6,740 | **−6.8%** (~3.7σ) |

     Recalibration *hurt* uniformly, not helped. The mixed-distribution
     BN running stats were either neutral or positively contributing
     to gameplay. Lesson: don't over-correct based on theory alone.
     The diagnostic was easy to write (~150 LOC) and gave a definitive
     answer in ~15 min M5.

147. **Multi-loss gradient diagnostic must use a SINGLE concat-forward,
     not two separate forwards.** Our first gradient-norm diagnostic
     ran V12 and oracle forwards separately on their own batches, then
     captured g_v12 and g_oracle. Result said:
     `‖g_v12‖=0.31, ‖g_oracle‖=4.89` at λ=0.05, cos ≈ −0.06 (mild
     conflict in deep backbone). ChatGPT flagged: training does ONE
     concat forward, so BN normalizes mixed-batch statistics across
     both losses. Re-running with proper single-forward +
     `loss.backward(retain_graph=True)` for both gave:
     `‖g_v12‖=0.66, ‖g_oracle‖=1.94, cos = +0.51`. Headline reversed:
     oracle was in "effective range" and POSITIVELY aligned with V12,
     not orthogonal or conflicting. Generalizable lesson: when
     measuring component gradients of a multi-loss training, match
     the actual training forward exactly. BN snapshot/restore across
     batches keeps the diagnostic non-mutating.

148. **MCTS@100 sims now hurts the strong policy.** Ran B_ep12 through
     `eval_parallel` with `value_head_v12_v12targets` + q=2.0 + 100
     sims + 100 seeds:

     | metric | policy-only | MCTS@100 |
     |---|---|---|
     | mean | 9,051 | 8,700 (−4%) |
     | P50 | 6,670 | 8,178 |
     | P95 | **24,965** | **16,510** (8K-turn cap) |

     The value head saturates at max_score ≈ 30K. B_ep12 routinely
     reaches 25K+ in long games. MCTS sees `V(state_A) ≈ V(state_B)`
     because the head can't distinguish "this leads to 25K" from "this
     leads to 40K" — both clipped at max_score. So MCTS prunes B's
     best moves. The asymmetric MCTS effect (floor improves, ceiling
     collapses) is consistent with "value head still discriminates bad
     states but can't tell good states apart." This finally proves
     empirically what the project had suspected since Pillar 3a-v3:
     value head is the bottleneck for further MCTS-driven progress.

149. **Score-regression value targets cannot scale past expert-level
     policy.** Color Lines is effectively an infinite game for a
     strong-enough player (board resets via line clears). Any
     score-based regression target (final score, TD return γ→1,
     cumulative survival) is bounded by `max_score`. Once the policy
     reaches scores near that bound (B_ep12 reaches 30K+ in long
     games), the value head's prediction saturates and the loss
     landscape provides no gradient distinguishing "good" from
     "great." All four trained value heads (v11 single-trajectory,
     v12 V11-targets, density head, spatial ranking head v3-v3a)
     saturated for the same reason. **The next value head must use a
     bounded-but-non-saturating target: pairwise/BPR ranking, finite
     survival probability, or discounted TD with γ ≤ 0.9.** Tracked
     in memory `project_path_b_v2.md`.

150. **Pivot decision: Path B v2 — oracle mining for POLICY, not value
     head.** With value head saturated and deployment target being
     policy-only (30K policy-only median per TODO), the cleanest path
     forward is direct policy improvement via state-level rollout
     judgment, applied on B_ep12's own distribution. Two-stage mining:
     cheap K=16-24 screen on ~6-10K stratified anchors (60% crisis,
     25% uncertain, 15% diverse-healthy), then K=128 on the ~1-1.5K
     promising candidates. Train next iteration with hard CE on
     rollout-winner at high-margin anchors. Decision criterion: ≥+8%
     over B_ep12 (≥8,700 mean on 500-seed policy eval). If it fails,
     we've hit the single-iteration ceiling for this corpus and need
     a structurally different ingredient (bigger model, different
     observation, retrained non-saturating value head). Scripts staged
     (`gen_b_selfplay.py`, `sample_b_anchors.py`, audit running);
     mining + training to follow.

151. **K=128 audit on existing 16K oracle anchors — labels collapsed.**
     Tested whether K=32 → K=128 changes ranking on the high-margin
     subset (Δcap ≥ 0.15 at K=32, 1,525 anchors). B-vs-oracle agreement
     on top-1 was 3.5%. On the 1,472 disagreements: **B wins 49.3%,
     oracle wins 45.9%, ties 4.9%, mean Δcap = +0.003, median = 0**.
     The "high-margin oracle labels" were sampling noise at 1.5σ. At
     K=32, cap-rate sampling noise std ≈ 0.088; margin 0.15 is barely
     above the noise floor. Path B v1's failure is now mechanistically
     explained: it trained on labels that were noise. Plus: explicit
     evidence that re-mining with K=128 on B-distribution states won't
     help unless margins are gated much harder.

152. **K=128 calibration on 100 fresh B-distribution anchors — oracle
     mining is dead even at K=128.** Per ChatGPT's pre-commit rule, ran
     a 100-anchor probe before spending 8h on full mining. Result:
     **split-half winner agreement = 44%**, **Δcap≥0.25 stable yield
     = 4%** (gate was ≥10% to proceed, <5% to abandon). Below every
     threshold. Path B v2 (rollout mining on B's own distribution)
     is structurally infeasible — the policy is uniform enough over
     legal top-30 that K=128 rollouts can't separate moves with
     statistical confidence at any reasonable margin tier. Closed
     the oracle path entirely. Total saved compute: ~8h Colab + ~12h
     training that wouldn't have produced learnable signal.

153. **The under-commitment diagnostic.** Adding `analyze_target_
     alignment.py` to compare model output vs V12 training targets on
     the SAME action set revealed: V12 targets have top1_prob mean
     0.261, but B_ep12 puts only **0.021** at the same action — **12×
     less than the target asked for**. B's argmax agrees with target
     argmax 62.5% of the time (good!), but its probability mass is
     spread across non-target actions: only 9.7% of mass on the 5
     trained target moves. The model has learned ranking direction
     but not commitment scale. This is the actual training pathology
     the whole oracle detour was a symptom of — V12 visit distributions
     are soft (top1=0.26 on MCTS@400 visit counts), and training on
     them with cross-entropy + dihedral × 8 + color × ~7! augmentation
     produces a model that learns to be even FLATTER than the targets
     demand (entropy invariance pressure).

154. **Target temperature sharpening pilot — the breakthrough.** Test:
     does sharpening V12 targets via `--target-temperature T` push the
     model to commit? Pre-training entropy diagnostic on V12 targets
     showed target top1 at T=1.0/0.75/0.50/0.25/0.10 = 0.26/0.28/0.31/
     0.38/0.50. Ran sharp_75/sharp_50 (T=0.75 and 0.5) on Colab
     G4-Blackwell, warm-start from B_smoke_epoch_12, otherwise identical
     recipe. The `prob @ target_top1` metric stayed flat across all 24
     epochs (0.021 → 0.021 for sharp_75, 0.022 for sharp_50) — looked
     like the mechanism wasn't engaging. **Then gameplay eval revealed
     the truth:**

     | run | T | policy mean (500 seeds) | %>10K |
     |---|---|---|---|
     | B_ep12 baseline | — | 8,043 | 27% |
     | sharp_75 | 0.75 | 8,817 (+10%) | 33% |
     | **sharp_50** | **0.50** | **12,622 (+57%)** | **45%** |

     Sharp_50 P95 hit 39,965 and seed 493 individually scored 89,508.
     Project's 30K policy-only median target is now within reach. The
     diagnostic metric was measuring the wrong axis — sharpening
     improved feature quality at argmax even though softmax tail
     stayed diffuse. Lesson: when a metric stays flat across
     hyperparameter sweeps, eval gameplay directly before concluding
     the mechanism didn't engage.

155. **MCTS regime restored with stronger policy.** With B_ep12, MCTS@
     100 + value_head_v12 + q=2.0 was net-NEGATIVE (mean dropped 4%):
     value head saturated at max_score=30K and pruned B's better
     moves. With sharp_50, the same MCTS recipe is **net-positive**:

     | | sharp_50 policy | sharp_50 + MCTS@100, cap=20K |
     |---|---|---|
     | mean | 11,789 | **14,602 (+24%)** |
     | P10 | 1,802 | 3,246 (+80%) |
     | P50 | 8,964 | 12,080 (+35%) |
     | P95 | 39,965 | 41,208 |
     | %<1000 | 4.0% | 1.0% (−75%) |

     The value head didn't change — the policy did. With a strong
     enough prior, MCTS uses search to refine borderline cases rather
     than wasting sims on obvious-good vs obvious-bad. The 8K-turn cap
     was the binding constraint; raising to 20K shows MCTS uncapped at
     14.6K mean (matches the prior pillar2z+MCTS@400 number of 15,465
     at 1/4 the simulation cost — equal strength achieved by stronger
     policy, not stronger search).

156. **The teacher path is now visible.** Six weeks ago: pillar2z
     policy mean 7,460, MCTS@400 plateau at 15,465, value head dead,
     three iterations of DAgger/oracle/v3-head failed. Today: sharp_50
     policy mean 11,789, sharp_50+MCTS@100 mean 14,602, sharp_50 P95
     39,965 (single-game), 30K policy-only median achievable in 2-3
     more iterations. The unlock was a **single hyperparameter**
     (`--target-temperature 0.5`) applied on the existing pipeline.
     Mechanism: V12 visit-distribution targets were softer than
     necessary; the model trained to mimic that softness; sharpening
     just before the CE loss forced commitment. None of the oracle /
     DAgger / value-head experiments mattered for this discovery —
     they only mattered for diagnosing where to look. The audit, the
     gradient diagnostic, the K=128 calibration, the BN recalibration,
     and the legal-distribution check all pointed at "the policy
     isn't committing." Sharpening was the cheapest possible fix and
     it engaged in the way ChatGPT had warned might happen: not via
     the diagnostic-visible metric, but via the actual play quality.

157. **Pillar3a = sharp_25_epoch_12.** T=0.25 sharpening completed
     (12 epochs on Colab G4-Blackwell, ~8h). Standalone policy mean
     **14,294** on 100 seeds (1,000-seed run on sharp_25_ep9 was 14,186;
     ep12 numbers consistent). +12% over sharp_50_ep12 (12,622) at
     the same compute. Picked as pillar3a — the new project baseline.

158. **Value head retrain on pillar3a backbone: +30% MCTS lift.** Per
     HISTORY 138's "retrain when backbone moves" rule, trained a new
     value head on pillar3a's frozen backbone using V11 survival
     targets (~48 min on M5 with channels-last + bf16 AMP). Recipe:
     `train_value_head.py --epochs 5 --batch-size 4096 --lr 1e-3`.
     A/B test on identical 100 seeds, MCTS@100, cap=25K:

     | head | mean | P50 | %<1000 | %>10K |
     |---|---|---|---|---|
     | OLD (value_head_v12_v12targets) | 16,340 | 12,859 | **7.0%** | 59% |
     | **NEW (value_head_sharp25_ep12)** | **21,310** | **17,644** | **1.0%** | **63%** |

     +30% mean, +37% P50, **floor cut by 86%**. The old head was
     actively HURTING on crisis states — its saturated max_score
     value estimates over sharp_25's stronger play space pruned
     B's crisis-escape moves. The retrained head shares pillar3a's
     backbone features and stops pruning correct play. Generalizable
     lesson: any iteration that changes the backbone (even mild
     distillation refinement, let alone a sharpening regime change)
     should mandatorily include a value-head retrain BEFORE deploying
     MCTS-based evaluation OR using that combo as a teacher.

159. **30K policy-only median is now within 1.7× reach.**
     pillar3a + MCTS@100 + new_head: P50 = 17,644, P75 = 32,231
     (25% of games already above target), P90/P95 ≥ 51K (cap-bound
     at 25K turns).

     | iter | policy mean | + MCTS@100 mean | gain |
     |---|---|---|---|
     | pillar2z | 7,460 | 9,138 (old head) | — |
     | sharp_50 | 12,622 | 14,602 (old head) | +57% policy |
     | **pillar3a** | **14,294** | **21,310** (new head) | **+49% MCTS** |

     One iteration on V12 (sharpening + value-head retrain) lifted
     MCTS from 15K → 21K. If pillar3b (V13 corpus + sharpening) gives
     another +30-50% via the now-properly-paired teacher signal,
     we land at 27-31K mean — at project target. The "2-3 more
     iterations" estimate looks conservative now.

160. **Pillar3b lands (2026-05-22): policy mean 17,255 on
     out-of-sample seeds 777000..777999, +15% over pillar3a.**
     Decision gate cleared.

     Training recipe: V13 corpus (9.16M states from selfplay@MCTS@400
     + crisis@MCTS@600/400), warm-start from pillar3a (sharp_25_ep12),
     17 epochs, target_temperature=0.5, lr=3e-4, batch 32K.

     Per-epoch policy eval (500 seeds 0..499):

     | ckpt | mean | P25 | P50 | P75 | <1000 | >10K |
     |---|---|---|---|---|---|---|
     | pillar3b_ep5  | 13,417 | 4,767 | 9,652  | 18,406 | 3.6% | 49% |
     | pillar3b_ep10 | 17,489 | 5,488 | 13,199 | 24,190 | 5.6% | 59% |
     | pillar3b_ep15 | 16,776 | 5,600 | 12,057 | 23,252 | 2.0% | 57% |
     | pillar3b_ep20 | **18,863** | 5,815 | 14,232 | 26,442 | 2.4% | 60% |

     Best by mean: epoch 20 (18,863). val_loss=2.1836 vs pillar3a's
     2.2550 — drop is consistent with the gameplay lift.

     Fresh out-of-sample comparison (seeds 0..499 were used during
     pillar3a tuning, so they're overfit-suspect for SOA claims).
     1000-seed eval on seeds 777000..777999:

     | ckpt | mean | P10 | P25 | P50 | <1000 | >10K |
     |---|---|---|---|---|---|---|
     | sharp_25_ep12 (pillar3a) | 15,002 | 1,842 | 4,076 | 9,596  | 4.8% | 49% |
     | **pillar3b_ep20**        | **17,255** | **2,476** | **5,483** | **12,567** | **2.5%** | **58%** |

     Pillar3b gains: +15% mean, +35% P25, +31% P50, **floor halved**
     (4.8% → 2.5%), +9pp on >10K rate. Both mean and floor moved.
     The floor improvement is the most important signal — that's
     where the gap to 30K+ best games lives.

161. **Image-3 diagnostic (2026-05-22): the value head is the
     leaf-evaluation bottleneck, not the search.**
     See `docs/image3_value_head_diagnosis.md` for full write-up.

     Concrete state where the only legal line-clearing move is
     ranked #2-#3 by both pillar3a and pillar3b instead of #1.
     Swept MCTS sims 100 → 3200, q_weight 0 → 3, c_puct 1.5 → 6.0,
     Dirichlet noise. **Nothing flips the argmax to the clearing
     move.** At ≥800 sims the top 3 candidates converge to within
     ±0.1pp of each other — MCTS reports them as equivalent.

     Crucially: at q_weight=0 (pure prior, no value head signal),
     both pillar3a (clear at 20.0% behind pink-setup at 25.5%) and
     pillar3b (clear at 20.7% behind pink-setup at 22.5%) show the
     same bias. The bias is **in the policy prior, not the value
     head**, but the value head also can't break the tie at q>0.

     Likely root cause: value_head_sharp25_ep12 was trained on
     V11-style **survival targets** (turns-until-death). It doesn't
     differentiate "post-clear: +5pts, -5 balls, free turn" from
     "post-pink-setup: line one closer to potential complete" — a
     strong policy survives roughly equally from either state.

     What this rules out for V14: more sims, q tuning, c_puct
     tuning, more sharpening, "force the clears" hard supervision
     (user-rejected — we want the model to learn *when* to clear).

     What's likely required: a **score-aware value head**. Retrain
     value head with targets = score gained over next H turns
     (or score-per-turn density), not survival horizon. If the
     post-clear state ranks measurably higher under this target,
     the bias unlocks. Independent of pillar3c training. Cost ~3-5h.

162. **Image-3 hypothesis REFUTED (2026-05-22 Phase 1 rollout judge).**
     Common-RNG paired rollouts K=256 H=500 from the image-3 state
     with pillar3b show the green-clear and pink-setup moves are
     **statistically tied on score-over-horizon**:

     | branch | mean Δscore (H=500) | A vs B win rate | die rate |
     |---|---|---|---|
     | A: green clear | 1004.9 | 43.8% | 2.0% |
     | B: pink setup  | 1005.4 | **54.7%** | 3.1% |

     Mean Δscore (A − B) = −0.5 ± 6.6 SE — CI crosses zero.
     B's win-rate edge is statistically real (~3.5σ); the
     +5 from immediate clear is recouped over horizon by bigger
     clears in the patient branch. Pillar3b's preference for the
     pink-setup is **correct** on this state. The visual "obvious
     clear" intuition was confirmation bias on a near-tie.

     This kills the score-aware-value-head retrain (HISTORY 161,
     task #109) — based on a false premise. Engine mechanics
     verified (calculate_score=n×(n−4), spawn suppressed on clear)
     so the rules were never the issue.

     Methodology lesson: visual analysis ≠ correct play. Always
     run common-RNG rollouts before redesigning training
     objectives. ChatGPT (2026-05-22) was right to push for a
     direct rollout judge before approving compute on N=1.

     Phase 2 (aggregate diagnostic across 100-300 clear-available
     states) still worth running with corrected framing — but
     priors now strongly suggest no systematic clear-undervaluation.

163. **pillar3d-v2.2 — data-scaling iteration (2026-06-06, launched).**
     The crisis-corrections floor pipeline as a *repeating pattern*:
     mine more crises → rebuild corpus → retrain (recipe fixed) →
     compare. v2.1 was iteration #1; v2.2 is #2.

     Rebuilt `crisis/corrections_corpus.pt` from all mined
     corrections (new crises on fresh seeds, corr_50xxx/51xxx):

     | corpus | games | anchors | size |
     |---|---|---|---|
     | v2.1   | 956   | 16,547  | 5.4 MB |
     | **v2.2** | **1,520** | **26,310** | **9.0 MB** |

     +59% data. Recipe **byte-identical to v2.1** (warm-start
     pillar3b_epoch_20, `train_path_b.py`, lr 5e-5 gentle throughout,
     target-temperature 0.5, aux-lambda 0.03, 10 epochs) — only the
     corpus changed, so it's a clean *data*-scaling measurement.
     Notebook `train_pillar3d_v2_2_colab.ipynb`, plan
     `docs/pillar3d_v2_2_plan.md`.

     Bar to beat (same 777k held-out, 777000..778999):
     - control pillar3b: mean 17,581, P10 2,377, <1000 2.9%
     - v2.1 (deployed): mean **20,609** (+17.2%), P10 2,736,
       <1000 2.5%, `<500` tail untouched

     Read (results pending): beats v2.1 (esp. `<500` moves) → more
     data still helps, keep the loop; plateaus → saturation is a
     *method* limit (next lever = teacher/objective, not corpus);
     regresses → marginal-data dilution (raise `--min-margin` /
     lower `--aux-lambda`).

164. **GPU batched-MCTS miner — PARKED (2026-06-06), no win shipped.**
     A multi-day effort to lift crisis-fork mining throughput by
     running K MCTS trees as GPU tensors (one L4 vs the M5 16-worker
     CPU miner @ 3.56 trees/s). Thoroughly characterized; **shipped
     no throughput win**. All code isolated in `batched_*_gpu.py` /
     `batched_mcts_closed.py` — the CPU scalar miner
     (`gen_corrections_parallel.py`) was never touched and remains
     production. Full trail: `docs/batched_mcts_perf_for_chatgpt.md`.

     What was learned (all measured, nothing hand-waved):
     - Engine + search ported to torch, golden-tested bit-identical
       (`legal_priors_t`, `build_observation_t` EXACT vs numba).
     - **Open-loop** GPU search is eager-**dispatch**-bound (L4:
       46k tiny ops/sim, GPU ~13% utilized). CUDA-graph capture
       works (block-capture 8× over eager) but the search then
       **compute-saturates at ~0.8× M5** — flat in K *and* W,
       because fixed-depth descent re-runs CC + reachability +
       apply_move every step. Can't beat M5 while faithful.
     - **Closed-loop** (cache board + legal children per node;
       descent = pure gather; one expansion/sim) cuts the engine
       ~depth-fold → 3.5× faster eager, and + capture would beat M5
       on *speed*. BUT it determinizes spawns per node →
       **biased teacher**: TV ~0.63 vs scalar's ~0.27 (it optimizes
       one frozen future, not the spawn expectation). ChatGPT
       confirmed bias, not noise. Single closed-loop labels are
       **not safe to distill**.

     Resume point if revisited: run
     `scripts/test_closed_ensemble.py` (built, not yet run) — M
     independent closed-loop determinizations/root, averaged, vs
     scalar. Ensemble TV → ~0.27 ⇒ viable with M-batching (then
     build capture); stays > 0.45 ⇒ semantics wrong for this
     stochastic game (keep CPU miner; GPU only as a batched prior
     server). Untried levers: open-loop prefix (1-3 plies) + cached
     closed-loop tail; judging a determinized teacher by downstream
     floor-lift rather than TV-vs-scalar.

     Lesson: micro-benchmarks (the 9.5× "bundle" graph speedup)
     overpromised; only the full sims-honest end-to-end number
     (×M5 at the real 4800 sims) told the truth. Measure end-to-end
     before celebrating a kernel-level win.

165. **pillar3d-mC = new deployable best (2026-06-07). Crisis
     distillation, median-targeted recipe.** `pillar3d_mC_dec_T05_epoch_2`.
     5k held-out (775000-779999): **mean 23,434, median 16,504,
     P10 2,855, <1000 2.0%, <500 0.3%, >10k 66%, max 297,214.**
     vs pillar3b base (mean 17,255): **+36% mean.** vs the prior
     deployed v2.1-ep2 (5k: mean 21,195 / median 15,121 / <1000
     2.6%): **~+10% mean, +9% median, floor improved.**

     **Recipe (the keeper):** warm-start pillar3b_epoch_20, lr 5e-5,
     re-distill V13 + aux crisis corrections, **decisive corpus only**
     (`build_corrections_corpus --min-margin 0.05`, ~13.8k from 1837
     games), **λ=0.01, aux_T=0.5, aux-warmup 0.5**, 6 epochs, peak ep2.

     **How we got here (the path that mattered):**
     - **Target pivot:** stopped chasing the floor as a special
       target (it's RNG-chaotic — a better move can draw a worse
       spawn; the policy plays the best move at the cliff). Switched
       to optimizing the **median**. Result: the floor *followed the
       median up* (<1000 2.6%→2.0%) — no floor-targeting needed.
     - **Grad-audit** (`train_path_b --grad-audit`): the supposedly-
       gentle λ=0.03 was **aux-dominated** (effective share
       λ|g_aux|/|g_main| = 2.09; |g_main|≈0.23 since pillar3b is
       converged). Right-sizing to λ=0.01 (~0.7 share) was the
       unlock; v2.2's "more data" regression was over-indexing, not
       bad labels.
     - **Decisive filter beats "use them all":** across a full run
       matrix, decisive corpus ≫ full (~22-23k vs ~19-21k mean) and
       sharpening the full corpus (aux_T 0.3) *hurt* (it peaks the
       marginal corrections' weak preferences into confident-bad
       targets). The low-margin 26% genuinely drag.
     - **Evals must be 5k+** (2k misled us on v2.2 — reversed at 5k).

     **Next:** data-scaling study — mine pillar3b to 3.5k/5k games,
     same recipe, median-vs-#games curve (coverage-limited or method
     ceiling?). Then compounding (warm-start the improved policy) +
     self-play re-anchor. The teacher's saturated linear leaf value
     remains the label-quality ceiling (rollout-grounded/learned
     value = the bigger future lever).

166. **Eval-protocol reset + the channel autopsy (2026-06-08..10).
     Three confounds excavated; pillar3d channel declared saturated.**
     - **Seed-list confound:** the matrix notebooks evaluated on
       777000.. while the deployable-best bar lived on 775000.. —
       batched-fp16 path divergence makes cross-list medians
       incomparable. Every conclusion drawn across lists (incl. the
       "19.6k regressed vs 13.8k" scare AND entry 165's "decisive ≫
       full" matrix read) was suspect. Fix: ONE protocol —
       scripts/eval_policy.py (single-process batched greedy player,
       fp16, batch 256), fixed 1k list 775000-775999 on the M5, 5k
       (775000-779999) for finalists; per-seed scores saved
       (logs/eval_scores/) + scripts/compare_evals.py paired test.
     - **ctrl0 (λ=0 control):** the gentle re-distill alone = ZERO
       (13,316 vs base 13,421). mC's +25% median is 100% corrections.
       (Nobody had ever run the control.)
     - **Clean ladder (1k):** base 13,421 / mC-13.8k **16,738** /
       mC-19.6k 15,924 / mC-27.6k 14,943; λ axis flat (mF .012 →
       14,782, mG .014 → 16,008). mF-vs-mG = near-replicates 1,226
       apart ⇒ **σ_train ≈ 1k** (trainer unseeded); the whole
       corpus-size "regression" is at/below the noise floor.
     - **Trust-region (pillar3e) NEGATIVE:** teacher-primary CE +
       β·KL(pillar3b‖student) on sampled broad anchors, NO V13
       rehearsal (alphatrain/train_trust_region.py). All β∈{5,15,45}
       lost to mC by ~2k; β=45@ep120 collapsed to 8,787 (cosine-lr
       endgame: Adam favors the consistent anchor gradient,
       un-learns the teacher). Held-out corr match hit 0.27 > mC's
       0.215 yet played worse ⇒ held match ≠ gameplay; full-coverage
       rehearsal ≫ KL-regularization on samples (replay > EWC).
     - **Forensics tools:** fingerprint_corpus_membership.py
       (behavioral which-corpus-trained-this test; rescued the
       19.6k checkpoint identity), diag_corpus_slices.py (mining
       slices statistically identical — composition ruled out),
       margin_diagnostics.py (27.6k model = best softCE/margin,
       worst match: calibrated-but-less-decisive; λ raises
       commitment and drift in equal measure).
     - Miner now records per-candidate **root Q** (q/q_min/q_max;
       visits-Q corr 0.79) — zero changes to mcts.py.

167. **TASK ARITHMETIC = the breakthrough (2026-06-10..11).
     +37% median over the bar from the SAME corrections.**
     Gemini peer review (docs/crisis_distill_review_for_gemini.md)
     named the disease: gradient competition. mC's dual loss made
     the tiny aux fight a 32k-batch status-quo gradient (and
     pillar3e's KL anchor was a rubber band). Cure (Ilharco et al.,
     task arithmetic): **decouple learning from preservation.**
     1. scripts/train_crisis_ft.py — fine-tune pillar3b ONLY on the
        27.6k decisive corrections (soft-CE T0.5, weighted, lr 1e-4,
        **frozen-BN so running stats stay bit-identical to base**).
        ~17s/epoch on the M5. Held-out match 0.350 @ep15 — best ever
        (mC 0.215, trust-region 0.28). Let it forget general play.
     2. scripts/merge_checkpoints.py — θ(α) = θ_base + α·(θ_crisis −
        θ_base); BN stats verified identical; each merge = seconds.
     3. Sweep α with the 12-min local eval. **Noiseless dose-response**
        (same two endpoints, deterministic merge):
        α:      0    0.1    0.2    0.4    0.5    0.6    0.7    0.8
        median: 13.4k 17.0k 19.4k 22.1k  22.8k  22.5k  23.0k  19.7k
        mean:   18.9k 23.6k 26.7k 32.7k  32.7k  32.2k  32.1k  27.2k
        Broad plateau α∈[0.4,0.7]: median ~22-23k, mean ~32-33k,
        P10 up to 4,932, <1000 down to 1.0%, max 288,695 (1k seeds,
        ft_epoch_15 vector). vs the mC bar (16,738/24,249): **+37%
        median, +35% mean. vs base: +71% median, +73% mean.**
     The corrections were never the limit — the channel was.
     **5k CONFIRMED (775000-779999):** α=0.4 mean 31,790 / median
     21,591; α=0.7 mean 31,617 / median 22,181 / max 301,050; both
     <1000 1.6%, P10 ~3.9k. vs the mC bar's 5k (16,504 / 23,434 —
     exactly reproducing HISTORY 165's numbers, so old and new
     protocols agree on this list): **+34% median, +36% mean, equal
     floor. NEW DEPLOYABLE BEST.** α 0.4-vs-0.7 statistically tied;
     pick via the paired per-seed compare. Reopened in
     the new channel (now ~10-min LOCAL trainings): full-63.8k
     corpus ("use them all"), corpus-size scaling, scrambled-label
     control (content vs perturbation), Q-advantage weighting.
     V14 retention per review: 5% permanent crisis replay in every
     batch, not a low-λ aux. Old-channel experiments (E1 subsample,
     mCrep replicates, mE2 λ-scaling) cancelled as moot.

168. **Scrambled-label control + "use them all" verdict (2026-06-11).
     The breakthrough is 100% content; the margin filter retires.**
     - **Scrambled control** (same 27.6k states, labels permuted,
       near-identical vector magnitude 0.0325 vs 0.0331, same α=0.4):
       median 719 / mean 916 / 67.5% of games <1000 — a DESTROYED
       policy, vs the real vector's +9k. The fine-tune couldn't even
       fit the random labels (train match 0.002, CE 8.3). No
       perturbation/rut-escape effect exists; every point of the
       task-arithmetic gain is 4800-sim knowledge. Mining justified.
     - **Full 63.8k corpus vector ≈ decisive 27.6k vector** (1k, α
       sweep): full@0.4 = 22,048/31,389 vs decisive@0.4 =
       22,084/32,722 — overlapping curves. The low-margin 26% that
       hurt the AUX channel (entry 165's matrix) are harmless in the
       MERGE channel: the filter was compensating for the channel,
       not the data. "Use them all" vindicated where it matters.
     - Decisions: **V14 permanent replay = the FULL corpus**;
       canonical policy stays pillar3f (= pillar3b + 0.5·decisive
       vector, the 5k-confirmed one). Paired-eval lesson: different
       policies on the same seed play UNCORRELATED games (corr
       ≈ −0.03) — per-seed pairing only helps near-identical
       variants; cross-model comparisons need raw N.

169. **V14/V15 self-play TRUNK distillation FAILED. FACT: overfits from epoch 1
     (val rises, train ~flat), NOT recipe/teacher/data-quantity. MECHANISM (why it
     overfits) STILL OPEN — my "no learnable signal" theory was refuted (2026-06-16).
     Facts locked so we stop relitigating; mechanism deliberately left open.** Two attempts to distill the V14/V15 self-play teacher onto
     pillar3f both regressed: pillar3g (visit-distill, train_path_b, T=0.5) and
     pillar3g2 (completed-Q "Gumbel" target). pillar3g2 eval (eval_policy, 200
     seeds 779000-779199, cap 20000t): mean 23,042→11,411 (ep1)→7,684 (ep2),
     <1000 0.5%→8% — halved, monotonic with training, floor WORSE.

     **REFUTED hypotheses (each killed with data — do NOT raise again):**
     - *"Student caught the teacher / gap closed."* FALSE. Teacher still ~2× on
       the floor: teacher self-play P10 7,127 / P5 3,508 vs raw pillar3f P10
       3,355 / P5 2,479. Matches 2z (MCTS 15,465 / pol 7,460 = 2×) and 3a (+49%).
       The 1.5-2× outcome gap has held the whole project.
     - *"Visits flattened / dilution on a strong base."* FALSE. Top-5-normalized
       top-1 visit share is IDENTICAL across generations: V12=0.22, V13=0.22,
       V14=0.22, V15=0.22 (top1-top2 gap ~0 in all). 2z distilled a target this
       "flat" to a new best. (The earlier 0.22-vs-0.14 was a top-5-vs-top-15
       recording artifact.)
     - *"Sharpening / wrong trainer broke 3g."* FALSE. **3g IS the 3b recipe** —
       same `train_path_b`, same `--target-temperature 0.5`, lr 3e-4, bs 32K,
       color-aug ON (train_path_b default). 3b (V13→pillar3a) gave +15% policy
       with this exact recipe.
     - *Base (pillar3a vs pillar3f merge)*: already tested — 3b-base vs 3f-base
       gave identical curves. Not it.
     - *Move cap*: V13/V14 selfplay both ~8.6-9.1k avg states/game (same ~cap);
       affects ceiling not floor. Not it.
     - *completed-Q (3g2) specifically* distilled `q_arg` = the high-Q LOW-visit
       move the search VISITED AND REJECTED (chosen_move==q_arg only 2.4%;
       ==prior pick 84.3%). Anti-teacher target → its extra regression. Value head
       is fine; the target was wrong.

     **METRIC RULES (stop getting confused):** (a) crisis-game "scores" are
     MEANINGLESS — a crisis game = snapshot score + a 200-500 move replay, so
     higher crisis score = STRONGER generator, not better data. (b) self-play
     mean/median are CAP-PINNED (most games hit the turn cap) — only the FLOOR
     (P1-P25) is meaningful. (c) NEVER per-seed/paired (RNG forks the game once
     players differ); distribution-over-a-seed-range only ([[feedback_eval_path_divergence]]).

     **"Floor-poor corpus" was WRONG (retracted).** V14 self-play floor is HIGHER
     than V13 (P10 10,348 vs 5,130) — but that's just the stronger generator
     (everything scales up); same distribution SHAPE, strictly-better data. A higher
     floor is good, not deficient. Do not raise "floor-poor" again.

     **ROOT CAUSE (evidenced by the train log): OVERFITTING from epoch 1 = the
     warm-start already fits the targets; nothing learnable.** pillar3g train log:
     train loss 2.2944→2.2935→2.2893→2.2871 (barely moves at full lr 3e-4), VAL
     loss RISES from ep1: 2.3371→2.3496→2.3516. So pillar3f (epoch 0) is the
     val-loss MINIMUM — every epoch makes val worse. Contrast 3b at the same lr:
     train loss DROPPED to ~2.0, val fell with it (real signal → +15% over 17 ep);
     2z trained 40 ep on a falling val. The diagnostic is the LOSS TRAJECTORY:
     decreasing val = learnable signal = improvement; rising val from ep1 = nothing
     to learn = overfit residual noise = regress. Floor scores (eval_policy, no cap,
     779000-779199) confirm: pillar3f P1 1057/P10 3349/<1000 0.5%; pillar3g ep2 P1
     1437/P10 3852/<1000 1.0%; ep3 P1 808/P10 3578/<1000 2.0% (accumulating
     regression as val climbs over 20 ep).

     **WHY it overfits — UNVERIFIED, my "no learnable signal" hypothesis was REFUTED.**
     I claimed pillar3f's search≈prior (nothing to learn). Direct test (forward each
     warm-start model on its own self-play, measure prior-vs-visit gap): V13/pillar3a
     argmax-agree 67% / KL 0.91; V14/pillar3f argmax-agree 79% / KL 1.06. My claim
     predicted V14 has a SMALLER gap; it has a BIGGER KL → refuted. There IS plenty of
     distributional signal in V14. The only directional hint: V13 had MORE argmax-level
     corrections (33% vs 21%) — search picked a different BEST move more often; V14's gap
     is more "visits flatter than the confident prior" than "different best move." NOT a
     verified cause. **Do NOT invest expensive compute (sim escalation) on this — it was
     justified only by the refuted theory.** Real lead (cheap): 3b and 3g are the SAME
     recipe yet 3b val FELL over 17 ep and 3g val ROSE from ep1 — compare the two training
     curves + what in the data flips val's direction. Mechanism still open.

     Also fixed: `alphatrain/train.py:68` bf16 crash (`scaler.scale` with
     scaler=None) — gate on `scaler is not None` (bf16 uses autocast, no GradScaler).

170. **pillar3f crisis re-mining (harvest2) + the TWO-PIPELINE clarification
     (2026-06-17..18). Provenance discipline starts here (write every model's
     exact commands + why).**

     **TWO DISTINCT crisis pipelines (do not conflate — this cost hours):**
     - *MCTS pipeline (made pillar3f, the +37% breakthrough, HISTORY 167-168,
       commit 76d1fed):* `gen_corrections_parallel.py` runs widened MCTS@4800
       (3 determinizations, q_weight 2.0, FV value net — NO neural value head)
       on each death game's crisis band, keeps states where MCTS top != policy
       move, target = soft visit dist. Writes `crisis/corrections/corr_<seed>.json`
       (keys: pol_share, mcts_top_share, visits). → `build_corrections_corpus.py`
       → `corrections_corpus.pt` (63.8k corrections / 3,676 games ACCUMULATED over
       many sessions; "decisive" subset mm05 = 27.6k). → `train_crisis_ft.py`
       (frozen-BN fine-tune) → `merge_checkpoints.py` (θ_base + α·Δ). **This is the
       teacher that produced pillar3f = pillar3b + 0.5·decisive_vector.**
     - *Catastrophe pipeline (this session's action-risk direction):*
       `overnight_systematic.py` → `mine_crisis_sweep.py` rolls out each policy
       candidate to death (R_curve 100 / R_screen 50 / R_confirm 500), labels by
       P(died-within-300) = catastrophe rate. Writes `mine_<seed>.json` (keys:
       pol_cat, best_cat, cand_rates). Held-out R=500 CI guards winner's-curse.
       Different teacher (policy self-judgment via rollouts, NOT a stronger searcher).

     **harvest2 = a 67h catastrophe re-mine on pillar3f (NOT MCTS).** Command:
     `caffeinate -s -i python scripts/overnight_systematic.py --model
     alphatrain/data/pillar3f.pt --rec-cap 25000 --out-dir logs/harvest2
     --seed-start 300000 --n-try 4000 --max-seconds 172800 --device mps
     --workers 10 --r-screen 50` (added --model/--rec-cap/--out-dir flags this
     session; depths stayed the default 15-45). Result: **575 natural-death pillar3f
     games mined (725 recorded to death_games/death_300*.json), 510/575 with ≥1
     CI-confirmed fork, 1,682 confirmed forks, 12,712 band states, 256,515 per-move
     labels.** Bigger than the prior catastrophe harvest v1 (`logs/mine_*.json`,
     443 games / 1,209 forks). NOT smaller — the earlier "less than 27.6k" alarm was
     a wrong cross-pipeline comparison (catastrophe 575 games vs MCTS 3,676 games).

     **Corpus builds from harvest2** (`scripts/build_catastrophe_corrections.py`,
     NEW — converts catastrophe forks → the soft-corrections format train_crisis_ft
     reads: tgt_prob ∝ softmax(−catastrophe/T=10) over candidates, weight = gap pp):
     - `--all-rows` (proven "use them all" scope, min_margin=0 analog) →
       `catastrophe_corrections_pillar3f_h2_all.pt`: **10,925 states (7,595 with
       weight>0), top-share P50 0.20** — ≈ the proven mm10 corpus size (10,922).
     - confirmed-only (1,682) and flagged-clean (3,371) variants also built; the
       all-rows is the one we train (drops 1,787 R=500-negative screen false-positives).

     **OPEN at write time:** task-arith on the catastrophe corpus is a NEW teacher
     through the PROVEN channel — untested vs the MCTS teacher's +37%. Running the
     500-seed α-sweep (`run_crisis_taskarith.sh`, CORPUS=...all.pt, EVAL 775000-775499)
     to test transfer; results + the chosen model → next entry. Fallback if it
     under-transfers: MCTS-relabel the SAME 725 recorded pillar3f death games via
     `gen_corrections_parallel --model pillar3f --death-glob
     'alphatrain/data/death_games/death_300*.json'` (the recording is reusable —
     not wasted).

171. **Catastrophe-corrections task-arith on pillar3f FAILED (wash/toxic) — the
     rollout teacher does NOT transfer through the merge channel. Pivot to MCTS.
     (2026-06-18.)**

     **Production commands (the catastrophe attempt):**
     1. `scripts/build_catastrophe_corrections.py --mine-glob 'logs/harvest2/mine_*.json'
        --temp 10 --all-rows --out catastrophe_corrections_pillar3f_h2_all.pt`
        (10,925 states / 7,595 weight>0; soft target = softmax(−catastrophe/T=10)
        over candidates, weight = R=500 gap pp).
     2. `scripts/run_crisis_taskarith.sh` (CORPUS=...all.pt, EVAL 775000-775499):
        `train_crisis_ft.py --base pillar3f --epochs 15 --lr 1e-4
        --target-temperature 0.5 --weighted` (frozen BN, ft_epoch_15) →
        `merge_checkpoints.py --base pillar3f --crisis ft_epoch_15.pt --alpha {α}` →
        `eval_policy 775000-775499` + `normal_play_drift.py`.

     **Dose-response (500 seeds 775000-775499; base pillar3f mean 35,810 / P50 27,434
        / P10 4,546 / <1000 1.2%):**
        α=0.05  mean 35,117 / <1000 1.8% (9)  — slightly WORSE both
        α=0.2   mean 33,700 / P50 25,326 / P10 4,623 / <1000 0.6% (3) — floor↓ but
                mean −6% / median −8%; drift gate clean (healthy argmax-Δ 3.8%)
        α=0.4   mean 16,253 — COLLAPSE; healthy argmax-Δ 26.9% (forgetting)
        α=0.5   mean 6,529   — destroyed
        α=0.7   mean 761     — destroyed; healthy argmax-Δ 53.8%
     **OPPOSITE of the MCTS dose-response** (HISTORY 167: MCTS plateau α∈[0.4,0.7]
     was the +37% sweet spot). The catastrophe task vector is TOXIC at the MCTS-proven
     doses and survivable only at α≤0.2, where it's a WASH (no dose is a clean win:
     0.05 slightly worse, 0.2 trades −6% mean for a noise-level floor edge).

     **VERDICT:** catastrophe-rollout corrections (policy self-judgment via rollouts,
     R=500-confirmed) are a genuinely WEAKER/different teacher than widened MCTS@4800
     ("grandmaster moves"), and do NOT transfer through the task-arith merge channel.
     Confirms HISTORY 168's scrambled-control conclusion that the +37% gain is
     specifically 4800-sim knowledge — not just "any crisis correction." The 67h
     harvest2 catastrophe mine is therefore NOT the lever; its 725 recorded pillar3f
     death games ARE reused as the MCTS input.

     **PIVOT (running): the PROVEN recipe — `gen_corrections_parallel.py --model
     pillar3f --death-glob 'alphatrain/data/death_games/death_300*.json' --sims 4800
     --mcts-seeds 3 --q-weight 2.0 --top-k 300 --topk-visits 20 --lo 15 --hi 85
     --workers 16 --out-dir crisis/corrections_pillar3f`** (FV value net, no neural
     head; script logic unchanged since commit 76d1fed — fac2009 only ADDED clean-prior
     recording). ~1.5 days on M5, resumable, additive corpus. Then build_corrections_
     corpus → train_crisis_ft(pillar3f) → merge α∈[0.4,0.7] → eval + drift gate →
     pillar3f' = pillar3f + α·MCTS_vector. Bar to beat = base pillar3f above.

172. **MCTS task-arith on pillar3f FIRST CUT = NEGATIVE (the +37% does NOT reproduce
     on a second iteration). 2026-06-22.**

     **Commands:** recorded ~2,000 more pillar3f deaths (batch_record --model pillar3f
     --tail 100, full 15-85 band) → 2,725 death games; `gen_corrections_parallel`
     (MCTS@4800, FV value net) labeled 902 games so far → `build_corrections_corpus.py
     --glob 'crisis/corrections_pillar3f/corr_*.json'` = **10,990 corrections / 902
     games (12.2/game, 27% marginal)**; `train_crisis_ft.py --base pillar3f --epochs 15
     --lr 1e-4 --target-temperature 0.5 --weighted` (frozen BN, ft_epoch_15, held-match
     0.26 vs proven 0.35) → `merge_checkpoints.py α∈{0.2,0.4,0.5,0.7}` →
     `eval_policy 775000-775499` (500 seeds) + `normal_play_drift.py`.

     **Dose-response (base pillar3f mean 35,810 / P50 27,434 / P10 4,546 / <1000 1.2%):**
        α=0.2  34,129 / P50 23,255 (−15%) / P10 4,750 / <1000 1.4%
        α=0.4  32,542 / P50 23,026 / P10 4,233 / <1000 1.6%
        α=0.5  30,615 / P50 22,290 / P10 4,512 / <1000 1.0%
        α=0.7  23,398 / P50 15,231 / P10 2,580 / <1000 2.4%
     **Every α mean+median NEGATIVE, monotonic. Even α=0.2 drops median 15% for a
     noise-level P10 bump.** NOT the +37% — a wash-to-harmful (less violent than the
     catastrophe vector but same shape). No α worth 5k-confirming.

     **Two compounding hypotheses (honest, not yet isolated):** (1) DIMINISHING RETURNS
     — pillar3f IS pillar3b + 0.5·decisive_MCTS_vector (HISTORY 167), so this is the
     SAME recipe applied a 2nd time to a base that already absorbed 4800-sim crisis
     knowledge; remaining corrections are marginal and perturb a converged policy.
     (2) VALUE-HEAD-LIMITED TEACHER — gen_corrections uses the FV LINEAR value net
     (value_head_path=None). FV-MCTS@4800 dominated the weaker pillar3b (+37%) but
     pillar3f is ~2× stronger now; FV-MCTS may no longer beat it, so some "corrections"
     are FV-favored moves WORSE than pillar3f's → distilling hurts. Caveat: partial
     corpus (902/2,725); floor counts noisy at 500 seeds; but the mean/median decline
     is robust/monotonic — more data is very unlikely to flip it to +37%.

     **NEXT:** (a) finish the overnight mine (2,725 games) → rebuild → retest (cheap
     sanity; prediction: still negative). (b) THE REAL LEAD: re-mine with the NEURAL
     value head value_head_pillar3f.pt (gen_corrections hardcodes value_head_path=None
     — needs a small change) so MCTS@4800 is a genuinely STRONGER teacher than the
     converged pillar3f. (c) accept the crisis-correction lever is near its ceiling at
     pillar3f (~30-35k mean). pillar3f stays the deployed best.

173. **pillar3k — DECISIVENESS-WEIGHTED distillation BEATS pillar3f (+18% mean, +28%
     floor). FIRST model past pillar3f; the de-peak wall is cracked.** (2026-06-23)

     **Problem solved:** every full-corpus distillation into pillar3f (3g; 3k-v1 uniform)
     de-peaked it → regression. Measured root cause: pillar3f policy peakedness ~0.40;
     the crisis+selfplay corpus is dominated by FLAT quiet/recovered positions (selfplay
     98% flat / 0% decisive; crisis 55% flat / 15-22% decisive >0.40). NO target-temperature
     clears 0.40 — v14_rev2 mix sharpened top-share: T=1.0→0.23, T=0.5→0.25, T=0.25→0.30,
     all <0.40. A strong base cannot be sharpened by a flat corpus (why 3b lifted WEAK
     pillar3a ~0.25 but de-peaks pillar3f).

     **Fix — decisiveness weighting (train_path_b.py `--decisiveness-power P`):** weight
     each state's policy-CE by (visit top-share)**P, mean-1 normalized. Decisive
     escape-or-die states (top-share 0.6-0.9) drive the gradient (64× a flat state at P=3);
     flat quiet states (~0.2) get ~0 weight → CANNOT de-peak. Separates the escape signal
     from the flat quiet bulk that sank every uniform attempt.

     **Corpus v14_rev2.pt (70% crisis / 30% selfplay = 1,822,799 states):**
       `python -m alphatrain.scripts.build_expert_v2_tensor --games-dir
       data/crisis_v14_s600 data/crisis_v15 data/crisis_v16 data/selfplay_v14_30pct
       --policy-only-data --output alphatrain/data/v14_rev2.pt`
       crisis (all, the escape signal): crisis_v14_s600 (2857 games / 1.15M, @600,
       pillar3f-generated) + crisis_v15 (132 / 54k, @600-800) + crisis_v16 (164 / 68k,
       @4800). selfplay anchor (down-sampled 30%): 60 of 960 selfplay_v14 games (every
       16th → symlinks data/selfplay_v14_30pct, 0.54M, pillar3f+MCTS@400). (all-zero-Q
       warning benign: Q is Gumbel-only; 3b uses pol_values.)

     **Train (Colab, train_pillar3k_colab.ipynb, 1.5h, 8× color+dihedral aug):**
       `train_path_b --tensor-file v14_rev2.pt --amp --compile --resume pillar3f.pt
       --warm-start --epochs 16 --batch-size 32768 --lr 3e-4 --warmup-epochs 1
       --target-temperature 0.7 --decisiveness-power 3.0`

     **Eval (eval_policy 775000-775499, 500 seeds, uncapped, fp16) vs pillar3f bar
       (mean 35,810 / P50 27,434 / P10 4,546 / <1000 1.2%):**
       ep10: mean 40,229 (+12%) / P50 27,182 / P10 4,392 / <1000 1.4% / max 238,539
       ep16: mean 42,265 (+18.0%) / P50 28,329 (+3.3%) / P10 5,823 (+28.1%) /
             <1000 1.0% / max 348,413
     **WIN on EVERY metric — floor (P10 +28%, <1000 down) lifted WITHOUT flattening quiet
     play (median UP +3.3%).** Still improving at ep16 (last epoch, LR→6e-6) → not converged.

     **Val doubly unreliable here:** rose ep1 2.2758 → ~2.285 (looks like de-peak) yet
     gameplay +18% — EXPECTED: val is unweighted CE dominated by the flat quiet states the
     model now de-emphasizes by design. Gameplay is the only truth.

     **NEXT:** (a) cheap (~3h): retrain current corpus more epochs (28-32) + sweep
     DECISIVENESS {3,4} × T {0.5,0.7} → ceiling + best operating point. (b) generate more
     v16 @4800 (decisiveness upweights the cleanest escapes; current corpus is 90% @600 /
     under-resolved → quality of decisive states is now the lever), rebuild 70/30, retrain.
     (c) iterate: best pillar3k → re-mine 4800 crisis on it → repeat.

174. **pillar3k v2 (v14_rev3) = NEW BASELINE — beats v1 on a 5k eval. Crisis distillation
     UNBLOCKED after months of failed attempts. Many lessons.** (2026-06-24)
     Model: **pillar3k_r3_dw3_T0.7_epoch_22**.

     **Result (eval_policy 775000-779999, 5000 seeds — 500 seeds was too noisy, showed
     v2≈v1; only the 5k reveals the real gap):**
       v2 ep22:  mean 43,390 / P50 31,016 / P10 5,010 / P25 13,247 / <1000 1.3% / max 337,411
       v1 ep16:  mean 41,180 / P50 28,653 / P10 4,693 / P25 12,562 / <1000 1.2% / max 389,307
     **v2 ep22 BEATS v1: mean +5.4%, median +8.2%, P10 +6.8%, P25 +5.5% (<1000 tied).**
     Both ~+15-21% over the pillar3f bar (500-seed: mean 35,810 / P50 27,434 / P10 4,546).
     **NEW BASELINE = pillar3k_r3_dw3_T0.7_epoch_22.**

     **Corpus v14_rev3.pt (2,259,096 states, 70% crisis / 30% selfplay):** v14_rev2 crisis
     pool + crisis_v16_1600 (699 games / 288k states, sims=[1600,2400] mix, cand_visits).
       `build_expert_v2_tensor --games-dir crisis_v14_s600 crisis_v15 crisis_v16
         crisis_v16_1600 selfplay_v14_30pct(74 games) --policy-only-data --output v14_rev3.pt`
       `train_path_b --resume pillar3f --warm-start --epochs 28 --batch-size 32768 --lr 3e-4
         --warmup-epochs 1 --target-temperature 0.7 --decisiveness-power 3.0` (Colab, ~2.5h)

     **=== THE LESSONS (this arc broke a long impasse: 3g/3i/3k-v1 all REGRESSED pillar3f) ===**
     1. **DECISIVENESS WEIGHTING is the unlock** (`train_path_b --decisiveness-power P`):
        weight each state CE by (visit top-share)**P, mean-1 normalized. Decisive escapes
        drive the gradient (~6.6× at P=3), flat quiet states get ~0.15× → they CANNOT de-peak
        a strong base. First thing to beat pillar3f via full-corpus distillation.
     2. **NO TEMPERATURE sharpens a flat corpus into a strong base.** Measured: v14_rev2 mix
        sharpened top-share T=1.0→0.23 … T=0.25→0.30, ALL below pillar3f's 0.40. 3k-v1
        (uniform, T=0.7) regressed −14% mean / −25% median. T is NOT the lever; decisiveness is.
        (Why 3b lifted WEAK pillar3a ~0.25 but de-peaks pillar3f ~0.40.)
     3. **FLATNESS IS POSITION TYPE, NOT SIMS:** selfplay 98% flat / 0% decisive (genuine
        multi-good-move quiet); crisis 15-22% decisive (escape-or-die). The floor signal IS
        the decisive escapes. Selfplay alone canNOT lift the floor (0% escapes) at any T.
     4. **VAL IS DOUBLY UNRELIABLE under decisiveness weighting:** (a) it ROSE while gameplay
        went +18% (val = unweighted CE dominated by the flat quiet states the model
        de-emphasizes BY DESIGN); (b) in v2, late-epoch val DROPPING tracked the CEILING
        (high-score tail), while the FLOOR was degrading. Gameplay is the ONLY truth.
     5. **OVERFITTING TRADES FLOOR FOR CEILING:** v2 P50 peaked ep17 (30,237), P10 peaked
        ep22 (5,362), BOTH declined by ep27 (P10 crashed to 4,316) while MEAN kept rising
        (tail/max grew). Late training sharpens into longer high-score games but more brittle
        crises (over-commits to noisy @600/1600 escape labels). PICK A MID-TRAINING epoch by
        GAMEPLAY FLOOR — not val, not the last epoch. (ep22 won the 5k.)
     6. **500 SEEDS IS TOO NOISY for close comparisons.** It showed v2≈v1 (wash); the 5k
        eval revealed v2 > v1 by +5-8%. ALWAYS 5k (775000-779999) to choose between candidates.
     7. **@600 CRISIS WORKS; 1600/2400 adds real gain; 4800 is overkill.** v1 was 90% @600
        → +18%; cleaner @1600/2400 → another +5-8%. Cost/quality sweet spot is 1600-2400.
     8. **SELFPLAY ANCHOR is partly redundant under decisiveness weighting** (@400 selfplay
        ~0.15× weight) — may just dilute batch capacity. OPEN: test 85/15 crisis ratio.
     9. **Don't be categorical from a proxy:** I claimed the new data "only added ~29k
        decisive" from the 10% position-mix figure; the 5k eval PROVED it genuinely helped.
        Position-mix counts peaked states, NOT escape quality/value. Validate, don't assert.

     **NEXT — the iteration loop is LIVE (unblocked):** mine **crisis_v17 ON THE NEW BASELINE**
     (pillar3k_r3_dw3_T0.7_epoch_22) — the improved policy's OWN crises = fresh decisive signal
     (the AlphaZero self-improvement loop), unlike re-mining pillar3f's now-saturated crises.
     Rebuild + distill → pillar3k iter 2. Side-bets: 85/15 crisis ratio (test selfplay
     dilution, lesson 8); DECISIVENESS=4 sweep.

175. **pillar3k_small128_hardce_epoch_87 — the 4× smaller student (10b×128ch, 3.0M
     params), distilled from pillar3k. LAUNCH PAD for the small-model improvement
     loop.** (2026-07-05)

     **Provenance (exact):** from-scratch distillation of pillar3k_r3_dw3_T0.7_epoch_22
     (teacher, 256ch) into 10b×128ch. Corpus `distill_pillar3k.pt` = 3,846,619 states
     (selfplay_v14 240 games ~2.16M broad + all crisis ~1.69M escapes, 56/44) relabeled
     with the teacher's top-5 legal-softmax policy via `alphatrain/scripts/distill_relabel.py`.
     Train (Colab, train_pillar3k_small_colab.ipynb, RUN=pillar3k_small128_hardce):
       `train_path_b --channels 128 --epochs 100 --batch-size 4096 --lr 1e-3
        --warmup-epochs 3 --flat-epochs 0 --target-temperature 0.5 --blend-alpha 0.5`
     (0.5·soft-CE + 0.5·hard-CE on teacher argmax; plain cosine; from-scratch recipe
     batch 4096/lr 1e-3 — larger batches step-starve from-scratch runs).
     **Eval (eval_policy 775000-775499, 500 seeds, fp16 @1024):** ep87 mean 12,987 /
     P50 8,889 / P10 1,799 / <1000 4.8% / max 131,845. Teacher-match 73.0% top-1,
     96.6% top-3 (diag_student_match, 20k states).
     **Key findings of the distillation arc:** (a) an ascending-vs-descending bug in
     diag/relabel top-K reads faked "not learning" for a week (true match was 67-69%);
     (b) match and gameplay DECOUPLE — match flat ~72% from ep19 while gameplay climbed
     6,237→12,987 (mimicry limit ≠ capacity limit); (c) cosine tail consolidated
     (ep49 10,877 → ep87 12,987); (d) truncation warm-start from the teacher's first
     128 channels = random init (0.2% match) — trained channels aren't importance-ordered.

176. **C++ engine (alphatrain/inference_cpp) = production for selfplay + crisis mining
     + policy eval. Python RETIRED for those three (user directive). First loop
     experiment sp1 (100% selfplay corpus) REGRESSED — lesson-3 replication.** (2026-07-06)

     **C++ engine, all golden/distribution-validated:** game engine bit-exact vs Python
     (obs/legal/clear 0.0e+00); 27-feature FV leaf evaluator bit-close (V 2.4e-07);
     MCTS (PUCT/min-max-Q/virtual-loss/open-loop determinized) strength-matched vs
     Python MCTS on a 100-seed A/B (median 11,112 vs 10,060, same seed list, noise);
     eval ~1.1-1.4× Python; selfplay + crisis recorders emit the moves-schema JSON that
     `build_expert_v2_tensor --policy-only-data` consumes directly (integration-tested;
     zero illegal actions; real Q). Throughput (M5, 14 threads): ~20.8k leaf evals/s,
     ~36-47k selfplay states/hour @1600 sims. Binaries: eval, mcts_eval, mcts_selfplay,
     mcts_crisis (+ game_test/feature_test goldens). RNG note: C++ plays different
     per-seed games than Python (own SplitMix64) — same distribution; compare
     distributions over a seed range, never per-seed.
     **MCTS-vs-greedy on the small model (n=100, 100 sims, q=1.0, fv-leaf):**
     median +15-25% over greedy (early +61% was an n=48 lucky draw).
     **sp1 NEGATIVE (controlled, cheap):** corpus = 100 C++ selfplay games @1600 sims
     (seeds 920000-99, cap 1500) = 142,844 states, `selfplay_cpp128_v1_slim.pt`; train =
     `train_path_b --resume ...epoch_87 --warm-start --channels 128 --epochs 16
      --batch-size 4096 --lr 1e-4 --warmup-epochs 1 --target-temperature 0.7
      --decisiveness-power 3.0`. Gate evals (775000-775499): ep1 10,865/6,800 →
     ep8 9,066/6,032 → ep16 8,192/6,265 (mean/P50) vs base 12,987/8,889 — MONOTONIC
     regression. Pre-flight had predicted it: ep87 prior top-share P50 0.565 (hard-CE
     made it decisive) vs 1600-sim visit target P50 0.335 → de-peak regime; and the
     corpus was 100% selfplay = ~0% decisive escapes, so decisiveness weighting had
     nothing to upweight. REPLICATES lesson 3 (entry 174) on the small model: the
     crisis (escape-or-die) share IS the load-bearing mitigation, not the trainer flags.
     **NEXT:** mine ep87's own crises with `mcts_crisis` (recovery 15@1600 / prevention
     75@2400 / continue 500), build ~70/30 crisis/selfplay corpus (selfplay anchor
     already in hand), retrain dw3/T0.7 on Colab, floor-gated 500-seed then 5k eval.

     **ERRATUM to 176 (2026-07-06, same day):** `data/policy_ts.pt` (the C++ tools'
     default model) was a stale Jun-26 export of the PRE-hardce model (~7.4k mean) —
     never re-exported after ep87 was chosen. Therefore: (a) selfplay_cpp128_v1
     (142,844 states) was generated with the OLD model, not ep87 → sp1's regression is
     CONFOUNDED (weaker-teacher data + de-peak; cannot attribute cleanly — the lesson-3
     replication claim is weakened, both causes plausible); (b) the C++-vs-Python MCTS
     "validation" A/B was CROSS-MODEL (C++ on old model vs Python on ep87) → redone
     clean below; (c) engine-level validations unaffected (greedy C++-vs-Python used
     the same model explicitly; goldens model-independent). Fix: ep87 re-exported and
     VERIFIED (logit diff 0.0 vs checkpoint); selfplay_cpp128_v1 + its slim tensor
     retired from training use; corpus to be regenerated with ep87. LESSON: never rely
     on a default model path in generators — pass --model explicitly and verify the
     export against the checkpoint before generating (a one-line logit-diff check).

177. **small128 improvement loop UNBLOCKED: value_head_small128 + head-guided crisis
     micro-corpus beats ep87 on the 5k gold standard. Three-gate validation ladder,
     all gates passed.** (2026-07-09)

     **Model: `checkpoints/gate3/epoch_1.pt`** (pick: ep1; ep3/5 overtrain).
     **5k eval (775000-779999, C++ eval, fp16):**
       ep1:  mean 13,080 / P50 9,323 / P5 1,222 / P10 1,889 / <1000 3.5% / >10k 48%
       ep87: mean 12,895 / P50 8,917 / P5 1,103 / P10 1,810 / <1000 4.2% / >10k 46%
     Better on EVERY metric (P50 +4.6%, <1000 −17%). The 500-seed gate eval had shown
     P50 −6% — inverted at 5k (lesson 6 again: 500 seeds mislead close calls).

     **Why this worked when iter-1 regressed 3× (recipe-invariant):** the FV-mined
     corpus's corrections were 90% survival-neutral (rollout judge; signal only in the
     danger band +2.6pp). Every historically-working corpus (V13/V14) was mined with a
     NEURAL SURVIVAL HEAD @ q=2.0 (verified from notebook commands); every FV-mined
     one failed on a strong base (HISTORY 171/172 + ours). Mechanism: chance-node
     branching → sims carry ~no horizon; the leaf value function is the knowledge
     carrier (600 sims + head beats 2400 + FV).

     **Provenance (exact):**
     1. value_head_small128.pt: survival ValueHead on ep87 FROZEN backbone;
        labels = build_value_targets on data/crisis_cpp128_v1 + selfplay_cpp128_v2
        (2.97M states, censoring-aware); train_value_head 5ep/bs4096/lr1e-3 (7.8 min).
        Death-balanced K=64 val: r 0.79-0.84 (H25-200). NOTE: the old "128ch backbone
        can't host a head" (val_acc 0.52) was a TRANSFER-DATA artifact — matched data
        fixed it (HISTORY 158 rule). Afterstate-ranking acc 0.567 ≈ the production
        value_head_pillar3f's 0.554 on the identical protocol (working heads consume
        state-level calibration at diverged leaves, not sibling-afterstate deltas).
     2. Gate-2 target audit (600 sims, no Dirichlet, q sweep): q=0 control = 0
        corrections; q=1/2 corrections judge at +4.5-5.1pp died-rate gap, 27-29%
        clear-wins, 0 phantoms (FV corpus: +0.1pp). Quiet states: still ties → the
        head is a danger-band amplifier, not a broad-corpus unlock. q=2.0 = op point.
     3. Micro-corpus: crisis_mining (Python, validation exception) ep87 +
        value_head_small128, q=2.0, recovery 15@600 / prevention 30@400, continue 500,
        seeds 960000-960099 → 84 deaths → 184 replays / 55,548 states (36 min).
     4. Train: gate3_crisis.pt + mix_tensors REHEARSAL 3:1 from distill_pillar3k.pt
        (222,192 states, 25% new signal); train_path_b --resume ep87 --warm-start
        --channels 128 --seed 42 --epochs 5 --batch-size 4096 --lr 1e-4
        --warmup-epochs 1 --target-temperature 1.0 --decisiveness-power 0
        --blend-alpha 0.5.
     **C++ engine gained NN-value MCTS** (export_policy_value.py fused policy+value
     TS module; InferenceServer tuple mode; MctsConfig.nn_value; --value-module flag
     on mcts_eval/mcts_selfplay/mcts_crisis) — production mining no longer needs the
     Python exception.
     **NEXT (scale + iterate):** full campaign ~1500-3000 probes @600/400 q=2 with
     --value-module (C++), rehearsal mix, train, 5k gate; then RE-TRAIN THE HEAD on
     the new model's own games each iteration (HISTORY 158) and repeat.

178. **iter-2 + the distill-better grid: REJECTED end to end — and the cause measured.
     vh1's per-move marginal gap is CONSUMED after one absorption step.** (2026-07-10/11)

     **What was tried (all 5k-rejected or grid-rejected; vh1 stays baseline):**
     - iter-2: 2.51M head-guided states (vh1's own deaths, 1200/800 sims, q=2.0,
       C++-mined) + full rehearsal (6.35M, 39% signal), gate-3 recipe → 5k mean 11,599
       (−11%). Epoch soup: no rescue.
     - GRID (6 seeded arms × step-checkpoints every 500): γ-disagreement weighting
       {0,2,6} × lr {1e-4,3e-4} × blend {0.5,0.3}, from vh1 on iter2_mixg
       (418,802-correction mask). Control step-axis: best point = s500 (floor ties,
       median −7%), monotonic-ish decline after — no winning window at ANY step.
       Best arm g2_lr1_s1000 (only floor-above-bar read at 500 seeds) → 5k mean
       11,156 (−15%). γ=6 collapses the floor; lr 3e-4 hurts everywhere.
       (500-seed reads flipped at 5k for the THIRD time — coarse screen only.)
     - New trainer machinery (kept): `--disagree-gamma` (+ add_disagree_mask.py),
       `--save-every-steps`, `--seed`; mix_tensors rehearsal blending.

     **The measured cause (rollout judge, vh1 continuation, R=64 H=300):**
       gate-2 reference (600/400 from ep87):  gap +4.5-5.1pp, 27-29% win, 0 phantom
       iter-2 corpus (1200/800 from vh1):     gap +0.3pp, 4% win, 94% tie
       isolation (600/400 from vh1):          gap +0.2pp, 4% win, 93% tie
     Sims exonerated; recipe exonerated (grid); volume exonerated. **After vh1
     absorbed gate-3's corrections, the teacher's remaining advantage is no longer
     expressible as single-move corrections** — yet its PLAY still escapes 84% of
     vh1's deaths (prevention band): the residual edge is multi-step / rolling-search
     knowledge. Decisive-share among corrections fell 20% → 5-6%.

     **Levers for iteration 3 (ranked, untested):** (a) sharpen the value head
     (fresh vh1-era labels, longer horizons) so finer Q differences become
     expressible as corrections — cheap, judge-auditable before any mining;
     (b) WIDENED deep search (the +37%-era teacher was 4800 sims with top-k~300
     widened roots — our miner has only run top-k 30; "sim-limited not
     value-limited" precedent, HISTORY 167/172); (c) multi-step distillation
     (sequence targets) — new machinery. Loop yield after 1 step: ep87→vh1 +4.6%
     median 5k; vh1→? requires one of the above.

179. **Levers (a) and (b) falsified in one day — single-move distillation from this
     teacher family into vh1 is CLOSED, with receipts from every angle.** (2026-07-11)

     Judge protocol identical throughout (300 corrections vs vh1 argmax, R=64, H=300,
     vh1 greedy continuation):
     | teacher config                     | gap    | genuine/tie/phantom |
     | 600/400 + fresh-label head (a)     | −0.1pp | 5% / 92% / 3%       |
     | 4800 sims, top-k 300 widened (b)   | +0.3pp | 2% / 97% / 1%       |
     | (prior: 600-1200 sims, old head)   | +0.2pp | 4% / 93-94% / 2-3%  |
     | (reference vs ep87, gate-2)        | +4.5pp | 27-29% / — / 0%     |
     Fresh-label head calibrates fine (r 0.76-0.84 on fresh death-balanced val) —
     calibration was not the constraint. The widened teacher disagrees MORE (22%
     correction rate, 16% decisive) but its alternatives are survival-neutral for a
     greedy executor. The teacher's play edge persists (escapes ~84% of vh1's deaths
     with rolling search): the residual advantage is structurally MULTI-STEP.

     **State of the small model: `small128_vh1` (5k: mean 13,080 / P50 9,323 /
     <1000 3.5%). The loop delivered exactly one iteration (+4.6% median over ep87)
     and every subsequent lever was measured to zero: corpus volume, mix ratio,
     step count, γ/lr/blend, head freshness, sim depth, tree width.**
     Remaining directions: (c) multi-step/sequence distillation (research bet,
     new machinery); (d) a stronger distillation SOURCE (the 256ch line's future
     improvements, re-distilled); (e) ship vh1 (browser MVP was the point of the
     4x compression). Strategy decision, not a compute decision.

180. **Phase 0b (DAgger gate): MICRO-GO — pillar3k's single-move corrections ARE
     greedy-cashable by vh1; the effect concentrates in confident disagreements;
     burst ladder is FLAT (single-move labels valid).** (2026-07-26)

     Context: single-move distillation from vh1's OWN MCTS teacher closed (179).
     New hypothesis (peer-reviewed, docs/small128_dagger_for_review.md): the
     256ch teacher pillar3k_ep22 (5k mean 43,390, GREEDY eval) is a different
     signal — its per-move choices are cashable by a greedy executor.

     **ChatGPT review corrections, all verified then adopted (commit 3e19e19):**
     crisis tensors are ~99.7% MCTS-replay states (only first-of-replay rows are
     true student-visited anchors — original 0a measured the wrong distribution);
     old judge exporter restricted base move to stored teacher top-5; the 2.5M
     "3:1" mix is really 39.5% new + its ep1 = ~28.6x the winning run's optimizer
     steps; a hard 500-seed ep1 gate would have rejected vh1 itself (HISTORY:3375).

     **Corrected 0a (7,287 true anchors from 3,653 vh1 deaths, full-legal argmax
     both policies):** agree 70.7% all / 69.0% recovery — NO aggregate drop, but
     disagreements sharpen near death: teacher logit-gap median 0.95 in recovery,
     49% >=1.0 (vs 0.40 / 27% in prevention). Aggregate agreement was the wrong
     shift diagnostic.

     **The gate (2,135 disagreement anchors x 2 arms x 64 common-seed reps,
     H=300, 7 continuation conditions, seed-cluster bootstrap 95% CIs):**
       ./scripts/run_dagger_judge.sh  (rollout_judge + new --burst-model/--burst-len)
       python -m alphatrain.scripts.export_dagger_anchors   (anchors + .bin + meta)
       python -m alphatrain.scripts.dagger_judge_analysis   (CIs + bars)
     Models: vh1_policy_ts.pt / pillar3k_ep22_policy_ts.pt, both logit-diff
     verified 0.00e+00 vs checkpoints (verify_ts_export.py — provenance rule).

     Condition S (vh1 continuation, PRIMARY): **+1.69pp [+1.32,+2.09]** died-
     within-300 uplift (turns +4.9). Genuine 18% : phantom 9% (own-MCTS levers
     in 178/179 read 2-5% genuine, <=+0.3pp).
       Strata: gap>=1.0 (n=822) **+3.73pp [+3.03,+4.42]**, turns +11.5 — near
       the gate-2 win-predicting +4.5pp reference; gap<0.5 (n=934) +0.34pp = 0;
       recovery +2.25pp vs prevention +1.06pp.
       T (teacher continuation) +1.81pp; bursts L=1/2/4/8/16: +1.55/+1.30/+1.30/
       +1.43/+1.54 — **FLAT ladder**: the advantage cashes at the single move,
       is continuation-robust, and teacher control after the swap adds nothing.
       So: single-move DAgger labels are VALID; short-sequence cloning adds no
       value ON THESE anchors.

     **Verdict per pre-registered bars: MICRO-GO** (lower CI > 0, point >= +1pp;
     Strong GO needed lower CI > +2pp).

     **NEXT (Phase 1, per review recipe + concentration finding):** 50-75k
     genuinely on-policy corpus — add --record-games to eval.cc (slim JSON via
     game_json.h) -> fresh vh1 greedy games -> harvest recovery/prevention/broad
     bands, mostly disagreements (gap<0.5 carries ~nothing; emphasize confident
     corrections), minority agreement states for calibration; pillar3k top-5 +
     argmax labels; 3:1 rehearsal; warm-start vh1; checkpoints at ~100/250/400/
     800 optimizer STEPS (absorption optimum, not epochs); 500-seed screens only
     reject catastrophes; the 5k eval decides.

181. **DAgger round-1 corpus BUILT (dagger_v1_mix.pt) + Colab package ready.**
     (2026-07-26)

     Follows the HISTORY 180 MICRO-GO. Master remains FROZEN by user directive
     (pillar3k = static label source only; all evolution on the small line).

     **On-policy generation (C++ eval, new --record-dir/--record-every/--record-tail):**
       ./build/eval --model data/vh1_policy_ts.pt --device mps --batch 512
         --seed-start 860000 --seed-end 862000 --max-turns 40000
         --record-dir data/dagger_games_v1
     2,000 vh1 greedy games in 443s. Distribution matched vh1's 5k bar (mean
     12,860 / P50 9,005 / <1000 4.3%) = on-policy sanity OK. Record format:
     full last-160-turn death band + every-8th-turn broad samples (~350/game),
     with the PLAYED move per state (fp16 batched argmax = deployment truth).

     **Harvest (alphatrain/scripts/harvest_dagger_corpus.py, seed 0):** 703,788
     candidates -> teacher selection pass (pillar3k fp16, full-legal argmax +
     logit gap vs the recorded student move) -> selection per HISTORY 180
     concentration: band disagreements gap>=0.5 all (25% sample below), broad
     only gap>=1.0, agreements downsampled to 18%; per-game caps 30/4/6; dedup.
     **66,917 states**: recovery 23,139 (21,119 dis) / prevention 31,992
     (27,255 dis) / broad 11,786 (6,543 dis, all confident) / agreements 12,000.
     Meta sidecar dagger_v1_states_meta.npz (seed/turn/band/gap/moves).

     **Labels:** distill_relabel.py (pillar3k ep22, top-5 legal softmax — the
     IDENTICAL convention as the rehearsal corpus). Label top-share P50=0.41.
     **Mix:** mix_tensors 3:1 -> 267,668 states, EXACTLY 25% new signal (the
     proven gate-3 ratio; no rehearsal-cap distortion at this size). Mask:
     patch_mix_mask.py -> disagree_mask on 47,308 confident-correction rows
     (rehearsal rows 0). ~523 optimizer steps/epoch at batch 4096 aug 8.

     **Colab (train_small128_dagger_colab.ipynb, RUN=small128_dagger1):**
     gate-3 winner recipe: warm-start vh1, blend 0.5, T=1.0, dw 0, lr 1e-4,
     bs 4096, seed 42, 3 epochs + --save-every-steps 100 (absorption window
     100-1000 steps). Upload: colorlines_pillar3d_v4.tar.gz (523,429 B),
     dagger_v1_mix.pt.gz (16,158,887 B), small128_vh1.pt (36,156,933 B).
     Gate: 500-seed = catastrophe filter ONLY; floor-first shortlist -> 5k
     decides vs vh1 bar (13,080 / 9,323 / P5 1,222 / <1000 3.5%).
     Expectation (HISTORY 180 calibration): +3-8% median.

182. **dagger1 REGRESSED at 5k despite the validated gate — postmortem measured,
     peer review requested.** (2026-07-26)

     Trained per 181 (RUN=small128_dagger1, gate-3 recipe, 3ep + step saves).
     500-seed screens: no catastrophe across the whole 100-1,570-step grid.
     5k evals (775000-779999) of the floor-first shortlist — ALL regress vh1
     (bar: 13,080 / 9,323 / P5 1,222 / <1000 3.5%):
       e2_s200: 12,312 / 8,554 (-8.2% P50) / 1,021 / 4.9%
       epoch_2: 12,678 / 8,816 (-5.4%) / 1,124 / 4.2%   (val 1.90 < vh1's 2.05
                — val IMPROVED while gameplay regressed; the val trap again)
       e3_s400: 12,610 / 8,917 (-4.4%) / 1,170 / 3.8%

     **Diagnostics (scripts diag_dagger1_regression.py / _direction.py):**
     - ABSORPTION on the 47,308 trained confident-correction rows: vh1 10.8%
       (near-tie fp16/fp32 baseline) -> trained 18% (+7pp only). keep_vh1
       80->66%; THIRD moves 9->16% (corruption on the very states we fixed).
     - DRIFT: 6.8-7.9% of argmaxes changed on held-out quiet states AND the
       rehearsal sample; fully formed at step 100 (warmup LR ~2e-5), partially
       heals later.
     - DIRECTION: mimicry-pull REFUTED. Match-to-pillar3k: ep87 73.0 / vh1
       73.0 / e3_s400 72.8 / e1_s100 72.4 — drift is AWAY from the teacher,
       incoherent, not re-distillation.
     - Near-tie contamination: vh1 keeps only 80.2% of its own recorded moves
       under fp32 recompute -> part of the "confident disagreement" set is
       tie-flips, not real preference conflicts.

     **The puzzle:** judge-validated +1.69pp single-move signal (HISTORY 180)
     fails to install (18%) and costs -3..-6% at 5k, while the structurally
     identical gate-3 recipe (same warm-start/LR/blend/rehearsal tensor) had
     WON +4.6% with MCTS-on-student labels at natural (low) correction density.
     Deltas: label function (search-on-self vs bigger-net policy), correction
     density (82% vs natural), contested-ness (gap median 0.65, soft top-share
     0.41 labels). Echo of pillar3c ("argmax-flip too aggressive").

     Peer-review brief: docs/small128_dagger1_postmortem_for_review.md
     (mechanism candidates: contradiction-gradient churn / BN stats shift /
     hard-CE aggression; round-2 arm menu with pre-registered early-abort).
     NO round-2 compute before review + arm selection. vh1 REMAINS the
     deployed best.

183. **R2 diagnostics VINDICATE the data (row-level judge + fp16 re-audit);
     round-2 arm = task-vector margin fine-tune, machinery built.** (2026-07-26)

     ChatGPT review of the 182 postmortem: all checkable claims verified
     (gap counts 31,659/25,116/15,649/7,609 of 54,917; HISTORY 146 BN-recal
     precedent; HISTORY 168-170 task-vector channel). Its key corrections:
     fp32 diagnostics were off-protocol; "mimicry-pull refuted" overclaimed
     (val CE 2.05->1.90 IS distributional movement toward the teacher);
     42% of trained disagreement rows were never judge-validated; effective
     top-1 target mass was already 0.705 (one-hot hardening = wrong arm);
     its own BN swap audit: amplifier, not primary cause. Mechanism verdict:
     OBJECTIVE/PROJECTION MISMATCH — judge validated action replacement;
     training demanded the 256ch teacher's full truncated top-5 logit
     geometry, which a 128ch net cannot fit pointwise without rotating
     unrelated rankings (the 7% quiet churn).

     **New measurements (scripts diag_dagger1_fp16.py, rowjudge_analysis.py):**
     - fp16 protocol: recorded corpus actions = vh1's deployment argmax 98.9%
       (the fp32 "80.2% near-tie contamination" was MY measurement artifact);
       true baseline teacher-play 0.6%; true absorption 11.8%; third 12.6%
       (67% inside teacher top-5).
     - ROW-LEVEL judge (750 actual corpus rows, exact fp16 label actions,
       vh1 continuation, seed-cluster bootstrap):
         gap>=1.0 x margin-large : +2.42pp [+1.15,+3.70]
         gap>=1.0 x margin-small : +2.80pp [+1.67,+4.02]
         gap 0.5-1.0 (both)      : +0.7pp, CIs cross zero
         gap<0.5                 : +0.08pp
       Student top-2 margin does NOT gate installability. Labels are GOOD
       where validated; 42% of the trained corpus was dead weight.
     - THIRD-action judge (300 rows): the trained model's invented moves are
       +1.53pp [+0.59,+2.50] BETTER than vh1's originals. Per-move learning
       was fine; the -4% regression is the COLLATERAL quiet-state churn.

     **Round-2 build (committed):** dagger_r2_gap1.pt = 25,116 judged-domain
     gap>=1.0 recovery/prevention rows, 1,997 seeds, gap median 1.94
     (build_dagger_r2_corpus.py). train_crisis_ft.py extended with
     --loss margin --margin 0.15 (pairwise hinge teacher-vs-vh1 action,
     frozen BN, corrections-only) + pref metric. Plan: fine-tune ->
     merge_checkpoints.py alpha={0.05,0.1,0.2,0.4} -> gate_dagger_r2.py
     (pre-registered: adoption >= +10pp, quiet drift <= 3%, third <=
     adoption, margin median up) -> 500-seed catastrophe screen -> 5k.
     Arm 2 control (labels-vs-loss-geometry): current 0.5 soft/hard +
     rehearsal on the SAME gap>=1 corpus. Venue for the ~minutes fine-tune
     (M5 per ta15 precedent vs Colab per standing rule) = user's call.

184. **R2 arm-1 (task-vector margin) — NO-GO at the pre-registered gate; the
     correction-vs-collateral frontier is bad at EVERY alpha. Cross-function
     per-state installation is now closed with receipts from two loss
     geometries.** (2026-07-26)

     Run: run_dagger_r2.sh — train_crisis_ft --loss margin --margin 0.15 on
     dagger_r2_gap1.pt (25,116 judged gap>=1.0 rows), frozen BN (verified
     bit-identical), 20ep @ 4s/ep on M5; merges alpha={0.05,0.1,0.2,0.4};
     gate_dagger_r2.py (fp16, 3,799 by-seed held-out corrections + 18,321
     quiet holdout states).

     **Training curves — the decisive fact:** train pref (logit_t > logit_s)
     climbed 0.11 -> 0.77, but HELD pref plateaued at ~0.49 from ep5 onward.
     The corrections DO NOT generalize across seeds — each is ~a lookup
     entry. Argmax adoption stayed low even on TRAIN rows (~0.08; the hinge
     lifts teacher-vs-vh1 without making teacher top-1 overall).

     **Gate table (bars: adoption >= +10pp, drift <= 3%, third <= adoption):**
       model    adopt%  pref%  marg_med  drift%  third%
       vh1         0.5    0.0    -1.062     0.0     0.4
       a005        3.7    3.6    -0.953     3.8     5.0
       a01         6.8    7.5    -0.875     8.2    10.6
       a02        10.9   13.7    -0.719    18.2    19.4
       a04        15.7   25.9    -0.438    40.5    35.6
       ft(a=1)     7.4   50.4    +0.008    89.6    85.6   (fabric destroyed;
                                                           adoption NON-monotonic in alpha)
     Collateral >= useful at every point (needed 2:1 the other way). NO
     gameplay evals run — the bars exist to stop here.

     **Where this leaves the theory:** labels are good (+2.4..+2.8pp row-
     validated, 180/183) but pillar3k's contested-state preferences are
     OFF-MANIFOLD for vh1's 128ch feature geometry — pointwise supervised
     installation (dense soft/hard R1, pairwise-margin task-vector R2) buys
     <=1 useful flip per collateral flip at any dose. Contrast gate-3's WIN:
     MCTS-on-vh1 corrections are ON-manifold (search amplifies the net's own
     latent preferences) and installed cleanly (+4.6%).

     **Named next options:** (a) same-forward gradient-cosine audit among
     corrections (documents interference; cheap); (b) STRATEGIC: re-arm the
     on-manifold channel — retrain the survival value head on the 2k+ recorded
     vh1 games (generate more overnight, ~4.5 games/s), gate-1 calibration,
     gate-2 judge of MCTS-on-vh1 with the better head (value-function law:
     leaf value = teacher strength); if positive -> the proven gate-3 recipe;
     (c) 192ch capacity control (standing contingency — multiple channels now
     read "uninstallable at 128ch/3M").

185. **legalmax (hard-negative + KL anchor) — best frontier points REGRESS at
     5k. The supervised-corrections channel at 128ch is CLOSED by the review's
     own criterion, with a complete frontier map.** (2026-07-27)

     R2c/d/e per the follow-up review: loss = relu(0.15 + max_{legal!=teacher}
     logit[a] - logit[teacher]) + lambda * CE-to-vh1 on 13,958 disjoint quiet
     states, frozen BN, 3 shuffle seeds (reproducible to ~0.3pp), lambda x
     epoch frontier (fp16 gate, 3,799 by-seed held-out corrections):

       lambda/ep   adopt%  drift%  ratio(useful:collateral)
       1 / 20       16.6    11.1      1.4:1
       3 / 20       14.7     8.0      1.7:1
       6 / 5        11.4     5.6      2.0:1
       10 / 5       10.0     4.5      2.1:1   <- best; ratio bar met,
                                                 drift bar (<=3%) missed
     Monotone, smooth, seed-stable. Epoch-invariance of the PAIRWISE vectors
     also confirmed (ep5/10/15 merges == ep20 within noise at every alpha —
     the vector direction stabilizes by ep5).

     **5k verdicts (775000-779999) vs vh1 (13,080 / 9,323 / P5 1,222 / 3.5%):**
       l100_ep5: mean 12,650 / P50 8,780 (-5.8%) / P5 1,126 / <1000 4.1%
       l60_ep5:  mean 12,108 / P50 8,347 (-10.5%) / P5 1,065 / <1000 4.5%
     NOTE: l100_ep5's 500-seed screen read mean 13,115 / P50 9,421 / <1000
     3.0% — BETTER than vh1 — then flipped at 5k. The FOURTH 500<->5k flip.
     500 seeds decide NOTHING.

     **Closure:** three loss geometries (dense soft/hard + rehearsal; pairwise
     task-vector across epoch x alpha; legal hard-negative + KL anchor across
     lambda x epoch), all on row-validated +2.4..+2.8pp labels, all regress at
     5k. 4.5% quiet-state drift costs more than 10pp of installed validated
     corrections buy. Per the review: "if that still has no adoption-versus-
     drift operating point, closing this channel at 128 is justified." Closed.
     (192ch control: OFF THE TABLE by user directive — 128ch only; fallback
     is keeping pillar3k, not intermediate widths.)

     **LIVE next:** the on-manifold re-arm — 20k on-policy games (seeds
     870000-890000) generating for the value-head scaling replication;
     protocol: disjoint splits (head-train / calibration / judge), decision-
     level bars (judge LCB > +1pp, genuine >= 15%, phantom < 5%; winning
     reference +4.5pp / 27-29%). Calibration = sanity gate only (the r=0.76
     fresh head already failed the judge at -0.1pp once — this is a 5x-data
     replication, not a guaranteed unlock).

186. **vh5x decision judge: FIRST POSITIVE GATE SINCE vh1 — the on-manifold
     channel re-arms with the 5x-data value head. Iteration-3 mining launched.**
     (2026-07-27)

     Value-head scaling replication (per follow-up review, disjoint seed
     splits): 20k on-policy vh1 games (seeds 870000-890000; mean 12,810 ~=
     the 5k bar). Targets: build_value_targets_slim.py, death-dense
     (--broad-keep 0.15 after the raw build read 82-97% positive — the
     saturated-val alarm), train <886000 (4.40M rows) / val 886000-888000
     (551k) / judge >=888000 (untouched). Head: value_head_vh5x.pt (12 min,
     frozen vh1 backbone). SANITY: AUC per-H on 100k identical val rows —
     vh5x 0.9730/0.9443/0.8973/0.8596 vs the falsified fresh head
     0.9709/0.9412/0.8890/0.8453 (better at every horizon).

     **Decision judge (gate-2 protocol @600 sims, 400 identical direct-sampled
     death-band states from reserved seeds, vh1 continuation, 64 paired reps,
     seed-cluster CIs):**
       q=0 control:        1/400 disagreements (all signal flows via the head)
       q=2 + OLD head:     49 dis, +2.84pp [+0.75, +5.06], 16%/2%
       q=2 + vh5x:         51 dis, **+3.37pp [+1.17, +5.73]**, 22%/6%
     Bars: LCB>+1pp PASS, genuine>=15% PASS, phantom<5% marginal (3/51).
     **KEY REVISION: the OLD head also reads positive on directly-sampled
     fresh death-band states — iter-2's "consumed" verdict was partly a
     property of the OLD miner's band selection, not the whole distribution.**
     Fresh deaths still carry per-move signal; vh5x finds more of it.

     **LAUNCHED: iteration-3 micro-corpus mine** — mcts_crisis, fused
     pv_vh1_vh5x_ts.pt (verified 0.00e+00), seeds 900000-900150, recovery
     15-back@1200 / prevention 30-back@800 / q=2.0 (the exact winning gate-3
     settings) -> data/crisis_vh5x_micro. Next: build_expert_v2_tensor
     (policy-only) -> mix 3:1 rehearsal -> gate-3 recipe on Colab
     (RUN=small128_vh2try) -> 500 catastrophe screen -> 5k vs vh1.

187. **Iteration-3 (vh2try) REGRESSED at 5k despite the positive judge — and
     the cross-round pattern now reads as a SIGNAL THRESHOLD, not a slope.**
     (2026-07-27)

     Local end-to-end round (M5, first fully-local loop turn): train_path_b on
     MPS, 3 epochs / 24 min (gate-3 recipe exactly; no --compile/--amp).
     500-screens: all 14 checkpoints pass (no catastrophe). 5k verdicts vs
     vh1 (13,080 / 9,323 / 1,222 / 3.5%):
       epoch_1: 12,351 / 8,579 (-8.0% P50) / 1,103 / 4.1%
       e1_s400: 12,134 / 8,701 (-6.7%) / 1,118 / 4.2%
       e3_s100: 11,697 / 8,342 (-10.5%) / 1,047 / 4.7%

     **The (judge -> 5k) record, all rounds ever measured:**
       gate-2 -> vh1 : +4.5pp / 27-29% genuine  ->  +4.6%   (WON)
       dagger R1     : +1.69pp / 18%            ->  -4%
       vh5x iter-3   : +3.37pp / 22%            ->  -7%
       iter-2 levers : <=+0.3pp                 ->  -11..-15%
     Reading: warm-start training pays a ~fixed collateral tax (~4-6% at 5k;
     the quiet-state churn measured directly in 182/184); mined per-move
     signal must EXCEED the tax. +4.5pp cleared it; +3.37pp did not.
     Expectation-setting by linear interpolation was wrong — it's a threshold.

     **DUE DILIGENCE LAUNCHED:** (a) 5k on the untested earliest checkpoints
     (e1_s100/s200); (b) the decisive measurement: gate-2 judge @2400 sims
     (vh5x head, same 400 reserved states) — can search on vh1 produce
     +4.5pp-class corrections AT ALL? If yes -> iteration-3b mines at those
     settings. If no -> vh1 is measured as sitting where its search-teacher's
     per-round yield < training's per-round cost (a real plateau statement,
     to peer review with the full table).

188. **Tax sweep: rehearsal ratio is a REAL tax knob — 6:1 recovers 2/3 of the
     iteration-3 regression. Iteration-3b (deep corpus x 6:1) launched.**
     (2026-07-28)

     Three arms on the UNCHANGED iteration-3 corpus (local M5, ~3h total):
     5k vs vh1 (13,080 / 9,323 / 3.5%):
       3:1 (187 ref) : 12,351 / 8,579 (-8.0%) / 4.1%
       6:1 r6/ep1    : 12,764 / 9,098 (-2.4%) / 3.9%   <- best post-vh1 yet
       10:1 r10/ep1  : 12,788 / 8,897 (-4.6%) / 4.3%   (diminishing past 6:1)
       3:1 lr 5e-5   : 12,336 / 8,693 (-6.8%) / 4.4%   (LR is not the lever)
     Screens again over-promised (r6/ep1 screened mean 14,074 = +7.6%; 5k
     says -2.4% — flip #5). Monotone tax curve 3:1 -> 6:1 confirms the
     threshold model's tax side; LR halving does nothing.

     **Ledger: 6:1 tax ~2-3% at 5k; current corpus signal +3.37pp — just
     short. Deep judge (187 diligence): @2400 sims +3.88pp [+2.37,+5.45],
     17.5% disagreement, 0% PHANTOMS.** Iteration-3b = both knobs:
     LAUNCHED mcts_crisis @ recovery 2400 / prevention 1600, q=2.0, seeds
     901000-901150 -> data/crisis_vh5x_deep; then 6:1 mix -> gate-3 recipe
     (2 epochs, step saves) -> screens (catastrophe only) -> 5k.

189. **Iteration-3b (deep corpus x 6:1) FAILED — deep corpus trained WORSE
     than shallow at the same tax point. CAMPAIGN HALTED for full review;
     pivot question = put the small line on the 256ch track (dw/T0.7).**
     (2026-07-29)

     5k vs vh1 (13,080 / 9,323 / 1,222 / 3.5%):
       vh2c/epoch_1 (deep 2400/1600 @ 6:1): 12,543 / 8,791 (-5.7%) / 4.1%
       vh2c/e1_s700:                        12,348 / 8,671 (-7.0%) / 4.9%
     vs r6 (shallow 1200/800 @ 6:1):        12,764 / 9,098 (-2.4%) / 3.9%

     **THE TWIST:** the better-JUDGED corpus (+3.88pp, 0% phantoms) trained
     worse than the +3.37pp one — the simple signal-minus-tax model breaks.
     Matching receipt: HISTORY 174 lesson 2 — deeper search = FLATTER visit
     distributions; the 256ch line's cure was DECISIVENESS WEIGHTING (dw=3).
     Every small-line run since gate-3 used dw=0. Per-move argmax quality
     improved while target-distribution quality for a greedy student fell.

     **Campaign record (vh1 = bar, 6 challenger rounds, all failed):**
       iter-2 (own-MCTS, 3 methods)      : judge <=+0.3pp -> -11..-15%
       dagger R1/R2 (pillar3k labels, 3
       loss geometries)                  : +1.69pp row-valid -> -4..-8%
       iter-3 (vh5x @1200/800, 3:1, dw0) : +3.37pp -> -7%
       tax arms (same corpus)            : 6:1 -2.4% (best), 10:1 -4.6%
       iter-3b (deep @2400/1600, 6:1)    : +3.88pp/0-phantom -> -5.7%
     Screens 0-for-5 on close calls. vh1 REMAINS best.

     USER DECISION: stop and review (docs/small128_campaign_review.md).
     Hypothesis on the table: the 256ch line progressed fine on dw3/T0.7
     full-corpus deep-visit distillation — put the small model on that track.

190. **Iteration-4 pilot (composite weighting + KL anchor) DEAD at every
     lambda; pivot to the review's fallback = advantage-filtered policy
     improvement. Per-row judging of all 15,655 disagreements launched.**
     (2026-07-29)

     Pilot arms (vh2c_crisis only, w=top_share^P*(1+gamma*disagree), lambda=1):
     screens DEGRADE MONOTONICALLY (arm A ep1 10,502 -> ep6 5,839; arm B
     10,481 -> 5,949) — concentrated correction gradient destroys the policy;
     lambda=1 anchor far too weak. Lambda sweep (P1.5/gamma2): lambda=10 best
     ~11.3k, lambda=30 best ~11.7k, still 15-20% below the vh1 screen range
     and decaying by ep3. NOT close-call territory (screen noise is ±5-8%);
     no 5k spent. The design has no viable operating point on this corpus.

     Running (overnight): export_advantage_judge.py -> rollout_judge on ALL
     15,655 full-legal fp16 disagreement rows of vh2c_crisis (64 paired reps,
     vh1 continuation) -> per-row advantage estimates. Next: train ONLY
     judged-positive executable disagreements (hard CE, advantage-weighted,
     strong quiet anchor via train_crisis_ft machinery), screens, 5k.
     Permanent asset either way: the corpus's actual row-level value,
     closing the judge!=corpus gap the review flagged.

191. **METHODOLOGICAL LANDMARK: the 5k eval's resolution is ~±480 mean /
     ~±550 P50 (95%) — measured via paired-seed bootstrap. The advantage-
     filtered candidates are AHEAD but inside noise; 20k evals running.**
     (2026-07-30)

     Advantage-filtered round (review #5 fallback): judged all 15,655 vh2c
     disagreement rows individually (overnight, 550M evals) -> mean row
     uplift only +0.5pp (roots read +3.88pp — judge!=corpus now QUANTIFIED,
     ~7x). Filtered to the 676 judged-genuine rows (uplift>=0.08, mean
     0.145), advantage-weighted soft-CE fine-tune (train_crisis_ft, frozen
     BN, lambda=3 quiet anchor) -> alpha-merges. 5k point estimates:
       m02 (a=0.2): 13,318 / 9,475 (+1.8% / +1.6%)  <1000 3.8%
       m04 (a=0.4): 13,474 / 9,553 (+3.0% / +2.5%)  <1000 4.2%
     FIRST candidates ever AHEAD of vh1 at 5k.

     **But: eval --scores-out (new) + paired_bootstrap.py (new) shows ALL
     CIs cross zero** (m04 mean +394 [-120,+903]; P50 +232 [-349,+759]);
     pairing doesn't help because same-seed games diverge into unrelated
     trajectories (the butterfly effect, now measured). m04 floor warning:
     <1000 +0.7pp [-0.0,+1.4] — near-significant worsening.

     **Consequence for the whole campaign: differences under ~8% P50 were
     never resolvable at 5k.** The -2.4% "near miss" (188), the +1.6-3%
     "wins" here, and possibly vh1's own +4.6% (177) sit in or near the
     noise band. Only the >=8-15% failures were conclusively measured.

     RUNNING: 20k-seed evals (775000-795000, ~65 min each, C++ engine) of
     vh1 / m02 / m04 -> CI halves to ~±240/±280 -> resolves the promotion.
     Promotion bar: paired mean AND P50 CIs > 0 with no confirmed floor
     loss (<1000 CI must not be clearly positive).

192. **small128_vh2 PROMOTED — the advantage-filtered channel wins at 20k
     paired seeds. First loop turn since vh1, and the first promotion under
     the rigorous instrument.** (2026-07-30)

     **small128_vh2 = alphatrain/data/small128_vh2.pt** (TS export
     vh2_policy_ts.pt) = theta_vh1 + 0.2 * (theta_ft - theta_vh1), where the
     fine-tune is train_crisis_ft on alphatrain/data/advfilt.pt: the 676
     judged-genuine rows (per-row uplift >= 0.08 over 64 paired reps; mean
     0.145) of vh2c_crisis's 15,655 full-legal disagreements, advantage-
     weighted soft-CE T=0.5, frozen BN, lambda=3 quiet-anchor, 15 epochs.

     **20k-seed paired verdict (775000-794999, paired_bootstrap):**
       m02 vs vh1: mean +527 [+280,+774] WIN (+4.1%); P50 +322 [+46,+573]
       WIN (+3.6%); P5/P10/<1000 all neutral (no floor loss).
       m04: mean WIN but P50 CI crosses zero, floor leans worse — NOT promoted
       (the 5k point estimates had ranked m04 first: noise, again).
     **New bar (20k): mean 13,334 / P50 9,364 / P5 1,150 / P10 1,853 /
     <1000 3.9%.** (vh1 @20k: 12,807 / 9,042 / 1,130 / 1,787 / 4.0% — note
     the old 5k seed-set read vh1 ~2% friendly.)

     **What the win validates end-to-end:** row-level judging (the corpus was
     96% ties — training ONLY the verified 4% flipped nine rounds of
     regression into a win), advantage weighting, alpha-metered task-vector
     integration with a quiet anchor, and 20k paired bootstrap as the
     promotion instrument. The channel's economics: 676 rows bought +4%;
     judging is the bottleneck (~3.5h per 15k rows, C++).

     **NEXT (the alternation, round 2 of this channel):** generate fresh vh2
     games -> retrain the survival head on vh2's backbone -> mine + judge
     MORE candidate rows (30-50k) -> advantage-filter -> fine-tune + merge ->
     20k paired bar vs vh2. All machinery exists; no new methods needed.

193. **Round 2 of the advantage-filtered channel: NO-GO at 20k — the recipe
     did not compound against the improved base. vh2 remains best.**
     (2026-08-01)

     Pipeline (all base=vh2, resume after session kill): 20k fresh games ->
     value_head_vh2_5x (inner-val 0.2979) -> 599 replays @1200/800 ->
     181,074 states -> 26,179 full-legal disagreements -> row-judged (vh2
     continuation): mean +0.4pp, GENUINE 1,137 (4%) / phantom 583 (2%) —
     same profile as round 1, 1.7x the genuine count (well refilled).
     advfilt2: 1,137 rows, mean uplift 0.142 (~= round 1's 0.145).
     Fine-tune identical to the winner (soft T0.5, adv-weighted, frozen BN,
     lambda=3 anchor on vh2's own quiet states); merges alpha={0.1,0.2,0.4}.

     **20k paired vs vh2 (pair20k_m02 reference): ALL NEGATIVE-LEANING.**
       m01: mean -231 [-476,+16]; <1000 +0.4pp [+0.0,+0.7] = floor LOSS
       m02: mean -226 [-479,+22]; P50 -128 [-375,+166]
       m04: mean -195 [-444,+54]; P50 -11 [-265,+289]
     Nothing near the bar (mean AND P50 CIs > 0).

     **Reading (recorded, not asserted):** the channel's round-1 yield
     (+527 mean) did not repeat with MORE verified rows of equal judged
     quality against the improved base. Candidate explanations for review:
     (a) vh2's merge vector already occupies the correction direction —
     new vectors interfere; (b) per-round true yield is small (~+0-300)
     and both rounds' outcomes are within the joint noise of judge
     selection + 20k resolution; (c) genuine-row value doesn't transfer
     once the base has moved. The loop now operates at the edge of
     measurability: ~1 day/round vs ±240 instrument resolution.

     STATE: small128_vh2 (20k: 13,334 / 9,364 / <1000 3.9%) = deployed
     best. Round-2 assets kept (26k judged rows, head, games). PAUSED for
     stocktake with the user.

194. **MECHANISM PROBE: vh2 installed ~NONE of its winning corrections — the
     +4.1% came from a sub-argmax "field tilt," and the fuel is still live.
     Dose hypothesis for round-2's failure; low-dose merges testing now.**
     (2026-08-01)

     probe_vh2_absorption.py on the 676 rows that created vh2:
       ABSORBED (vh2 plays teacher move): 7.2%
       kept vh1's original move        : 89.1%   third: 3.7%
     Judge of the 627 still-contested rows UNDER VH2 (64 paired reps):
       teacher 0.321 vs current 0.385 = **+6.4pp, 36% genuine / 3% phantom**
       (selection regression from 14.5 avg -> 6.4, but strongly alive).

     **Reframe:** the alpha-merge does NOT install moves; it tilts logits
     sub-argmax in a rollout-verified direction. Round-1's dose-response
     already peaked at alpha=0.2 (0.4 was worse). Round-2 merges stacked a
     heavily-overlapping direction ON TOP of vh2's existing 0.2 -> total
     dose past the peak -> the observed negative lean (193). RUNNING:
     round-2 vector at alpha={0.02,0.05} on vh2 -> 20k paired (the direct
     dose test). Also on the table if confirmed: dose-response mapping of
     theta_vh1 + beta*(D1+D2).

195. **Dose hypothesis FALSIFIED — the round-2 vector helps at NO dose.
     Loop halted at vh2; rounds 1+2 receipts to peer review.** (2026-08-01)

     20k paired vs vh2, round-2 vector at all doses:
       a=0.02: mean -254 [-501,-9] LOSS;  a=0.05: -26 [-277,+227];
       a=0.1: -231;  a=0.2: -226;  a=0.4: -195.
     Same recipe, same judged row quality (uplift 0.142 vs 0.145), 1.7x the
     rows — but THIS vector's direction is unhelpful at every strength,
     where round-1's identical construction bought +527 [+280,+774].

     Open explanations (for review, not asserted): round-1's effect partly
     fortunate (high side of a smaller true effect + favorable vector draw);
     an unidentified asymmetry (fine-tune base vh1-vs-vh2, anchor set,
     corpus band mix); or the channel fundamentally yields one-shot gains
     whose repeatability is below instrument resolution. The field-tilt
     probe (194) stands: vh2's gain was sub-argmax, its fuel unabsorbed
     and still judged +6.4pp — knowledge the training methods cannot yet
     cash twice.

     STATE: small128_vh2 = best (20k: 13,334 / 9,364 / <1000 3.9%).
     Compute HALTED. Brief: docs/small128_round2_for_review.md.

196. **ITERATION-5 LAUNCHED: the literal 256ch recipe at literal scale on the
     small model (user directive — capacity assumption stands, target =
     parity with 256ch then beyond).** (2026-08-02)

     Design (matched to the +18% round, HISTORY 173): base vh2; crisis
     @600 sims (the winning corpus's bulk was @600), 2,600 seeds, recovery
     15 / prevention 30, pv_vh2_ts (vh5x-class head) at the leaves, q=2.0
     -> ~1.6M states; selfplay @400 sims, 80 games -> ~0.7M (70/30 mix);
     train_path_b warm-start, dw=3, T=0.7, PURE soft (no blend, no anchor —
     the selfplay fraction IS the anchor), lr 3e-4, step-matched (~15.5k
     optimizer steps ~= 552x28 of the 256ch run), step checkpoints.
     Kept protections: screens = catastrophe filter only; promotion = 20k
     paired bootstrap vs vh2 (bar unchanged). ~28h fully local
     (scripts/run_iter5.sh, resumable; waits for the review-#6 rejudges).

     Known risk on record (review #5): dw=3 routed 4.7-9% of gradient to
     corrections on our PRIOR corpora — but those were @1200-2400; the @600
     corpus's top-share profile may differ (the 256ch winner's did). The
     experiment decides, not the extrapolation.

197. **Review-#6 forensics: winner's curse CONFIRMED (x0.44-0.50 shrinkage)
     but BOTH corpora equally good fresh; replica extraction COHERENT (cos
     0.95-0.97) — suspects eliminated; the final-candidate experiment
     (r1 vector re-derived at vh2) is running.** (2026-08-02)

     Independent rejudges (R=256, --seed-offset NEW, fresh seeds):
       r1 rows: 0.145 -> 0.063 fresh (x0.44), corr 0.65, 75% positive,
                phantoms 0.3%
       r2 rows: 0.142 -> 0.071 fresh (x0.50), corr 0.63, 78% positive,
                phantoms 0.2%
     -> Selection curse exactly as predicted (1.5 SE threshold), AND the
     two corpora are statistically indistinguishable in true row quality
     (r2 slightly better) — selection noise CANNOT explain r1-won/r2-lost.

     Shuffle-replica coherence (8+8, scripts/replica_coherence.py):
       r1 corpus: pairwise cos 0.951-0.962, |d| cv 1%
       r2 corpus: pairwise cos 0.969-0.973
     -> Extraction is nearly deterministic; the 63-row-tail lottery is
     REFUTED. Same-quality fuel + deterministic extraction + opposite
     gameplay outcomes = the difference lives in WHERE the corpus points
     the vector, not in noise.

     RUNNING (scripts/run_r1_at_vh2.sh, alongside iter-5 mining): the
     review's discriminator — r1 rows rejudged under vh2 (R=256, fresh
     seeds), corpus rebuilt with FRESH weights and vh2-current base moves,
     4 fixed-batch replicas averaged, alpha=0.2, ONE 20k paired vs vh2.
     Decision tree: wins -> channel repeatable, r2's direction was bad;
     loses -> one-shot/integration limit -> ship vh2, close the line.
     (Iter-5 scale ladder A/B/C mining in parallel throughout.)

198. **Final-candidate NO-GO: the r1 vector re-derived at vh2 — with every
     methodological correction — does not improve vh2. Per the pre-registered
     decision tree, the MICRO-CHANNEL IS CLOSED; vh2 is its deliverable.**
     (2026-08-02)

     The cleanest extraction the channel can produce: r1 rows rejudged under
     vh2 (R=256, fresh seeds: +6.0pp mean, 32% genuine, 1% phantom — the
     fuel is REAL), corpus of the 199 fresh-verified rows weighted by
     curse-free estimates against vh2's current moves, 4 fixed-batch
     replicas averaged (coherence 0.95+), alpha=0.2. 20k paired vs vh2:
       mean -86 [-332,+166]; P50 -156 [-407,+140];
       <1000 +0.4pp [+0.0,+0.8] = confirmed FLOOR LOSS.

     Review #6's tree: "rows remain valuable but retraining fails -> the
     improvement was one-shot or the new base cannot integrate another
     edit; stop." Both conditions met exactly. vh2 took one edit; a second
     edit — any corpus, any dose, any extraction — does not integrate.

     THE LINE'S LEDGER: vh1 (+4.6% over ep87) -> vh2 (+4.1% over vh1,
     20k-paired-verified) -> closed. Remaining open thread: the iter-5
     scale ladder (A 2.3M / B 4.3M / C 8.6M, literal 256ch recipe),
     mining now — the user's main directive and the sole active program.

199. **Scale A verdict: the literal 256ch recipe DEGRADES the small model
     monotonically from the start — epoch 3 (step-matched) already -32%,
     epoch 8 catastrophic. Scale axis (B/C) proceeds; recipe axis opens.**
     (2026-08-05)

     iter5a (Colab A100, user ran 12 epochs): corpus 6,175,263 states
     (5,866,500 train @ aug 8 = 46.9M samples/ep, 5,730 steps/ep).
     Train loss 1.68 -> 1.63 (barely moves); val FLAT 2.267 -> 2.257 while
     gameplay COLLAPSES — the corpus targets are largely already-fit;
     what IS learned (dw-emphasized/sharpened tails) hurts play:
       ep3  (~17k steps, the matched budget): 5k mean 9,092 / P50 6,438
             / <1000 6.3%   (vh2: 13,318 / 9,475 / 3.8%)  = -32% mean
       ep8: 5k mean 3,579 / P50 2,561 / <1000 17.6%        = -73% mean
     Same monotone-destruction signature as iter-3/pilot, now at 6.2M
     states — scale alone (at this recipe) did not bend the curve at A.

     USER PLAN: complete B (9M) and C (15M) at the SAME recipe to measure
     the scale axis cleanly even if negative; open a RECIPE axis on A's
     corpus in parallel. Arm a2 = the small line's own winning recipe at
     6M scale (UNTESTED: gate-3 was 394k): blend 0.5 hard-CE, T=1.0,
     dw=0, lr 1e-4, 3 epochs — train_iter5a2_colab.ipynb, same Drive data.
     Also requested: iter5a step checkpoints (e1_s500/1000/2000, epoch_1)
     for the destruction-onset ladder screen.

200. **ERRATUM for 196/199 + review #7 verdict: "good teacher, wrong corpus
     weighting and loss." Faithful-corpus arms built.** (2026-08-05)

     ChatGPT review (docs/small128_iter5_for_review.md) caught MY errors:
     (1) iter5.pt was NOT 70/30 — it is 23.5% crisis / 76.5% selfplay
     (1,453,708 + 4,721,555; the 80 UNCAPPED selfplay games ballooned;
     "70/30 by construction" in run_iter5.sh was written unverified);
     (2) "step-matched" conflated total steps with examples/update, warmup
     length and LR trajectory (bs8192x3ep vs the winner's bs32768x28ep);
     (3) "monotone collapse from step one" overclaimed (early ckpts uneval;
     ep3/ep8 from DIFFERENT runs — user's ep8 came from an epochs=12 run);
     (4) my "95% of gradient" was a row-weight proxy (its diagnostic: dis
     rows ~5.4% row-weight but 7.8% of weighted logit-grad norm);
     (5) the table supports RECIPE-wrong (dw3 anti-selects corrections;
     a2's effective top-1 target 0.656 vs iter5a's 0.360 explains a2's
     stability), not corpus-inert; (6) a2 1k eval cannot distinguish
     gain/parity/loss; (7) B/C composition confounds the scale axis.

     Verified myself: tensor boundary 1,453,708/4,721,555 exact (selfplay
     matches to the digit; crisis dir has since grown from B's mining).

     BUILT per its prescription: iter5_bal.pt = ALL 1,453,708 crisis +
     623,018 sampled selfplay = 2,076,726 @ TRUE 70/30, + full-legal vh2
     disagree_mask (230,370 rows, 11.1%). train_path_b --freeze-bn added
     (eager-module frozen-BN main forward; guarded against --compile).
     Tarball v5 (26,277,598 B). Two arms:
       iter5a3 EXACT CONTROL: bs32768, lr3e-4, 28ep, dw3/T0.7/pure-soft
               — the actual 256ch recipe, finally literal.
       iter5a4 EXTRACTION: bs8192, lr1e-4, 3ep, dw0/T1/blend.5,
               --disagree-gamma 3 (~33% of mass on the 11.1% disagreement
               rows; stratification proxy), --freeze-bn, no compile.
     User reframe accepted: all big-model wins were WARM-START (my
     from-scratch/warm-start dichotomy in #7 was wrong for the 256ch line
     — its warm starts had 22% teachable rows; the constraint is corpus
     geometry, not warm-starting per se).

201. **Faithful-corpus arms: a3 (the ACTUAL literal 256ch recipe) collapses
     from inside the warmup epoch (ep1 8,592 -> ep10 6,606); a4 (extraction:
     gamma3+freeze-BN+70/30) flat at -28%. a2's mild imitation remains the
     only non-destructive point. Brief #8 out.** (2026-08-05)

     Screens (vh2 ref ~13,700): a4 ep1-3: 9,653/9,231/9,661. a3 ep1/2/3/5/
     7/10: 8,592/6,859/7,732/6,757/6,897/6,606 — ep1 is 508 steps at WARMUP
     LR (~3e-5) and already -35%. Composition was NOT the fix: the literal
     recipe is WORSE on true 70/30 than on 23.5/76.5 (selfplay was DILUTING
     the destructive dw3/T0.7-on-crisis interaction). Cross-campaign law
     sharpened: gradient CONCENTRATION of any kind damages this model
     roughly in proportion; only mild uniform argmax-anchored self-imitation
     (a2) is stable, and it extracts ~nothing. vh2 = a local optimum whose
     measured neighborhood is downhill in every tried direction.
     docs/small128_iter5_arms_for_review.md (Q: a2->a4 single-variable
     decomposition; any full-corpus recipe left; next teacher change; or
     terminus = ship vh2 + close with B/C dose-response under a2).
     Scale-B corpus BANKED (iter5b.pt.gz, 576,807,301 B); C mining.

202. **BN-swap hybrids: BOTH components of a3's first epoch are severely
     damaging ALONE, and the intact checkpoint is better than either —
     weights and BN stats moved far off-manifold in entangled, partially
     compensating ways. Running statistics are a first-class damage
     channel.** (2026-08-05)

     Screens (vh2 ref ~13,700; intact a3_ep1 = 8,592):
       a3 weights + vh2 BN buffers: mean 4,160  (<1000 12.8%)
       vh2 weights + a3 BN buffers: mean 6,475  (<1000 8.8%)
     One warmup-LR epoch moved BN buffers enough to cost ~-53% BY
     THEMSELVES (8.37% buffer displacement, per review #8's measurement),
     while the weight update alone costs ~-70%; their entanglement partly
     cancels. Confirms review #8's CPU-probe prediction at gameplay level.
     Implication: any viable full-corpus recipe must handle BN running
     stats explicitly AND fix the loss direction — protecting BN alone
     would not have saved a3.

203. **a2 demoted (5k ladder: -9..-12% at every epoch — the 1k "parity" was
     noise); a5 pins composition toxicity; THE SELFPLAY DISCOVERY: the
     search teacher's uncapped games score median 99,144 / max 486,992
     (~10x greedy vh2, 2-10x the frozen master). Recipe dossier out for
     dual review (ChatGPT + Gemini).** (2026-08-06)

     a2 5k: ep1 11,663/8,302, ep2 11,620/8,263, ep3 12,157/8,707.
     a5 (same recipe, 70% crisis): screens 10,591 -> 8,950 monotone.
     Matrix complete: every visit-target recipe negative; damage scales
     with crisis fraction; BN stats a first-class channel (202); targets
     carry zero Q/prior (slim format) = deliberation-vs-decision
     unresolvable in-corpus.

     LEAD PROPOSAL (owner's): imitate DECISIONS not deliberation — hard
     one-hot CE on the PLAYED moves, selfplay-only corpus (4.7M states of
     median-99k demonstrations; blend_alpha=0), warm-start and/or the
     proven from-scratch hardce recipe; mask temperature moves.
     docs/small128_recipe_dossier.md (self-contained, for two reviewers).
     NO training until reviews land.

204. **Dual review (ChatGPT + Gemini) CONVERGES: crisis states are mandatory;
     the poison was soft visit targets + BN exposure. Stage-1 arms built:
     H70/H44 pure hard-CE with full sampling hygiene.** (2026-08-06)

     Shared verdict: (a) played-move hard CE (blend_alpha=0) replaces soft
     visit distributions — "in danger boards, deliberation is noise"; visit
     mass on delaying death-traps flips greedy argmaxes (Gemini's mechanism);
     (b) BN must not learn a crisis-heavy world: ChatGPT = train normal BN
     then post-hoc recalibration grid on deployment-like states; Gemini =
     split-batch training (crisis pass under eval-mode BN) — staged as
     take-5 if H-arms show promise; (c) sampling hygiene: per-replay caps
     (successful: first 128; failed: first len-20 capped at 96 — keep hard
     prefixes, drop terminal tails), stratify prevention/recovery ×
     success/failure, 10k-row cap per uncapped selfplay game, mask first 30
     temperature moves, split train/val by death game; (d) Q-recording
     canary before any big re-mine; (e) evaluation: add tail metrics +
     held-out crisis action accuracy; a neutral-mean/better-tail candidate
     is a WIN here.

     BUILT (build_hardce_corpora.py): pools crisis=926,719 (830k successful-
     escape rows, 97k failure prefixes) broad=1,911,185 (10k/game capped
     uncapped + 1k-capped games). h70.pt = 1,323,884 rows @70% crisis;
     h44.pt = 2,106,179 @44%. Arms: warm-start vh2, blend 0.0, dw0, T1,
     lr1e-4, bs8192, NORMAL BN, save-every-100. H70 = one-variable label
     test vs a3/a5; H44 = historically credible composition.

205. **The hall arc: hard-CE imitation is the first full-corpus recipe that
     CLIMBS — but converges to a mimicry ceiling ~1.3k below vh2 (path- and
     optimizer-independent). h_big built: ALL data, 10.69M rows, crisis-
     amplification mask. hall4 gamma arms out.** (2026-08-06)

     hall (bs8192/lr1e-4, 8+10ep spliced): 1k screens 10,690 -> 12,454
     rising; ep7 at 20k paired = mean -1,493 / P50 -1,111 vs vh2 (LOSS —
     the 1k flattered by ~600; screen-optimism instance #6). hall2: plateau
     11.4-12.6 band. hall3 (bs32768/lr3e-4, 20ep, clean cosine): FLAT from
     ep1 (~11.1-11.9) — arrived instantly at the same optimum hall crawled
     to. READING: pure imitation of this teacher at 3M params has a fixed
     point ~11.8-12k true — BELOW vh2 (13.3), which = mimicry ceiling +
     verified-edit stack. Echo of ep87 (mimicry of the 43k master also
     capped ~13k). Val loss under blend-0 is structurally anti-correlated
     (soft-CE vs a sharpening policy) — ignored by design.

     USER PRINCIPLE (standing): use ALL data, AMPLIFY the useful, discard
     nothing; test and measure. h_big.pt = 10,687,089 rows: every replay
     (full continuations) + every selfplay game UNCAPPED (the 10k/game cap
     was reviewer heuristic, dropped on challenge — it had discarded 4M
     rows); only label-quality trims (temperature moves, terminal death
     rows of failed replays). disagree_mask repurposed as CRISIS-provenance
     amplification dial (44.8% of rows; augmentation-invariant).
     Arms (hall3 geometry, 12ep): hall4g0 = uniform (pure 2x-data test);
     hall4g2 = crisis x3 (amplification test). One upload (605,588,900 B).

206. **hall4 finals + the CROSS-EXPERIMENT verdict: x3k (the 256ch winning
     corpus, native recipe) collapses vh2 to ~8.5k — failure follows the
     soft-target family + student, not corpus origin. And the hard-CE
     ceiling MOVES with data: 20k gap −1,493 (5.3M) → −424 (10.7M).**
     (2026-08-06)

     hall4g0 (10.7M, 12ep): 1k ladder 11.3→13.4(ep12, mirage #7 — 20k says
     12,910); 20k paired: ep12 −424 [−674,−181], ep5 −1,070. hall4g2
     (crisis ×3): flat 11.0-12.0 — amplification closed. x3k: ep1-12 flat
     ~8.1-9.1 — the historically-winning corpus destroys THIS student under
     soft targets exactly like ours did.

     LIVE AXES: (1) brute demonstration-scaling (mint more uncapped search
     selfplay; naive extrapolation crosses vh2 at ~2× more rows; unlimited
     locally); (2) owner's master-teacher proposal (search-on-256ch
     demonstrations — denser signal/row, ~1M-turn games; needs a pillar3k-
     backbone survival head ~1h). Dossier v2 FINALIZED for dual review
     (docs/small128_recipe_dossier_v2.md) with updated theory questions +
     pre-registered closure criteria ask. vh2 remains best.

207. **Dual review of dossier v2 — verdicts + queue.** (2026-08-06)

     ChatGPT's fresh measurements: (a) hall4 hard-target agreement FLAT
     across ep5->ep12 (83.7% both; vh2 86.5%) while gameplay rose +646 —
     the late gain lives BELOW the argmax (margins/BN/consolidation; the
     field-tilt mechanism again); (b) vh2-vs-v14_rev3 target agreement only
     65.0% (pillar3f's was 79.0%) — x3k confounded soft targets with denser
     contradiction; defensible claim = "soft visit CE robustly destructive
     for vh2 across two lineages"; (c) paired seeds: score correlation
     -0.001/-0.002 — pairing adds NOTHING post-butterfly (drop the "paired"
     language; independent bootstrap equivalent); (d) hall-vs-hall4 is NOT
     a clean 2x-data point (composition 64->44.8% crisis, caps, geometry);
     gamma2 = 71% weighted-loss on crisis and LOSES -> composition shift is
     part of hall4's gain; (e) hall4 floor STILL worse than vh2 (<1000
     +0.4pp) — naive selfplay minting would dilute crisis further.
     Gemini: brute-scale #1 (predicts parity ~20-25M rows, log curve), EMA
     cheap, master-teacher high-risk/high-reward #2, Q = the ceiling-
     breaker if scaling walls.

     ADOPTED QUEUE: (1) zero-training battery (interpolations alpha
     .1/.25/.5 of hall4ep12 into vh2, ep10-12 average; BN recal pending)
     — RUNNING; (2) tiny-LR continuation (3-5ep @ 3e-6 from ep12);
     (3) CONTROLLED half-vs-full corpus pair (matched composition/schedule)
     for a real scaling exponent; (4) controlled scaling arm: fresh
     independent game bank WITH fixed crisis quotas + fresh anchors;
     (5) bounded master-search pilot (20-40 games + master replaying vh2's
     crisis anchors; measure before training); (6) Q-canary re-mine
     (measure stability/Q-gaps first, no blind weighting). Closure only per
     ChatGPT's 5-condition rule (controlled 20M fail + master pilot fail +
     no hidden checkpoint via consolidation tricks).

208. **small128_vh3 PROMOTED: vh2 + 0.5·(hall4g0_ep12 − vh2), full-state
     interpolation — the consolidation channel works. 20k: mean +431
     [+178,+686], P50 +294 [+49,+591], floor neutral. NEW BEST.** (2026-08-07)

     **small128_vh3 = alphatrain/data/small128_vh3.pt** (TS: vh3_policy_ts)
     = interp_checkpoints --alpha 0.5 of iter5_hall4g0_ckpts_epoch_12 (the
     12-epoch pure hard-CE run on h_big: 10.69M rows, ALL data uncapped,
     bs32768/lr3e-4, which by ITSELF was -424 vs vh2) into vh2, ALL params
     AND BN buffers interpolated. **NEW 20k BAR: mean 13,765 / P50 9,658 /
     P5 1,140 / P10 1,908 / <1000 3.9%.**
     Dose curve: alpha=0.1 also passed (+289/+283); 0.5 took the mean.
     Epoch-average with base-BN self-destructed (801 — BN/weight
     entanglement, consistent with 202).

     **THE WORKING CRANK (two independent wins now):** (1) mint all-data
     demonstrations from the student's own search; (2) bulk hard-CE train
     (the endpoint may LOSE — it is a VECTOR GENERATOR, not a candidate);
     (3) alpha-interpolate the displacement into the current best;
     (4) 20k bar. Credit chain: owner's all-data/hard-label instincts +
     ChatGPT's "test vh2 + alpha(hall4-vh2)" + the field-tilt mechanism
     (gains live below the argmax; agreement stayed 83.7% flat while this
     vector was earning +431).

     NEXT (round 2 of the crank): fold C's finishing selfplay into the
     corpus, bulk-train FROM vh3, interpolate into vh3, re-bar. Optional
     same-day: finer alpha sweep 0.3-0.75.

209. **Round-2 (from vh3) corpora + vector notebooks shipped.** (2026-08-08)

     Fresh generation (loop_vh3, all with the NEW vh3-backbone head per the
     HISTORY-138 rule): 20k games -> value_head_vh3 -> 5,192 bulk replays
     @600 (2,596 fresh vh3 deaths) + 598 DEEP replays @1600/2400 + 1,500
     capped selfplay. Corpora:
       r2_bulk.pt     = 12,512,995 rows — ALL data ever generated (both
                        teacher eras, uncapped), hard-CE targets, plus a
                        CONTINUOUS danger score per row (1-P(survive H100),
                        vh3 head; median 0.067 / P90 0.471 / P99 0.927)
                        stored in disagree_mask -> w = 1 + gamma*danger.
       r2_frontier.pt = 576,576 rows — first-96-move escape windows of all
                        fresh SUCCESSFUL replays (incl. deep tranche) + 50%
                        broad anchor.
     Notebooks (all FROM vh3, bs32768/lr3e-4/blend0/12ep): r2bulk_g0
     (control), r2bulk_dg (gamma=6 danger amplification — critical MOMENTS,
     not provenance), r2frontier. Then local merges theta = vh3 + a*Dbulk
     + b*Dfrontier, dev/test 20k split. Pre-registered: exact-crank ~+2-5%;
     bulk+frontier central ~+14%; owner's bar +15-20%.
     (Stale iter5c selfplay generator killed at 81% — its 3,409 banked
     games retained; only generation of MORE stale-teacher data stopped.)

210. **Round-2 design AMENDED per design-review #10 + Gemini cross-check.**
     (2026-08-08)
     Adopted: era-weighting (fresh-era rows x2.1 -> ~40% of gradient mass;
     era sidecar RECONSTRUCTED via deterministic shuffle replay, verified
     against strata on all 12,512,995 rows — no rebuild/re-upload of the
     740MB corpus); danger arm kept at gamma=6 with intermediate doses via
     LOCAL vector interpolation t in {0,.5,1} (no 4th run); frontier kept
     whole-window and — after Gemini flagged the hard-CE failed-label risk
     and ChatGPT RETRACTED — restored to SUCCESSFUL-ONLY (my failed-prefix
     build reverted same-day; capability gated behind FRONTIER_FAILED=1
     pending independent label validation); merge = vh3 + a*Db + b*Df with
     norm-scaled beta grid + BN-recalibrated merge variants; epochs
     shortlisted by vector stability before dev evals. DISCLOSED: bulk
     build accidentally applied the 10k/game cap to OLD uncapped selfplay
     (env default slip) — shipped as-is, coincides with de-emphasize-old.
     **TEST BANK RESERVED: seeds 1100000-1119999 (never used in any
     training/generation/eval; first use = final confirmation only).**
     ChatGPT predictions: exact design +12%, amended +16% central (9-23%).

211. **H1/H2 discriminator pair queued (owner challenge: prove or retract
     "the great-epoch regime is over for 128ch").** (2026-08-08)
     Claim audit: NOTHING measured ties the stall to the 128ch architecture;
     the 256ch line also stopped producing promotable epochs near ITS
     frontier (task-arithmetic era). The real 128ch puzzle: stall DESPITE a
     3x teacher. Hypotheses: H1 capacity / H2 warm-start-basin+params (all
     recent runs warm-start vh3, <=12ep, one LR) / H3 corpus composition
     (76% old-era rows; measured today: era+danger weighting leaves the
     gradient direction cos 0.976 with unweighted — homeopathic).
     New notebooks (from-scratch, 40ep, UNWEIGHTED r2_bulk, val meaningful
     again): train_scratch128_colab.ipynb / train_scratch192_colab.ipynb
     (192 = owner-proposed DIAGNOSTIC, not a capacity pivot).
     Readout: both climb past vh3 => H2; 192 climbs, 128 stalls => H1;
     both stall ~vh1 => H3 (next arm: fresh-teacher-heavy corpus).
     Also: vector geometry of r2 arms (vector_stats.py) — ep1 is a warmup
     transient in a NEW direction (cos 0.55 w/ round-1 vector), ep2-4
     settle INTO the round-1 direction (0.85/0.93/0.955), g0 vs dg ep4
     cos 0.976. The 12.5M corpus keeps prescribing the medicine vh3
     already half-dosed. Merge screen (8 candidates, merge_vectors.py)
     running on dev slice 775000-775999 (vh3 same-slice bar 14,429).

212. **Corrected-recipe scratch curves substantially retract the early 128ch
     capacity story: update-rich/high-LR 128 keeps climbing, a strong-corpus
     128 matches the old 192 distribution, and endpoint-only reads hid both
     runs' best region.** (2026-08-09)

     All downloaded checkpoints were exported to TorchScript with exact eager
     agreement and the legal-move golden probe: `small128_strong` epochs 1--39,
     `scratch128_lr3e3` all 37 available epochs (1--4, 6, 8, 10--40), and
     `scratch128_bs2048` epochs 1, 6--16.  Export now accepts an explicit
     `--output`, so batch conversion no longer races through a shared
     `policy_ts.pt`.  Gameplay remained FP16 MPS, aggregate-only, and all
     distribution comparisons used independent bootstrap resampling; matching
     numeric seeds were deliberately not paired.  Reserved seeds
     1,100,000--1,119,999 remain untouched.

     **Update-rich bs2048 (lr=3e-4):** the 999-game curve kept improving after
     its epoch-6 parity point: means ep7--16 = 4,376, 4,866, 5,336, 6,116,
     7,045, 7,007, 7,526, 7,537, 7,883, 7,846.  Epoch 15 at 5k is mean 7,946 /
     median 5,712 / P10 1,248 / <1000 7.00% / >10k 28.16%.  Versus epoch 6
     at 5k it gained mean +3,752 [3,518,3,982], median +2,664
     [2,410,2,882], P10 +446 [351,519], and cut <1000 by 7.28pp
     [6.08,8.48].  It had 696,525 Adam updates by epoch 15; the small-batch
     effect is therefore batch noise plus vastly more updates, not an isolated
     capacity result.

     **Uniform Pillar3k-relabeled strong corpus:** sparse 999 screens rose from
     mean 1,202 (ep3) to 7,404 (ep18), 8,280 (ep27), and a real ep30--32 peak
     of 8,966 / 9,411 / 9,339 before collapsing to 6,874 at ep33 and recovering
     only to 7,897 at ep39.  Thus the supplied ep29/ep39 endpoints had hidden
     the best checkpoint.  Epoch 32 at 5k is mean 9,430 / median 6,524 / P10
     1,409 / <1000 5.72% / >10k 34.88%.  It is distributionally
     indistinguishable from old scratch192 epoch 19 (mean 9,457 / median 6,588)
     on every reported metric.  Strong data therefore DID teach transferable
     policy structure to 128ch; its final-checkpoint regression is an
     optimization/gating problem, not evidence that expert trajectories are
     unusable or that the student lacks capacity.

     **R2 scratch128, bs32768/lr3e-3:** the late envelope continued upward:
     999-game means ep12/15/18/21/24/27/30/33 = 4,702 / 5,680 / 6,308 /
     6,666 / 7,589 / 9,132 / 8,458 / 9,143, followed by a broad ep35--36
     peak (10,490 / 10,428), ep37--39 noise/regression, and ep40 10,523.
     At 5k, ep36 = mean 9,942 / median 6,978 / P10 1,482 / <1000 5.24% /
     >10k 36.94%; ep40 = 10,136 / 6,950 / 1,455 / 5.32% / 36.62%.
     Ep40 minus ep36 is a wash: mean +194 [-187,575], median -27
     [-420,362], P10 -27 [-166,100], <1000 +0.08pp [-0.80,0.96].  The final
     checkpoint is therefore valid, but not demonstrably better than the
     balanced late plateau.

     High-LR ep40 beats strong32 by mean +706 [334,1,079] and median +427
     [59,784], while its P10/failure deltas are unresolved.  It still trails
     vh3 at 5k: mean -3,722 [-4,187,-3,261], median -2,664
     [-3,085,-2,224], P10 -513 [-663,-365], <1000 +1.66pp
     [+0.84,+2.46], >10k -11.84pp [-13.78,-9.92].  So 128ch has NOT caught
     vh3 and infinite play is not established, but there is no observed
     scratch-128 plateau: the old low-LR result was a recipe/pipeline failure
     with optimization clearly a major factor, and scratch128 high-LR now
     exceeds old scratch192 ep19 despite the width difference.  Exact credit
     between LR/batch geometry and corrected D4 still requires the corrected
     bs32768/lr3e-4 control.

     Fixed turn-survival rates make the remaining indefinite-play gap clearer.
     High-LR ep40 reaches turns 1k / 2k / 5k / 10k / 20k in 85.16% / 68.70% /
     35.98% / 12.92% / 1.96% of games; vh3 reaches them in 89.88% / 76.70% /
     47.66% / 22.06% / 5.26%.  The scratch recipe has learned meaningful
     long-horizon reliability, but neither finite distribution supports an
     "infinite" claim.  Future gates should report this survival curve in
     addition to heavy-tailed score summaries.

     Width remains scientifically open because the old scratch192 run used
     lr=3e-4 AND the corrupt directional D4 views, whereas all three new 128
     recipe arms use corrected D4.  Added `train_scratch192_lr3e3_colab.ipynb`:
     identical R2 corpus/seed/batch/40-epoch lr=3e-3 schedule/objective and
     corrected D4, only width changed, with wall-clock logging.  Compare it to
     scratch128 both at equal optimizer steps/examples and equal wall time;
     capacity evidence requires a tuned 128 envelope to flatten while 192
     continues improving broadly, especially P5/P10/crisis failure.

     **Operational verdict:** retain `small128_vh3` as the lineage champion;
     the late high-LR scratch checkpoints are recipe/capacity references, not
     promotions.  The next primary experiment is the lineage-pure capped play
     + clean reanalyse + weighted all-data replay loop in `FLYWHEEL_PLAN.md`:
     many independent
     ~1k-turn games, broad anchors retained, explicit crisis/deep quotas and
     per-game caps, all 30 clean search candidates, behavior/teacher actions
     separated, corrected D4, update-rich training, frequent step checkpoints,
     and floor-aware 5k promotion.  Old big-model data may be tested only as a
     separately declared rehearsal arm.  The 43k disagreement slice remains a
     diagnostic/judge set, never the training corpus.

213. **Corrected high-LR 192 absorbs labels faster per update, but its early
     advantage disappears at the owner's measured 2x wall-clock/inference
     ratio; keep 128 as the flywheel actor pending the decisive e20/e40
     comparison.** (2026-08-09)

     `scratch192_lr3e3` epochs 1--5 are the intended control: 10b x 192ch,
     bs32768, lr3e-3, corrected D4, 2,903 Adam steps/epoch.  Their 999-game
     means are 248 / 1,631 / 2,798 / 4,133 / 5,097.  Epoch 5 distribution:
     median 3,691 / P10 843 / P90 11,160 / P95 14,837 / <1000 12.5%.

     Equal-update fixed-state audit at epoch 4 (50k R2 states, exact legal,
     FP16): 192 clearly learns faster than 128 — target match 72.89% vs 69.79%,
     legal NLL 0.797 vs 0.943; on danger>=.60, match 72.60% vs 67.72% and legal
     NLL 0.857 vs 1.066.  This is width-dependent sample efficiency, not yet a
     ceiling.

     At approximately equal training time, 192e5 and 128e10 are functionally
     identical: target match 73.69% vs 73.61%, legal NLL 0.762 vs 0.768, and
     on their 11,351 action disagreements the recorded target favors 192 in
     39.52% versus 128 in 39.15%.  Gameplay is also a screen-level wash:
     192e5 minus 128e10 mean +359 [-43,+753], median +314 [-77,+710], P10 +7
     [-110,+149], <1000 -0.10pp [-3.00,+2.80].

     Inference throughput is the binding system cost because training is rare
     and the flywheel spends most compute on selfplay/crisis/search.  Therefore
     width/depth changes must be judged on a strength-versus-production-FP16
     inference frontier, not equal epochs or parameter count.  The decisive
     existing-run ladder is 192 epochs 10/15/20 versus 128 epochs 20/30/40.
     If 192e20 does not beat 128e40 materially and broadly, stop widening the
     actor.  Do not add residual blocks speculatively: first exhaust same-cost
     levers (current-policy flywheel, clean/full-support targets, legal-support
     training, and symmetry canonicalization).  A larger model may remain a
     selective reanalysis/crisis teacher without paying its cost on every bulk
     selfplay forward.

214. **Started the first clean, current-lineage 128ch policy-iteration pilot;
     the 8k-state end-to-end canary validated label separation and exact
     mid-epoch resume, while exposing and fixing three latent integration
     bugs before the multi-million-state run.** (2026-08-09)

     The frozen pilot is declared in
     `flywheel/vh3_iteration_1_pilot.json`: 3,000 noise-free MCTS@400 exploit
     games, 750 exploratory MCTS@200 games with a separate noise-free
     MCTS@400 label tree, and 500 greedy probes feeding noise-free crisis
     replays (recovery@600/prevention@1600). Games remain independently capped
     at 1,000 turns; expected fresh search data is 3.5--4.5M states, combined
     during training with a deterministic 1M sample of the existing 19.6M
     current-vh3 anchors. All generation is FP16, terminal-fixed, top-30,
     full-root-record, and excludes big-model targets and reserved seeds.

     Native generation now writes immutable `run_config.json` and
     `generation_complete.json`, creates nested output directories, stores the
     run ID/model/value/precision/full-record fields in every game, rejects
     unknown CLI arguments, and remains atomic per game. Corpus construction
     validates every file against the manifest and uses flushed disk-backed
     arrays plus `progress.json`, so interruption resumes at the next file.
     Base-policy annotation likewise resumes by inference batch and records
     checkpoint/corpus identities. Training now has exact mid-epoch resume:
     optimizer, scheduler, CPU/device RNG, deterministic epoch permutation,
     next batch, and partial scalar/source diagnostics are atomically stored in
     `latest.pt`.

     Canary: four exploit plus four explore games all reached the 1,000-turn
     cap, yielding 8,000/8,000 clean full-record rows and zero invalid targets.
     Exploit behavior equaled its teacher on 100%; explore behavior/teacher
     agreement was 21.7% in the first 15 temperature turns and 95.7%
     thereafter, proving behavior and labels are genuinely separated. Root
     top-share P10/P50/P90 was roughly .25/.39/.60 with all 30 candidates
     retained. Exact FP16 all-legal annotation found 333/8,000 (4.16%) clean
     teacher disagreements: 4.10% exploit and 4.23% explore. Only 98 rows
     combined both disagreement and visit top-share >=.30, so the correction
     signal is rare and often visit-close rather than a huge high-confidence
     class.

     This measurement explains the weak old warm starts quantitatively. With a
     4:1 fresh-target/anchor mix and anchor weight .1, the plastic aggregate
     arms use 16x original-view disagreement weight, putting edits near 40% of
     effective loss while retaining every agreement/anchor row. Critically,
     edit membership comes from the immutable original-view sidecar: online
     disagreement rose to ~17% under D4 because vh3 is not equivariant, and
     must remain a separate symmetry diagnostic rather than receiving the edit
     dose.

     Added the missing plastic objective: unconstrained hard/soft search CE on
     rolling-search target trajectories plus full-distribution base KL only on
     replay anchors. It supports both warm and same-architecture scratch init;
     the matched basin test is bounded warm versus aggregate warm versus
     aggregate scratch. Two tensor bugs found by the real smoke were fixed:
     fp16 sparse probabilities are explicitly reconstructed into fp32 dense
     targets, and `scatter_add_` prevents zero-valued padding at action 0 from
     overwriting a genuine action-0 label. A deliberately interrupted MPS
     scratch smoke resumed epoch 1 at batch 6 and reproduced batches 7--8's
     loss/KL/gradient/NLL exactly before completing both epochs (RNG tensors
     are now restored to CPU before PyTorch RNG setters).

     The full exploit stream is now running under `caffeinate`. At 14
     concurrent games it measured effective inference batch 55 and about
     20.5k leaf evaluations/s, implying roughly 17 hours for its 3M states.
     The exact restart/build/annotate/train/audit commands and gates are in
     `flywheel/VH3_ITERATION_1_PILOT.md`.

215. **High-LR 192 has pulled ahead at equal training wall-clock by epoch 12;
     launch a clean 10b x 96ch third scaling point, while keeping parameter-
     matched depth/width reshaping as a separate diagnostic.** (2026-08-09)

     Owner's aggregate epoch-12 screen for `scratch192_lr3e3` reports
     P1/P5/P10/P25/P50/P75/P90/P95 = 478/940/1453/3256/7253/14511/24372/31245.
     At the measured 2x training-time ratio, the relevant 128ch comparator is
     epoch 24: 443/898/1287/2534/5644/10301/16887/21210. Thus the 192ch model
     is only modestly separated at the lower floor but about 29% ahead at the
     median and 44--47% ahead at P90/P95. It already broadly matches the late
     128ch epoch-40 screen (531/923/1530/3315/6785/14370/23371/30832). This
     overturns the early equal-wall-clock wash, but remains a 999-game screen;
     confirm epoch 12 at 5k and continue the 192 trajectory before declaring
     a higher asymptote. Its 2x production-inference cost still controls actor
     selection.

     Added `train_scratch96_lr3e3_colab.ipynb`: 10 blocks x 96 channels,
     unchanged 128ch policy head, 1.702M parameters, same corrected-D4 R2
     corpus/seed/batch/lr/schedule/objective as the 128/192 high-LR controls.
     It resumes from the last completed Drive epoch. The notebook's gate is
     aggregate-only FP16: sparse epoch screens, then 5k only for the best
     apparent plateau. The generator now accepts selected arm names and makes
     block count explicit, avoiding rewrites of unrelated notebooks.

     A parameter-matched 96ch model would be 18b x 96ch (3.032M params) versus
     10b x 128ch (3.003M). This is scientifically useful only as a second
     question: if 10b x 96 loses, does restoring parameter count via depth
     restore strength? Equal parameters do not mean equal production cost:
     18b x 96 has 39 convolution stages versus 23 for 10b x 128, narrower
     kernels, and substantially more sequential BN/ReLU barriers. Measure
     native FP16 latency at the effective MCTS batch rather than assuming it
     is equal. Other near-parameter-matched shapes are 13b x 112ch (2.985M)
     and 8b x 144ch (3.044M). Do not launch the shape sweep until the pure
     10b x 96 width point shows which ambiguity is worth resolving.

216. **Owner elected to run the exact parameter-matched deep/narrow control in
     parallel: 18b x 96ch versus 10b x 96ch and 10b x 128ch.** (2026-08-09)

     Added `train_scratch18b96_lr3e3_colab.ipynb`: 18 residual blocks, 96
     channels, 3.032M parameters, otherwise exactly the corrected-D4 high-LR
     scratch recipe (R2, seed 42, bs32768, lr3e-3, 40-epoch schedule, unchanged
     policy head/objective). It has its own Drive lineage and completed-epoch
     resume. This intentionally overrides entry 215's conservative sequencing
     because independent Colab capacity is available and the result closes a
     useful ambiguity while the 10b x 96 control runs.

     Predeclared readout: 18b96 versus 10b96 isolates the effect of adding
     sequential nonlinear computation at fixed width; 18b96 versus 10b128 is
     nearly parameter-matched and tests depth/width substitution. If 18b96
     restores 128-level broad performance, total representational computation
     matters and depth can substitute for width. If it especially improves
     P5/P10, crisis-stratum target fit, and long-turn survival, that supports
     the owner's deeper-clearance hypothesis. If it remains near 10b96 while
     10b128 wins, width/parallel feature rank is the likely bottleneck. If
     training optimization is unstable or still rising at epoch 40, do not
     mislabel that as a capacity ceiling; inspect fixed-state loss/gradient and
     consider a depth-specific LR/schedule control. Gameplay comparisons remain
     independent aggregate distributions, with 5k reserved for the best
     apparent envelopes.

     Even a gameplay win is not automatically an actor win: 18b96 has 39
     sequential convolution stages versus 23 for 10b128. Measure native FP16
     throughput at the flywheel's effective inference batch before promotion.

217. **Epoch-5 architecture screen: parameter-matched 18b x 96ch already
     beats 10b x 128ch broadly, while 10b x 192ch remains far ahead.**
     (2026-08-09)

     Owner's common aggregate screen, reported as
     P1/P5/P10/P25/P50/P75/P90/P95:

     - 10b96 (1.702M): 199/290/365/540/862/1366/2002/2547
     - 10b128 (3.003M): 243/414/534/911/1721/3144/4965/5952
     - 18b96 (3.032M): 269/459/644/1130/2150/3653/5967/7470
     - 10b192 (6.710M): 336/601/843/1761/3691/7120/11160/14837

     At identical data/steps/recipe, 18b96 beats parameter-matched 10b128 at
     every percentile: about +11% at P1/P5, +21% at P10, +25% at the median,
     and +20--26% at P90/P95. Against equal-width 10b96, adding eight blocks
     gives +35% P1, +76% P10, +149% median, and roughly +193% P95. Architecture
     shape therefore materially changes early absorption; parameter count
     alone does not explain learning efficiency. Width still matters: 10b192
     remains +31% at P10, +72% median, and roughly +99% P95 over 18b96.

     This is not yet evidence specifically for deeper clearance reasoning. The
     18b96 advantage is broad and relatively smaller at P1/P5 than at the
     center/upper tail, consistent with generally faster policy learning. Raw
     full-softmax validation loss is actively non-diagnostic here: e5 18b96
     and 10b128 are nearly equal (4.491 vs 4.495), while much weaker 10b96 is
     lower at 4.340. Use legal fixed-state match/NLL stratified by danger and
     later survival/crisis distributions instead.

     Do not spend a 5k gate on these still-moving epoch-5 models. Continue the
     trajectories and screen sparse later epochs (roughly 8/12/18/24/30/36/40),
     compare best envelopes, then run 5k on the credible finalists. Record
     per-epoch wall time and benchmark native FP16 at effective MCTS batch ~55:
     the deeper shape is an actor candidate only if its strength gain survives
     its additional sequential-kernel latency.

218. **The corrected 192ch curve continues beyond the tuned 128ch plateau:
     epoch 17 is clearly stronger than the equal-wall-clock and late-128
     envelopes, and is approaching vh3.** (2026-08-09)

     Owner's aggregate `scratch192_lr3e3` epoch-17 screen reports
     P1/P5/P10/P25/P50/P75/P90/P95 =
     528/1132/1568/3962/8676/15960/26294/35211. Versus its own epoch 12 this
     is broad continued improvement: +10% P1, +20% P5, +8% P10, +22% P25,
     +20% median, +10% P75, +8% P90, and +13% P95. It has not merely reached
     the 128 result in fewer epochs and stopped.

     With 192 training measured at 2x the 128 wall time, epoch 17 corresponds
     approximately to 128 epoch 34. Against the neighboring strong 128 epoch
     35 screen, 192e17 is +17% median, +10--13% P90/P95, +5% P5, and -4% P10;
     against final 128e40 it is +28% median, +11--14% P75/P90/P95, +23% P5,
     and essentially tied at P1/P10. Thus the early equal-wall-clock wash has
     decisively broken in 192's favor over most of the distribution. This is
     the strongest width/capacity evidence so far for the fixed 10-block
     scratch recipe.

     It remains below the 5k vh3 reference by roughly 10% median, 15--16%
     upper tail, and 20% P10, while P1/P5 are already close; this cross-sample-
     size comparison is only directional. Screen epoch 20 as predeclared, then
     take the better e17/e20 envelope to a 5k independent-distribution gate.
     Do not infer a paired result from common numeric seeds. Raw validation
     loss again moves the wrong way (4.526 at e12 to 4.576 at e17) while
     gameplay improves, so it remains unsuitable for checkpoint selection.

     Production conclusion is separate: 192 still costs 2x inference and is
     not automatically the flywheel actor. It is now a credible stronger
     teacher/reference. Whether a ~3M-parameter actor can recover the useful
     depth through architecture rather than width is the purpose of the
     running 18b x 96 control.

219. **Queued the corrected high-LR 10b x 256ch upper scaling control; define
     what would and would not constitute a capacity proof.** (2026-08-09)

     Added `train_scratch256_lr3e3_colab.ipynb`: 10 blocks x 256 channels,
     11.893M parameters, otherwise exactly the scratch96/128/192 R2 control
     (seed 42, corrected D4, bs32768, lr3e-3, 40 epochs, unweighted objective,
     unchanged 128ch policy head). It has a separate Drive lineage, completed-
     epoch resume, sparse aggregate-only FP16 screens, and 5k finalist gates.
     This is also exactly pillar3k's backbone architecture, so comparison to
     scratch192 isolates upper-width scaling while comparison to pillar3k
     exposes how much strength comes from data/iteration rather than nominal
     architecture.

     Expected readout: if 256 beats 192 at the same epochs, it demonstrates
     continuing width-dependent sample efficiency. If its best late envelope
     also beats a converged 192 envelope broadly, while the smaller envelopes
     remain flat under adequate updates, that is strong capacity evidence for
     the fixed 10-block representation. If 256 and 192 converge together,
     corpus/target or recipe saturation is more likely. A weak/divergent 256
     at lr3e-3 is not evidence against width until a width-appropriate LR
     control is checked; the identical LR arm is deliberately first because it
     is the clean comparison and already works for 128/192.

     One R2 ladder cannot prove that a 128-class actor is intrinsically unable
     to reach the goal. R2 contains mixed historical target eras, and the
     running 18b96 result already shows that parameter arrangement changes
     absorption at fixed size. The stronger closure standard is: (1) tuned
     small-model envelopes demonstrably flatten; (2) larger widths continue to
     lower exact legal training/heldout NLL and improve 5k gameplay broadly,
     especially crisis/survival strata; (3) the ordering persists on a clean,
     single-lineage, consistently reanalysed 3--6M-state corpus; and (4) close
     architecture differences survive another training realization. Until
     then call the result a capacity signal, not a definite impossibility
     proof.

     Actor selection remains a strength/FP16-production frontier. A 256 model
     is expected to cost roughly 3--4x 128 inference and is primarily an upper
     bound/teacher candidate unless its strength gain is exceptional.

220. **The tuned 10b128 run has plateaued under its completed cosine, but
     10b192 epoch 20 now exceeds vh3 and 18b96 remains a credible near-128-
     latency actor; static scratch scaling still cannot answer whether either
     small model can self-improve.** (2026-08-09)

     The 128 plateau statement is deliberately narrow. On 5,000 games,
     `scratch128_lr3e3` epoch 36 versus epoch 40 is a distributional wash:
     mean 9,942 versus 10,136, median 6,978 versus 6,950, P10 1,482 versus
     1,455, and `<1000` 5.24% versus 5.32%. Their fixed-state actions agree on
     97.22% of rows and legal NLL only changes 0.635 to 0.632 while the
     schedule reaches lr=3e-5. More tail epochs on the same decayed schedule
     and R2 corpus are therefore low value. This is an optimization/data-run
     plateau, not proof that 10b128 cannot move after a learning-rate restart
     or fresh on-policy search targets.

     `scratch192_lr3e3` epoch 20 completed a 5,000-game FP16 MPS gate:
     mean 15,012 / median 10,578 / P5 1,239 / P10 2,059 / P90 33,550 /
     P95 43,341 / `<1000` 3.52% / `>10000` 52.24%. Independent bootstrap
     against `small128_vh3` gives mean +1,154 [+587,+1,706], median +962
     [+482,+1,510], P90 +2,238 [+373,+4,056], and `>10000` +3.78pp
     [+1.86,+5.66]. P10 +90 [-78,+279], `<1000` -0.14pp [-0.88,+0.58],
     and turn-survival at 1k +0.70pp [-0.50,+1.88] remain unresolved; 2k/5k/
     10k survival improves significantly by +2.02/+3.78/+3.38pp. Against
     192e18 it improves mean +2,234 [+1,721,+2,754], median +1,604
     [+1,072,+2,162], and P10 +382 [+200,+550]. Thus 192 is still learning
     at e20 and establishes a higher fixed-10-block envelope than 128, although
     its approximately 2x production inference cost keeps it a reference or
     teacher rather than the default actor.

     Fixed-state R2 fit is not a sufficient strength objective. From 192e18
     to e20, all-state legal NLL improves 0.619 to 0.607 and match 77.80% to
     78.02%, but the `danger>=.60` match is flat at 77.85% and NLL is
     effectively flat/slightly worse (0.648 to 0.650). More strikingly, vh3
     still matches 88.61% of the historical R2 labels with NLL 0.550 while
     losing to 192e20 in gameplay. The old mixed-era labels reward imitation
     of vh3's lineage, not every transferable ingredient of policy quality.
     Do not select checkpoints or declare capacity solely from this loss.

     `scratch18b96_lr3e3` is still moving at epoch 10: mean 6,295 / median
     4,409 / P10 957 on the 999-game screen. Its turn survival at 1k/2k/5k
     rises from 71.37%/43.24%/10.31% at e7 to 74.27%/52.95%/18.32% at e10;
     it already resembles 10b128 epoch 18 rather than equal-epoch 128. On the
     50k fixed audit it beats 128e10 (match 74.63% vs 73.97%, legal NLL 0.733
     vs 0.753; danger NLL 0.808 vs 0.846), and on their 11,110 disagreements
     the label favors 18b96 40.83% versus 128 37.87%. Its worst-score P10,
     however, barely moves from 933 to 957. Continue sparse e15/e20/e30/e40
     gates and take only its apparent late envelope to 5k; the present curve
     makes a higher landing than 10b128 plausible, not proven. Native FP16
     latency at effective MCTS batch 55 is 2.648ms versus 2.513ms for 10b128,
     so a sustained strength win would improve the production frontier.

     The decisive capacity question now moves to closed-loop attribution.
     Scratch training only asks whether a model can absorb frozen historical
     targets. A candidate proves self-improvement only if: its own frozen
     search produces causally better actions under fixed-state common-random-
     number rollouts; the same architecture absorbs those edits without broad
     regression; a 5k independent gameplay distribution improves; and the
     promoted model repeats the cycle. Search-win/student-fail implicates
     capacity/objective/optimization; search-fail implicates the teacher/value
     operator, not student width. The running vh3 iteration-1 clean pilot is
     the first direct 128ch test. Use many independent ~1k-turn capped games
     and staged 1k/2k/5k survival gates rather than uncapped play. Updated
     `compare_score_distributions.py` to report independent-bootstrap turn-
     survival deltas; its focused tests pass.

221. **New architecture checkpoints strengthen 18b96 as the small-actor
     candidate, expose a depth/width distribution-shape tradeoff, and show
     192e22 improving old-target fit while gameplay regresses.** (2026-08-09)

     Newly landed 18b96 epochs 11--14, 10b96 epochs 18--23, and 192 epochs
     21--22 were exported to TorchScript with exact eager/trace agreement and
     the legal golden probe. Gameplay remained aggregate-only FP16 MPS. Sparse
     endpoint screens were used first; the only backfilled intermediate was
     192e21, justified by the e22 endpoint regression. All statistical
     comparisons independently resampled whole-game distributions and did not
     pair matching numeric seed IDs.

     **18b96 remains clearly unconverged.** Epoch 14 over 999 games reaches
     mean 7,619 / median 5,260 / P10 1,072 / P90 16,838 / `<1000` 9.31% /
     `>10000` 25.63%. Versus e10, independent bootstrap gives mean +1,324
     [+692,+1,943], median +851 [+115,+1,544], P90 +2,151
     [+538,+4,395], and `>10000` +7.11pp [+3.40,+10.71]. P10 +113
     [-61,+267] and `<1000` -1.50pp [-4.10,+1.00] are directionally better
     but unresolved. Fixed-state legal NLL improves 0.733 to 0.709 and
     `danger>=.60` NLL 0.808 to 0.767; on e10/e14 disagreements the recorded
     label favors e14 41.26% versus e10 37.36%. The earlier P10 stall was not
     a plateau.

     At roughly strength-matched endpoints, 18b96e14 and 10b128e24 have tied
     mean score/lifetime (7,619/3,749 turns versus 7,589/3,735), but different
     shapes. The deep/narrow model is worse at P10 and early survival (turn
     >=1k 77.88% versus 81.58%; >=2k 57.26% versus 62.06%), while it is better
     at very long survival (>=10k 7.01% versus 5.31%; >=20k 0.70% versus
     0.10%). The 128 endpoint also fits old R2 labels better (legal NLL 0.671 /
     danger NLL 0.737 versus 18b96's 0.709 / 0.767), despite tied gameplay
     means. Treat this as a hypothesis, not an architectural proof: depth may
     learn more transferable sustainable-play structure while width supplies
     robust crisis feature rank and/or historical-label imitation. If the
     crossing persists at late 5k gates, targeted crisis replay is the natural
     complement to the low-latency 18b96 shape.

     **10b96 is also still learning but remains far behind.** Epochs 17/20/23
     means are 2,377/2,835/3,181 and medians 1,812/2,169/2,408. E17 to e23 is
     broad: mean +804 [+597,+1,012], median +596 [+357,+806], P10 +89
     [+23,+152], and `<1000` -6.71pp [-10.31,-3.20]. Fixed legal NLL improves
     0.823 to 0.788 and danger NLL 0.929 to 0.894. Thus 96ch has not supplied a
     capacity-ceiling point yet, but the enormous gap to same-width 18b96
     confirms that sequential depth changes usable representation, not merely
     parameter count.

     **192 has a local gameplay peak/tradeoff around e20--21, not a declared
     asymptote.** Epoch 21's 999 screen is mean 14,937 / median 9,830 / P10
     2,046 / P90 35,662; every reported e20/e21 delta is unresolved. Epoch 22
     is mean 13,557 / median 9,503 / P10 2,183 / P90 30,204. Versus e21 its
     mean falls -1,381 [-2,614,-150], P90 -5,508 [-10,091,-1,000], and mean
     lifetime -662 [-1,271,-74], while the lower floor remains unresolved.
     Nevertheless e20 to e22 fixed-state legal NLL improves 0.607 to 0.596,
     danger NLL 0.650 to 0.630, and on their disagreements the old target
     favors e22 42.00% versus e20 39.37%. This is direct evidence that further
     optimizing historical R2 targets can hurt gameplay. Retain e20 as the
     5k-validated reference, skip an e22 5k gate, and continue sparse later
     screens to detect recovery or a genuinely new envelope.

     Operationally, continue 18b96 at approximately e18/e22/e28/e34/e40 and
     use a 5k gate only near its apparent envelope, with floor and staged turn
     survival primary. Continue 192 sparsely rather than chasing every local
     epoch. None of these frozen-corpus curves answers self-improvement; the
     clean vh3 play/reanalyse/train pilot remains the decisive closed-loop
     experiment.

222. **Completed scratch architecture envelopes: 18b96 surpasses vh3 at
     near-128 inference cost, while 192e39 establishes a materially higher
     crisis and long-play ceiling; the smallest model still needs a closed-
     loop test.** (2026-08-10)

     All remaining checkpoints landed: 10b96 and 18b96 through epoch 40,
     10b192 through epoch 39 (no e40 supplied), and no 256 checkpoints. Sparse
     late candidates were exported with exact eager/trace and legal-golden
     agreement. To avoid spending almost all evaluation time extending games
     which were already sustainable, the late sweep used a predeclared staged
     gate: 5,000 independent games capped at 1,000 turns for early catastrophe
     reliability, followed by ordinary full-length 999 screens and 5,000-game
     gates only for apparent envelopes. All inference was FP16 MPS under
     caffeinate; comparisons independently resampled complete game
     distributions and never paired matching numeric seed IDs.

     **The cap-1k trajectories cleanly separate the architectures:**

     - 10b96 e28/e34/e40: 57.84% / 62.98% / 65.58% capped;
     - 18b96 e18/e22/e26/e30/e34/e37/e40: 84.24% / 86.78% / 88.06% /
       89.72% / 89.10% / 89.54% / 89.26%;
     - 10b192 e25/e28/e31/e34/e37/e39: 90.74% / 90.42% / 91.64% /
       91.48% / 91.96% / 92.72%.

     Thus 10b96 was still improving at its endpoint, 18b96 reached a broad
     early-hazard plateau around e30, and 192 continued improving to the last
     available checkpoint. E39 minus e25 cap rate is +1.98pp
     [+0.90,+3.08]; e39 minus the best-point 18b96e30 is +3.00pp
     [+1.90,+4.10]. Width therefore buys real crisis reliability under this
     frozen-corpus recipe, not merely a heavier upper tail.

     **Full-length finalist distributions (5,000 games):**

     - 18b96e30: mean 13,917 / median 9,926 / P5 1,172 / P10 1,927 /
       P90 31,125 / `<1000` 3.90% / `>10000` 49.56%;
     - 18b96e40: 14,766 / 10,384 / 1,099 / 1,873 / 33,982 / 4.16% / 51.52%;
     - 192e39: 18,500 / 13,040 / 1,458 / 2,532 / 41,478 / 2.72% / 59.34%.

     E30 is distributionally identical to vh3 on every reported metric:
     mean +58 [-474,+575], median +310 [-155,+748], P10 -42
     [-208,+156], 1k survival -0.16pp [-1.38,+1.02]. This is a from-scratch
     model trained only on R2, not a warm-start promotion. E40 trades a tied
     early floor for significantly more sustainable-play strength. Versus
     vh3 it gains mean +907 [+363,+1,450], median +766 [+264,+1,257], P90
     +2,669 [+563,+4,424], `>10000` +3.06pp [+1.10,+5.00], 5k survival
     +2.90pp [+0.92,+4.88], and 10k survival +2.94pp [+1.32,+4.62]; P10 -97
     [-279,+84] and 1k survival -0.52pp [-1.72,+0.68] are unresolved. From
     e30 to e40, mean and 10k/20k survival improve significantly while the
     early floor remains tied. The 18b96 architecture therefore had not
     globally plateaued—only its historical-data early-hazard channel had.

     E40 is broadly tied with the old 192e20 envelope despite near-128
     inference cost. The exceptions favor 192e20 at the crisis floor: P10
     -188 [-394,-2] and 1k survival -1.22pp [-2.38,-0.08] for 18b96e40;
     means, medians, upper tails, and longer survival are unresolved. This is
     the key parameter-shape result: 18b96 (3.032M parameters, measured only
     ~5% slower than 10b128 at MCTS batch 55) recovers nearly all of a 6.71M-
     parameter 10b192e20 policy and decisively exceeds parameter-matched
     10b128e40.

     192 nevertheless keeps scaling. E39 versus e20 improves mean +3,488
     [+2,836,+4,096], median +2,462 [+1,848,+3,179], P10 +474
     [+245,+674], `<1000` -0.80pp [-1.52,-0.16], and turn survival at
     1k/2k/5k/10k/20k by +2.14/+4.20/+7.24/+8.26/+4.32pp, all significant.
     Against 18b96e40, e39 gains mean +3,734 [+3,098,+4,403], median +2,658
     [+2,083,+3,390], P10 +661 [+418,+875], `<1000` -1.44pp
     [-2.14,-0.70], and 1k/2k/5k/10k/20k survival by
     +3.36/+5.78/+8.12/+8.70/+4.20pp. The user's production measurement of
     roughly 2x inference remains decisive: 192 is the stronger reference and
     potential selective teacher, not automatically the bulk actor.

     Fixed-state audits agree on continued absorption but also delimit what
     old-target fit means. From 18b96e30 to e40, legal NLL improves 0.618 to
     0.605 and danger NLL 0.670 to 0.657 while cap-1k reliability is flat;
     actions already agree 94.72%. From 192e31 to e39, legal NLL improves
     0.568 to 0.555 and danger NLL 0.602 to 0.583 while both reliability and
     full gameplay improve. Thus 18b96's remaining early hazard may be either
     representational capacity or a missing crisis-target recipe; the frozen
     R2 run cannot distinguish them.

     **Capacity verdict:** a roughly 3M-parameter/128-cost model is plainly
     sufficient to catch and exceed the previous vh3 champion. The old
     10b128 stall was substantially architecture/recipe dependent, not proof
     that the client budget was too small. The evidence does not yet show that
     18b96 can drive catastrophe hazard toward zero: about 10.6% of games still
     fail before 1,000 turns, versus 7.3% for 192e39. The smallest credible
     actor is now 18b96; 192 is the positive capacity control. Test 18b96 with
     clean on-policy crisis corrections before adding an intermediate model or
     declaring its floor plateau intrinsic.

     The clean vh3 exploit stream completed all 3,000 capped games. Started
     the separately declared 750-game explore-200/clean-label-400 stream under
     caffeinate after the evaluation sweep; it remains lineage-pure and does
     not mix 18b96/192/big-model games into the vh3 pilot. Crisis generation
     follows only after explore completes.

223. **The canonical exploit bank passes its generation audit and shows a
     significant search-policy reliability gain; exact actor-root edits are
     now preserved instead of being reconstructed at another batch shape.**
     (2026-08-10)

     The completed vh3 exploit stream contains 3,000/3,000 independent games
     and 2,928,729 states. Streaming validation found 100% clean-label and
     full-search-record coverage, no malformed targets, all 30 root candidates
     on essentially every row, and broad turn coverage: 282,760 states remain
     in turns 900--999. The 400-simulation search policy capped 2,816/3,000
     games at 1,000 turns (93.87%). Against the independent 5,000-game greedy
     vh3 distribution's 4,494/5,000 turn-1k survival (89.88%), the difference
     is +3.99pp with independent Bernoulli-bootstrap 95% interval
     [+2.79,+5.17]. This does not pair numeric seeds or claim every recorded
     edit is causal; it establishes that the declared search policy is a
     stronger aggregate actor, consistent with the earlier fixed-state
     common-random-number crisis result.

     Search edits are sparse but no longer tiny. Because native `LegalPriors`
     selects the top 30 from the complete legal support before PUCT, the
     maximum recorded clean prior is the actor's exact legal greedy action at
     that root. The clean visit winner differs on 118,031/2,928,729 rows
     (4.03%). The rate is 4.02% in capped-game rows, 4.33% across failed-game
     rows, and 6.98% in the final 20 turns of failed games. Thus failure tails
     are enriched, but contain only 257 such edits in this broad exploit bank;
     dedicated crisis mining remains necessary. Only 6.06% of visit winners
     are also maximum-Q among visited candidates (16.14% in failed tails), so
     raw root Q argmax is not promoted as a target without calibration.

     The corpus previously retained `cand_prior` but omitted its exact argmax,
     then `add_fulllegal_mask` recomputed the base action with a different MPS
     batch shape. Added `base_move` to schema 4 tensors, stored both recorded-
     root and deployment-recomputed disagreement masks in the sidecar, and
     made training/checkpoint audits select `recorded_disagree` explicitly.
     The original recomputation remains an inference-parity diagnostic. The
     focused data/trainer suite passes 47 tests (4 device-specific skips). A
     restart-safe continuation now waits for explore completion, audits it
     alone, runs and validates a separate eight-probe crisis canary, and only
     on success starts full crisis generation, the full audit, corpus build,
     FP16 annotation, and target audit; it does not start training
     automatically. Corrected the manifest's leaf-value provenance as well:
     this iteration uses the historically successful 27-feature geometry
     evaluator, including its naturally low terminal-board value; the neural
     terminal-survival-zero path is not active, and the teacher recipe was not
     changed after exploit generation.

224. **The explore stream and an independent crisis canary pass; native MPS
     requests now fail closed, and the full crisis mine is underway.**
     (2026-08-11)

     The 750-game explore-200/clean-label-400 stream completed with 724,807
     valid full-record rows, for 3,653,536 accepted fresh rows together with
     exploit. It capped 697/750 games at 1,000 turns (92.93%), covered every
     100-turn band broadly, and had zero malformed targets. The clean teacher
     differed from the actor-root prior argmax on 28,007 rows (3.86%); failed
     final-20 rows were enriched to 5.94%, while remaining a small stratum.

     The first eight-probe crisis canary exposed an operational safety bug:
     LibTorch reported MPS unavailable in the restricted process and all five
     native inference/search tools silently converted an explicit `mps`
     request to CPU. The generated game records correctly said fp32, so the
     manifest audit rejected every file before corpus assembly. Preserved
     those non-training artifacts under
     `crisis_cpu_fp32_rejected_20260811`; none entered the pilot. Explicit MPS
     requests now terminate before model loading or output creation when MPS
     is unavailable, unsupported device names are rejected, and run-config
     precision is derived from the actual execution device rather than only
     the fp32 flag.

     The same PyTorch build sees one available MPS device outside sandbox
     isolation. A fresh canary therefore ran under caffeinate as verified MPS
     fp16: eight probes produced seven deaths, one 10k-turn capped probe, 14
     recovery/prevention games, and 5,662 full clean rows in 468 seconds. Its
     strict audit found zero errors or invalid targets. This opened the gate
     for the resumable 500-probe production mine. Probe phase produced 402
     deaths and 98 capped probes, hence 804 deep replay tasks. A separate
     fail-fast continuation waits for the completion marker, then performs the
     full generation audit, resumable corpus build, corpus audit, MPS-fp16
     base-policy annotation, and 50k-row target audit. It intentionally does
     not start training.

225. **The vh3 bounded-target branch is closed; 18b96e40 becomes the first
     fully on-policy compact flywheel actor, and its iteration-1 pilot passes
     the end-to-end gate.** (2026-08-14)

     Exact dense-target diagnostics separated optimization failure from target
     failure. Low-dose hard vh3 edits did not approach their bounded optimum;
     16x exposure repaired much of that absorption problem but gameplay was a
     wash. The strongest mixed arm (`alpha=0.5`, `eta=0.1`, 8x whole-row edit
     exposure) reduced exact all/edit/high-confidence/crisis target residuals
     by roughly 8--10% and retained 98.61% of anchor actions. Its independent
     5k cap-1k result still failed promotion: cap-rate delta -0.70pp
     [-1.90,+0.50], mean -5 [-16,+7], and P10 -79 [-169,+42]. Thus a student
     can measurably fit this intended vh3 target without improving the game;
     more optimizer tuning on the same stale labels is not the next step.

     The compact actor handoff uses `scratch18b96_lr3e3` epoch 40. Against vh3
     its independent 5k cap-1k distribution is unresolved (cap -0.58pp
     [-1.78,+0.60], mean -7 [-19,+4]), while full play is significantly
     stronger: mean +907 [+358,+1,456], median +766 [+275,+1,246], P90 +2,669
     [+522,+4,423], and `>10000` +3.06pp [+1.12,+4.98]. This preserves the
     near-128 inference cost while moving the actor to the strongest available
     small-model basin.

     The disjoint e40 canary generated four exploit games, four explore games,
     and eight crisis probes. Strict audit accepted all 20 files and 11,219
     rows with zero invalid actions and 100% clean/full root records. Six
     probes died and two reached 10,000 turns. Search recovery survived 500
     turns in only 1/6 deaths, but prevention survived in 5/6; failed tails had
     about 6.45% actor/search corrections. This is the first direct on-policy
     reason to prioritize a correction class for this actor.

     Generated a separate actor-native preservation bank from 5,000 fresh
     greedy games capped at 1,000 turns, under MPS FP16 and caffeinate. The
     distribution capped 4,464/5,000 (89.28%) with mean score 1,934. Sparse
     broad plus dense tail recording produced 1,798,692 states; the built
     tensor has exactly 5,000 source-game groups and 4.99% validation rows.
     Extended the corpus provenance gate from a hard-coded `small128` string to
     an explicit `small_policy` family plus an enforced generator prefix and
     base-checkpoint SHA-256. Foreign-model sources remain rejected. Native
     `final_score/final_turns/died` anchor metadata now maps to the same tensor
     diagnostics as MCTS `score/turns/capped`; the focused builder suite passes
     7/7 tests.

     Full target generation is now active from the immutable
     `18b96e40_iteration_1_pilot.json`: 3,000 exploit games, 750
     behavior/clean-label-separated explore games, and 500 crisis probes, all
     on disjoint non-reserved seeds. The exploit stream started at about 20.2k
     leaf evaluations/s with effective batch 55 and writes one atomic game per
     completion, so it is safe to interrupt and rerun. No vh3, 192/256-channel,
     or Pillar game labels enter this lineage. After audit/build/annotation,
     the two live hypotheses are a crisis-focused bounded update and aggregate
     rolling-search distillation; only independent 5k distributions can
     promote either branch.

     The 3,000-game exploit tranche subsequently completed and passed strict
     standalone audit: 2,945,600 clean/full-record states, 3,000 unique seeds,
     and zero errors or invalid actions. Search capped 2,851/3,000 games
     (95.03%), versus 4,464/5,000 (89.28%) for the independent greedy anchor
     distribution: +5.75pp with independent Bernoulli-bootstrap 95% interval
     [+4.59,+6.91]. Exact actor-root edits were 2.32% overall and 6.04% in
     failed final-20 states. The rolling search operator therefore works
     aggregate-wise on the new actor even though broad single-edit utility is
     not assumed. The sequential launcher handed off to the explore stream as
     intended; crisis remains queued behind it on the same single MPS job.

226. **The first fully on-policy 18b96 student improves the development floor;
     the complete 3.9M-state corpus is now being replicated at a matched update
     dose.** (2026-08-16)

     Iteration-1 generation finished cleanly. Strict combined audit accepted
     all 4,528 files and 3,903,888 full-root rows: 2,945,600 exploit, 723,704
     behavior/teacher-separated explore, and 234,584 crisis. There were no
     invalid targets, incomplete records, mixed run settings, or provenance
     failures. The resumable tensor build retained a 5.03% group-held-out
     split. Exact MPS-fp16 base annotation found 94,491 recorded actor-root
     edits (2.42%) and agreed with those recorded base actions on 99.88% of
     rows. The separate actor-native anchor tensor contains 1,798,692 states
     from exactly 5,000 games.

     Search itself supplies a real improvement operator. The 3,000 exploit
     games capped at 95.03%, versus 89.28% for an independent 5,000-game
     greedy actor distribution: +5.75pp [+4.59,+6.91]. The 500 crisis probes
     produced 389 deaths and 111 10k caps. Their distinct-state replays
     survived 500 turns in 335/389 prevention trajectories versus 104/389
     recovery trajectories; this prioritizes the prevention class without
     misreporting it as a paired single-action estimate.

     Four 200k-target/50k-anchor smokes separated objective mechanics. The
     bounded crisis update remained safe but adopted only about 5% of exact
     edits where its analytic target implied roughly 53%, so it was underfit.
     Removing color permutation increased drift for both bounded and aggregate
     objectives. Aggregate hard-teacher epoch 1 with color augmentation was
     the safest meaningful candidate: target/anchor retention 96.32%/96.25%,
     target legal KL 0.0259, and 32.3% true-edit adoption.

     On the standard 5,000-game cap-1k development distribution, this student
     improved mean +14 [+2,+25], `<1000` -0.86pp [-1.60,-0.12], mean turns
     +7 [+1,+12], and cap rate +1.22pp [+0.04,+2.40]. A matching 100k-turn
     evaluation retained the floor gain without a detected long-tail cost:
     P10 +209 [+7,+384], mean +297 [-260,+854], median +244 [-274,+784], and
     survival to 2,000 turns +1.70pp [+0.08,+3.32]. A previously unused
     non-reserved cap-1k bank was neutral but directionally consistent: mean
     +7 [-4,+18], `<1000` -0.58pp [-1.30,+0.12], and no significant regression.
     All comparisons independently resample the two distributions; matching
     numeric seeds are never treated as paired games.

     This is evidence against an already-reached 18b96 capacity ceiling, but
     the selected 200k smoke is not the iteration-2 actor. A one-pass
     replication is active on all 3,903,888 targets plus a deterministic 1M
     anchor sample. Batch 8,192 and LR 2.14e-6 produce 575 steps, matching the
     integrated update dose of the successful 123-step, LR-1e-5 smoke while
     greatly increasing state coverage. It is MPS-fp16, caffeinated,
     color-augmented, frozen-BN, deployment-legal, and resumable every 25
     steps. The reserved 1,100,000--1,119,999 final bank remains untouched.

227. **The apparent iteration-1 student win does not replicate; causal mining
     isolates crisis edits, whose current corpus is too correlated to test
     capacity.** (2026-08-16)

     Entry 226 recorded the then-current development result and active
     replication. That replication is now complete, and it retracts the
     promotion interpretation. The full 3.9M-target/1M-anchor dose-matched run,
     a lower-dose intermediate checkpoint, and an exact smoke replica on a
     second deterministic data sample were all neutral on previously unused
     5,000-game distributions. An edit-only bounded arm independently repeated
     the pattern: mean +15 [+3,+26] on the development bank, then mean -3
     [-14,+8] and cap -0.74pp [-1.88,+0.42] on a fresh bank. These are
     checkpoint-selection effects, not reproducible policy improvement. No
     student is promoted and the final seed bank remains untouched.

     Common-random-number fixed-state judging then separated target classes.
     Broad exploit edits were neutral, and an explore pilot did not repeat on
     fresh rollout RNG. A fresh 64-repetition rejudge of 353 crisis edits did
     repeat: death-within-200 uplift +1.24pp [+0.68,+1.84] and teacher-turn
     delta +2.18 [+1.25,+3.21]. Recovery edits were strongest locally at
     +2.02pp [+1.06,+3.04] and +3.81 turns [+2.07,+5.74]. This is valid paired
     evidence because the state, first-action alternatives, and rollout random
     numbers are held fixed; whole-game evals remain independent-distribution
     comparisons even when numeric seeds match.

     The exact crisis mask contains 9,603 edits, only 9,094 in the training
     split. A stratified low-LR 18b96 update includes every training edit and
     reduces its exact target residual from 0.01264 to 0.01119, but held-out
     residual is unchanged/slightly worse (0.01169 to 0.01175). High LR
     oversteps even in-sample. Therefore the immediate result is neither
     "18b96 has enough capacity" nor "18b96 lacks capacity": it can absorb
     some seen corrections, while 500 source probes provide too little
     independent crisis diversity to generalize. The next experiment expands
     same-actor crisis mining on disjoint seeds to tens of thousands of edits,
     repeats the exact train/held-out audit, and only then uses a larger model
     as a positive-control student on the identical target if the gap remains.

228. **Fivefold crisis-edit expansion replicates causal search advantage, but
     neither 18b96 nor a 192ch positive control generalizes sparse one-hot
     corrections.** (2026-08-17)

     Ran 2,000 new scratch18b96-e40 probes on disjoint seeds
     2,210,000--2,211,999. The caffeinated native MPS-fp16 job finished 1,478
     deaths and 2,956 recovery/prevention replays in 16.46 hours at about
     20.5k leaf evaluations/s. Strict audit accepted 905,645 clean full-record
     states and rejected nothing. The expansion contains 36,717 actor/search
     edits (28,282 prevention, 8,435 recovery); combined with the pilot this is
     1,140,229 states, 46,320 edits, and 1,867 independent death seeds. The
     group split keeps recovery/prevention siblings together and reserves
     57,161 rows from 95 source seeds with no leakage.

     The new labels independently pass the causal gate. On 600 held-out edits
     with 64 common-random-number repetitions per arm and horizon 200, the
     search action improves death rate by +1.31pp [+0.82,+1.81] and lifetime by
     +2.26 turns [+1.46,+3.12]. Prevention is +0.84pp [+0.35,+1.36]; recovery
     is +1.77pp [+0.94,+2.64]. This nearly reproduces the prior 353-state
     fresh-RNG result. The class is real, but individual labels are noisy:
     89.3% are ties under the predeclared 8pp verdict threshold, 8.5% genuine,
     and 2.2% phantom.

     A full 18b96 bounded pass used every 1.14M crisis state and every 1.80M
     anchor, `eta=.05`, hard teacher, 8x edit exposure, LR 1e-5, frozen BN,
     legal-support loss, and no D4. The endpoint is functionally safe (about
     99.2--99.3% retention and legal KL 3.4--3.6e-4). It reduces exact
     training-edit target KL only 0.00399 -> 0.00392, while every held-out
     checkpoint is worse than base: best step 500 is 0.00420 and endpoint
     0.00430 versus base 0.00381. It was correctly stopped before gameplay.

     The promised width control used scratch192-e39 as its own frozen base on
     the identical immutable edit mask. Because 192 and 18b96 differ on 15.6%
     of these states, this is an absorption control, not a direct lineage
     comparison. It fits training edits more readily (0.00921 -> 0.00875) but
     likewise worsens every held-out dose (best 0.00869, endpoint 0.00915,
     base 0.00841). Width buys plasticity but does not solve generalization;
     this result rejects a simple 18b96-capacity diagnosis for the current
     target.

     NEXT: replace equal one-hot labels with causal magnitude. Judge several
     thousand edits at cheaper R16/R32, validate any confidence/recovery
     routing on a fresh fixed-state set, and train teacher-vs-base pairwise
     margins or an advantage/value head weighted by rollout uplift, while
     retaining broad frozen-base KL anchors. Do not evaluate gameplay until a
     held-out source-seed target/ranking gate passes. The final 20k evaluation
     seeds remain untouched.

229. **Course correction: the sparse-edit branch was a diagnostic, not the
     promised aggregate selfplay+crisis Colab run.** (2026-08-17)

     The crisis expansion and width control answered a narrow question: equal
     one-hot isolated edits do not generalize, even when the student is wider.
     They did not execute the other live flywheel hypothesis—coordinated
     distillation of complete rolling-search trajectories. Treating the local
     proxy gate as the main path was therefore a planning drift.

     Built the non-overlapping current-lineage aggregate tensor from all four
     intended sources: 2,945,600 exploit states, 723,704 explore states,
     234,584 pilot-crisis states, and 905,645 expansion-crisis states. The
     resulting `flywheel_18b96e40_i1_mixed_targets.pt` has exactly 4,809,533
     rows, 5,617 source-seed groups, 239,905 group-held-out validation rows,
     and zero duplicate crisis tranche. All 4,809,533 roots retain clean
     top-30 visits/priors/Qs; all labels and generators are scratch18b96-e40.
     The separate preservation tensor contributes 1,798,692 states from 5,000
     actor-native games.

     The primary notebook is now
     `train_flywheel_18b96e40_i1_colab.ipynb`. It warm-starts 18b96 epoch 40,
     pools every target and anchor within batches, uses hard clean-search CE
     plus frozen-base KL, weights exploit/explore/pilot-crisis/new-crisis
     `1/1/2/2`, keeps exact legal support, FP16 forward/FP32 loss, frozen BN,
     color augmentation, and no D4 in the first arm. Batch 4,096 for 12 epochs
     supplies about 18k updates, with exact mid-epoch resume and per-epoch
     model/functional diagnostics. Promotion still requires independent
     5,000-game distributions; matching numeric seeds are never paired game
     evidence, and the reserved final 20k seeds remain untouched.

230. **The first full mixed selfplay+crisis epoch is a clean gameplay wash;
     checkpoint granularity, not the KL guard, was wrong.** (2026-08-17)

     Colab completed 1,533 updates over all training-split mixed targets and
     anchors. The epoch-1 student retained 96.3%/96.2% of target/anchor
     actions, reached target/anchor legal KL 0.0676/0.0680, adopted 28.8% of
     audited target edits, and introduced no new deployment BN nonfinites. It
     stopped solely because the predeclared maximum legal KL was 0.05. The
     checkpoint is a valid high-dose model, not a failed job.

     Exact MPS-fp16 export passed with zero traced/eager difference. On fresh
     seeds 1,000,000--1,004,999 at a 1,000-turn cap, the untouched e40 base was
     mean 1,940, median 2,015, P10 1,938, `<1000` 3.78%, and cap rate 90.14%;
     epoch 1 was mean 1,943, median 2,017, P10 1,943, `<1000` 3.48%, and cap
     rate 90.16%. Independent 10k-bootstrap differences were mean +3
     [-9,+14], median +2 [+0,+4], P10 +5 [-59,+50], `<1000` -0.30pp
     [-1.04,+0.44], and cap +0.02pp [-1.16,+1.20]. This is a wash, not a
     promotion or regression.

     The notebook's mistake was `SAVE_STEPS=0`: the earliest dose after base
     was a complete 1,533-step epoch. Resumed epochs now save model-only
     checkpoints every 250 global steps. Keep the existing exact-resume state;
     use intermediate dose screens before deciding whether another endpoint
     deserves a 5k distribution.

231. **Label-level teacher measurement + corpus r3_tail: the student is saturated
     on teacher-trajectory imitation; the 10x teacher (vh2 + pv_vh2 value head,
     corrected PUCT) relabels the student's OWN death windows with 3-4x the
     judge-confirmed corrections of any prior teacher. From-scratch retrain
     queued.** (2026-09-15/16)

     New tooling: `inference_cpp/src/mcts_relabel.cc` (re-search STORED states
     under any search controls -> CSV; 0.022 s/state @400 sims, 14 thr, MPS),
     `--virtual-mean`/`--q-range-floor` added to `mcts_selfplay`, scripts
     `export_relabel_states.py`, `relabel_compare.py`, `relabel_judge_summary.py`,
     `audit_student_match.py`, `export_disagreement_judge.py`,
     `export_greedy_tail_states.py`, `export_relabel_flips_judge.py`,
     `build_r3_tail_corpus.py`, `compare_uncapped_csv.py`, `selfplay_dir_stats.py`.

     Facts read from data/code (not from earlier entries):
     - r2_bulk composition (`r2_bulk_strata.npz`): 56.3% crisis-replay rows,
       37.8% capped-1k selfplay, 5.9% of the 80 uncapped vh2+MCTS games
       (`rows_of_game` applied its 10k/game cap; 84% of those moves unused).
     - `data/selfplay_iter5` (vh2 + pv_vh2, q2.0, 400 sims, UNCAPPED, n=80):
       median score 99,144 / P25 37,786 / P75 158,790 / max 486,992 vs vh2
       greedy P50 9,475 -> the search teacher is ~10x the greedy student.
     - 18b96e40 argmax-match vs its own r2_bulk targets (200k rows, MPS):
       99.6% (top-share>=.8), 98.5% (.6-.8), 93-95% (.4-.6), 85.8% (.3-.4),
       68.2% (<.3, = 48% of rows). Disagreements on the 10x teacher's own
       trajectories judged (student continuation, R64 H200): 4 genuine / 2
       phantom of 700 flat rows = noise. => saturated; no LR/batch/epoch fix.
     - Teachers relabelling 5,980 stored 18b96e40 roots (2,980 final-20 of
       dying cap-1k games + 3,000 broad), judged R64/H200 student continuation,
       excess = genuine - phantom per 1k tail roots: own FV search VL-1 c2.5
       q1 @400 = 3.7; virtual-mean = 5.4; vm c1.5 q1 = 14.1; vm c1.5 q2 =
       18.5; @800 sims VL-1 = 8.1; pillar3k greedy = 22.5 (62 phantoms of
       2,033 flips); **vh2+pv_vh2 vm c1.5 q2 @400 = 55.0** (201 genuine / 31
       phantom of 2,141 flips). Broad states: noise for every teacher at H200.
     - Throughput: greedy eval 30 games/s; MCTS 19.6k leaf evals/s (18b96) /
       11.4k (pillar3k 256ch) -> 5k MCTS games @400 sims cap-1k ~= 27 h.

     Corpus r3_tail (`data/r3_tail.pt`, 12,712,930 rows, gz 741,873,349 B):
       eval --model scratch18b96_lr3e3_ckpts_epoch_40_ts.pt --seed-start 2600000
         --seed-end 2601000 --record-dir data/greedy_18b96e40_uncapped_2600k
         --record-every 1 --record-tail 200   (uncapped greedy; mean 15,164 /
         P50 10,683 / P10 1,961 / <1000 3.6%; 6 min)
       export_greedy_tail_states.py --tail 200 -> 199,935 states
       mcts_relabel --model vh2_policy_ts.pt --value-module pv_vh2_ts.pt
         --sims 400 --c-puct 1.5 --q-weight 2.0 --virtual-mean --keep 15
         (70 min) -> 64,483 flips (32.3%; 40.9% in the last 20 turns)
       judge of 1,200 stratified flips (R64 H200): excess 13.7% (<20 turns to
         death), 18.0% (20-60), 6.0% (60-120), 0.7% (120-200).
       build_r3_tail_corpus.py: r2_bulk (mask 0) + relabelled rows, pack()
         semantics (top-5 by visits, teacher argmax forced), mask = (1 flip /
         .25 agree) x (1.0 <60, 0.5 <120, 0.25 <200 turns-to-end); mask sum 54,330.
     Run: `train_scratch18b96_r3tail_colab.ipynb` = scratch18b96_lr3e3 recipe
     exactly (18b96, bs32768, lr3e-3, 40 ep, T1, dw0, blend0 = hard CE) with
     --disagree-gamma 20 (new rows ~8.5% of gradient mass). From scratch, not
     warm-start: every corrections-only warm start of the small model collapsed.
     GATE (policy-only, uncapped): greedy 1k on seeds 2,600,000-2,600,999 vs
     18b96e40's 15,164 / 10,683 on the same bank, +20% mean AND P50 with
     independent-bootstrap CI clear; 5k only to confirm a pass.

232. **r3_tail (hard CE, gamma 20) REGRESSES the from-scratch curve by a constant
     ~30% at ep4 and ep6; diagnosis = 80% of the weighted flip rows are ties
     forced into single argmax labels. Fix = set-valued loss on relabelled rows;
     run scratch18b96_r3tail_set queued.** (2026-09-16)

     Gate bank 2,600,000-2,600,999, uncapped greedy (EVAL.md):
       baseline ep4 2,368 / P50 1,816 / <1k 26.1%   r3tail ep4 1,684 / 1,254 / 39.2%  (-28.9% [-35,-22])
       baseline ep6 4,058 / P50 3,000 / <1k 15.7%   r3tail ep6 2,818 / 2,104 / 23.0%  (-30.6% [-37,-24])
     Absorption audit (scripts/audit_match_by_mask.py, ep4): bulk rows 71.5% vs
     72.8% (baseline); relabelled flips <60t: 34.7% vs 23.1% (chance) -> the
     weighting installs the labels at ~no bulk cost, and play still drops.
     Judge on 1,200 stratified flips (R64 H200, student continuation): excess
     13.7% (<20 turns to death) / 18.0% (20-60) / 6.0% (60-120) / 0.7% (120-200),
     i.e. ~80% of weighted flips are ties. Hard CE at 21x weight forces a
     coin-flip argmax on each tie against the student's consistent policy.
     Run stopped by owner after ep6.

     Fix (train_path_b.py, tests added): `--set-loss-on-mask --set-tau T`: rows
     with disagree_mask>0 use -log P(set), set = teacher candidates with target
     mass >= T*max; other rows unchanged. On r3_tail's relabelled rows at T=0.5:
     set size mean 2.11, 40% singletons; true contradictions (student move
     outside set) 12.3% of rows vs 32.3% hard-CE flips. M5 smoke (60k rows,
     2 epochs) passes end to end. Code tarball colorlines_pillar3d_v6.tar.gz
     (same file set as v5 + patched train_path_b). Notebook
     `train_scratch18b96_r3tail_set_colab.ipynb` = r3tail recipe + set loss.
     Same gate: +20% mean and P50 vs 15,164 / 10,683 on the 2.6M bank.

233. **Smoking gun: per-move judging is blind to what decides game length; the
     search's advantage is non-decomposable per move; hard-argmax targets on
     near-ties are noise and concentration amplifies it. Self-search with the
     linear leaf is only ~1.7x; survival head for 18b96 being trained.** (2026-09-17)

     - r3tail_set (set-valued loss, gamma 20) ep3: 1,440 / P50 1,075 vs baseline
       ep3 1,936 / 1,483 (-25.6%) on the gate bank (EVAL.md). Hazard by turn
       bucket (scripts/hazard_curve.py): 1.4-1.6x in EVERY bucket for all three
       r3tail pairs -> global damage.
     - set-ep3 vs base-ep3 moves on 2,208 held-out disagreements (pilot roots):
       H200 base-cont 99 vs 106; set-cont 107 vs 122; H1500 R32 n=300: 0.915 vs
       0.909. Single moves of the 25%-worse model are indistinguishable.
     - 10x-teacher flat-row moves vs student moves on the teacher's own games,
       H1500 R32 n=300, student continuation: died 0.178 vs 0.178 [+-0.01],
       mean turns 1,370 vs 1,372 -> single teacher moves carry ZERO value.
     - Arms (1M r2_bulk rows from scratch, 4ep bs4096 lr1e-3, 500-game evals):
       A control 318 / P50 282; C +16k RANDOM bulk rows at r3_tail weights
       287 / 267 (-10%); B +16k relabelled rows 256 / 239 (-19.5%).
       Decisiveness equal across arms (0.62/0.62/0.61) -> flattening is NOT
       the mechanism; concentration on hard-argmax noise (-10%) + foreign-policy
       content (-10%) is.
     - Self-search of 18b96e40 (FV leaf, virtual mean, c1.5, q1, 400 sims,
       cap 3000, 60 games, seeds 2,800,000+): 80% reach 3,000 turns vs greedy
       69% (P10 turns 2,306 vs 974) -> hazard ratio ~0.6, ~1.7x teacher. The
       10x (vh2) came with the NN survival head at q=2.
     - Built value_targets_18b96e40.pt (2,081,728 states from the 1,000 recorded
       uncapped greedy games, every 4th state + full final 300 turns, no
       censoring). Training value_head_18b96e40 (frozen backbone, 5 ep) ->
       pv_18b96e40_ts.pt -> 60 self-search games with the head at q=2, cap 3k.
     Scripts: hazard_curve.py, style_density.py, policy_decisiveness.py,
     export_model_pair_judge.py, build_gun_arms.py, build_value_targets_from_records.py.

234. **Self-search WITH a survival head is a strong self-teacher; flywheel-2
     corpus generation launched; soft-target pilot.** (2026-09-17)

     value_head_18b96e40 (3,268 params, frozen 18b96e40 backbone, 5 ep on
     value_targets_18b96e40.pt = 2.08M states of its own uncapped games): inner-val
     BCE 0.116 (per-H 0.022/0.051/0.121/0.271). Fused pv_18b96e40_ts.pt.
     Self-search, same 60 seeds (2,800,000+), cap 3,000, virtual mean, c1.5, 400 sims:
       greedy                : 69% reach 3,000 turns, P10 974 turns
       FV leaf, q=1          : 78% (score P10 4,653)
       NN head, q=2          : 92% (score P10 6,001)  -> hazard ratio ~0.22 vs greedy
     Visit distributions (scripts/visit_stats.py), corrected search: top-share
     mean 0.69 (old VL/c2.5 pilot 0.49), <0.3 on 1.9% of moves (old 8.6%),
     visit argmax != prior argmax 8% (old 2.2%), 30 cands saved.
     LAUNCHED: fw2_18b96e40_pv_cap3k = 300 games, seeds 2,810,000-2,810,299,
     same search, full record (~11 h M5). Corpus recipe (build_soft_visit_corpus.py):
     EVERY move, top-5 renormalized visits as SOFT targets, no argmax forcing, no
     per-game cap. Training: warm-start e40, soft CE, gentle LR, few epochs, on M5;
     gate greedy 1k on 2,600,000-2,600,999 (+20%) and the hazard curve.
     Pilot on the 60 head games (fw2_pilot60_soft.pt) first.

235. **Flywheel-2 recipe pilots on the 60 head-search games (175k rows, warm-start
     e40): soft visit targets regress 7-12%; sharpened (T0.7 dw3) is a wash.**
     (2026-09-17)  Gate bank 2,600,000-2,600,999 (baseline e40 15,164 / P50 10,683 /
     <1k 3.6%), all in EVAL.md:
       soft T1.0 lr1e-5 (scheduler quirk: --warmup-epochs 0 keeps LinearLR at 0.1x)
         ep1/2: 13,886 / 13,746 mean; soft T0.7: 14,103 / 13,386
       soft T1.0 lr1e-4 warmup1 bs2048 ep2/3/4: 13,648 / 14,103 / 13,496 (P50 9.6-9.9k)
       soft T0.7 dw3 ep2/3: 14,687 / P50 11,038 ; 14,144 / 10,338  (wash)
     Targets' top-share 0.69 < policy top-1 0.80 -> soft CE de-peaks; sharpening
     removes the regression but adds nothing. Visit argmax != prior on 8% of moves.
     Hard-CE (chosen-move) arm queued. Corpus generation fw2 (300 games) continues.

236. **Why own-search distillation washes: two thirds of the search's overrides
     are single-search noise; the distilled student churns more good moves than
     it gains.** (2026-09-22, Opus 5.5)
     Held-out 60 head-search games (58,243 states, every 3rd move):
       search overrides base prior on 8.1% (mostly occupancy <50).
       fw2_300_hardCE ep2/ep3 adopt 22.8% of overrides (e40: 4.8%) but keep only
       96.1-96.4% of agree rows (e40: 99.3%): agreement with search 91.6 -> 90.2%.
     Search self-consistency (new `--seed-salt` in MctsConfig/mcts_relabel;
     default 0 = unchanged; mcts_controls_test PASS): 4,697 override + 2,000
     agree states re-searched x3 independently @400 sims: override reproduced
     36.9%, prior picked 53.8%; agree rows 94.5%; pairwise agreement 57.7% vs
     92.3%; per-state reproductions 0/1/2/3 = 1,684/1,406/1,023/584; normalized
     Q(choice)-Q(prior) P50 +0.014. Robust (>=2/3) 34%, noise (0/3) 36%.
     hardCE ep3 adopts 33% robust / 12% noise / keeps 96% agree.
     Recorded stats cannot filter (scripts/override_predictors.py): best
     precision ~55%; recorded Q margin anti-predictive.
     RUNNING: 3x re-search of all 71k overrides in the 300-game corpus ->
     fw2_300_denoise.pt (override kept iff >=2/3 reproduce, else target=prior)
     and fw2_300_self.pt (all targets = prior; fine-tuning-tax control) -> same
     hard-CE recipe -> held-out absorption + gate evals (scripts/denoise_chain.sh
     in session scratchpad; builder scripts/build_denoised_corpus.py).

237. **SYMMETRY: the committed training code has broken D4 augmentation (the whole
     256ch line and vh1-vh3 trained on it); averaging the small net over the 8 exact
     board symmetries at inference = +50% with NO training.** (2026-09-22, Opus 5.5)

     D4 ground truth (scripts/verify_d4_transform.py: dataset LUT transform vs the
     observation rebuilt from the actually transformed board+preview, 300 r2_bulk
     states): committed HEAD (5b714b9) and the June 24 code (a7ce846, pillar3k era)
     rotate the 9x9 planes but NOT the line-direction channels 13-16 -> wrong in 6 of
     8 views on 95-100% of states. Working tree and both Aug code tarballs (used by
     18b96 and every scratch run since Aug 9): exact on all 8 views, obs and policy.
     The fix is still UNCOMMITTED.
     Symmetry consistency (scripts/d4_consistency.py, 5,980 pilot states): all 8
     views same argmax: 18b96e40 60.5%, pillar3k ep22 41.5%, pillar3b ep20 38.3%;
     TTA-8 changes the move on 13.2 / 20.0 / 23.2%.
     BN FP16 overflow: 18b96e40 backbone_bn ch 9/48 running_var 100,447/70,628
     (> 65,504 -> +inf in FP16); pillar3k/3b max 16k-18k (no overflow). FP32 eval of
     e40 = 14,732 vs FP16 15,164 (-2.9% [-10.5,+5.2], wash). export_ts.py now applies
     an exact reparameterization (fp16_safe_batchnorm; 486 ch rescaled; FP32 logits
     max diff 0.0015, argmax identical); e40 re-exported = 14,296.
     eval --tta 8 (new): greedy on the mean of the 8 views' logits mapped back to the
     original frame (observation built natively from each transformed board).
     Gate bank 2,600,000-2,600,999, uncapped, same export:
       e40 plain  14,296 / P50  9,944 / P10 1,806 / <1k 4.4%
       e40 TTA-8  21,436 / P50 14,776 / P10 2,449 / <1k 3.3%   +49.9% [+39.7,+61.4]
     Hazard 3000-5000 bucket 26.8% -> 15.4%. On held-out search states TTA-8 adopts
     34% of the search's robust overrides (single view 8%).
     Also measured today: interpolation of fw2_300_hardCE into e40 (a=.2/.3/.5):
     14,118 / 14,213 / 14,649 = wash. Trust-region imitation chain paused.
     NEXT (running): symmetric self-distillation — record 400 TTA-8 games (seeds
     2,900,000+), target = e40's own 8-view-averaged policy, correct D4 aug, frozen BN,
     warm e40 -> single-pass model; audit symmetry + gate.

238. **Anomaly hunt (after the D4 bug): pipeline verified against ground truth;
     one structural defect found — the policy head's "which ball" axis is 81
     absolute-position readouts.** (2026-09-22, Opus 5.5)
     Verified CLEAN: training obs builder (_build_obs_core) == reference
     build_observation on 20k r2_bulk states (all 18 channels); C++ Game::BuildObs +
     LegalMask (new tool build/obs_dump) == reference on 30k states incl. 10k student
     near-death boards (occupancy up to 80); stored r2_bulk target argmax legal on
     20k/20k; BN eval-mode vs train-mode argmax agreement 96.6% (target match 78.2 vs
     78.0); game rules (path reachability, 5+ clears, spawn clears, displaced spawns)
     match CL98; every input group is used (zeroing changes argmax 32-88%); r2_bulk has
     0 duplicate positions even up to symmetry (scripts/label_symmetry_audit.py, hash
     unit-tested); preview-ball ORDER is harmless (99.4% consistent); C++ TTA ==
     Python TTA 99.9%. All top-k>1 call sites of _legal_priors_jit (ASCENDING order)
     audited: handled correctly (my build_tta_corpus stat line was wrong, targets fine).
     Color relabeling consistency: 90.8% per relabel, 79.1% all-8 (vs D4 80.9 / 60.5).
     STRUCTURAL DEFECT: the policy head emits, at each destination cell, 81 logits
     indexed by ABSOLUTE source position (1x1 conv) -> the "which ball" axis shares no
     weights across positions or rotations. When rotated views disagree, it is a
     different ball to the same destination 36.5% of the time vs same ball to a
     different destination 17.0% (pillar3k: 46.7% vs 13.1%) (scripts/d4_src_dst.py).
     Added PolicyNet(head='pair'): logit(s,d) = u(s).w(d)/sqrt(k) + a(s) + b(d) from
     shared 1x1 convs (exactly permutation-equivariant; tests/test_model_pair_head.py);
     train_path_b --policy-head pair --pair-dim; loaders infer the head from keys; the
     C++ engine needs no change (same 6561 output).
     Small A/B queued (1M rows, identical to smoking-gun arm A = 318 / P50 282).

239. **Every warm-start fine-tune of e40 loses 10-18%, including one on its OWN moves;
     the +50% is an ensemble effect (canonical single-pass = baseline). From-scratch
     "born-again" runs on 8-view-averaged labels prepared for Colab.** (2026-09-23, Opus 5.5)
     Gate bank 2,600,000-2,600,999 (e40 re-export baseline 14,296 / P50 9,944), all in EVAL.md:
       canonical inference (eval --canon, D4 x color canonical form)      14,018  (-1.9%)
       sym1 soft 8-view targets, 360 TTA games (1.0M rows) ep2/ep4       12,509 / 13,009
       symhard (hard targets, same corpus) ep1/ep2/ep4                   13,575 / 13,537 / 12,339
       symcanon (canonical corpus + eval --canon) ep1/ep2/ep4            13,949 / 13,319 / 12,642
       r2tta_tta (ALL 12.5M r2_bulk rows relabeled, hard) ep1/ep2        12,609 / 12,329
       r2tta_hybrid (orig label if top-share>=0.4) ep1/ep2               12,281 / 12,793
       r2self (targets = e40's OWN view-0 move, D4 aug) ep1/ep2          12,202 / 11,692
       checkpoint average e33-e40 / + BN re-estimate                     14,762 / 13,834
     Students adopt ~24% of the 8-view overrides but change ~2.3% of already-agreed
     moves (scripts/tta_adoption.py); symmetry barely improves (all-8-agree 68->71%).
     r2self uses view-0 targets under D4 aug (non-equivariant), so clean controls without
     augmentation are running (C1 bs4096 lr5e-5, C2 bs32768 lr3e-5).
     Tools: alphatrain/canonical.py + inference_cpp/src/canonical.h (C++ == Python on 30k
     states), eval --canon / --tta 8, relabel_tta_tensor.py (GPU, 12.5M rows in 46 min, ==
     native TTA 1000/1000), build_tta_r2_corpus.py, build_canonical_corpus.py,
     average_checkpoints.py. Trainer: warmup-0 LR bug FIXED (+test); --legal-mask-loss
     (+test); PolicyNet head 'pair2' (geometry-gated, exactly D4-equivariant; +test).
     Colab package: colorlines_pillar3d_v7.tar.gz (1,267,609 B), r2_bulk_tta8_tta.pt.gz
     (650,115,072 B), notebooks train_scratch18b96_tta_colab.ipynb (abs head, born-again)
     and train_scratch18b96_pair2_tta_colab.ipynb (pair2 head + legal-mask loss).

240. **Fine-tuning damage is LR-driven drift, not the targets: e40 on its OWN moves (no aug, frozen
     BN, hard CE, 1 epoch of r2_bulk) loses 9.6% at bs4096/lr5e-5, 5.0% at bs16384/lr2e-5, 0% at
     bs16384/lr5e-6.** (2026-09-23)
       self_noD4_C1  bs4096  lr5e-5   12,921 / P50 9,093
       self_noD4_C2b bs16384 lr2e-5   13,580 / P50 9,745
       self_noD4_C3  bs16384 lr5e-6   14,350 / P50 10,378   (baseline 14,296 / 9,944)
     (bs32768 run died, likely MPS memory; bs16384 epochs took 3.0-3.4 h on the M5 vs 37 min at
     bs4096 — large batches are ~5x slower per sample locally.) e40 sits in a sharp optimum; any
     fine-tune above ~5e-6 perturbs decisions more than it teaches. Running: gentle fine-tune toward
     the 8-view-averaged moves, bs4096 lr2.5e-6 (same Adam noise per epoch as C3, 2x the directed
     movement), 3 epochs.

241. **ARCHITECTURE WIN: from-scratch 18b96 with the geometry-aware PAIR2 policy head +
     legal-mask loss plays +44% over e40 in a single pass at epoch 27 of 40, and captures most of
     the 8-view ensemble gain.** (2026-09-23, Colab runs by the owner; evals on the M5)
     Notebook train_scratch18b96_pair2_tta_colab.ipynb (tarball v7, r2_bulk_tta8_tta.pt.gz = r2_bulk
     relabeled with e40's 8-view-averaged move; e40 recipe bs32768 lr3e-3 warmup 1, 40 ep, hard CE,
     --policy-head pair2 --pair-dim 64 --legal-mask-loss). Gate bank 2,600,000-2,600,999, single pass:
       epoch   original run   old head + averaged labels   PAIR2 + mask + averaged labels
         18        9,280              7,113                     18,029 / P50 12,478
         23       10,463                -                       19,464 / P50 13,868 / <1k 1.9%
         27       12,437             10,361                     20,642 / P50 14,014 / <1k 2.9%
       e40 (original, finished) 14,296 / P50 9,944.
     8-view averaging: e40 14,296 -> 21,436 (+50%); PAIR2 e23 19,464 -> 21,681 (+11%) — one pass of the
     new head ~= eight of the old. Symmetry (all 8 views agree, 5,980 states): e40 60.5%, old head +
     averaged labels e27 65.0%, PAIR2 e23 70.4%.
     Averaged labels alone HURT the old head (-17% at e27): the win is the head (+ mask).
     Running (Colab): the PAIR2 winner to epoch 40; attribution arm without the mask
     (train_scratch18b96_pair2_nomask_tta_colab.ipynb); labels arm on the ORIGINAL r2_bulk labels
     (train_scratch18b96_pair2_orig_colab.ipynb). Old-head born-again run stopped (dead branch).

242. **NEW BEST: PAIR2 head + legal-mask loss on the ORIGINAL r2_bulk labels = 25,190 single pass
     at EPOCH 7 (P50 18,234, P10 2,722, <1k 2.2%).** (2026-09-23, Colab by the owner)
     train_scratch18b96_pair2_orig_colab.ipynb (tarball v7, r2_bulk.pt; e40 recipe; checkpoint args
     verified: r2_bulk.pt, head pair2, legal_mask_loss True). Gate bank, single pass, epoch 7:
       original run (old head)                        4,898 / P50 3,715 / <1k 11.6%
       PAIR2, no mask, 8-view-averaged labels        14,753 / P50 10,150 / <1k 3.5%
       PAIR2 + mask, ORIGINAL labels                 25,190 / P50 18,234 / <1k 2.2%
     vs e40 (finished old head) 14,296 and its 8-view average 21,436; the averaged-label PAIR2 run
     reaches 20,436-21,413 at epochs 32-34. The 8-view-averaged labels hurt BOTH heads; the head +
     mask on the original labels is the new line. Mask share still open (winner's ep7 missing; compare
     the no-mask arm's later epochs with the winner's ep18/23/27).

243. **The small model was never capacity-limited — the old policy head capped it at ~15k. PAIR2 +
     legal mask on the ORIGINAL labels = 40,154 single pass at epoch 33 (2.8x e40, same 3M 18b96).**
     (2026-09-24) Gate bank 2,600,000-2,600,999, single pass (mean / P50 / P10 / <1k):
       run \ epoch             7        18        23        27        33-35         40
       original (old head)   4,898     9,280    10,463    12,437         -        15,164 (e40)
       PAIR2+mask, ORIG     25,190    30,615    36,450      -       40,154 (e33)   pending
                             P50 18,234 / 21,164 / 24,135 / 28,817; P10 5,669 and <1k 0.7% at e33
       PAIR2+mask, TTA lab     -      18,029    19,464    20,642    21,413 (e34)   19,916
       PAIR2 no mask, TTA   14,753    17,745    18,110      -       18,850 (e35)     -
       old head, TTA lab       -       7,113       -      10,361         -            -
     Attribution: head ~2.5x (PAIR2 no-mask vs old head, same labels, ep18); mask +2%/+7%/+14%
     (ep18/23/34, same labels); labels: 8-view-averaged e40 labels cap students at ~21k (= the
     teacher's own strength) while the original search-teacher labels are absorbed to 40k+.
     Pillar3k (256ch, 11.9M params) greedy reference: 43,390.

244. **Final epoch 40: PAIR2 + mask on ORIGINAL labels = 41,938 / P50 28,442 / P10 4,552 / <1k 0.9%
     (plateau: e33 40,154). New small-model actor A1 = scratch18b96_pair2_orig_ckpts_epoch_40.**
     (2026-09-24) Mask attribution at e40: PAIR2 no-mask (averaged labels) 19,425 vs with mask 19,916
     (+2.5%, noise; +2..14% across epochs) — the head carries the win, the mask is a small bonus.
     e33 8-view average 47,622 (+19% over single) -> residual backbone asymmetry.
     Law observed: a PAIR2 student reaches its label source's strength (averaged-e40 labels 21.4k ->
     student ~21k), so the next gain must come from stronger labels: A1's own search.

245. **Flywheel generation 1 prepared (A1 = PAIR2 + mask e40).** (2026-09-24)
     A1 on fresh seeds 3,000,000-3,000,999: 41,667 / P50 30,387 / <1k 1.0% (reproduces the gate).
     A1 survival head value_head_A1.pt (frozen A1 backbone, 2.81M states of A1's own uncapped games,
     record-every 8 + final 300; inner-val BCE 0.0756, per-H 0.013/0.028/0.075/0.187); fused
     pv_A1_ts.pt (traced == eager); C++ search with the PAIR2 head verified (mcts_selfplay smoke).
     mcts_crisis gained --virtual-mean. Timing (10 seeds, recovery 15 @600 + prevention 30 @600,
     c1.5 q2 virtual mean): 9 deaths -> 18 replays, 3,780 rows in 583 s (39% of replays survive to
     the continue cap). Owner runs: gen1_crisis_A1 (seeds 3,200,000-3,200,999) and gen1_selfplay_A1
     (200 games, seeds 3,300,000+, 400 sims, cap 3000), both Dirichlet 0.

246. **Flywheel gen-1 crisis mining done; blending an A1 fine-tune on the WHOLE crisis corpus is a
     wash (alpha 0.5) and plain continued training loses 22% (alpha 1.0).** (2026-09-25)
     Mining (owner run, 24.3 h on the M5): mcts_crisis --model A1_pair2_orig_e40_ts.pt --value-module
     pv_A1_ts.pt, seeds 3,200,000-3,202,999, recovery 15 @600 + prevention 30 @600, c1.5 q2
     --virtual-mean, Dirichlet 0, threads 14 -> data/gen1_crisis_A1. 3,000 probes: 2,588 deaths, 412
     reached the 40k-turn probe cap. 5,176 replays: prevention escapes (survives the 500-turn
     continuation) 63.6%, recovery 34.0%; 72% of deaths escaped by at least one rewind.
     Corpus: build_expert_v2_tensor --policy-only-data -> alphatrain/data/gen1_crisis_A1.pt
     (1,353,669 rows). Checks (anomaly_checks, 20k rows): obs builder == reference, 0 illegal
     targets, A1 argmax == search move on 84.6%.
     Fine-tune (task-arithmetic vector, M5, 12 min): train_path_b --resume A1_pair2_orig_e40.pt
     --warm-start --freeze-bn --policy-head pair2 --pair-dim 64 --legal-mask-loss --epochs 2
     --batch-size 1024 --lr 1e-4 --warmup-epochs 0 --target-temperature 0.5 --augment-factor 1
     (D4 + color aug) -> checkpoints/gen1_crisis_ft/epoch_2.pt (val 1.2549 -> 1.2531).
     Merge + gate (scripts/ta_sweep.sh = scripts/merge_checkpoints.py + eval_log run), gate bank
     2,600,000-2,600,999 uncapped (A1 41,938 / P50 28,442 / P10 4,552 / <1k 0.9%):
       alpha 0.5   41,146 / P50 28,123 / P10 4,918 / P95 119,852 / <1k 1.2%   (wash)
       alpha 1.0   32,758 / P50 22,095 / P10 3,949 / P95  95,764 / <1k 1.9%   (-22%)
     Why (scratchpad crisis_composition.py): 86% of rows are the post-escape 500-turn continuation
     (quiet late-game states under search), i.e. own-search distillation, which washed before
     (entry 236). The corrections (moves up to the death turn) are 106k rows (8%).
     Running: the pillar3f analogue on the corrections only (alphatrain/scripts/build_crisis_windows.py
     --after 15 --escaped-only -> 100,515 rows from 2,527 escaped replays; A1 matches 84.9%).

247. **The pillar3f analogue (fine-tune on the corrections only, then blend) is also a wash on gen-1
     data.** (2026-09-25)
     Corpus: alphatrain/scripts/build_crisis_windows.py --in-dir data/gen1_crisis_A1 --out-dir
     data/gen1_crisis_A1_windows --after 15 --escaped-only (2,527 escaped replays, moves up to the
     original death turn + 15) -> build_expert_v2_tensor --policy-only-data ->
     alphatrain/data/gen1_crisis_A1_windows.pt (100,515 rows; 0 illegal targets; A1 == search move
     84.9%; BN train-mode vs eval-mode argmax agreement only 61.7% on these dense boards, so the
     frozen-BN fine-tune matters). Fine-tune: same flags as entry 246 but --epochs 8 (val 1.1481 ->
     1.1345; max |task-vector| element 0.029 vs 0.128 for the full corpus). Gate bank, 1k uncapped:
       alpha 0.4   41,962 / P50 29,526 / P10 4,928 / <1k 1.0%
       alpha 0.7   41,236 / P50 29,054 / P10 5,411 / <1k 1.7%      (A1 41,938 / 28,442 / 4,552 / 0.9%)
     Unlike pillar3f (+36%, MCTS@4800 widened labels), 600-sim labels on A1's crisis windows carry
     little A1 doesn't already play: 85% agreement, and the rest is largely single-search noise
     (entry 236). The escapes the search finds (72% of deaths) don't transfer move by move.
     Running: AlphaZero-style continuous training (new train_path_b --mix-tensor/--mix-share replay
     mix + --ema-decay weight EMA; tests added), treatment r2_bulk + 30% gen-1 crisis rows vs control
     r2_bulk only, from A1, constant lr 3e-5, bs 4096, hard CE, frozen BN, EMA 0.999, 1 epoch.

248. **Continuous training is safe but gen-1 data adds nothing, and the reason is measured: A1's
     600-sim search is barely stronger than A1 greedy at A1's own crisis states.** (2026-09-25..26)
     AlphaZero-style continuous training (new train_path_b options, tests added): --mix-tensor /
     --mix-share (MixedLoader: every batch = base rows + a fixed share of the new corpus) and
     --ema-decay (WeightEMA, validated and saved as ema_*.pt). From A1, 1 epoch of r2_bulk,
     constant lr 3e-5 (--flat-epochs 1 --warmup-epochs 0), bs 4096, hard CE (--blend-alpha 0),
     --freeze-bn --legal-mask-loss --augment-factor 1 --seed 42, EMA 0.999, gate on the EMA weights:
       control   r2_bulk only                          40,971 / P50 28,437 / P10 4,912 / <1k 1.1%
       treatment r2_bulk + 30% gen-1 crisis per batch  42,250 / P50 29,407 / P10 4,710 / <1k 0.9%
     (A1 41,938.) The constant-LR + EMA regime keeps A1's strength (unlike every plain fine-tune of
     e40); the crisis rows barely moved (crisis val CE 1.532 -> 1.507).
     Teacher-advantage control (new eval --anchors mode: Game(seed) + SetState from each mcts_crisis
     rewind state, same spawn seed as the replay; alphatrain/scripts/crisis_anchors.py export/compare):
       prevention (rewind 30)  search@600 escapes 63.6%   A1 greedy 54.7%   (+8.9 pts)
       recovery   (rewind 15)  search@600 escapes 34.0%   A1 greedy 27.1%   (+6.9 pts)
     Paired: search-only escapes 502 vs greedy-only 271 (prevention), 402 vs 223 (recovery). Most of
     the "72% of deaths escaped" (entry 246) is the replay's new spawn stream, not the search. With a
     teacher this close to the student, no training channel can extract much (entries 246-248).
     NEXT: measure teacher strength vs search budget cheaply (search only the crisis window, then
     greedy, from the same anchors) and mine only with a configuration that clearly beats greedy.

249. **BUG: mcts_crisis never used the survival head. Since 2026-07-08 every crisis corpus mined with
     --value-module (gen-1, and r2_bulk's crisis_iter5 / crisis_vh3 / crisis_vh3_deep / crisis_vh2_r2
     = its ~56% crisis-replay rows) was searched with the 27-feature leaf evaluator at q 2.0, while
     the JSONs said value_kind "neural". Fixed, the teacher's edge over greedy triples.** (2026-09-26)
     Cause: the replay worker built its MctsConfig without `cfg.nn_value = nn_value` (mcts_selfplay,
     mcts_relabel and mcts_eval set it; the server loaded the fused module, whose policy logits were
     used, but leaf values came from data/feature_value.bin). Introduced with --value-module in
     87ee45a (2026-07-08). Fix: set the flag; value_kind now reports cfg.nn_value.
     Measurement (new build/anchor_search: from each saved rewind state, Game(seed) + SetState exactly
     as the replay, search the first 45 moves, then greedy to the 500-turn cap; --search-turns 0
     reproduces eval --anchors, 54.9% vs 54.7%). 1,294 prevention anchors (30 moves before A1's death),
     A1 prior, 600 sims, c1.5 q2 virtual mean:
       A1 greedy                                       54.9% escape
       feature leaf (the bug), 45 searched moves      60.2%   (+5.3)
       gen-1 replays as mined (feature leaf, 500)     63.5%   (+8.6)
       survival-head leaf (fixed), 45 searched moves  72.3%   (+17.4)
     So entries 246-248 trained on the weak teacher, and A1 itself learned its crisis states from
     feature-leaf labels. NEXT: re-mine gen-1 with the fixed binary; later, relabel r2_bulk's crisis
     states with the fixed search.

250. **BREAKTHROUGH: the same 45-move crisis windows, labeled by the FIXED search (survival-head leaf),
     lift A1 by +60% mean: 67,240 / P50 46,422 / P10 8,240 / <1k 0.5% / max 539,980 (alpha 0.7).**
     (2026-09-26)
     Mining (owner run, 4.1 h, fixed binary from entry 249): mcts_crisis --run-id gen1b_crisis_A1_w45
     --model A1_pair2_orig_e40_ts.pt --value-module pv_A1_ts.pt, seeds 3,200,000-3,202,999 (same
     probes: 2,588 deaths, identical to gen-1), recovery 15 @600 + prevention 30 @600, c1.5 q2
     --virtual-mean, Dirichlet 0, --continue-turns 45 -> data/gen1b_crisis_A1_w45 (value_kind
     neural checked). Corpus alphatrain/data/gen1b_crisis_A1_w45.pt: 194,302 rows, 0 illegal
     targets, A1 == teacher move 86.3% (buggy windows: 84.9% -> the gain is override QUALITY).
     Orchestration: scripts/gen1b_after_mining.sh.
     Arm 1 (fine-tune + blend): train_path_b --resume A1 --warm-start --freeze-bn --policy-head pair2
     --legal-mask-loss --epochs 4 --batch-size 1024 --lr 1e-4 --warmup-epochs 0
     --target-temperature 0.5 --augment-factor 1 --seed 42 -> checkpoints/gen1b_ft/epoch_4.pt
     (val 1.3674), scripts/ta_sweep.sh (merge_checkpoints + eval_log). Arm 2 (continuous): r2_bulk +
     30% new rows per batch, 1 epoch, constant lr 3e-5, bs 4096, hard CE, frozen BN, EMA 0.999.
     Gate bank 2,600,000-2,600,999, 1k uncapped (A1 41,938 / P50 28,442 / P10 4,552 / <1k 0.9%):
       blend alpha 0.4        53,877 / P50 38,312 / P10 6,973 / P95 164,800 / <1k 0.5%   (+28%)
       blend alpha 0.7        67,240 / P50 46,422 / P10 8,240 / P95 202,981 / <1k 0.5%   (+60%)
       continuous EMA, 1 ep   54,320 / P50 38,738 / P10 5,408 / P95 166,313 / <1k 1.1%   (+30%)
     Same windows from the buggy teacher (entry 247): wash. The flywheel works once the teacher is
     right. Running: alpha 0.85 / 1.0 (dose-response still rising at 0.7) and a fresh-bank
     confirmation of alpha 0.7 (seeds 3,000,000-3,000,999, A1 = 41,667 there).
     CONFIRMED on fresh seeds 3,000,000-3,000,999 (A1 there: 41,667 / P50 30,387 / <1k 1.0%):
       blend alpha 0.7        70,782 / P50 49,390 / P10 7,171 / P95 213,013 / <1k 0.5% / max 578,521
     (+70% mean, +63% median). Candidate A2 = alphatrain/data/ta_gen1b_e4_a0.7.pt.
     Dose-response keeps rising through the plain fine-tune (gate bank, 1k uncapped):
       alpha 0.85             75,353 / P50 52,067 / P10 8,954 / P95 236,692 / <1k 0.4%
       alpha 1.0              81,033 / P50 54,926 / P10 9,670 / P95 252,156 / <1k 0.3%   (+93%)
     alpha 1.0 == checkpoints/gen1b_ft/epoch_4.pt itself (frozen BN): with a correct teacher, plain
     "resume from A1 and continue" on the crisis windows is the best channel (the -22% of entry 246
     was the buggy teacher's labels). Running: alpha 1.0 on the fresh bank, alpha 1.3 extrapolation.
     alpha 1.0 CONFIRMED on fresh seeds 3,000,000-3,000,999: 77,311 / P50 52,597 / P10 7,795 / P95
     242,302 / <1k 0.6% / max 580,375 (A1 there 41,667 / 30,387: +86% mean, +73% median; alpha 0.7
     there 70,782). **A2 = checkpoints/gen1b_ft/epoch_4.pt** (== alphatrain/data/ta_gen1b_e4_a1.0.pt,
     TS inference_cpp/data/ta_gen1b_e4_a1.0_ts.pt): A1 fine-tuned 4 epochs on 194k fixed-teacher
     crisis-window rows.
     EXTRAPOLATION keeps improving (theta = A1 + alpha * task vector, gate bank, percentiles; ">=100k" =
     share of games reaching 100k turns, the new infinite-play metric):
       model       P10     P25     P50      P75   >=100k
       A1          4,552  12,468  28,442   56,904    1.0%
       alpha 1.0   9,670  25,768  54,926  112,766    7.9%
       alpha 1.3   9,621  25,719  62,498  122,076   10.4%
       alpha 1.6  10,776  30,179  71,431  144,984   13.8%
       alpha 2.0  12,204  33,234  79,481  166,507   18.4%
     The fine-tune points the right way but moves too little. New eval protocol (owner, 2026-09-26):
     every eval --max-turns 100000 (eval_log default), judged by percentiles + capped share, never the
     mean; EVAL.md gained a `capped` column (older rows uncapped -> 0.0%). Evals are C++ only.
     Running: alpha 2.5 / 3.0 (capped 100k).
       alpha 2.5  16,085  43,272 102,938  203,505   25.3%   (cap 100k)
       alpha 3.0  16,249  43,321 100,123  194,884   23.3%   (cap 100k)
     Peak around alpha 2.5-3.0 (P75 is now at the cap's ceiling). Candidate A2 = alpha 2.5
     (alphatrain/data/ta_gen1b_e4_a2.5.pt): P50 3.6x A1, a quarter of games survive 100k turns.
     gen1_selfplay_A1 (owner) finished: 200 games, 45,859 s (not yet used).
     alpha 2.5 CONFIRMED on fresh seeds 3,000,000-3,000,999 (cap 100k): P10 15,884 / P25 46,727 /
     P50 104,629 / P75 203,185 / 24.9% reach 100k turns (A1 there: 4,785 / 13,211 / 30,387 / 59,487
     / 0.9%). **A2 = alphatrain/data/ta_gen1b_e4_a2.5.pt** = A1 + 2.5*(checkpoints/gen1b_ft/epoch_4.pt
     - A1), TS inference_cpp/data/ta_gen1b_e4_a2.5_ts.pt. Next turn: scripts/flywheel_turn.sh from A2.

251. **MCTS config audit after the entry-249 bug: no other tool has it; parsers and the inference
     server hardened.** (2026-09-26)
     Every MctsConfig site (mcts_selfplay cfg + clean_cfg, mcts_crisis cfg + clean_cfg, mcts_relabel,
     mcts_eval, anchor_search) copies every parsed search flag, and every tool derives the server's
     fused mode and cfg.nn_value from the same bool. Confirmed minor defects: (1) label_dirichlet_weight
     is written as a constant 0.0 (no consumer relies on it); (2) policy_model / run_config "model"
     record --model, but with --value-module only the fused module is loaded, so the recorded policy is
     right only if pv was exported from that checkpoint (flywheel_turn.sh does); (3) eval, mcts_eval
     and rollout_judge silently ignored unknown flags (a typo like --max_turns would run uncapped), and
     mcts_relabel ignored a trailing value-less flag. Fixed (3): all four now exit on unknown or
     value-less flags. InferenceServer now aborts if a search asks for leaf values from a policy-only
     module (it used to hand back zeros: a flat-Q, prior-only search, the reverse of entry 249).
     Noted, not changed: per-tool defaults differ (c_puct 2.5 vs 1.5, q_weight 1.0 vs 2.0), so every
     script passes the operating point explicitly (c1.5 q2 --virtual-mean).

252. **Flywheel turn 2 (A2 -> A3): more than half the games now survive 100k turns.** (2026-09-26..27)
     Owner run: scripts/flywheel_turn.sh alphatrain/data/ta_gen1b_e4_a2.5.pt A2 3400000 3500000
     "1.0 2.0 3.0" (binaries with the entry-251 hardening; no previous-generation windows).
     - Own games: 1,000 greedy games, seeds 3,400,000-3,400,999, cap 100k, 38 min: P10 16,470 /
       P25 42,529 / P50 104,883 / 27.0% capped (third bank agreeing on A2).
     - Survival head value_head_A2.pt on the frozen A2 trunk (5 ep, 19.6 min, inner-val 0.0270);
       fused pv_A2_ts.pt.
     - Mining: mcts_crisis --run-id A2_crisis_w45, seeds 3,500,000-3,502,999, probe cap 100k, 600 sims,
       c1.5 q2 virtual mean, 45-move windows: 2,240 deaths (760 probes survived 100k turns), 4,480
       replays (3,441 survived their 45 moves), 5.4 h (probes 2.3 h).
     - Fine-tune: 4 ep, frozen BN, lr1e-4, T0.5 -> checkpoints/A2_ft/epoch_4.pt (val 1.6859).
     Gate bank 2,600,000-2,600,999, cap 100k (A2: P10 16,085 / P25 43,272 / P50 102,938 / 25.3%):
       alpha 1.0   P10 23,070 / P25  74,148 / P50 178,426 / capped 44.9%
       alpha 2.0   P10 39,270 / P25 107,085 / P50  at cap / capped 55.4%
       alpha 3.0   P10 33,314 / P25 113,973 / P50  at cap / capped 57.2%
     A3 candidate = alpha 2.0 (best floor): alphatrain/data/ta_A2_e4_a2.0.pt. With >50% of games at
     the cap, P50 is censored; the capped share, P10 and P25 are the live metrics. Running: fresh-bank
     confirmation (3,000,000-3,000,999).
     A3 CONFIRMED on fresh seeds 3,000,000-3,000,999 (cap 100k): P1 5,236 / P5 14,160 / P10 31,404 /
     P25 104,974 / P50 at cap / 56.3% capped (A2 there: 1,434 / 6,972 / 15,884 / 46,727 / 104,629 /
     24.9%). **A3 = alphatrain/data/ta_A2_e4_a2.0.pt** (TS ta_A2_e4_a2.0_ts.pt).

253. **Escape benchmark from a weaker greedy player's imminent-death states (owner's choice of
     "troubled positions"; no human games).** (2026-09-27)
     eval --anchors on A1's 5,176 gen-1 rewind states (Game(seed) + SetState, the replay's spawn stream;
     escaped = survives 500 more turns), greedy, cap 500 per anchor, ~1 min per model:
       player                         30 moves before death   15 moves before death
       A1 greedy                      54.7%                   27.1%
       A2 greedy                      63.7%                   34.1%
       A3 greedy                      68.0%                   36.6%
       (A1 + buggy 600-sim search      63.6%                   34.0%)
     A2's greedy play already equals A1's (buggy) search. Escape ability grows far slower than
     on-policy survival (capped share 1% -> 25% -> 56%): the actors mostly learn to avoid trouble.
     Track this next to the gate for the hint use case. Owner's stop criterion for the flywheel:
     continue until P1 reaches the 100k cap; track P1-P5 from now on.

254. **Flywheel turn 3 (A3 -> A4 candidate): smaller gain, probing is now the bottleneck; off-policy
     crisis tooling (weak probe players, mouse slips).** (2026-09-27..28)
     Owner run: PROBES=5000 scripts/flywheel_turn.sh alphatrain/data/ta_A2_e4_a2.0.pt A3 3600000 3700000
     "1.0 2.0 3.0". A3 own games (seeds 3.6M, cap 100k): P1 3,675 / P10 37,439 / 57.4% capped. Mining:
     5,000 probes -> 2,159 deaths, 4,318 replays; probe phase 5.2 h of 8.3 h. Fine-tune val 1.4603.
     Gate bank, cap 100k (A3: P1 3,459 / P5 16,380 / P10 39,270 / P25 107,085 / 55.4%):
       alpha 1.0   P1 5,583 / P5 21,611 / P10 45,152 / P25 124,695 / capped 62.3%   <- A4 candidate
       alpha 2.0   P1 3,908 / P5 18,811 / P10 40,468 / P25 115,615 / capped 60.5%
       alpha 3.0   P1 2,411 / P5 16,905 / P10 32,463 / P25  90,680 / capped 50.4%
     Diminishing returns (capped share 1% -> 25% -> 55% -> 62%) and the optimal alpha fell to 1.0.
     Owner's direction: two goals (infinite play on a small model; recover from real human positions);
     death spirals may be alike, so cheap probes by a human-level player (scratch18b96_lr3e3_ckpts_epoch_4,
     mean 2,368 = average human; games die in ~1-2k turns) could replace most strong-actor probing; bad
     play simulated with mouse slips; quiet play stays in the data.
     New mcts_crisis options (tested on MPS, scratchpad test_probe_model2.sh): --probe-model (a weaker
     policy plays the probes, the actor's search replays; recorded as probe_model), --probe-slip p (per
     move, a uniformly random legal move; reproducible per seed; recorded), --anchors-out (anchors.h
     lines, seed = ReplaySeed = original*37+rewind), --probe-only. Checks: same-model --probe-model and
     --probe-slip 0 reproduce the default anchors exactly; 5% slips cut epoch-4's mean death turn from
     1,399 to 244. Bug caught in testing: slip RNGs seeded with seed*gamma made consecutive games'
     streams overlap (4.37% instead of 5%); hashed seeding gives 5.05% over 99k moves.
     Escape horizon: failed rescues die fast (95-98% by move 100, 99-100% by move 200, gen-1 search and
     A1-A3 greedy), so escape = survive 200 moves from now on.

255. **Off-policy experiment: crises from a human-level player's deaths teach as much as the actor's
     own (death spirals are alike) at ~1/150 of the probing cost; no arm moves the escape benchmarks.**
     (2026-09-28) Owner run: scripts/offpolicy_experiment.sh (all arms from A3 = ta_A2_e4_a2.0.pt,
     A3's search: pv_A3_ts.pt, 600 sims, c1.5 q2 virtual mean; turn-3 fine-tune recipe).
     A4 (= ta_A3_e4_a1.0) CONFIRMED on fresh seeds 3,000,000-3,000,999: P1 2,314 / P5 19,580 / P10
     52,388 / P25 127,099 / 63.9% capped (A3 there: 5,236 / 14,160 / 31,404 / 104,974 / 56.3%).
     Arms (gate bank, cap 100k; A3: P1 3,459 / P5 16,380 / P10 39,270 / P25 107,085 / 55.4%):
       arm  source (probes)                  replays   probe time   alpha  P1     P5     P10    P25     capped
       a    A3's own deaths (turn 3, 5,000)   4,318     5.2 h        1.0    5,583  21,611 45,152 124,695 62.3%
       b    epoch-4 deaths (2,160)            4,320     134 s        1.0    8,556  26,608 49,879 118,313 60.0%
                                                                     2.0    5,025  23,010 50,416 121,056 60.6%
       c    epoch-4 deaths, 200-move replays   1,000     51 s         1.0    5,079  28,629 47,018 121,092 61.5%
            (crisis + quiet play; 500 probes)                         2.0    5,440  17,277 36,080  97,327 56.2%
     (A3's search survived its window in 82% of arm-b replays, 80% of arm-c.) Fine-tune val: b 1.4772,
     c 1.2228. Escape benchmarks (200-move horizon; greedy): expert trouble (A1's 5,176 death states,
     30/15 moves before death) A3 68.2/36.7%, A4 68.7/38.5%, b1.0 69.2/38.5%, c1.0 69.3/37.9%;
     human trouble (epoch-4 deaths on held-out seeds 3.9M; built WITHOUT --prevention-turns 30, so its
     deep anchors sit 75 moves before death and escape_benchmark labels them "1" = 75 % 37) A1 90.6/
     34.8%, A2 93.3/47.4%, A3 94.8/47.3%, A4 95.2/47.6%, b1.0 95.2/48.6%, c1.0 94.7/47.4% (75/15 before).
     Reading: the gate is the same for all three sources, so future turns can probe with the epoch-4
     player. No arm improves escapes (all within ~2 pts), even arms trained on human-level crises.
     NEXT: rebuild the human benchmark at 15/30; data scaling (A3 fine-tuned on a+b+c together);
     teacher edge on the human benchmark (A3 greedy vs A3 search @600 and @1,600 for 45 moves).

256. **Gate switch: the per-turn death rate (survival analysis), because the hazard is flat over game
     age; escape-rate benchmarks retired; gate bank 2k seeds.** (2026-09-28, owner's direction)
     alphatrain/scripts/survival_stats.py (every game's turns count as exposure, capped or not; hazard by
     game-age bucket; S(t); --cap re-censors uncapped runs). Gate bank, cap 100k:
       model   deaths/1k  deaths per 100k turns [95% CI]  MTBF turns  S(10k)  S(50k)  S(100k)  hazard 0-5k/5-20k/20-50k/50-100k
       A1        990      4.9 [4.6, 5.2]                     20,567   61.9%    9.0%    1.0%   4.7 / 5.1 / 4.9 / 4.2
       A2        747      1.4 [1.3, 1.5]                     71,984   87.2%   50.5%   25.3%   1.4 / 1.4 / 1.4 / 1.4
       A3        446      0.6 [0.5, 0.6]                    170,727   94.4%   75.6%   55.4%   0.7 / 0.5 / 0.6 / 0.6
       A4        377      0.47 [0.43, 0.53]                 210,640   95.4%   78.2%   62.3%   0.5 / 0.5 / 0.5 / 0.5
     Deaths arrive at a constant rate per turn at every game age, so survival is exponential and the
     death rate alone determines every percentile (P_q = -ln(1-q) / rate turns). P1-P5 are the same
     information from 10-50 games; the rate uses all ~400 deaths (+-10% at 1k games, +-7% at 2k) and is
     comparable across caps (A1 uncapped 4.88 vs re-censored at 100k 4.9). Per-turn improvement: A1->A2
     3.5x, A2->A3 2.4x, A3->A4 1.2x. The stop criterion (P1 at the 100k cap) = ~0.01 per 100k turns,
     ~50x below A4; the eval cost to measure a rate r grows ~1/r (~400 deaths of exposure).
     EVAL.md gained "deaths/100k turns [95% CI]" and "MTBF turns" (all 121 rows backfilled from their
     CSVs); eval_log computes both; gate bank = 2,600,000-2,601,999 (eval_log + ta_sweep defaults).
     Gate: death rate down >= 20% with non-overlapping intervals; the 0-5k hazard must not rise.
     Escape-rate benchmarks retired as a metric (owner: escape depends on position badness + spawn luck).

257. **Eval 1.8x faster: the CPU game loop redone and parallelized (bit-identical), BatchNorm folded
     into the convolutions; the forward pass is now the limit (GPU-bound at every batch size).**
     (2026-09-29) Workload: A4, 4,000 games x 1,000 turns (4M positions), same seeds.
       old binary, batch 500                        155.3 s
       new CPU path, batch 500 / 1k / 2k / 4k      116.7 / 114.4 / 115.2 / 118.5 s   (CSV byte-identical to old)
       + BatchNorm folded (export --fold-bn), 1k     86.5 s
     CPU path (game.cc / obs.cc / eval.cc; build/eval_cpu_bench checks bit-exactness on 4,000 real
     states from A2's games and times it): the empty-cell flood fill ran 3x per game per step (obs,
     legal mask, move) -> once, shared (Game::Labels, BuildObs(labels), LegalMaskU8, Move(labels));
     the legal mask was 6,561 floats + a float->uint8 pass -> built as uint8 from 128-bit component
     sets; spawns allocated 3-4 heap vectors per move -> stack arrays with the same RNG draws
     (SimpleRng::ChoiceNoReplaceArr); per-game work split over a thread pool (--cpu-threads, default
     8). CPU per step at 4,000 games: 36.1 ms -> 17.2 ms single-thread -> 2.6 ms on 8 threads.
     game_test (Python goldens incl. spawns) and mcts_controls_test pass; the spawn change also speeds
     up MCTS. Byte-identical CSVs vs the old binary on 400 capped A4 games and 300 full epoch-4 games.
     Per-step time grows linearly with batch: the GPU saturates at ~35k positions/s (~17 TFLOPS for
     ~0.5 GFLOP/position), so bigger batches don't help. Folding the 20 conv->BN pairs (stem, each
     block's conv1->bn2, policy conv1->bn; bn1 and backbone_bn can't fold) removes memory-bound
     elementwise passes: 1.32x on the forward. Folded exports play the same policy with different fp16
     rounding (3,604 of 4,000 games diverge somewhere within 1,000 turns; deaths 18 vs 17), so every
     model in a comparison must use the same export. eval_log now exports folded by default ([fold]
     tag, _fold_ts.pt, --no-fold for the old) and keeps all games in flight (batch = #games, <= 4k).
     Core ML potential (Python throughput test, not an eval): GPU 61k positions/s, GPU+Neural Engine
     66k/s at batch 1k-4k (Neural Engine alone 13.5k/s) vs ~50k/s folded MPS -> another ~1.3x if the
     C++ eval gets a Core ML (Objective-C++) inference path.

258. **Data scaling at a fixed teacher: 3x the crisis data ~= 1x (union 0.48 vs A4 0.49 deaths per 100k
     turns). Teacher strength is the lever. The union model teaches a 9x64 student.** (2026-09-28..29)
     A3 fine-tuned 4 epochs on arms a+b+c together (A3 deaths 45 moves + epoch-4 deaths 45 + 200 moves;
     alphatrain/data/union_abc.pt, 511,834 rows, val 1.3773), alpha 1.0 = ta_union_abc_a1.0. Folded 2k
     gates (seeds 2,600,000-2,601,999, cap 100k, optimized eval, all games in flight):
       union (3x data)  0.48 [0.45, 0.52]/100k turns  MTBF 207,927  P1 5,815  P5 23,452  P10 47,413  61.7%
       A4 (arm a only)  0.49 [0.46, 0.53]/100k turns  MTBF 202,343  P1 5,643  P5 20,184  P10 41,715  61.1%
     Intervals overlap: at the same teacher (A3 + 600-sim fixed search) more rows don't buy a lower
     death rate. Next flywheel turns: stronger teacher (owner: 800 sims) rather than more rows.
     Owner's decision: shrink to a 9 blocks x 64 channels PAIR2 student (~4x cheaper forward -> ~4x faster
     evals and mining). Teacher = union (lower rate, picked automatically). Corpus: scripts/prep_distill.sh
     alphatrain/data/ta_union_abc_a1.0.pt union -> teacher games (3,000, seeds 4,300,000+, every 25th
     move + last 300), epoch-4 games (1,000, seeds 4,400,000+, every move), and the five crisis corpora's
     states, all labeled with the union model's 8-view TTA top-5 -> alphatrain/data/distill_union.pt.
     Training on Colab (A1 recipe, soft and hard arms): alphatrain/train_scratch9b64_pair2_distill_colab.ipynb.
     Corpus (2026-09-29): 12,417,115 rows, .pt 1,986,742,177 B, .gz 842,220,899 B:
       teacher games  10,368,754 rows  3,000 games at cap 100k (P1 5,138, P50 203,931, 64.1% capped)
       epoch-4 games   1,167,348 rows  1,000 games (P1 299, P50 1,818, 26.4% < 1,000)
       crisis states     881,013 rows  gen1b_crisis_A1_w45 194,302, A2_crisis_w45 174,877,
                                       A3_crisis_w45 170,483, offp_b_ep4_w45 175,107, offp_c_ep4_w200 166,244
     Label checks (logs/distill_union_verify.log, _redo_crisis.log, _crosscheck.log): rows realign to the
     recorded games 100%; every target's top move is legal; it equals the move played on 92.5% of the
     teacher's rows (single pass vs 8-view average) and 70.4% of epoch-4's; 5 moves per target, mean
     top-share 0.747 / 0.712 / 0.65-0.70 (teacher / epoch-4 / crisis rows).
     BUG caught before training: relabel_tta_tensor.py took the softmax of the SUM of the 8 views' logits
     (temperature 1/8; fp16 then zeroed the tail: crisis targets had 2.7 moves, top-share 0.94) while
     build_tta_corpus uses the MEAN. The argmax is unaffected, and every earlier use trained hard CE; the
     soft arm would have mixed two label temperatures. Fixed (acc / 8), crisis rows relabeled: top-5 moves
     identical on 100% of rows; values match build_tta_corpus's labeller on 2k-row samples (same top move
     99.7%, median |diff| 1.7e-3 = fp16 deploy net vs fp32). Also: _legal_priors_jit returns its top-k
     ASCENDING, so build_tta_corpus's "argmax == recorded" stat compared the 5th-best move (printed 0.0%);
     stat fixed, labels were never affected.
     Smoke (M5, 40k rows, 1 epoch): 9x64 PAIR2 = 734,722 params; both arms train; the folded export runs
     in the C++ eval.

259. **9x64 student distilled from the union model: best = soft epoch 35 at 1.52 deaths per 100k turns
     (teacher 0.48, 3.2x). Soft labels beat hard at every epoch; the soft arm plateaued from epoch 30.**
     (2026-09-29) Colab (owner): alphatrain/train_scratch9b64_pair2_distill_colab.ipynb (tarball v8,
     distill_union.pt of HISTORY 258): from scratch, 9 blocks x 64 ch PAIR2 + legal-mask loss (734,722
     params vs the teacher's 3,071,170), 40 ep, bs 32768, lr 3e-3, warmup 1, AMP + compile, seed 42;
     soft = --blend-alpha 1.0 (the teacher's top-5), hard = 0.0 (its move). Val loss: soft 0.8047 /
     0.7974 / 0.7945 at ep 30/35/40 (still falling); hard best 1.0294 before ep 30, then 1.0443 /
     1.0427 / 1.0447 (rising). Gates (2k, seeds 2,600,000-2,601,999, cap 100k turns, folded), scores:
       model              P1     P5     P10     P25     P50      P75      P90      P95   capped  per 100k turns
       teacher (union)  5,815 23,452  47,413 124,318 203,897  204,276  204,541  204,691  61.7%  0.48 [0.45, 0.52]
       soft ep30        1,598  6,802  12,761  34,102  83,652  167,863  204,058  204,313  18.4%  1.69 [1.61, 1.78]
       soft ep35        2,217  7,461  14,838  38,970  94,584  184,449  204,130  204,331  21.8%  1.52 [1.45, 1.60]
       soft ep40        1,555  6,584  12,974  34,126  81,711  164,782  203,995  204,270  17.9%  1.72 [1.64, 1.80]
       hard ep30        1,486  4,606   9,711  26,685  65,875  137,482  203,926  204,195  12.7%  2.09 [2.00, 2.19]
       hard ep35        1,315  5,096  10,977  29,615  70,371  143,984  203,854  204,154  14.0%  1.98 [1.89, 2.08]
       hard ep40        1,223  5,589  10,549  28,933  70,781  138,826  203,778  204,144  12.2%  2.05 [1.96, 2.15]
     (1) Soft > hard at every epoch (disjoint CIs): the teacher's full top-5 carries what a small student
     can use. (2) Epoch 35 beats 30 and 40 while val loss kept falling: val loss does not rank
     checkpoints; gate several. (3) The 18b96 law "a PAIR2 student reaches its label source" (HISTORY
     241-244) does not hold at 1/4 the size. Gate throughput 91k positions/s vs the teacher's 36k (not a
     controlled benchmark: batch compositions differ). A 2k gate of the student takes ~17-19 min.
     Next (owner): flywheel on the student from soft ep35 with 600 sims. scripts/flywheel_turn.sh now
     reads the trunk shape from the checkpoint (alphatrain/scripts/model_shape.py) and takes SIMS
     (default 600); value head, fused export, fine-tune and merge smoke-tested on the student.

260. **Mining 6.1x faster on the 9x64 student: 256 replay threads + a batching window in the GPU server
     (15,003 -> 91,762 network evals/s). Autonomous student flywheel started.** (2026-09-29)
     Diagnosis: at 14 threads mining used 0.7 CPU cores; the GPU ran ~55-position forwards of ~3.7 ms
     each, and a forward costs ~2.8 ms even at that size (build/infer_bench: fused pv_S1, fp16, round
     trip 2.9 ms at batch 55, 4.3 at 300, 6.4 at 600, 10.5 at 1,200, 19.3 at 2,400; reading back all
     6,561 logits costs <= 0.5 ms, fp16 readback gains nothing). Fixes: more games in flight
     (--threads) and InferenceServer::SetBatchWindow: the server fired the moment the first request
     arrived, so after every forward the first thread back triggered a near-empty forward while the
     rest queued (average batch 340 of 512 at 64 threads). mcts_crisis --batch-wait-us N now waits until
     every active replay thread has submitted or N us pass; the target shrinks as threads run out of
     work. Benchmark (S1 policy + survival head, 600 sims, epoch-4 probes, 400 probe seeds; rate
     between the last two [GPU] lines):
       threads  window   evals/s  batch  ms per forward
          14      -      15,003     55     3.7
          32      -      24,300    138     5.7
          64      -      40,516    340     8.4
          64     3 ms    60,381    504     8.3
         128     3 ms    73,593  1,008    13.7
         256     3 ms    91,762  2,019    22.0
     Each game's search is unchanged (8 leaves per request, same sims); only the fp16 batch composition
     differs. scripts/flywheel_turn.sh defaults THREADS=256 BATCH_WAIT=3000; --threads left out of the
     locked run config (an execution setting: a resume may change it); steps 2-3 skip when done, so a
     stopped turn resumes by rerunning it (S1's mining resumed: 291/3,000 seeds kept).
     Owner's directive: run the flywheel on the student autonomously; stop when it reaches the 18x96
     level or a turn gains < 15%. scripts/flywheel_loop.sh: per turn, promote the alpha with the lowest
     2k death rate (alphatrain/scripts/best_gate.py; the largest swept alpha winning gates one more);
     stop at rate <= 0.52 (the union teacher's 95% CI upper end) or a death-rate cut < 15%. Running:
     scripts/flywheel_loop.sh alphatrain/data/scratch9b64_pair2_distill_union_soft_ckpts_epoch_35.pt
     1.52 S 1 4500000 4600000 (turn k: seeds +500,000 per turn) -> logs/flywheel_loop_S.log.

261. **Student flywheel turn S1 (own search, 600 sims): 1.52 -> 1.36 deaths per 100k turns (-10.5%), under
     the owner's 15% bar, so the loop stopped. The 9x64 is capacity-limited: on rows it trained on, its top
     move matches the labels' on ~80.5% of crisis rows vs ~89.6% for the teacher's own single pass.**
     (2026-09-29) scripts/flywheel_loop.sh <soft ep35> 1.52 S 1 4500000 4600000 -> logs/flywheel_loop_S.log:
     own games (seeds 4,500,000+, 1k) 1.62 [1.51, 1.74]; survival head (inner-val 0.0310); mining 3,000
     epoch-4 probes -> 6,000 windows (resumed at 291 seeds; 5,416 mined in 1,879 s, 68,915 evals/s);
     corpus 241,211 rows (anomaly B 0/20,000; C: eval-mode BN matches target 79.48%, vs A1's 86.34%:
     the search overrides the student more); fine-tune 4 ep (val 1.4876). Gates (2k, cap 100k, folded):
       model            P1     P5     P10     P25     P50      P75      P90      P95   capped  per 100k turns
       S1 actor       2,217  7,461  14,838  38,970   94,584  184,449  204,130  204,331  21.8%  1.52 [1.45, 1.60]
       S1 alpha 1.0   2,118  7,007  15,860  43,860  106,406  203,595  204,373  204,585  25.4%  1.36 [1.29, 1.43]
       S1 alpha 2.0   1,441  7,445  15,673  43,117  100,465  203,786  204,534  204,759  25.4%  1.38 [1.31, 1.45]
       S1 alpha 3.0   1,735  5,919  12,943  34,654   84,509  169,320  204,597  204,880  19.8%  1.65 [1.57, 1.73]
     Loop: "STOP: gain 0.105 < 0.15. Best actor: alphatrain/data/ta_S1_e4_a1.0.pt (1.36 per 100k turns)".
     Fit gap (logs/fit_gap_9b64.log; 20k distill_union.pt rows per source; KL(labels || model) over legal
     moves, nats / top-1 agreement with the label's top move):
       source               H(labels)  union single pass   9x64 soft ep35    S1 alpha 1.0
       teacher games          0.642     0.042 / 92.4%      0.121 / 85.6%     0.134 / 84.6%
       epoch-4 games          0.728     0.065 / 91.4%      0.150 / 83.9%     0.161 / 83.4%
       crisis (A1,A2,A3,b)    ~0.86     ~0.12 / ~89.6%     ~0.21 / ~80.5%    ~0.22 / ~80.3%
       crisis offp_c w200     0.767     0.071 / 91.6%      0.154 / 83.4%     0.164 / 83.1%
     The student sits 7-9 agreement points below the teacher's single pass on every source (most on crisis
     rows) after 35 epochs on these very rows: a fit (capacity) limit, not generalization. The own-search
     turn moved S1 slightly away from the union's labels while cutting deaths 10.5%.

262. **9x64 bug hunt: nothing found; the 9x64 is parked and the 18x96 flywheel resumes (autonomous,
     800 sims).** (2026-09-30) Owner doubted the capacity reading (the 18x96 was called capacity-limited
     before and then improved ~10x after bug fixes). Checks, all clean:
     - fp16 / folded export fidelity (logs/fp16_fidelity.log, 20k game + 20k crisis rows, argmax vs fp32):
       student folded 99.77 / 99.73%, unfolded 99.71 / 99.67%; union folded 99.29 / 98.77%, unfolded
       99.29 / 98.75%. The student is not hurt by fp16 (its 63 rescaled BN channels vs the union's 550
       are normal for this architecture).
     - Recipe: identical to A1's (bs 32768, lr 3e-3, wd 1e-4, warmup 1, 40 ep, augment x8, ~4B samples)
       except size and soft targets. PolicyNet has no width-dependent constants (policy head 128 ch,
       pair_dim 64; 9 blocks see the whole board).
     - Flywheel teacher: S1 value head inner-val 0.0310 / fine-tune val 1.488 vs A2's 0.0270 / 1.686 at
       similar strength (A2 1.38).
     - Covariate shift (DAgger potential): 41 of S1's own recorded games labeled with the union's 8-view
       policy (230,864 rows, logs/own_state_gap_9b64.log): top-1 agreement by balls on board
       <40 / 40-54 / 55-64 / 65+: student on its OWN states 85.6 / 85.0 / 79.1 / 80.2%, on the teacher's
       states 85.7 / 85.4 / 78.8 / 79.6% (union single pass ~92 / 92 / 87 / 86-88%). No shift: DAgger
       would not add information.
     - Labels: verified in HISTORY 258 (the only bug, sum-vs-mean TTA temperature, fixed before training).
     The student's gap is a uniform ~7-point imitation deficit on every source and fullness. Decisive
     capacity test, not run: an 18x96 from scratch on distill_union.pt. Owner: park the 9x64, find the
     18x96's ceiling. Running: SIMS=800 ALPHAS="0.5 1.0 2.0" MIN_GAIN=0.05 TARGET=0 scripts/flywheel_loop.sh
     alphatrain/data/ta_union_abc_a1.0.pt 0.48 A 5 5000000 5100000 -> logs/flywheel_loop_A.log (A5 := the
     union, best 2k gate 0.48 vs A4 0.49; turn k seeds +500,000; stop when a turn cuts the rate < 5%).

263. **18x96 flywheel turn A5 (actor = the union, 800 sims): 0.48 -> 0.38 deaths per 100k turns (-21%,
     CIs disjoint). A stronger teacher broke the plateau that 3x the data at 600 sims could not
     (HISTORY 258). A6 = ta_A5_e4_a2.0.** (2026-09-30) scripts/flywheel_loop.sh (SIMS=800 ALPHAS="0.5 1.0
     2.0" MIN_GAIN=0.05 TARGET=0) alphatrain/data/ta_union_abc_a1.0.pt 0.48 A 5 5000000 5100000:
     own games (seeds 5,000,000+, 1k) 0.48 [0.43, 0.53], P1 4,544 / P10 38,601 / 62.4% capped (2,544 s at
     500 in flight; later turns record all 1,000 at once); survival head 27.6 min, inner-val 0.0147; mining
     3,000 epoch-4 probes -> 6,000 windows at 800 sims in 5,847 s (33,229 evals/s, 3.6x the 18x96's old
     ~9.3k); corpus 245,254 rows (anomaly B 0/20,000; C: the actor's move matches its 800-sim search on
     78.39% of crisis rows vs A1's 86.34% at 600 sims: the stronger teacher corrects more); fine-tune 4 ep
     (val 1.4953). Gates (2k, cap 100k turns, folded):
       model          P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A5 (union)   5,815 23,452  47,413 124,318  203,897  204,276  204,541  204,691  61.7%  0.48 [0.45, 0.52]
       alpha 0.5    5,568 22,020  47,206 132,202  203,934  204,294  204,518  204,668  64.8%  0.43 [0.40, 0.47]
       alpha 1.0    4,326 30,176  58,963 145,423  203,944  204,307  204,535  204,705  66.5%  0.41 [0.38, 0.44]
       alpha 2.0    6,750 25,161  52,397 150,986  203,957  204,273  204,504  204,630  68.1%  0.38 [0.36, 0.42]
       alpha 3.0    8,200 32,225  57,227 151,921  203,880  204,230  204,452  204,583  67.0%  0.40 [0.37, 0.43]
     MTBF 207,927 -> 259,857 turns. The loop promoted A6 = ta_A5_e4_a2.0 (alpha 3.0 statistically tied,
     best tail) and started turn A6 (seeds 5,500,000 / 5,600,000).

264. **Turn A6 (800 sims again) stalls: best alpha 0.5 = 0.37 vs the actor's 0.38 (-2.6%); alpha 2.0, the
     winner of turn A5, is worse (0.52). Same teacher strength twice -> the second turn finds nothing,
     as at 600 sims. Next: 1,600 sims.** (2026-09-30) Loop turn A6 from ta_A5_e4_a2.0 (seeds 5,500,000 /
     5,600,000): own games 0.37 [0.33, 0.41] (P1 4,974, 69.2% capped); survival head inner-val 0.0082;
     mining 6,000 windows in 5,806 s (32,769 evals/s); corpus 241,570 rows (B 0/20,000; C: actor matches
     its search on 76.82% of crisis rows); fine-tune train 1.2327 / val 1.5328 (A5: 1.1719 / 1.4960, the
     corrections are harder to fit). Gates (2k, cap 100k, folded):
       model          P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A6 actor     6,750 25,161  52,397 150,986  203,957  204,273  204,504  204,630  68.1%  0.38 [0.36, 0.42]
       alpha 0.5    5,588 27,986  56,355 155,380  203,976  204,273  204,519  204,667  69.1%  0.37 [0.34, 0.40]
       alpha 1.0    6,947 29,348  59,616 156,701  203,997  204,311  204,532  204,662  68.7%  0.37 [0.35, 0.41]
       alpha 2.0    3,491 19,274  40,489 110,320  204,582  204,963  205,244  205,417  59.7%  0.52 [0.48, 0.56]
     Loop: "STOP: gain 0.026 < 0.05. Best actor: alphatrain/data/ta_A6_e4_a0.5.pt".
     Fine-tune anatomy (scratchpad task_vector_anatomy.py): the largest element of BOTH turns' task
     vectors is 0.0415 = pair_dbias/pair_sbias, scalar biases that add one constant to every logit
     (softmax-invariant, no effect on play); a near-zero gradient of constant sign is normalized by
     Adam into a full step every update (sum of lr over the run). Otherwise A5 and A6 move the same
     tensors (last trunk blocks + head; norms 3.54 vs 3.86): no pipeline anomaly.
     Pattern: 600 sims gave A2->A3 -58%, A3->A4 -19%, then 3x data ~0; 800 sims gave A5 -21%, A6 -2.6%.
     Each teacher strength buys one or two turns. Running: SIMS=1600 loop from ta_A6_e4_a0.5 as A7
     (seeds 6,000,000 / 6,100,000).

265. **Why the flywheel stalls: the strong actor's deaths are 60-120-move SLIDES from a healthy board, and
     every crisis window ever mined starts 15/30 moves before death -- past the slide. A slide is a loss of
     line potential, and the survival head sees it but the leaf value hides it.** (2026-09-30, CPU-only
     analyses while turn A7 ran; owner: "we can't keep doubling sims ... explore different options")
     Death anatomy (scratchpad death_anatomy.py, logs/death_anatomy.log; recorded own games, last 300 moves
     of every death):
                                         A6 (actor)   A5      A3      epoch-4 (probe player)
       healthy mid-game balls mean/p99   32.9/47      33.1/48 33.3/48 37.0/56
       balls 300/150/100/45/15 before    34/38/43/53/63  same   34/38/42/52/62  37/41/44/51/62
       slide length (moves since <=45)   64 (p75 89)  58      54      52
       slide length (moves since <=40)   90 (p75 122) 84      77      82
     The slide lengthens as the actor improves (54 -> 58 -> 64), so a fixed late window sees less of it
     every turn. mcts_crisis anchors windows at death-15 (recovery) and death-30 (prevention): ~57 balls
     for A6, beyond the healthiest games' p99 of 47. No experiment so far (incl. the 200-move arm, which
     kept the late anchors) searched the 40-50-ball slide.
     Survival head sees slides (scratchpad value_sees_slide.py, A6 head on A6's records, AUC within
     fullness bins for "dies within H"): 46-50 balls H=100 0.87 / H=200 0.79; 41-45 balls H=200 0.74
     (P(survive 200) 0.978 doomed vs 0.991 recovering); 36-40 balls H=200 0.68. But the leaf value weights
     the horizons 25/50/100/200 by 1.0/0.8/0.5/0.25: in a slide P(survive 25/50) ~ 1 for every move, so
     move-to-move differences are ~0.003 and the search just echoes the prior.
     What a slide is (scratchpad slide_features.py, 35,933 doomed states 60-150 moves before an A5/A6
     death vs 1.07M recovering states >300 moves from death, AUC within fullness bins): same-colour runs
     of >= 3 AUC 0.19-0.22 (doomed 1.0-1.3 vs 1.5-1.8), longest run 0.18 (3.1 vs 3.5); empty-space
     fragments 0.40, movable balls 0.39-0.42, legal moves / colour mix ~0.5. Doomed boards have lost
     their partial lines: no setups -> no clears -> the board fills.
     Tooling: export_policy_value --horizon-weights (leaf value weights); scripts/flywheel_turn.sh takes
     REC_TURNS / PREV_TURNS / CONT_TURNS (window starts before death and length), HW (leaf weights) and
     RUN (arm name: windows, fine-tune and merges), reusing the actor's recorded games and survival head.
     Defaults reproduce the old turn exactly. Next GPU slot (after turn A7): slide arm on the same actor
     and probe deaths: RUN=A7s REC_TURNS=60 PREV_TURNS=120 CONT_TURNS=75 HW=0.1,0.2,1.0,1.25 SIMS=800.
     Own-death slide anchors: alphatrain/scripts/death_anchors.py --before 120 over greedy_A5/A6/A7_cap100k
     -> alphatrain/data/slide_anchors_A5A6A7_k120.txt (988 anchors, 40.7 balls mean, p10 31, p90 51);
     mcts_crisis --anchors-in replays them (flywheel_turn.sh ANCHORS=...), tested on CPU in both modes.
     1-ply preview (scratchpad one_ply_preview.py, logs/one_ply_preview_A6.log; A6 policy top-5 x 4 random
     spawn futures, afterstates scored by the A6 survival head; 800 slide states 60-150 moves before an
     own death vs 800 recovering states, 38-52 balls):
                                            slide (doomed)        recovering
       value spread over top-5, default     median 0.0076 (p90 0.104)   0.0008 (p90 0.009)
       value spread over top-5, slide W     median 0.0225 (p90 0.198)   0.0030 (p90 0.027)
       best afterstate != policy's move     59% (both weightings)       57%
       runs>=3 of that choice - policy's    +0.16                       +0.26
     The head separates moves ~10x more in slides than in recovering states, the slide weights triple
     that, and where it disagrees with the policy it keeps more partial lines. (1-ply, 4 futures: noisy.)

266. **Turn A7 at 1,600 sims (one-off experiment): 0.37 -> 0.32 deaths per 100k turns (-13.5%). Owner: no
     further sim doubling ("we won't get to infinite play if we have to keep doubling sims"); the slide arm
     (same actor, 800 sims) runs next for comparison.** (2026-09-30..10-01) Loop turn A7 from
     ta_A6_e4_a0.5 (seeds 6,000,000 / 6,100,000): own games 0.36 [0.32, 0.41] (P1 5,608, 69.5% capped);
     survival head inner-val 0.0106; mining 6,000 windows at 1,600 sims in 11,552 s (33,829 evals/s);
     corpus 247,143 rows (B 0/20,000; C: actor matches its 1,600-sim search on 75.47% of crisis rows).
     Gates (2k, cap 100k turns, folded):
       model          P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A7 actor     5,588 27,986  56,355 155,380  203,976  204,273  204,519  204,667  69.1%  0.37 [0.34, 0.40]
       alpha 0.5    8,747 33,983  69,153 174,996  204,029  204,334  204,572  204,676  71.8%  0.33 [0.30, 0.36]
       alpha 1.0    5,500 30,081  63,906 178,216  204,047  204,339  204,555  204,698  72.5%  0.32 [0.30, 0.35]
       alpha 2.0    4,395 28,917  66,709 176,101  204,072  204,354  204,594  204,735  70.8%  0.34 [0.32, 0.37]
     The loop promoted ta_A7_e4_a1.0 (A8 candidate) and started turn A8; scratchpad after_A7_slide_arm.sh
     stopped it within seconds, removed A8's export and records, and launched the slide arm A7s on the A7
     actor: 988 own-death anchors 120 moves before death, 120 searched moves, leaf weights 0.1,0.2,1.0,1.25,
     800 sims -> logs/slide_arm_A7s.log.

267. **Slide arm A7s (own deaths, 800 sims) matches the 1,600-sim turn for ~1/4 the mining: 0.37 -> 0.33
     deaths per 100k turns (A7 at 1,600 sims: 0.32). Summing both task vectors does not stack (0.34).**
     (2026-10-01) RUN=A7s ANCHORS=alphatrain/data/slide_anchors_A5A6A7_k120.txt PREV_TURNS=120
     CONT_TURNS=120 HW=0.1,0.2,1.0,1.25 SIMS=800 scripts/flywheel_turn.sh alphatrain/data/ta_A6_e4_a0.5.pt A7
     6000000 6100000 "0.5 1.0 2.0" (A7's actor, recorded games and survival head; pv exported with the slide
     weights): 988 own-death anchors, 120 moves before death, 120 searched moves -> 988 windows in 2,632 s
     (35,955 evals/s; A7's 6,000 crisis windows at 1,600 sims took 11,552 s); corpus 118,171 rows (B 0/20,000;
     C: the actor's move matches its slide search on 88.55% of rows, BN eval/train agreement 83.91%);
     fine-tune val 1.1459. Search escaped 984 of 988 slides (120 moves); the no-search control from the
     same anchors with the same replay spawns (anchor_search --search-turns 0, CPU, pv module) escaped 979:
     slides are not doomed positions (the original deaths needed bad spawns too) but they carry ~24x the
     actor's hazard (9 deaths in ~118k turns = ~7.6 per 100k turns; search: 4, ~3.4). Gates (2k, cap 100k):
       model                 P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A7 actor            5,588 27,986  56,355 155,380  203,976  204,273  204,519  204,667  69.1%  0.37 [0.34, 0.40]
       A7s alpha 0.5       8,064 31,314  69,098 171,094  203,984  204,293  204,539  204,702  70.3%  0.35 [0.32, 0.38]
       A7s alpha 1.0       8,309 29,804  67,520 171,179  203,982  204,303  204,527  204,671  71.7%  0.33 [0.31, 0.36]
       A7s alpha 2.0       5,711 28,782  53,566 146,466  203,921  204,234  204,490  204,607  67.5%  0.39 [0.36, 0.42]
       A7 alpha 1.0        5,500 30,081  63,906 178,216  204,047  204,339  204,555  204,698  72.5%  0.32 [0.30, 0.35]
       A7 + A7s, 1.0+1.0   9,270 34,243  64,444 172,227  204,018  204,328  204,564  204,684  71.4%  0.34 [0.31, 0.37]
     The crisis (A7) and slide (A7s) task vectors are nearly orthogonal (cosine 0.09, norms 3.58 / 2.36;
     alphatrain/scripts/merge_task_vectors.py) yet their sum is no better than either: no stacking at 1+1.
     Next: does the slide lever repeat? Turn A8 slide-only at 800 sims from ta_A7_e4_a1.0, anchors pooled
     from the A5-A8 own deaths.

268. **The slide lever does not repeat: turn A8s (slide-only, 800 sims) on A8 = ta_A7_e4_a1.0 gives 0.36 /
     0.31 / 0.35 at alpha 0.5 / 1.0 / 1.5 vs the actor's 0.32.** (2026-10-01) RUN=A8s SLIDE_DEATHS=<A5-A7
     records> PREV_TURNS=120 CONT_TURNS=120 HW=0.1,0.2,1.0,1.25 SIMS=800 scripts/flywheel_turn.sh
     alphatrain/data/ta_A7_e4_a1.0.pt A8 6500000 6600000 "0.5 1.0 1.5": own games (seeds 6,500,000+, 1k)
     0.34 [0.30, 0.38], P1 10,740, 70.9% capped; 1,279 own-death anchors (A5-A8); 1,279 windows in 3,632 s;
     corpus 153,112 rows (C: actor matches search 88.83%); fine-tune val 1.1106. Gates (2k, cap 100k):
       model            P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A8 actor       5,500 30,081  63,906 178,216  204,047  204,339  204,555  204,698  72.5%  0.32 [0.30, 0.35]
       A8s alpha 0.5  5,684 27,931  57,707 159,418  204,003  204,308  204,571  204,706  69.9%  0.36 [0.33, 0.39]
       A8s alpha 1.0  5,532 31,021  68,315 191,028  204,019  204,302  204,533  204,651  73.6%  0.31 [0.28, 0.34]
       A8s alpha 1.5  6,228 28,385  62,131 164,657  203,967  204,271  204,509  204,638  70.5%  0.35 [0.32, 0.38]
     With "no stacking" (HISTORY 267) this says the crisis (1,600 sims) and slide (800 sims) corrections
     fix the same marginal deaths: once A8 holds one, the other adds nothing. Both teachers share one
     judge of danger: the survival head, trained on the current actor's ~300 own deaths. Next: does a
     head trained on pooled own deaths (A3-A7, every 32nd state + full death tails;
     build_value_targets_from_records now takes several --games-dir) separate danger better on A8's games
     than A8's own head?

269. **More death data does not sharpen the survival head: a head trained on 4,000 pooled own games (A3-A7,
     every 32nd state + all death tails; 11.2M states) separates doomed from recovering boards on A8's games
     exactly as well as A8's own head trained on those very games.** (2026-10-01) value_head_A8_pooledA3A7.pt
     (train_value_head on ta_A7_e4_a1.0, 5 ep, inner-val 0.0340) vs value_head_A8.pt; scratchpad
     value_sees_slide.py on greedy_A8_cap100k (same 60k-state sample), AUC within fullness bins:
       balls     H=200 own / pooled     H=100 own / pooled
       36-40     0.701 / 0.694          -
       41-45     0.765 / 0.761          0.826 / 0.821
       46-50     0.717 / 0.725          0.830 / 0.836
       51-55     0.783 / 0.778          0.835 / 0.828
       56-62     0.810 / 0.817          0.847 / 0.862
     ~0.7-0.85 looks like the predictability of death 100-200 moves ahead from the board (with HISTORY
     267's control: 99% of slide positions survive under fresh spawns, so which ones die is mostly spawn
     luck). Survival is too rare an event to learn more from. Next: a DENSE danger target -- P(the board
     stays below ~55 balls for H moves) -- slides are ~100x more frequent than deaths; same 4-output head,
     so the fused export and the C++ search take it unchanged.

270. **The 8-view ensemble of the best actor dies 38% less than its single pass (0.21 vs 0.34 deaths per
     100k turns): a search-free teacher that renews with every actor. Calm (slide-risk) head: no better at
     predicting death. Status since A4: 0.49 -> 0.32 deaths per 100k turns (-35%).** (2026-10-01)
     Calm head (alphatrain/scripts/build_calm_targets_from_records.py --calm-balls 50: label = survive H
     AND stay below 50 balls; 3-6x denser than death) on A8's backbone, death AUC on A8's games (same
     sample as HISTORY 269) H=200: 0.709 / 0.764 / 0.698 / 0.731 / 0.698 over the fullness bins (survival
     head 0.701 / 0.765 / 0.717 / 0.783 / 0.810): no better. Three heads (own, pooled deaths, calm) hit the
     same ~0.7-0.85 ceiling; danger 100-200 moves ahead is mostly spawn luck.
     TTA headroom: eval_log run --model alphatrain/data/ta_A7_e4_a1.0.pt --tta 8 --max-turns 25000 (2k,
     seeds 2,600,000-2,601,999, folded): 102 deaths, 0.21 [0.17, 0.25] per 100k turns, MTBF 476,883,
     94.9% reach 25k; the single pass on the same seeds over its first 25k turns (survival_stats --cap
     25000 on the 100k gate CSV): 164 deaths, 0.34, MTBF 292,257. -38%, CIs disjoint.
     Best model: alphatrain/data/ta_A7_e4_a1.0.pt, 0.32 [0.30, 0.35] (MTBF 309,782, 72.5% capped; fresh
     seeds 0.34 [0.30, 0.38]) vs A4 0.49 [0.46, 0.53] (MTBF 202,343, 61.1%): P5 20,184 -> 30,081, P10
     41,715 -> 63,906, P25 118,015 -> 178,216.
     Running: scripts/tta_distill_turn.sh alphatrain/data/ta_A7_e4_a1.0.pt alphatrain/data/greedy_A8_cap100k
     A8t "0.5 1.0 2.0" (8-view labels of the actor's own recorded states, frozen-BN soft-CE fine-tune at lr
     3e-5 for 1 epoch, bs 4096, then the alpha sweep) -> logs/tta_distill_A8t.log.

271. **Warm-start 8-view self-distillation hurts (0.37 / 0.38 vs the actor's 0.32), as in HISTORY 239-240:
     fine-tuning a sharp model on its own ensemble drifts it more than it teaches.** (2026-10-02)
     scripts/tta_distill_turn.sh alphatrain/data/ta_A7_e4_a1.0.pt alphatrain/data/greedy_A8_cap100k A8t:
     10,733,498 own states labeled with the actor's 8-view average (alphatrain/data/tta8_A8t.pt; the
     ensemble's move differs from the recorded single-pass move on 7.7%), frozen-BN soft-CE fine-tune lr 3e-5,
     1 epoch, bs 4096 (val 0.6376). Gates (2k, cap 100k):
       model          P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       actor          5,500 30,081  63,906 178,216  204,047  204,339  204,555  204,698  72.5%  0.32 [0.30, 0.35]
       alpha 0.5      6,103 30,563  55,308 162,595  203,982  204,304  204,552  204,664  69.2%  0.37 [0.34, 0.40]
       alpha 1.0      4,436 22,765  52,688 156,793  203,927  204,257  204,483  204,642  68.5%  0.38 [0.35, 0.41]
     alpha 2.0 gate stopped (extrapolates the harmful direction). The ensemble's 38% headroom (HISTORY 270)
     stays on the table; the route that captured an ensemble before is born-again (HISTORY 241: a
     from-scratch PAIR2 18x96 on 8-view labels beat its teacher's single pass by 44%).

272. **Born-again corpus for a from-scratch 18x96 on the best actor's 8-view policy: 12,906,096 rows,
     Colab notebook ready.** (2026-10-02) scripts/prep_born_again.sh alphatrain/data/ta_A7_e4_a1.0.pt
     alphatrain/data/tta8_A8t.pt A8 <A5/A6/A7 crisis w45, A7s/A8s slide w120, distill_union_ep4>: own
     recorded games 10,733,498 (from HISTORY 271), crisis 733,967, slide 271,283, epoch-4 (human-level)
     game states 1,167,348, all relabeled with the actor's 8-view average (relabel_tta_tensor, mean logits,
     top-5 soft) -> alphatrain/data/born_again_A8.pt (2,064,979,169 B; .gz 873,544,987 B). Checks
     (logs/born_again_A8_check.log): top move legal 100% on every source, 5 moves per target, top-share
     0.765 own / 0.633 crisis / 0.726 slide / 0.711 epoch-4. Notebook
     alphatrain/train_scratch18b96_pair2_born_again_A8_colab.ipynb (tarball colorlines_pillar3d_v8.tar.gz;
     A1 recipe, 40 ep, bs 32768, lr 3e-3, PAIR2 + legal mask; TARGET soft (recommended: soft beat hard for
     the 9x64) or hard). Target: the ensemble's 0.21 deaths per 100k turns in a single pass.

273. **Sims scaling saturates: a 2,400-sim turn on A8 (itself the 1,600-sim product) gives 0.318 / 0.317 /
     0.314 deaths per 100k turns at alpha 0.5 / 1.0 / 1.5 vs the actor's 0.323 (-3% at best). 800 sims gave
     -21%, 1,600 -13.5%, 2,400 ~-3%. The death rate stays the right gate: the hazard is flat in every model.**
     (2026-10-02; owner: "do 1600 or 2400 sims ... better understanding if increasing sims helps")
     RUN=A8c SIMS=2400 scripts/flywheel_turn.sh alphatrain/data/ta_A7_e4_a1.0.pt A8 6500000 6700000
     "0.5 1.0 1.5" (A8's records + value head reused; pv_A8_ts.pt default weights): mining 6,000 windows
     in 16,671 s (35,200 evals/s); corpus 247,644 rows (B 0/20,000; C: the actor matches its 2,400-sim
     search on 74.56% of crisis rows: 800 sims 78.39%, 1,600 75.47%); fine-tune val 1.5554. Gates (2k, cap 100k):
       model          P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A8 actor     5,500 30,081  63,906 178,216  204,047  204,339  204,555  204,698  72.5%  0.323 [0.296, 0.351]
       alpha 0.5    6,647 35,133  69,046 185,845  204,013  204,308  204,522  204,655  72.8%  0.318 [0.292, 0.346]
       alpha 1.0    4,824 36,498  76,040 182,551  204,012  204,300  204,536  204,662  72.8%  0.317 [0.290, 0.344]
       alpha 1.5    9,350 35,226  71,274 187,645  203,958  204,242  204,467  204,585  73.0%  0.314 [0.288, 0.341]
     Metric check (owner: "is the death rate the right metric?"; scratchpad hazard_bands.py, deaths per
     100k turns by game age 0-5k / 5-20k / 20-50k / 50-100k): A4 0.46/0.53/0.48/0.49, A7 actor
     0.38/0.38/0.38/0.36, A8 actor 0.36/0.33/0.34/0.30, alpha 1.0 0.34/0.28/0.29/0.34, alpha 1.5
     0.24/0.34/0.30/0.32 -- flat within every model (all bands inside each other's CIs). Under a flat
     hazard the rate fixes the whole survival curve (S(100k) = e^-0.32 = 72.6% vs 72.5-73.0% observed;
     P5 = -ln 0.95 / rate ~ 32k points, P10 ~ 67k vs observed 30-36k / 64-76k), and uses all ~550
     deaths (CI +-8%) where P10 rests on the 200th-worst game (SE ~7%), P5 ~10%, P1 ~25%. The +10-20%
     at P5/P10 above is a lower 5-50k hazard offset by a higher 50-100k one, each within its band's CI
     (and the actor's own gate sits below its predicted P5): noise around one rate. Per-band hazards
     will accompany gate reports so a real age-dependent change is not pooled away; 4k seeds (a second
     2k bank, 2,602,000-2,603,999) for close calls -- not needed for this verdict (a 2% difference).
     Born-again (HISTORY 272) epoch 20 copied from Colab (val 0.6837; training continues): gating.
     Born-again epoch 20 (2k, cap 100k, folded): P1 8,691, P5 34,781, P10 65,500, P25 178,421, 72.2% capped,
     0.324 [0.298, 0.352] per 100k turns = the actor's single pass (0.323) at half training; hazard by age
     0.28/0.34/0.31/0.34 (flat). First 25k turns of the same games: actor 0.342 (164 deaths), born-again
     0.316 (152), the 8-view ensemble 0.210 (102). Later epochs pending from Colab.

274. **Gemini review (owner-shared, 2026-10-02): verified points, corrections, fixes.** Corrections to my
     claims: (1) HISTORY 241 does not show born-again capturing an ensemble: the same-head born-again
     control lost 17% (10,361 vs 12,437 at ep27); the +44% came from the PAIR2 head. (2) "danger is
     ~70-80% predictable (spawn luck)" (HISTORY 269-270) was premature: all three heads compared were the
     same 3,268-param ValueHead (1x1 conv 96->32, global average pool, linear) -- the head, not luck, is
     the first suspect. (3) prep_born_again relabeled the crisis rows with the search-free ensemble,
     discarding the MCTS labels, in a corpus that is 83% quiet own states. Verified in code: ValueHead
     GAP (value_head.py:42-73); SpatialValueHead (3x3 residual blocks + mean/max pool) existed but
     train_value_head hard-coded ValueHead; export_policy_value skipped fp16_safe_batchnorm (A8 max
     backbone_bn running_var 59,666 of fp16's 65,504; e40 once hit 100,447); MCTS min-max Q normalization
     with q_range_floor 0 (mcts.cc:187, mcts.h:40); --decisiveness-power / --set-loss-on-mask / --ema-decay
     exist in train_path_b; preview balls encoded as color/7 in three ordered channels (observation.py:147).
     Fixed (commit 991fa0f): fp16-safe BatchNorm in export_policy_value and evaluate.load_model;
     train_value_head --arch spatial (save_spatial records the survival target + horizons; frozen features
     under no_grad -- inference tensors broke CPU training for either head); the export takes spatial
     survival heads. Smoke-tested on CPU; 53 tests pass. Reservations: a q_range_floor only acts when all
     leaves agree, so it does not address noise extremes stretching the range; 8-view averaging inside the
     teacher costs 8x the network compute (like the sims scaling the owner ruled out); "D4-equivariance by
     averaging each 3x3 kernel over D4" makes every filter isotropic (3 free weights), unable to tell
     horizontal from vertical lines -- true equivariance needs group convolutions. Next: SpatialValueHead on
     the pooled A3-A7 targets, death AUC on A8's games vs the GAP heads (queued after the epoch-30 gate);
     decisiveness-weighted fine-tune on mined corpora; checkpoint-averaged born-again once 35/40 land.

275. **Born-again stays at the actor's level (ep30 0.344); a SPATIAL value head does not break the danger
     ceiling either (Gemini's GAP-head hypothesis falsified on the frozen backbone).** (2026-10-02)
     Born-again gates (2k, cap 100k, folded): ep20 0.324 [0.298, 0.352], ep25 0.457 [0.425, 0.492] (hazard
     up in every age band: mid-schedule checkpoint), ep30 0.344 [0.317, 0.373] (P1 6,548, P5 35,595, P10
     67,217, P25 171,361, 70.8% capped; hazard by age 0.35/0.29/0.37/0.35). Owner stopped Colab at ep35
     (val 0.6731); gating ep35 and a 30+35 weight average with BN re-estimated (average_checkpoints.py
     --recalib-tensor born_again_A8.pt).
     SpatialValueHead (train_value_head --arch spatial, 162,756 params: 1x1 conv + two 3x3 residual blocks
     + mean/max pool + MLP) on the pooled A3-A7 survival targets, A8 backbone: best inner-val 0.0346 (GAP
     pooled head 0.0340; later epochs overfit: train 0.019, inner-val 0.054). Death AUC on A8's games
     (same 60k sample; scratchpad value_sees_slide.py), H=200 / H=100 by balls 36-40, 41-45, 46-50, 51-55,
     56-62: spatial 0.677/0.755/0.722/0.783/0.823 and -/0.812/0.825/0.836/0.854 vs the GAP own head
     0.701/0.765/0.717/0.783/0.810 and 0.826/0.830/0.835/0.847 and the GAP pooled head
     0.694/0.761/0.725/0.778/0.817 and 0.821/0.836/0.828/0.862 -- all within +-0.02 (32-65 doomed states
     per bin). Head size, death data and target density all leave danger detection unchanged on the frozen
     policy backbone; untested: an end-to-end value network. Queued: decisiveness-weighted fine-tune A/B
     (same A8c corpus, --decisiveness-power 2.0 --blend-alpha 0.5; scratchpad decisive_ab.sh).
     Born-again final (2026-10-03): ep35 (val 0.6731; owner stopped Colab here) P1 8,116, P5 36,435, P10
     74,442, P25 197,602, 74.0% capped, 0.301 [0.276, 0.328] per 100k turns (MTBF 332,259; by age
     0.28/0.30/0.32/0.29) -- the lowest rate measured, -7% vs the actor, inside the CIs at 2k. Uniform
     average of ep30+35 with BN re-estimated on born_again_A8.pt (augmented): 0.333 [0.307, 0.362], early
     hazard 0.54 (53 deaths in 0-5k turns vs ep35's 28) -- BN re-estimation hurts again (as HISTORY 239).
     The late, low-LR epochs improved the run (ep30 0.344 -> ep35 0.301): run Colab arms to the end.
     Next: confirm ep35 vs the actor on a second 2k bank (seeds 2,602,000-2,603,999 -> 4k each).

276. **Decisiveness-weighted crisis fine-tune (Gemini Step 2) hurts: same 2,400-sim corpus and recipe as turn
     A8c, only the loss changed (--decisiveness-power 2.0 --blend-alpha 0.5): 0.350 / 0.380 at alpha 0.5 /
     1.0 vs the unweighted 0.318 / 0.317.** (2026-10-03) scratchpad decisive_ab.sh: A8 fine-tuned 4 ep on
     alphatrain/data/A8c_crisis_w45.pt (frozen BN, lr 1e-4, T 0.5, augment 1; val 1.6803), ta_sweep:
       model          P1     P5     P10     P25      P50      P75      P90      P95   capped  per 100k turns
       A8d alpha 0.5  9,061 28,968  60,471 166,450  203,993  204,283  204,535  204,662  70.5%  0.350 [0.323, 0.380]
       A8d alpha 1.0  5,961 28,795  50,788 149,008  203,878  204,204  204,450  204,587  68.8%  0.380 [0.35, 0.41]
     (alpha 1.5 gate stopped: the trend worsens with alpha). Concentrating the gradient on decisive search
     states makes play worse, not better; the flat-state soft targets carry useful information.
     Running: 4k confirmation of born-again ep35 vs the actor (bank 2 = seeds 2,602,000-2,603,999).
