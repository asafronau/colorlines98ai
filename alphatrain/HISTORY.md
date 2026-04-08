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

### Pillar 2k: Heavy Value Head + Adversarial Ranking (NEXT)
Fixing "Value Hallucination" with bigger value head, adversarial training, and dropout.

**Changes from 2j:**
| Component | Pillar 2j | Pillar 2k |
|---|---|---|
| value_conv | 8 channels | **32 channels** |
| value_fc1 | 648→256 | **2,592→512** |
| Dropout | None | **0.3** |
| Value head params | 183K | **1.37M (7.5x)** |
| Ranking pairs | top-1 vs top-5 | **top-1 vs random move** |
| Total model | 12.1M | **13.3M** |

**Why each change:**
1. **Bigger head (183K→1.37M):** 8-channel conv bottleneck squeezes 256 backbone channels
   to 8 before FC layers — value head couldn't see enough features. 32 channels + 512
   hidden gives 7.5x more capacity to learn geometric patterns.
2. **Adversarial ranking:** top-1 vs top-5 pairs compare two GOOD moves. The value head
   never learned what "bad" looks like. top-1 vs random move forces it to distinguish
   master play from random play — teaching the "strategic floor."
3. **Dropout 0.3:** Epoch 2 overfitting with 12.8M states indicates memorization.
   Dropout forces the value head to learn generalizable features instead of specific boards.

**Success criterion:** MCTS must NOT regress when increasing from 400 to 1,600 sims.
If more search helps, the value head is trustworthy.

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
