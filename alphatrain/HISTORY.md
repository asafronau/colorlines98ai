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

**Fix:** Reverted MCTS stack to last known-good commit (87e2d5f). The value head
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
