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
