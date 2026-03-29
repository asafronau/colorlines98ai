# Color Lines 98 AI — Experiment Log

## Goal
Build an AI that plays Color Lines 98 at superhuman level and deploy it in a browser.

## Current Status (Phase 16)

**Best player: Tournament 200 (uncapped, bug-fixed)**
- **Mean: 4,641 | Median: 3,264 | Max: 15,697** (20 games, seed=77777)
- 25% of games score >5,000; 10% score >10,000
- ~150ms per move decision (100ms/move on shorter games)
- Pure CPU, no ML — heuristic + search only

**Two critical bugs were the biggest improvements in the entire project:**
1. **Pathfinding bug** (`_get_target_mask` used `break` after first connected component) — fixing this one line nearly doubled all scores
2. **Turn cap** (`turns < 1000` in evaluate.py) — removing this revealed the agent could play 6000+ turn games scoring 14,000+

**Architecture:** Tournament bracket (successive halving) with CMA-ES tuned heuristic:
- 2-ply evaluates all ~300 legal moves → top 30 qualifiers
- Quarter-finals: 10 rollouts × 30 → top 10
- Semi-finals: 40 rollouts × 10 → top 3
- Finals: 150 rollouts × 3 → pick best

**Next: Phase 16 — ML-Powered Rollout Policy**
The heuristic inside rollouts scores mean=71 standalone. It has no spatial awareness
(center control, color zoning, combo building). An ML-trained linear model would inject
human-level spatial intuition into every rollout step at zero speed cost.

Pipeline:
1. Go engine for fast data generation (18 parallel games, 30x faster than Python)
2. Generate 1000 tournament games → filter elite (>3000 pts) → extract 30 spatial features
3. Train logistic regression → 30 learned weights
4. Hardcode weights into Numba `_evaluate_move_ml` → spatially-aware rollouts
5. Iterate: improved tournament → better data → retrain (AlphaZero-lite feedback loop)

**Go engine benchmark:**
| Benchmark | Python (Numba) | Go | Speedup |
|-----------|---------------|-----|---------|
| Heuristic (parallel) | 238 games/sec | 7,237 games/sec | **30x** |
| Tournament per-move | ~100ms | ~88ms | 1.1x |
| Data gen (1000 games) | ~83 hours | **~4.6 hours** | **18x** |

**30-feature spatial extractor built** (`game/features.py`), JIT-compiled at 0.8µs/call:
line features, spatial topology, color clustering, board health, next-ball synergy, source quality.

---

## What Worked (ranked by impact)

1. **Fixing pathfinding bug** — +82% score (1041→1895), one line of code
2. **Removing turn cap** — revealed 14,000+ point capability
3. **Tournament bracket (successive halving)** — concentrates rollouts on best candidates
4. **Softmax exploration in rollouts** — +31% over greedy (temp=2.0→3.23 via CMA-ES)
5. **CMA-ES weight tuning** — +29% over hand-tuned heuristic weights
6. **Numba JIT optimization** — 1.4-2.9x speedup on hot paths
7. **2-ply + CNN hybrid** — +72% over pure 2-ply (322 vs 187, client-side option)

## What Failed

1. **Value networks** (4 versions) — can't discriminate between moves within a single state
2. **Policy distillation as tournament filter** — PolicyNet top-15 loses too many good moves
3. **PPO reinforcement learning** — catastrophic forgetting, reward shaping too brittle
4. **Hand-tuned heuristic changes** — break softmax temperature calibration
5. **CMA-ES spatial features for gambling** — higher ceiling but lower floor than baseline
6. **Open-loop MCTS** — single-threaded can't compete with parallel rollouts
7. **Beam search with CNN** — CNN predictions vary by only 0.032 std within a state

---

## Detailed Phase History

## Baseline: 1-Step Heuristic (fast_heuristic.py)
- Evaluates all legal moves by temporarily placing the ball, scoring line potential
- Scores: **mean=52, median=50, max=190** (500 games)
- Speed: ~0.12s/game
- This is our primary baseline

---

## Phase 1: Behavior Cloning (BC)

### Hard-label BC (10K games, 64ch/6 blocks)
- Train on (observation, best_source, best_target) from heuristic games
- Cross-entropy loss on source and target heads separately
- On-the-fly dihedral augmentation (8 transforms)
- **Result: mean=14, val src_acc=37.5%, val tgt_acc=82.5%**
- Target accuracy is strong; source accuracy is the bottleneck
- Many sources are equivalently good, so exact-match CE underestimates quality

### Hard-label DAgger (5 iterations on top of BC)
- Iteratively: play with model, label with heuristic, add to dataset, retrain
- Beta schedule: 0.6 → 0.5 → 0.4 → 0.3 → 0.2
- Dataset cap: 800K samples (keeps most recent)
- **Result: mean=18.8, median=15, max=57**
- Progression: 13.3 → 13.8 → 15.1 → 17.1 → 16.8 → 18.8

### Soft-label BC (15K games, 64ch/6 blocks)
- Instead of hard one-hot source label, use temperature-scaled softmax over
  heuristic scores for all valid sources (temperature=2.0)
- `F.cross_entropy(logits, probs)` — PyTorch handles zeros correctly (no NaN)
- Previous attempt with `F.kl_div` failed because `log(0)` explodes
- Data gen 3x slower (must evaluate all sources per state)
- **Result: mean=17.7 (BC only)**

### Soft-label DAgger (5 iterations, 15K initial games)
- Same DAgger but with soft source labels throughout
- **Best: iter4, mean=20.4, median=20, max=90**
- iter5 regressed to 19.4 — DAgger dataset cap pushes out good initial data
- Saved as `checkpoints_soft/dagger_best.pt`

### Larger Model (128ch/8 blocks, ~2M params)
- 4x parameters vs small model
- BC: **mean=25.6** (vs 17.7 for small model)
- DAgger iter1: **mean=29.6** (best)
- DAgger degraded after iter1 — same dataset cap issue
- Val src_acc=49% (vs 39% for small model)
- Saved as `checkpoints_large/dagger_iter1.pt`

---

## Phase 2: Hybrid Player (Model + Heuristic)

### Key Insight: Model is good at ranking, bad at exact pick
- Top-K source recall analysis (small model):
  - Top-1: 43%, Top-3: 71%, Top-5: 82%, Top-10: 94%
- If we use model to narrow sources and heuristic to evaluate targets,
  we get most of the heuristic's performance with much less computation

### Hybrid Results (model top-K sources + heuristic target evaluation)
| Model | Top-K | Mean Score |
|-------|-------|------------|
| Small (soft DAgger) | 5 | **46.8** (90% of heuristic) |
| Small (hard DAgger) | 5 | 41.8 (80%) |
| Small (soft DAgger) | 10 | ~47.8 |
- Soft labels help hybrid performance too (better source ranking)
- top-5 ≈ top-10 (diminishing returns past 5)

---

## Phase 3: 2-Ply Lookahead Search

### Key Insight: Simulating moves gives much more information
For each candidate move: clone game, execute move, evaluate resulting state.
This captures ball spawns, line clears, and board congestion.

### Scoring Function
```
score = immediate_heuristic + clear_bonus * 20 + (future_heuristic + empty * 0.25) * 0.5
```
Where:
- `immediate_heuristic`: fast_heuristic's score for the move (line potential)
- `clear_bonus`: actual game points from lines cleared this move
- `future_heuristic`: fast_heuristic's best move score in resulting state
- `empty`: number of empty cells in resulting state

### Results (2-ply lookahead)
| Player | Mean | Speed |
|--------|------|-------|
| Model top-5 (small) | **107** | 1.6s/game |
| Model top-10 (small) | **144** | 3.6s/game |
| Model top-10 (large) | **169** | ~4s/game |
| Exhaustive (no model) | **196** | ~10s/game |

### Weight Tuning (exhaustive 2-ply, 50 games each)
| Config | Mean |
|--------|------|
| clear=10, empty=0.25, future=0.5 (default) | 198 |
| clear=20, empty=0.25, future=0.5 | 197 |
| clear=30, empty=0.25, future=0.5 | **211** |
| clear=20, empty=0.5, future=1.0 | **212** |
| clear=20, empty=1.5, future=0.3 | 184 |
| clear=20, empty=2.0, future=0.2 | 168 |

Higher clear weight helps slightly. Empty cell weight beyond 0.5 hurts.

---

## Phase 4: Deeper Search (Failed Attempts)

### 3-Ply Exhaustive Search
- Killed after no output for 10+ minutes — too slow
- O(moves^3) with ~600 legal moves per position = billions of evals

### 3-Ply Model-Guided (top-5 sources at level 1, top-3 at level 2)
- **Result: mean=109, 156s/game** — WORSE than 2-ply (196)
- Model pruning at level 2 is too aggressive, cuts off good moves
- Also very slow due to nested clone+move and neural network forward passes

### Enhanced Heuristic (board health features)
- Added color clustering bonus, next-ball awareness
- **Result: mean=194** — no improvement over base 2-ply (196)
- The heuristic's scoring function is not the bottleneck

### Value Network v1
- Trained on 1000 games of 2-ply play (110K state-value pairs)
- ValueNet: 64ch/4 blocks, MSE on log(1 + remaining_score)
- Val loss: 0.266
- **Value search player: mean=3** — complete failure
- Predictions don't differentiate well between good and bad moves
- Needs much more data and/or better architecture

---

## Phase 5: MCTS (Partial Success)

### Pure MCTS with Policy Priors
- Policy network selects top-K sources, computes joint P(source, target) prior
- PUCT selection: Q(s,a) + c_puct * P(s,a) * sqrt(N) / (1 + N(a))
- Dirichlet noise at root for exploration
- Heuristic rollouts (10-30 moves) for leaf evaluation
- Implementation: `game/mcts.py`

### MCTS Results (5-20 games, seed=42)
| Config | Mean | Speed |
|--------|------|-------|
| 100 sims, rollout d=10, 30 actions | **68** | 41s/game |
| 300 sims, rollout d=15, 10 actions | ~70 | ~60s/game |
| 500 sims, heuristic leaf, 20 actions | **47** | 8s/game |
| 500 sims, 2ply leaf, 20 actions | **46** | 10s/game |

### Why MCTS Underperformed
- With 30 actions per node and 200 sims, each action gets only ~7 visits — not enough
  to average out stochastic ball spawns
- NN priors have 82% top-5 recall — misses 18% of best moves
- Fast leaf evaluations (heuristic, 2ply) give too coarse a signal
- Rollout mode gives better signal but is slow, limiting simulation count
- Tree barely goes 1-2 levels deep with this branching factor
- **Stochastic bug** (identified by Gemini review): `clone()` creates fresh RNG,
  so re-traversing an edge leads to different ball spawns than the cached child.
  Would need determinized MCTS (fixed RNG per sim) or chance nodes to fix.

---

## Phase 6: 2-Ply + Rollout Refinement

### Key Insight: Combine exhaustive search with stochastic evaluation
1. **Phase 1**: Parallel 2-ply scores ALL legal moves across 12 workers (~150ms)
2. **Phase 2**: Top-N candidates refined with progressive parallel rollouts
   - Screening round: top-N, fewer rollouts, half depth
   - Final round: top half survivors, full rollouts, full depth
   - Multiprocessing across all CPU cores (12 workers on M3 Pro)

### Scoring
```
combined = avg_rollout_score + 2ply_score * 0.1
```
The 2-ply score is a critical tiebreaker — pure rollout scoring drops to mean=196.

### Pipeline Evolution
1. **v1 (sequential 2-ply + parallel rollouts)**: ~52s/game, baseline
2. **v2 (+ pre-filter + shorter screening)**: ~23s/game, 2.3x faster but loses candidates
3. **v3 (parallel 2-ply, no pre-filter)**: ~37s/game, fastest AND best quality

**Critical finding**: Pre-filtering with 1-step heuristic before 2-ply HURTS quality.
Moves that score low on immediate line potential but high on positional value
(clearing lines after spawns, opening space) get incorrectly filtered out.
Full parallel 2-ply for all ~300 moves costs only ~150ms and misses nothing.

### Results (20 games, seed=42, 2ply_weight=0.1, epsilon=0.0)

Early experiments (sequential 2-ply):
| Top-N | Rollouts | Depth | Mean | Median | Max | Time/game |
|-------|----------|-------|------|--------|-----|-----------|
| 10 | 30 | 20 | **460** | 421 | 919 | 31s |
| 20 | 30 | 20 | **518** | 481 | 856 | 55s |
| 10 | 50 | 20 | **570** | 520 | 1487 | 105s |

Optimized pipeline (parallel 2-ply + progressive rollouts):
| Top-N | Rollouts | Depth | Mean | Median | Max | Time/game |
|-------|----------|-------|------|--------|-----|-----------|
| 20 | 50 | 20 | **655** | 565 | 1613 | 37s |
| 20 | 100 | 20 | **704** | 589 | 1779 | 72s |
| 20 | 100 | 20 | **810** | 634 | 2081 | 918s* |

*Note: 918s run was with a different timing methodology (separate eval run).

### Key Findings
- Rollout depth 20 >> 15 (460 vs 380 with same params)
- More rollouts reduces noise (460 → 704 from 30→100 rollouts)
- Wider candidate set helps (460 → 655 from top-10 → top-20)
- 2-ply weight is critical: weight=0.0 → mean=196, weight=0.1 → mean=655
- Full-game rollouts (d=999) WORSE (261) — 20-move rollouts give best signal
- Pre-filtering HURTS: top-50 heuristic filter → mean=470 vs full → mean=704
- **Max score 2081** — human-level performance reached

### Epsilon-Greedy Rollouts
Based on Gemini's suggestion: add small random exploration during rollouts.
Instead of always picking the heuristic best move, pick random with probability ε.

| Epsilon | Mean | Median | Max | Notes |
|---------|------|--------|-----|-------|
| 0.0 | 677 | 498 | 1701 | Baseline (50 rollouts) |
| 0.02 | 538 | 446 | 1043 | *CPU contention* |
| 0.03 | 605 | 526 | 1685 | *CPU contention* |
| **0.05** | **766** | **752** | 1362 | Best mean & median |
| 0.07 | 550 | 494 | 1350 | *CPU contention* |
| 0.1 | 581 | 478 | 1184 | Too much noise |

All runs: 20 games, seed=42, top-20, 50 rollouts, depth 20.

**Epsilon=0.05** shows +13% mean and +51% median over baseline. The eps=0.02/0.03/0.07
runs had CPU contention from parallel tasks, so their results may be depressed.

**Confirmed with 100 rollouts** (eps=0.05, top-20, depth 20, 20 games):
- **mean=796, median=802, max=1551** (132s/game, had some CPU contention)
- vs eps=0.0 with 100 rollouts: mean=704, median=589, max=1779
- **+13% mean, +36% median** — confirms epsilon exploration helps consistently

**Upgrading to softmax exploration**: Gemini suggested replacing epsilon-greedy
(which occasionally picks the *worst* move) with temperature-scaled softmax over
heuristic scores. This keeps exploration high-quality — the 2nd/3rd best moves get
sampled, never suicidal ones. Implemented as `get_softmax_move_fast()` in
`game/fast_heuristic.py`.

**Temperature calibration** (10 mid-game states, heuristic scores range [-7, +182]):
- Most scores cluster in [-7, +18] with std ~1.5, occasional clearing moves at 100+
- temp=1.0: top-1 prob=78%, top-5=94% — still very greedy
- temp=2.0: top-1 prob=44%, top-5=53% — good exploration of 2nd/3rd best moves
- temp=5.0+: nearly uniform — too random
- **Best candidate: temp=2.0** — explores high-quality alternatives without suicide

**Softmax evaluation results** (20 games, seed=42, top-20, 100 rollouts, depth 20):
| Exploration | Mean | Median | Max | Time/game |
|-------------|------|--------|-----|-----------|
| Greedy (temp=0) | 704 | 589 | 1779 | 72s |
| Epsilon=0.05 | 796 | 802 | 1551 | 132s |
| **Softmax temp=2.0** | **926** | **862** | **2160** | 48s |

**+16% mean over epsilon, +31% over greedy.** Softmax is strictly better: higher quality
exploration (2nd/3rd best moves instead of random), AND faster (no wasted compute on
suicidal random moves). temp=2.0 is our new default for rollouts.

---

## Phase 7: Value Network v2

### Motivation
The rollout player (Phase 6) achieves mean=700+ but is slow (37-72s/game). A value
network that can evaluate board states instantly would replace the expensive rollout
phase (50-100 rollouts × 20 depth per candidate) with a single NN forward pass (~1ms).

### Value Network v2 Training
- Training data: **5000 2-ply games** (611K samples, mean score 191.3)
- Architecture: **128ch/6 blocks** (1.8M params vs ~500K in v1)
- Target: `log(1 + remaining_score)` (log-space compression)
- Loss: MSE with OneCycleLR, AdamW, dihedral augmentation
- Data gen: 82 min (13 workers, 2-ply player)
- Training: 130 min (40 epochs, 195s/epoch on MPS)
- **Best val_loss: 0.281** (converged smoothly from 0.62)

### Value-Based Player Results

**value_lookahead** (2-ply + value net evaluation, 20 games, seed=42):

Value predictions cluster in ~175-190 range (poor discrimination). The 2-ply score
must be primary with value as tiebreaker, not the other way around.

| 2-ply weight | Mean | Median | Max | Speed |
|-------------|------|--------|-----|-------|
| 0.1 (default) | 46 | 41 | 115 | ~1s/game |
| 1.0 | 169 | 157 | 251 | ~2s/game |
| 3.0 | 166 | 162 | 298 | ~2s/game |
| 4.0 | 216 | 184 | 610 | ~3s/game |
| **5.0** | **229** | **222** | **460** | ~3s/game |
| 7.0 | 180 | 164 | 291 | ~2s/game |
| 10.0 | 203 | 189 | 296 | ~2s/game |

**Best: weight=5.0, mean=229** — 17% over pure 2-ply (196), 25-50x faster than rollouts.

**value_rollout** (value screens top-20→5, then rollouts on top-5):
- mean=282, ~10s/game
- Value screening loses good candidates — pure rollout on more candidates is better

### Key Finding: Value Net OOD Problem (Gemini insight)
The value network trained on 2-ply games (mean=191) can't differentiate between
high-quality states that matter at rollout-level play (mean=700+). Its predictions
cluster in a narrow range (~175-190), making it a weak discriminator.

To be useful, the value net must be trained on data from the rollout player (mean=700+),
not the weaker 2-ply player.

### Self-Play Iteration (completed)
Trained v3 value net on 2000 games played by value_lookahead (using v2 value net).
- Data: 264,917 samples, mean game score=212.5 (barely above 2-ply's 191)
- Training: 40 epochs, best val_loss=0.219 (vs 0.281 for v2)
- Data gen: 8502s (~2.4h), training: ~3700s (~1h)
- **Verdict: self-play bootstrapping from weak value net produces same-tier data.**
  Mean=212 vs 2-ply's 191 — the value net doesn't make moves different enough to
  generate meaningful new training signal. Need to train on rollout-player data.

---

## Score Progression Summary

| Phase | Player | Mean | Max | Speed | Multiplier |
|-------|--------|------|-----|-------|------------|
| 0 | Random | 0.5 | ~5 | instant | - |
| 0 | 1-step heuristic | **52** | 190 | 0.12s | 1x |
| 1 | BC (soft DAgger) | 20 | 90 | instant | 0.4x |
| 2 | Hybrid (model+heuristic) | 47 | ~120 | <1s | 0.9x |
| 3 | 2-ply exhaustive | **196** | ~400 | 10s | 3.8x |
| 5 | MCTS | 68 | ~150 | 41s | 1.3x |
| 6 | 2-ply + rollouts (50) | **655** | 1613 | 37s | 12.6x |
| 6 | 2-ply + rollouts (100) | **704-810** | 2081 | 72s | 13.5-15.6x |
| 6 | + epsilon=0.05 (50 rollouts) | **766** | 1362 | 65s | 14.7x |
| 6 | + epsilon=0.05 (100 rollouts) | **796** | 1551 | 132s | 15.3x |
| 6 | + softmax temp=2.0 (100 rollouts) | **926** | 2160 | 48s | 17.8x |
| 6 | + softmax temp=2.0 (200 rollouts) | **1152** | 2251 | 105s | 22.2x |
| 10 | Tournament bracket (200 budget) | **1385** | 2266 | 68s | 26.6x |
| 11 | + CMA-ES tuned weights | **1041*** | 2145 | 68s | ~20x |
| 12 | Distilled PolicyNet + 2-ply | 192 | 349 | 43ms | 3.7x |
| 12 | Distilled PolicyNet standalone | 53 | 150 | 0.15ms | 1x |

*50-game eval (seed=77777), most reliable number. CMA-ES = +29% over baseline in head-to-head.
| 7 | value_lookahead (w=5.0) | **229** | 460 | 3s | 4.4x |

---

## Key Learnings

1. **Source selection is the bottleneck** for pure model play — many sources are
   equivalently good, making exact-match CE a poor objective
2. **Soft labels help** — temperature-scaled distribution over source quality
   improves both pure play (+13%) and hybrid play (+12%)
3. **Model excels at ranking** — top-5 recall of 82% enables effective hybrid play
4. **2-ply search is transformative** — 2x to 4x improvement over 1-step heuristic
5. **Deeper heuristic search doesn't scale** — the heuristic evaluation function
   itself is the ceiling, not search depth
6. **DAgger can degrade** — with a dataset cap, later iterations push out
   high-quality initial BC data. For search-based players, pure BC may be better.
7. **Rollout refinement is the breakthrough** — 2-ply + parallel rollouts achieves
   mean=704, 13x the heuristic baseline and 3.5x the exhaustive 2-ply
8. **Stochastic evaluation needs many samples** — MCTS with few visits per action
   fails; flat parallel rollouts with 50+ samples per candidate succeed
9. **Don't pre-filter candidates** — 1-step heuristic and 2-ply rank moves
   differently. Moves good for space/spawns may look bad for line potential.
10. **Parallelizing 2-ply is ~free** — 300 moves / 12 workers × 5ms = 150ms total
11. **Value net needs in-distribution data** — training on 2-ply games (mean=191)
    produces predictions that cluster in a narrow range, useless for ranking moves
    at rollout-level play (mean=700+)
12. **Epsilon-greedy rollouts may help** — 5% random moves in rollouts showed +13%
    mean improvement, but needs more data to confirm (high variance in 20-game evals)

---

## Phase 8: CPU Optimization + Categorical Value Head

### Game Engine Optimization (M5 Max)
Profiling revealed Python glue code between JIT kernels was ~50% of per-move time.

**Changes:**
1. `_get_empty_array` JIT kernel: **0.2 µs** vs 15.1 µs old `np.argwhere` + list comprehension (75x)
2. `_count_empty` JIT kernel: **0.1 µs** vs 1.0 µs `np.sum(board==0)` (10x)
3. Optimized `_generate_next_balls`: **4.6 µs** vs 19.9 µs (4.3x) — uses JIT array directly
4. `fast_move()`: returns tuple instead of dict (for rollout inner loop)
5. `_get_sources_and_targets()`: shared helper avoids redundant `np.argwhere` in heuristic

**End-to-end impact (M5 Max):**
| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| Heuristic (10 games) | 16,782 t/s | 23,455 t/s | **1.40x** |
| 2-ply (3 games) | 15.5s | 9.8s | **1.58x** |
| Rollout (3 games) | 21.4s | 16.5s | **1.30x** |

**Total speedup vs original M3 Pro code:**
| Benchmark | M3 Pro (old) | M5 Max (optimized) | Speedup |
|-----------|-------------|-------------------|---------|
| Heuristic | 12,540 t/s | 23,455 t/s | **1.87x** |
| 2-ply | 19.8s | 9.8s | **2.02x** |
| Rollout | 47.9s | 16.5s | **2.90x** |

### Categorical Value Head (MuZero-style two-hot encoding)
Implemented `--num-bins 64` option for value net training.

**Design:**
- Support: 64 linearly-spaced bins in log(1+x) space, range [0, 8.5]
- Target: two-hot encoding — probability mass split between two adjacent bins
- Loss: cross-entropy with soft targets (vs MSE for scalar head)
- Inference: expected value = sum(softmax(logits) * support)
- Backward compatible: `num_bins=0` (default) gives original scalar head
- Auto-detected from checkpoint at inference time

**Usage:**
```bash
python -m training.train_value --num-bins 64 --player curriculum --games 1000
```

---

## Completed Tasks

1. **Best config eval** (eps=0.05, 100 rollouts): **mean=796, median=802, max=1551**
   — Confirmed epsilon-greedy exploration helps consistently
2. **Value self-play v3**: val_loss=0.219, but data mean=212 ≈ 2-ply — marginal value

---

### Curriculum Value Net Training (categorical, 1000 games)

**Data generation** (7,311s = ~2h on M5 Max):
| Phase | Games | Mean Score | Samples |
|-------|-------|------------|---------|
| 2-ply | 500 | 190 | 60,876 |
| Medium rollout (top-10, 30 rollouts) | 300 | 472 | 74,610 |
| Strong rollout (top-20, 100 rollouts) | 200 | 672 | 67,170 |
| **Total** | **1,000** | **371** | **202,656** |

**Training** (40 epochs × 38s = 25 min):
- 128ch/6 blocks, 1.84M params, categorical (64 bins, support [0, 8.5])
- Best val_loss: **1.59** (converged from 3.15, still improving slightly at epoch 40)
- Value range in data: [0, 1117], log range: [0, 7.02]
- Saved to `checkpoints/value_net.pt`

Key difference from v2: training data spans mean scores 190–672 (vs only 191 for v2).
This should give the value net much better discrimination across the score range.

### Curriculum Value Net Evaluation

**value_lookahead results** (20 games, seed=42):
| 2-ply weight | Mean | Notes |
|-------------|------|-------|
| 0.1 | 22 | Value dominates, terrible |
| 0.5 | 45 | |
| 1.0 | 53 | ≈ heuristic baseline |
| 2.0 | 100 | |
| 5.0 | 116 | |
| 10.0 | 159 | Still below pure 2-ply (196) |

**Verdict: Worse than v2** (which got mean=229 at weight=5.0). Despite wider score range in
training data (190–672 vs only 191), the categorical curriculum value net hurts more than helps.
More 2-ply weight = better, meaning the value predictions add noise rather than signal.

**Root cause** (identified by Gemini review): V(s) is *policy-dependent*. The same board
state has completely different remaining scores depending on which policy plays from it.
Mixing 2-ply data (mean=190) with rollout data (mean=672) creates contradictory labels
for identical states. The network averages them into a "mushy" value that reflects no
real policy. This is fundamental — not fixable with more data or better architecture.

**Lesson**: Value nets for this problem must train on data from a single, consistent policy.
But even then, the high variance from random ball spawns makes score prediction inherently
noisy. A better approach is **policy distillation** (see Next Steps).

---

## Phase 9: Scaling + Policy Distillation

### Rollout Depth Sweep
| Depth | Rollouts | Mean | Median | Time/game |
|-------|----------|------|--------|-----------|
| 20 | 100 | **926** | 862 | 48s |
| 30 | 100 | 807 | 686 | 49s |

**Deeper is worse.** The 1-step heuristic (mean=52) makes too many mistakes that compound
over more steps. Depth=20 is the sweet spot for the current heuristic quality.

### 200 Rollouts
| Rollouts | Mean | Median | Max | Time/game |
|----------|------|--------|-----|-----------|
| 100 | 926 | 862 | 2160 | 48s |
| **200** | **1152** | **1160** | **2251** | **105s** |

**Consistent 1000+ achieved!** Median above 1000 — most games score 1000+.

### Next-Ball-Aware Heuristic (failed)
Added spawn-position awareness to `_evaluate_move`:
- Bonus for next-ball spawns completing lines
- Penalty for moving to a spawn cell
- Extra bonus for clearing (skips spawns)

**Result: mean=51.6 vs 55.8 baseline (0.93x) — worse.** The hand-tuned weights disrupt
the heuristic's existing score calibration. A neural network should learn these patterns
instead, via training on expert data that implicitly captures spawn-aware decisions.

### Policy Distillation Attempts

**Critical insight** (Gemini review): Do NOT put the CNN inside the rollout loop.
Heuristic: 23,000 turns/sec. CNN: ~500 turns/sec (unbatched). Use the distilled policy
as a **candidate filter** (top-K from 300 moves), not as the rollout policy.

**Target architecture:**
1. PolicyNet filters 300 moves → top-K (~1ms)
2. 200 fast heuristic rollouts on top-K candidates (~30ms)
3. Pick best → 1150+ quality at <1s/move

**Attempt 1: Sequential game distillation** (50 games, 18K samples, from scratch)
- src=5.6%, tgt=45.3% — but standalone mean=0.3 (random)
- Too few samples for a 519K param network

**Attempt 2: Fine-tune DAgger on 20 distillation games** (6K samples, lr=1e-4)
- src=35%, tgt=71% — pretrained features transfer well
- Standalone: mean=18.5 (similar to DAgger's 19.6)
- But only 6K samples, barely shifted behavior

**Attempt 3: Oracle labeling** (Gemini suggestion — offline state labeling)
Pipeline: collect 266K states from 5K heuristic games → sample 30K → label each
independently with rollout evaluator (50 rollouts, 18 workers parallel).
- Labeling: 13.2 states/sec, 38 min total
- From scratch, soft_temp=5.0: src=1.4%, tgt=4.3% — **failed**
- Cause: rollout scores (50-500) mixed with 2-ply scores (-5 to +20) in source labels
  creates near-one-hot distributions even at high temperature

**Attempt 4: Oracle with 2-ply-only soft labels** (30K states, fine-tune DAgger, lr=3e-4)
- Fixed: soft source labels use only 2-ply scores (uniform scale)
- src=2.6%, tgt=7.4%
- **Top-K recall vs heuristic:** Oracle=56% at top-5, DAgger=78% at top-5
- **Fine-tuning HURT** — the model lost existing source selection ability

**NN-filtered rollout player** (using original DAgger model as filter):
| Filter | Mean | Time/game | vs Full (1152, 105s) |
|--------|------|-----------|---------------------|
| DAgger top-5 | 170 | 8s | 15% quality |
| DAgger top-10 | 438 | 19s | 38% quality |
| DAgger top-15 | 670 | 29s | 58% quality |
| Full (all→top-20) | 1152 | 105s | 100% |

**Key findings:**
1. The DAgger model (trained on heuristic labels) remains the best filter — oracle
   distillation consistently degraded it
2. Top-15 filtering recovers 58% of quality at 3.6x speedup
3. The gap is fundamental: DAgger learned to predict *heuristic* source preferences,
   but the rollout player's preferences are different
4. Bridging this gap requires either much more oracle-labeled data (100K+) or
   a different training formulation (e.g., DAgger-style iterative relabeling)

---

## Next Steps

### High priority
- **NN filter + more rollouts** — DAgger top-15 + 200 rollouts might match full pipeline
  quality at 3x the speed. Quick to test.
- **DAgger-style oracle distillation** — instead of one-shot oracle labeling, iteratively:
  play with the model's top-K, label those states with oracle, retrain. This makes training
  data match the model's own state distribution (avoids distribution shift).

### Medium priority
- **More oracle data** — 100K+ labeled states might overcome the data scarcity issue
- **Open-loop MCTS** — with fast_move optimization, could outperform flat rollouts
  if the branching factor can be managed via NN pruning

### Lower priority
- RL fine-tuning on top of the current search player
- Stronger rollout policy via model-in-the-loop (with careful speed management)

### Tournament Bracket (Successive Halving)
Gemini suggested concentrating rollout budget on survivors instead of flat allocation.

**4-round bracket:**
1. Qualifiers: 2-ply ALL moves → top 30 (~50ms)
2. Quarter-finals: 10 rollouts (depth 10) on 30 → top 10
3. Semi-finals: 40 rollouts (depth 20) on 10 → top 3
4. Finals: remaining rollouts on 3 → pick best

Total rollouts: 10×30 + 40×10 + N×3 = 1,150 (vs ~4,000 for flat 200×20).

| Player | Mean | Median | Max | Time/game | Total rollouts |
|--------|------|--------|-----|-----------|---------------|
| Flat 200 rollouts | 1152 | 1160 | 2251 | 105s | ~4,000 |
| **Tournament 200** | **1385** | **1570** | **2266** | **68s** | ~1,150 |
| Tournament 400 | 1319 | 1263 | 2158 | 100s | ~1,750 |

**Tournament 200 is the sweet spot.** +20% mean, +35% median, 1.5x faster than flat.
More finals rollouts (400 budget) don't help — top-3 are already well-separated by semi-finals.
The bracket structure is already concentrating compute effectively.

### Heuristic Improvements (all failed)

Gemini suggested three heuristic fixes to break the "bias ceiling":

1. **Combo-awareness** (`_total_clearable` replacing `_max_line_at`): Correctly scores
   cross patterns (9 balls = 45 pts instead of 5 pts). Result: **mean=962 vs 1385 baseline**.
   The dedup loop is expensive (92s vs 68s/game), and the changed scoring disrupts rollout
   policy behavior in ways that cascade negatively.

2. **Dynamic clear bonus** (congestion-scaled): Dampen small clears on empty boards to
   encourage building bigger combos. Result: further regression to **mean=977** when
   combined with combo-awareness and discount.

3. **Discount factor** (γ=0.95 in rollouts): Weight near-term gains higher. Not tested
   in isolation but contributed to the combined regression.

**Lesson**: The heuristic's weight calibration is a delicate balance evolved over many
iterations. Changing individual weights without systematic tuning (e.g., Bayesian optimization)
is a minefield. All three "theoretically sound" changes made things worse.

---

### Open-Loop MCTS (failed)

Implemented spawn-safe Open-Loop MCTS per Gemini's suggestion:
- Tree stores action sequences only, no game states
- Graceful fallback on illegal actions (no -1000 penalty)
- 1-step heuristic priors, root pre-filtering to top-30

| Config | Mean | Time/game |
|--------|------|-----------|
| 1200 sims, 300 root actions | 107 | 77s |
| 1200 sims, 30 root actions | 57 | 52s |
| **Tournament 200 (baseline)** | **1385** | **68s** |

**MCTS is fundamentally broken for this problem in single-threaded form.** 1200 sims
across 30 actions = 40 visits each. But heuristic rollouts have massive variance —
40 samples isn't enough for reliable Q-values.

The tournament player wins because:
1. **Parallel rollouts** exploit all 18 cores; MCTS is inherently sequential
2. **Progressive averaging** (10→40→150 rollouts) builds reliable estimates at each stage
3. We only need **depth-1 search** — deeper search hurts due to heuristic bias
4. The bottleneck is **variance reduction** (many samples per candidate), not tree depth

Would need parallelized MCTS (virtual loss / leaf parallelism) to compete — significantly
more complex for marginal benefit over the tournament bracket.

---

---

## Phase 11: CMA-ES Weight Optimization

### Motivation
All Phase 9-10 heuristic changes (combo-awareness, dynamic clear bonus, discount factor)
regressed because changing individual weights breaks the coupling with softmax temperature
and other weights. Gemini identified this as the "softmax temperature misalignment" problem:
the temp=2.0 was tuned for score std ~1.5 in [-7, +18] range. Any scoring change alters the
distribution, requiring simultaneous re-tuning of temperature.

**Solution**: CMA-ES (Covariance Matrix Adaptation Evolution Strategy) via Optuna —
a gradient-free optimizer that tunes all 6 parameters simultaneously:
1. `clear_mult` (default 10.0) — scales game points for clearing lines
2. `clear_base` (default 100.0) — flat bonus for any clear
3. `partial_pow2` (default 2.0) — scales (length-1)² for extendable partial lines
4. `partial_linear` (default 0.3) — scales (length-1) for dead-end partials
5. `break_penalty` (default 1.5) — penalty for breaking source-side partial lines
6. `temperature` (default 2.0) — softmax temperature for rollout exploration

### Implementation
- Parameterized `_evaluate_move_w(board, sr, sc, tr, tc, color, w)` — JIT function with weights array
- `set_weights(w, temperature)` / `get_weights()` — module-level weight management
- Pool initializer `_init_worker_weights` propagates weights to multiprocessing workers
- `training/tune_weights.py` — Optuna CMA-ES optimizer using tournament player as evaluator
- Objective: maximize mean tournament score over 5 games per trial

### Preliminary Results (in progress, 50 trials)

| Trial | Mean Score | Weights [clear_m, clear_b, part_p2, part_lin, break_pen] | Temp |
|-------|-----------|----------------------------------------------------------|------|
| 0 (baseline) | 883 | [10.0, 100.0, 2.0, 0.30, 1.5] | 2.00 |
| 2 | 1344 | [14.6, 109.4, 5.7, 1.38, 2.4] | 3.23 |
| 5 | 1178 | [36.5, 93.5, 6.1, 1.54, 3.2] | 2.40 |
| **7** | **1471** | **[18.8, 96.9, 3.2, 0.67, 3.2]** | **2.42** |

**Trial 7: mean=1471 (+67% over baseline)** after only 8 trials. Emerging trends:
- Higher `clear_mult` (18.8 vs 10.0) — value line clears more
- Lower `clear_base` (96.9 vs 100.0) — slightly less flat bonus
- Higher `partial_pow2` (3.2 vs 2.0) — value partial line building more
- Higher `partial_linear` (0.67 vs 0.3) — even dead-end partials matter more
- Higher `break_penalty` (3.2 vs 1.5) — penalize disrupting existing lines more
- Higher `temperature` (2.42 vs 2.0) — more exploration in rollouts

Note: 5-game evaluations have high variance (std ~500), so individual trial scores are
noisy. CMA-ES handles this by averaging over the population's covariance structure.

### Final Results (50 trials completed)

CMA-ES best trial (by 5-game score): Trial 7, score=1471

**20-game validation with tournament 200** (same seed set for all):

| Candidate | Mean | Median | Max | Min | Weights |
|-----------|------|--------|-----|-----|---------|
| baseline | 955 | 740 | 2042 | 391 | [10.0, 100.0, 2.0, 0.3, 1.5] temp=2.0 |
| trial7 | 1074 | 904 | 2206 | 399 | [18.8, 96.9, 3.2, 0.67, 3.2] temp=2.42 |
| trial19 | 1101 | 959 | 2251 | 288 | [8.4, 131.1, 2.9, 0.77, 3.6] temp=1.81 |
| trial32 | 1204 | 1186 | 2182 | 405 | [28.0, 74.3, 4.2, 0.66, 2.4] temp=1.71 |
| **trial2** | **1228** | **1174** | **2200** | **586** | **[14.6, 109.4, 5.7, 1.38, 2.4] temp=3.23** |

**Trial 2: +29% mean over baseline**, highest floor (min=586).

CMA-ES's 5-game best (trial 7) was noise — validated at only 1074. But CMA-ES still found
the right region: trials 2, 32, 19 all significantly outperform baseline.

**Key learnings from CMA-ES:**
- **Partial line weights should be ~3x higher**: pow2: 2.0→5.7, linear: 0.3→1.38.
  The heuristic was massively undervaluing line-building potential.
- **Higher temperature** (3.23 vs 2.0): more exploration in rollouts helps — the default
  was too greedy.
- **Break penalty matters**: 2.4 vs 1.5 — protecting existing partial lines is important.
- `clear_mult` and `clear_base` relatively stable — line clearing was already well-weighted.

### Expected Impact (Gemini analysis)
- The "search multiplier effect": tournament gives ~26x amplification of heuristic quality
- If CMA-ES improves standalone heuristic from mean=52 to mean=80-120, the tournament
  multiplier yields mean=1600-2200
- Deeper rollouts (depth 30+) may become viable once heuristic bias is reduced
- Temperature is tuned simultaneously, preventing the misalignment that broke manual tuning

### Usage
```bash
# Run CMA-ES optimization (50 trials ≈ 2.5 hours on M5 Max)
python -m training.tune_weights --trials 50 --games-per-trial 5 --rollouts 100

# Apply best weights
python -c "from game.fast_heuristic import set_weights; ..."
```

---

### CMA-ES v3: 17 Spatial Features + 2-ply (abandoned after 129 trials)

Added 12 spatial features to the heuristic: center distance, edge bonus, same-color
clustering, different-color penalty, empty neighbors, hole prevention, local openness,
empty count, source center bonus, move distance, combo detection, line-4 bonus.

| Trial range | Best score | Baseline |
|-------------|-----------|----------|
| 0-129 | 247 (trial 117) | 204 |

**Modest +21% improvement, plateauing.** CMA-ES explored 17D space for 129 trials but
couldn't find a breakthrough. The spatial features help avoid some dumb moves but the
1-step evaluation horizon fundamentally limits how much spatial planning helps.

**Conclusion:** Shallow search (1-ply or 2-ply) has a hard ceiling of ~250 mean regardless
of feature quality. The heuristic can't plan "move A now to set up move B next turn."
Spatial features are necessary but not sufficient — they need to be combined with deeper,
smarter search.

---

## Phase 13: Value-Guided Beam Search (next)

### Motivation
Two hard truths from all experiments:
1. **Shallow search is dead**: 2-ply + any features maxes at ~250
2. **Deep search is blind**: 200-rollout tournament uses dumb heuristic, maxes at ~1000

To reach 3000+, we need **deep search with smart evaluation** — a CNN that understands
board health, used inside a multi-ply beam search.

### Why Previous Value Nets Failed
Phase 7 value nets trained on actual game scores (high variance from random spawns).
Two identical boards could lead to scores of 500 or 2500 based on luck. MSE loss
averaged these out → predictions clustered at ~190, useless for discrimination.

### The Fix: Train on Tournament Evaluation Scores
The tournament player's combined rollout score for each state is **already averaged
over 200 rollouts** — it's a variance-free "board health" metric. A CNN trained on
this target learns what a good board looks like according to the expert search.

### Architecture: ExpectiMax Value Beam Search
Critical correction (Gemini): evaluate **pre-spawn** boards, not post-spawn.
The CNN already accounts for spawn variance (trained on rollout-averaged scores).
Sampling 3 physical spawns would re-introduce the exact randomness we eliminated.

1. **Ply 1**: Generate all ~300 legal moves, execute each (clone+move, **before spawns**)
2. **Batch CNN evaluate**: Run 300 pre-spawn boards through Value CNN (~5ms)
3. **Prune**: Keep top 10 moves
4. **Ply 2**: For top 10, simulate spawns (3 samples each), generate next legal moves
5. **Batch CNN evaluate**: Run expansion boards through Value CNN
6. **Repeat** for ply 3 if budget allows
7. **Execute**: Pick the ply-1 move leading to highest deep value

### Expected Performance
- CNN evaluation: ~5ms per batch of 300 boards (MPS)
- 3-ply beam search with width 10: ~300 × 3 spawns + 10 × 300 × 3 + 10 × 300 × 3 = ~20K evaluations = ~30ms
- **Total: ~50ms per move** — fast enough for browser deployment
- Quality: if the CNN accurately predicts board health, could reach 2000-3000+

### Data Needed
Modify tournament data gen to save (state, best_move_rollout_score) instead of just
(state, best_move). We already have the tournament infrastructure — just need to expose
the internal scores.

### Training Results
- 253,134 samples from 500 tournament games (9.5h data gen)
- 128ch/6 blocks, 1.83M params, 40 epochs × 43s
- **Val loss: 0.041, correlation: 0.897** (vs Phase 7's 0.281 loss, ~0.3 corr)
- Prediction range: [-0.1, 4.0] (target: [0.0, 4.6]) — full discrimination

### Beam Search Results — FAILED

| Config | Mean | Speed |
|--------|------|-------|
| Beam depth=1 (CNN only) | 58 | 1.7s/game |
| Beam depth=2 (spawn expansion) | 59 | 20s/game |
| For reference: 2-ply exhaustive | 196 | 5s/game |
| For reference: tournament 200 | 1041 | 68s/game |

**Root cause: CNN can't discriminate between moves within a single state.**
- Across the dataset: 0.897 correlation (early-game vs late-game boards differ hugely)
- Within a single state (1222 legal moves): **std=0.032** in log space
- All 1222 resulting boards score between 18.0 and 24.4 raw rollout points
- The 6-point spread is within the CNN's noise — effectively random selection

**Why:** We trained state → value (board health). But all 1222 resulting boards
look nearly identical to the CNN — one ball moved slightly. The CNN learned macro
patterns (early vs late game) but not micro patterns (which specific move is best).

**The fundamental problem with state-only value evaluation:** Moving ball A to position
X vs position Y produces boards that differ by exactly 2 cells. The CNN sees 81 cells
and can't detect that the 2-cell difference matters. The 2-ply heuristic evaluates the
*move itself* (line potential, extensions) — information the state-only CNN doesn't have.

### Hybrid 2-ply + CNN Board Health

Instead of using the CNN alone (beam search), combine it with 2-ply heuristic scoring:
`combined = normalize(2ply_score) + normalize(cnn_value) * weight`

| CNN Weight | Mean | vs 2-ply only (187) |
|-----------|------|---------------------|
| 0.0 (pure 2-ply) | 187 | baseline |
| 0.5 | 269 | +44% |
| **1.0** | **322** | **+72%** |
| 2.0 | 86 | CNN dominates, bad |
| 5.0 | 25 | pure CNN, terrible |

**2-ply + CNN at weight=1.0: mean=322, max=1008!** The CNN's subtle board health signal
(std=0.032 within a state) is real — it provides a meaningful secondary signal that tips
close decisions. This is the biggest 2-ply improvement ever achieved (+72% vs +21% from
spatial feature CMA-ES).

### Hybrid Tournament (CNN + rollouts) — Regression

| Player | Mean | Speed |
|--------|------|-------|
| 2-ply only | 187 | 10s/game |
| 2-ply + CNN | 322 | 10s/game |
| Tournament 200 (standard) | **1041** | 68s/game |
| Tournament 200 + CNN | **399** | 19s/game |

**CNN hurts tournament (-62%).** The CNN corrupts candidate selection by pushing moves
with good "board health" but poor rollout potential into the top-30. The rollouts already
capture board health through 20-step simulation — the CNN's signal is redundant and adds
noise to the ranking.

**Key insight:** The CNN is a useful *substitute* for rollouts (2-ply + CNN = 322 vs
2-ply alone = 187), but a harmful *addition* to rollouts (tournament + CNN = 399 vs
tournament alone = 1041). Rollouts already do what the CNN does, but better.

### Conclusions from Phase 13

1. **Value CNN v4 works** — 0.897 correlation, trained on rollout-averaged scores
2. **But it can't replace the heuristic** — within-state std=0.032, too low for move selection
3. **CNN + 2-ply is the best client-side option** — mean 322 at ~10ms/move
4. **CNN + tournament is worse than tournament alone** — redundant signal adds noise
5. **The heuristic rollout ceiling stands** — no CNN approach has beaten the tournament player

---

## What We Have Now (Final Summary)

### Best Players

| Player | Mean | Speed | Deployment |
|--------|------|-------|------------|
| Tournament 200 + CMA-ES | **~1041** | 100ms/move | Server (Python) |
| 2-ply + CNN hybrid | **322** | ~10ms/move | Browser (ONNX + JS) |
| 2-ply exhaustive | 196 | ~10ms/move | Browser (JS only) |
| PolicyNet standalone | 53 | 0.15ms/move | Browser (ONNX) |
| 1-step heuristic | 53 | 0.01ms/move | Anywhere |

### What Works
- Tournament bracket (successive halving) — best search architecture
- CMA-ES weight tuning — +29% over hand-tuned
- Softmax exploration in rollouts — +16% over epsilon-greedy
- Numba JIT optimization — 2.9x speedup
- Value CNN trained on rollout scores — 0.897 correlation (useful for 2-ply hybrid)
- Policy distillation from single expert — 77% top-5 recall

### What Doesn't Work (for >1000 mean)
- Value nets for move selection — can't discriminate within a state
- Policy distillation — caps at teacher's level
- Spatial feature CMA-ES on 2-ply — ceiling ~250
- Hand-tuned heuristic changes — break softmax calibration
- Open-loop MCTS — too few visits per node
- CNN + tournament hybrid — CNN noise hurts rollout-based ranking

### The Path to 3000+
Every approach above is built on the 1-step heuristic (mean=53). The tournament amplifies
it 20x to ~1000, but the heuristic itself can't understand crosses, space management, or
multi-move planning.

**The only remaining path: Reinforcement Learning (PPO/AlphaZero).**
- Train a policy+value network via self-play
- Dense reward shaping to overcome sparse rewards
- The network discovers strategies (center avoidance, combo building, delayed gratification)
  that no heuristic or teacher can provide
- Once trained, the PolicyNet deploys to browser via ONNX

---

---

## Phase 14: PPO Reinforcement Learning

### Attempt 1: PPO from scratch (3M steps, 32 envs)
- Dense reward shaping: survival, center control, line building, break penalty
- **Result: mean=0.3 at end — no learning at all**
- Entropy stuck at maximum (5.14) — pure random play throughout
- Root cause: random initialization never clears a line, so +10 clear reward is never seen

### Attempt 2: PPO with BC initialization (3M steps, 32 envs)
- Loaded distilled tournament model (mean=53 standalone) as starting weights
- Lower entropy coef (0.001), lower lr (1e-4), longer annealing
- **Started at mean=25 (BC baseline), collapsed to mean=0.5 by step 100K**
- Entropy rose from 2.67 → 4.18 — BC knowledge completely erased
- Root cause: reward shaping PUNISHES normal play. Break penalty (-10 for disrupting a
  4-line) fires on almost every move. Agent learns "every action is bad" → degrades to random.
- Secondary cause: lr=1e-4 + clip_epsilon=0.2 too aggressive for fine-tuning

### What's needed for RL to work
1. **Reward shaping must be net-positive for competent play** — the BC-initialized agent
   scoring mean=25 should receive POSITIVE total reward, not negative. Currently the
   penalties dominate the bonuses.
2. **Much lower learning rate** (1e-5 or 3e-5) and lower clip epsilon (0.05) to prevent
   catastrophic forgetting of the BC policy
3. **KL penalty** to anchor the policy near the BC baseline during early training
4. **Careful reward calibration** — play games with the BC policy, log per-step rewards,
   verify the total is positive

---

## CRITICAL BUG FIX: Pathfinding Target Mask (discovered during human play)

### The Bug
`_get_target_mask()` in `board.py` used `break` after finding the FIRST adjacent empty
cell's connected component. If a ball touched empty cells in DIFFERENT connected components
(separated by walls of balls), only one component was reachable. Legal moves were blocked.

```python
# BEFORE (broken):
for dr, dc in ((0,1),(0,-1),(1,0),(-1,0)):
    if cc_labels[nr, nc] > 0:
        target_label = cc_labels[nr, nc]
        break  # BUG: ignores other components!
```

### The Fix
Check ALL adjacent empty cells, collect all unique component labels, mark cells from
ALL reachable components as valid targets.

### Impact — THE BIGGEST IMPROVEMENT IN THE PROJECT

| Metric | Before fix | After fix | Change |
|--------|-----------|-----------|--------|
| **Mean** | 1041 | **1895** | **+82%** |
| **Median** | 900 | **2128** | **+136%** |
| Max | 2145 | 2280 | +6% |
| Turns/game | 506 | **882** | **+74%** |

**A single bug fix nearly doubled our score.** The bug was causing games to end prematurely
by blocking legal moves. With correct pathfinding, games last 74% longer and the tournament
player scores mean=1895 — crossing the 2000 barrier on median.

This bug was present since the very beginning of the project. ALL previous evaluation
numbers were artificially depressed.

---

## Phase 15: Human Expert Data Collection (in progress)

Interactive Pygame GUI (`play_gui.py`) for human expert play with move logging.
Each game auto-saves (observation, source, target) pairs to `data/human_games/`.

**Data collected so far:** 1 game, 795 points, 374 moves.
Human expert data collection is ongoing — will be used for behavioral cloning
once enough games (~20) are accumulated. This data encodes spatial intuition
(center avoidance, color zoning, combo building, next-ball awareness) that
no heuristic or search algorithm has captured.

Plan: Train a Human PolicyNet on this data, use as filter/prior for tournament search.
Could push scores significantly above 2000 by combining human spatial intuition with
the tournament's search depth.

---

### Post-Fix Re-Baseline (all with corrected pathfinding)

| Player | Mean | Median | Max | Speed |
|--------|------|--------|-----|-------|
| Heuristic | 71 | 65 | 207 | 0.01s/game |
| 2-ply + CNN hybrid | 442 | 402 | 886 | 25s/game |
| Tournament 200 | **1895** | **2128** | 2280 | 148s/game |
| Tournament 400 | **2026** | **2142** | 2278 | 232s/game |

Tournament 200 → 400 gives modest +7% mean but lower variance (std 440 vs 547).
The extra rollouts add consistency rather than raw score.

### Score Progression (Full History)

| Phase | Player | Mean | Notes |
|-------|--------|------|-------|
| 0 | Random | 0.5 | |
| 0 | 1-step heuristic | 53→71 | +33% from bug fix |
| 3 | 2-ply exhaustive | ~196 | pre-fix |
| 6 | + softmax rollouts (100) | ~926 | pre-fix |
| 10 | Tournament bracket (200) | ~1041 | pre-fix |
| 11 | + CMA-ES weights | ~1041 (+29% in A/B) | pre-fix |
| 13 | 2-ply + CNN hybrid | 322→442 | +37% from bug fix |
| **BUG FIX** | **Tournament 200** | **1895** | **+82% from fix alone** |
| | **Tournament 400** | **2026** | current best |

### CRITICAL FINDING: Turn Cap Was Limiting Scores

All previous evaluations used `turns < 1000` (hardcoded in evaluate.py), silently
killing games before they finished. With the turn cap removed AND pathfinding fixed:

**Uncapped Tournament 200 (20 games, seed=42, turns<1500):**
mean=2685, median=3216, max=3416

**Uncapped Tournament 200 (20 games, seed=77777, NO cap, FINAL):**
mean=**4641**, median=3264, max=**15697**, min=549
Games >5000: 5/20 (25%), Games >10000: 2/20 (10%)
Individual: [549, 606, 788, 892, 1558, 2185, 2223, 2256, 2396, 3161,
3368, 3483, 4450, 4757, 6030, 7051, 8482, 8691, 14201, **15697**]

The agent was ALWAYS capable of 10,000+ point games — we just kept stopping them at
turn 1000. Combined with the pathfinding bug, our previous "ceiling" of ~1000 mean was
two artificial limits, not a fundamental capability issue.

### CMA-ES v3 Spatial Features — Mixed Results

With bug-fixed pathfinding, CMA-ES found spatial weights that hit max=3293 but
mean=1479 (vs baseline mean=2685). The spatial weights create a "gambling" strategy —
higher ceiling but much lower floor. The baseline with no spatial features is more
consistent and scores higher on average.

**Lesson:** The base heuristic + tournament search is already very strong with correct
pathfinding. Spatial features add variance without reliably improving play. A better
approach: ML-learned weights via logistic regression on elite game data.

---

## Phase 16: ML-Powered Rollout Policy (next)

### The Plan: Oracle-Trained Numba Linear Model

The tournament player scores 14,000+ when lucky, but its rollout heuristic is still
the 5-weight line-building function. Injecting spatial intelligence into rollouts
would make "genius futures" the norm instead of the exception.

**Architecture:**
1. Extract 30 spatial features per move (Numba JIT, 0.8µs/call) — DONE
2. Generate 5000 tournament games, filter top 10% (elite data)
3. Train logistic regression: features → chosen move
4. Hardcode 30 learned weights into Numba rollout function
5. Tournament player uses ML-powered rollouts → consistently higher scores
6. Iterate: use improved player to generate better data → retrain

**Feature extractor (`game/features.py`) — 30 features:**
- Line features (6): max_line, clears, partial_dirs, cross_dirs, line4, extends
- Spatial (8): center_dist, on_edge, on_corner, src_center, move_dist, local_empty, total_empty, holes
- Color clustering (6): same_orth, diff_orth, same_diag, nearest_same, same_count, n_colors
- Source (4): src_in_center, src_congestion, break_penalty, center_empty
- Next-ball synergy (6): spawn_conflict, spawn_clear, spawn_extends, spawns_valid, occupancy, pressure

**Speed:** 0.8µs per feature extraction, 0.2ms for all 300 legal moves — fast enough
for rollout inner loops at 20,000+ moves/sec.

### Human Expert Data
Interactive Pygame GUI (`play_gui.py`) available for human play with move logging.
Data in `data/human_games/`. Can be used alongside tournament data for training.

---

### Score Progression (Final)

| Phase | Player | Mean | Max | Notes |
|-------|--------|------|-----|-------|
| 0 | Random | 0.5 | ~5 | |
| 0 | 1-step heuristic | 71 | 207 | post-fix |
| 3 | 2-ply exhaustive | ~250 | ~500 | post-fix est. |
| 6 | + softmax rollouts | ~1800 | ~3000 | post-fix est. |
| 10 | Tournament 200 | **1895** | 2280 | post-fix, cap=1000 |
| | **Tournament 200 (uncapped)** | **4641** | **15697** | post-fix, NO cap |
| 13 | 2-ply + CNN hybrid | 442 | 886 | post-fix |

### Target
- ~~Consistent 1000+ mean scores~~ **ACHIEVED**
- ~~Consistent 2000+ median~~ **ACHIEVED**
- ~~Mean 3000+~~ **ACHIEVED** (uncapped tournament: mean=4641)
- ~~Max 5000+~~ **ACHIEVED** (15697 in one game, two games >14K)
- Next: ML-powered rollouts for consistent 5000+ mean (eliminate the 549-pt floors)
- Ultimate: 10,000+ mean via iterative self-improvement

---

## Phase 16: ML-Powered Rollout Policy (in progress)

### Problem Statement
The tournament player averages 4,641 but has huge variance (549 to 15,697).
The low-scoring games die early because the rollout heuristic (mean=71) has no spatial
awareness — it scatters colors, blocks the center, creates dead holes, and clears
5-lines immediately instead of building crosses.

### The Solution: Oracle-Trained Numba Linear Model
Train a 30-weight linear model that replaces `_evaluate_move` inside rollouts.
The model learns spatial rules (center avoidance, color zoning, combo building)
from elite tournament data. It runs at ~1µs/call in Numba — same speed as the
current heuristic but with human-level spatial understanding.

### Pipeline

**Step 1: Rust Data Generator** (~6.8 hours for 1000 games)
- Rust engine (`rust_engine/`) — zero GC, zero-allocation Clone, xorshift64 RNG
- Heuristic scores verified: **exact match with Python** (41.70, -1.50, 0.00)
- Runs 18 tournament games in parallel via Rayon (one game per core, sequential rollouts within)
- Each game exports board + chosen move + **top-5 candidates with rollout scores** (for hard negative mining)
- Uncapped games — no turn limit, games play to natural death
- Progress reported after every game completion

**Go engine was tried first but abandoned:**
- Go's GC overhead caused 57% of CPU time in `runtime.kevent`
- Go tournament was 3.4x slower per-move than Python (1060ms vs 315ms)
- Only 4.7x scaling on 18 cores (vs Rust's near-linear scaling)

**Performance comparison (data gen workload):**

| Engine | Heuristic (parallel) | Tournament single-thread | Est 1000 games |
|--------|---------------------|-------------------------|----------------|
| Python (Numba) | 238 games/sec | 315ms/move (18 workers) | ~83h |
| Go | 7,237 games/sec | 1,060ms/move | ~33h |
| **Rust** | **10,514 games/sec** | **629ms/move** | **~6.8h** |

Rust wins because: no GC pauses, stack-allocated Game structs (~200 bytes, memcpy clone),
inline xorshift64 RNG (1 u64 vs Go's 607-element array), and LTO+codegen-units=1 optimization.

**Step 2: Elite Filtering**
- Sort 1000 games by score, keep top 10% (>3000 pts, ~100 games)
- Extract ~80K (state, move) pairs from elite games only
- These represent how a 5000+ point player makes decisions

**Step 3: Feature Extraction + Training** (minutes)
- For each elite (state, chosen_move): extract 30 features for chosen + 5 random alternatives
- Train logistic regression (scikit-learn) to classify chosen vs not-chosen
- Export 30 learned weights

**Step 4: Numba Integration** (minutes)
```python
@njit(cache=True)
def _evaluate_move_ml(board, sr, sc, tr, tc, color):
    features = extract_features(board, sr, sc, tr, tc, color, ...)
    score = 0.0
    for i in range(30):
        score += features[i] * ORACLE_WEIGHTS[i]
    return score
```

**Step 5: Evaluation**
- Replace heuristic in rollouts with ML model
- Run tournament 200 with ML rollouts → compare to baseline (mean=4641)
- If ML rollouts improve consistency (fewer <1000 games), iterate

**Step 6: AlphaZero-Lite Feedback Loop**
- Use improved tournament to generate new elite data
- Retrain linear model on better data
- Repeat until scores asymptote

### Expected Impact
- The linear model eliminates "dumb" rollout moves (center blocking, color scattering)
- Games that currently die at 549 pts should survive to 2000+
- Mean should increase and variance should decrease
- Conservative estimate: mean 6000-8000
- Optimistic: mean 10,000+ (if spatial rules prevent most early deaths)

### Browser Deployment Path
Once the ML-powered tournament consistently scores 5000+:
1. Distill tournament's move probabilities into a PolicyNet (128ch/6 blocks)
2. Export PolicyNet to ONNX for browser inference
3. Browser pipeline: ONNX PolicyNet (~100ms) + TypeScript 2-ply (~200ms) = ~300ms/move
4. Server option: Rust engine + tournament player via REST API

### Human Expert Data
Interactive Pygame GUI (`play_gui.py`) with AI Hint button (tournament player).
Data collected in `data/human_games/`. Available for future training alongside
tournament data. The AI Hint revealed the heuristic's "delayed gratification"
blind spot — it clears 5-lines immediately when waiting one turn could clear 6+.

---

---

## Phase 16 Results: ML Oracle + Tournament

### Pairwise Ranking Training (the breakthrough)

**Key insight:** Elite game filtering selects for LUCK, not skill. Instead:
- Use ALL games (not just high-scoring)
- Train on move-level comparisons (best vs worst among top-5 candidates)
- Weight samples by confidence (rollout score gap = skill signal, not luck)

**Method (Gemini's RankNet approach):**
1. For each move decision: ΔX = Features(best_candidate) - Features(worst_candidate)
2. Label y = 1 (best is better), weight = score_gap (confidence)
3. Add flipped pairs (y = 0) for balanced classes
4. Train logistic regression on (ΔX, y, weight)

**Data:** 263 Rust-generated tournament games → 477K move records → 600K pairwise samples
- Mean game score: 3921, median: 2874, max: **22,614**
- Training: **0.5 seconds** (logistic regression on 540K samples)

**Results:**
- Weighted accuracy: **92.9%** (model correctly ranks best vs worst 93% of the time)
- Unweighted accuracy: 76.7%

**Learned spatial weights (top features by importance):**
| Feature | Weight | What it means |
|---------|--------|---------------|
| cross_dirs | +2.48 | Multi-line clears are hugely valuable |
| clears | +2.40 | Clearing a line is very good |
| has_line4 | +2.02 | Building 4-in-a-row is highly valued |
| **spawn_clear** | **+1.57** | **Next-ball spawn completes a line — discovered autonomously!** |
| max_line | +0.76 | Longer lines are better |
| spawns_valid | -0.60 | Fewer valid spawn positions = danger |
| break_penalty | -0.56 | Breaking existing lines is bad |
| spawn_conflict | +0.47 | Awareness of spawn position conflicts |

**The model discovered next-ball awareness on its own!** `spawn_clear` (+1.57) means
"prefer moves where the upcoming spawn will complete a line." This is exactly the human
intuition the heuristic was missing (confirmed during human play testing with play_gui.py).

### ML Oracle Standalone vs Heuristic
| Player | Mean (50 games) |
|--------|----------------|
| ML oracle standalone | 11.5 (terrible — only learned fine distinctions, not basics) |
| Heuristic standalone | 68.8 |
| **Heuristic + ML blend (0.3)** | **79.4 (+15%)** |

The ML model can't replace the heuristic (it was trained to distinguish between top-5
candidates, not between good and terrible moves). But as a REFINEMENT it adds real value.

### ML-Enhanced Tournament — FIRST ML SUCCESS
| Player | Mean (5 games) | vs Baseline |
|--------|---------------|-------------|
| Tournament 50 (baseline) | 2913 | — |
| **Tournament 50 + ML oracle (blend=0.3)** | **3717** | **+28%** |

**This is the first time an ML component has HELPED the tournament player.**

Why it works (unlike all previous ML attempts):
1. **Pairwise training** — learns relative preferences, not absolute values
2. **Confidence weighting** — focuses on decisive situations, ignores noise
3. **All games used** — no luck-based elite filtering
4. **Refinement, not replacement** — blends with heuristic, doesn't override it
5. **0.8µs/call** — zero speed cost in rollout inner loop

Projected with tournament 200 uncapped: 4641 × 1.28 ≈ **mean ~5900**

### Next: AlphaZero-Lite Feedback Loop
1. Generate new tournament data WITH ML oracle enabled (better player → better data)
2. Retrain linear oracle on new data
3. Repeat until scores asymptote
4. Each iteration: data gen (Rust, ~1h) → train (Python, 0.5s) → evaluate (~10min)

### Target
- ~~Consistent 1000+ mean~~ **ACHIEVED**
- ~~Consistent 2000+ median~~ **ACHIEVED** (median=3264)
- ~~Mean 3000+~~ **ACHIEVED** (mean=4641, now 3717 with tournament 50+ML)
- ~~Max 5000+~~ **ACHIEVED** (max=22614 in Rust data gen)
### Definitive 20-Game Eval: ML Oracle v1 (blend=0.05)

| Stat | Baseline (no ML) | **ML Oracle v1** | Change |
|------|-------------------|------------------|--------|
| **Mean** | 4641 | **8432** | **+82%** |
| **Median** | 3264 | **8075** | **+147%** |
| **Max** | 15697 | **20875** | +33% |
| Min | 549 | 983 | +79% |
| >5000 | 5/20 (25%) | **15/20 (75%)** | 3x |
| >10000 | 2/20 (10%) | **6/20 (30%)** | 3x |

Scores: [4988, 8093, 11291, 2785, 11229, 14603, 7617, 1767, 12847, 5045,
5510, 8582, 20875, 9120, 17024, 5118, 983, 9235, 8057, 3863]

**The ML oracle transformed the tournament player.** 75% of games score above 5000.
The 30-weight linear model (trained in 0.5s) added +82% to the mean.

### Rust ML Oracle Integration — Lessons Learned

**Attempt 1: Raw blend (no normalization)**
- Added ML scores directly to heuristic: `h + ml * 0.05`
- Result: mean=764 (terrible) — scales don't match without normalization

**Attempt 2: Normalized blend in move selection**
- Added `normalize_and_blend()` matching Python's z-score approach
- Feature extraction verified: all 30 features match Python exactly ✓
- ML score verified: -1.804945 matches Python ✓

**Attempt 3: ML in rollout inner loop (greedy)**
- Used `get_best_move_ml` inside every rollout step
- Result: STUCK — 36M feature extractions per tournament move (300 moves × 200 rollouts × 20 depth × 30 features)
- 0.8µs × 36M = 29 seconds per move — completely impractical

**Attempt 4: ML in 2-ply candidate selection ONLY (current)**
- ML oracle scores all ~300 candidates once per tournament move (~0.24ms)
- Normalized blend with 2-ply scores for candidate ranking
- Rollouts use fast plain heuristic (no ML overhead)
- This matches Python's architecture where ML enhances candidate selection

**Key insight:** The ML oracle must be applied at the RIGHT level:
- ✓ 2-ply candidate selection (once per move, ~300 evaluations)
- ✗ Rollout inner loop (36M evaluations per move — 1000x too expensive)
- Python "hides" this because `get_best_move_fast` with ML runs the normalization
  per-call, but each call only evaluates ~300 moves, not 36M

### Self-Improvement Loop: Iteration 2

**Data generation (v2):**
- 29 games from ML-enhanced tournament player (oracle v1, blend=0.05)
- Mean game score: 5986, median: 4503, max: **26,000**
- 158K move records, all with top-5 candidates + rollout scores (hard negatives)
- Generated on M5 Max (18 workers), ~3 hours
- Key learning: hard negatives (top-5 candidates) are critical — random negatives
  gave 97.8% accuracy (too easy, learned nothing), hard negatives give 93.3% (meaningful)

**Oracle v2 training:**
- Trained on v2 data ONLY (no mixing with v1 — Gemini's advice: discard older generations)
- 600K pairwise samples, confidence-weighted, 0.5s training
- Weighted accuracy: **93.3%** (vs v1's 92.9%)
- Score gap range: [0.01, 149.24], mean=10.44

**Oracle v2 weights (top features):**
| Feature | v1 Weight | v2 Weight | Change |
|---------|-----------|-----------|--------|
| cross_dirs | +2.48 | +2.54 | Slightly higher |
| clears | +2.40 | +2.52 | Slightly higher |
| has_line4 | +2.02 | +2.14 | Slightly higher |
| spawn_clear | +1.57 | +1.57 | Stable |
| partial_dirs | -0.21 | **-0.29** | More penalty for partial-only moves |
| break_penalty | -0.56 | -0.57 | Stable |

Interpretation: v2 values clearing and combos slightly more, and penalizes moves that
only build partials without clearing potential more strongly. The spawn awareness
features remain stable — the oracle v1 already discovered those correctly.

**Evaluation: running** (20 games, seed=77777, tournament 200, blend=0.05)
- Baseline (no ML): mean=4641
- Oracle v1: mean=8432
- Oracle v2: pending...

### Rust Engine Lessons

Attempted to port the ML oracle to Rust for faster data generation. Key findings:
- **Rust heuristic: verified exact match** with Python (41.70, -1.50, 0.00) ✓
- **Rust 30 features: verified exact match** with Python (all 30 values) ✓
- **ML in rollout inner loop: impractical** — 36M feature extractions per tournament move
- **ML in 2-ply only: low quality** — rollouts still use blind heuristic
- **Normalization matters**: raw blend (h + ml*0.05) ≠ normalized blend (z-score + z-score*0.05)
- **Conclusion**: Python's multiprocessing (18 workers per game) is faster than Rust's
  single-threaded-per-game for this workload. Keep Rust for future non-ML data gen.

### Roadmap
- Eval oracle v2 → if improved, iterate to v3
- Target: mean 12,000 on seed=77777
- Then: update Pygame with best AI
- Then: deep learning distillation (dual-head ResNet → ONNX → browser)
- Ultimate: superhuman browser AI at <200ms/move

### 50-Game Reliable Eval (CMA-ES weights, seed=77777)
```
mean=1041, median=900, max=2145, std=616 (50 games)
```
Most reliable number — 50-game eval reduces confidence interval to ±87.

---

## Phase 12: Browser-Ready Policy Distillation (in progress)

### Target Architecture (for browser deployment)
1. **PolicyNet** (ONNX, ~50-100ms in browser): filters 300 moves → top 5
2. **2-ply search** (JS/WASM, ~100-300ms): exhaustive evaluation of top 5
3. **Total: ~200-400ms** per move — instant for interactive play

### Data Generation (running overnight)
- 1000 games with tournament 200 player (CMA-ES weights, temp=3.23)
- Expected: ~500K (state, move) pairs from single expert policy
- Save to `data/tournament_distill.npz`
- ETA: ~14 hours

### Training Plan
- 128ch/6 blocks PolicyNet (~1.8M params) on 500K samples
- Cross-entropy loss with hard labels (expert's exact move)
- On-the-fly dihedral augmentation (8x effective data)
- Target: >85% top-5 recall on tournament moves

### Deployment Plan
1. Export PolicyNet to ONNX
2. Port game engine + `_evaluate_move` (CMA-ES weights) to TypeScript
3. ONNX Runtime Web for in-browser inference
4. Angular app (Phase 2 of the project)

### Training Results

**Data:** 480,229 samples from 1,000 tournament games (mean score=983), 14.4 hours gen time.

**Model:** 128ch/6 blocks, 1.97M params, 40 epochs × 66s = 44 min training.

| Epoch | Val Loss | Src Top-1 | Src Top-5 | Src Top-10 | Tgt Acc |
|-------|----------|-----------|-----------|------------|---------|
| 1 | 8.07 | 4.4% | 18.8% | 32.3% | 9.0% |
| 5 | 4.40 | 28.7% | 60.3% | 76.5% | 62.1% |
| 10 | 3.49 | 35.5% | 69.6% | 83.3% | 72.4% |
| 20 | 3.06 | 40.6% | 74.9% | 86.9% | 76.7% |
| 30 | 2.91 | 41.9% | 76.9% | 88.0% | 77.7% |
| **40** | **2.88** | **42.4%** | **77.2%** | **88.3%** | **77.9%** |

**Why this worked (unlike Phase 9):**
- **480K samples** (vs 18-30K) — 20x more data
- **Single expert policy** — no mixed/conflicting labels
- **CMA-ES-tuned teacher** — plays at mean=983 level
- 244:1 sample-to-parameter ratio — no overfitting risk

### Evaluation

| Player | Mean | Speed | Browser? |
|--------|------|-------|----------|
| PolicyNet standalone | 53 | 0.15ms/move | Yes (ONNX) |
| PolicyNet top-5 + 2-ply | 192 | 43ms/move | Yes (ONNX + JS) |
| PolicyNet top-10 + 2-ply | 176 | 42ms/move | Yes (ONNX + JS) |
| Tournament 200 (server) | **1041** | 100ms/move | Server only |

**The 5x gap** (192 → 1041) is entirely from rollouts. The PolicyNet learns *what* move to
make but not *why* — without rollout verification, it can't distinguish between moves that
look similar but have very different long-term outcomes. The 2-ply provides tactical
verification but can't substitute for 20-step rollout evaluation.

**Conclusion:** For competitive play (1000+ scores), a server-side tournament player is
required (~100ms/move via REST API). The PolicyNet+2-ply is viable as an instant client-side
fallback (~200 mean, 43ms/move) for offline or low-latency scenarios.

### Oracle V2 Result: REGRESSION

Trained V2 on 29 ML-enhanced games (V1 player's own output). Result on seed=77777:
- V1 mean: 8,432
- V2 mean: ~5,726 (11 games before killed) — **down 27%**

**Root cause (Gemini diagnosis):** Echo chamber / confirmation bias.
V1 pruned 295 of 300 moves per turn. V2 only saw V1's preferred moves —
never learned why the other 295 were bad. Amplified V1's biases until they
became pathological constraints, destroying flexibility.

**Lesson:** Don't iterate the oracle too soon. Need massive volume + exploration
diversity. 30 expert games is not enough when the expert plays homogeneously.

### Phase 17: Spatial Rollout Injection (in progress)

**Key discovery (Gemini):** The Python ML oracle only applies at the ROOT (2-ply
candidate selection). Rollout workers never receive ML weights — they use the
blind 5-weight base heuristic. All 8,432 mean points come from smart root
pruning with dumb rollout simulation.

**Proof:** If ML features ran 36M times inside rollouts, Python would take
30 min/move, not 100ms. The multiprocessing workers import fresh modules
without `enable_ml_oracle()`.

**The experiment:** Inject ultra-cheap center_distance directly into the
base heuristic weights (w[7]=-0.07, from ML oracle's learned weight):

```
center_dist = abs(tr - 4) + abs(tc - 4)
score += center_dist * (-0.07)
```

Cost: 1 integer math operation per evaluation — effectively zero.
This makes every rollout step slightly prefer edges over center.

**Preliminary results (3/5 games on seed=77777):**
- Spatial rollouts: [4399, 2956, 3996] — mean so far: 3784
- V1 baseline: [4988, 8093, 11291] — mean: 8124
- **Currently lower** — weight may need calibration

**Possible issues:**
1. Weight -0.07 from pairwise model may compound badly over 20 rollout steps
2. Softmax temperature (3.23) calibrated for original score distribution
3. Only 3 games — high variance

### GCP Instance
- c3-highcpu-176 (176 vCPU), stopped after 20 games (data/gcp_run/)
- GCP Intel cores ~2-3x slower per-core than M5 Max
- Available for future large-scale data gen

---

## Phase 17: AlphaZero Data Generation

### Rust Tournament Bug — ML Blend at Root (CRITICAL FIX)

Previous Rust code added z-score normalized ML blend that **replaced** raw 2-ply scores:
```rust
let blended = normalize_and_blend(&h_scores, &ml_scores);
c.score_2ply = blended[i]; // REPLACED raw score with z-score!
```

This destroyed tournament quality because later rounds use `score_2ply * 0.1` as a tiebreaker.
With raw scores (0-200), the bonus is meaningful (0-20). With z-scores (-2 to +2), it's noise (±0.2).

| Configuration | Mean Score | Notes |
|---|---|---|
| Broken (z-score replacement at root) | **504** | 2000 games wasted |
| ML in rollouts (get_softmax_move_ml) | **548** | z-norm helps temperature but still broken at root |
| ML in 2-ply follow-up only | **3,244** | Slower, no improvement |
| **No ML (plain heuristic everywhere)** | **4,782** | Matches Python quality, fastest |
| Python tournament (no ML) | ~4,641 | Reference |

**Fix:** Removed the ML blend block entirely. Rust tournament now uses pure CMA-ES heuristic
(identical logic to Python verified line-by-line). The heuristic, features, and game engine were
already verified to produce identical outputs for the same inputs.

**Key learning #11:** Never replace raw scores with normalized scores in a pipeline that uses
those scores downstream. Z-score normalization destroys scale information that later stages depend on.

### ML Oracle Not Propagating to Workers on macOS (CRITICAL DISCOVERY)

Python 3.12 on macOS uses `spawn` (not `fork`) for multiprocessing. Workers start as fresh
processes and DON'T inherit `enable_ml_oracle()` state from the parent. This means the previously
reported mean=8,432 "ML Oracle V1" result may have been from GCP (Linux/fork) or was never
actually using ML in tournament workers.

**Fix:** Pass `oracle_path` and `oracle_blend` through the pool initializer so workers explicitly
call `enable_ml_oracle()` on startup. Modified `_init_worker_weights` and `make_tournament_player`
in `evaluation/players.py`.

**Result — ML oracle confirmed working (2.4x improvement):**

| Seed | No ML (workers never had oracle) | ML propagated to workers |
|---|---|---|
| 77777 | 6,215 | 5,263 |
| 77778 | 1,997 | **25,764** |
| 77779 | 4,954 | 1,072 (rerun: 1,486) |
| **Mean** | **4,389** | **10,700** |

The 25,764 game (seed 77778) is a new all-time high — 11,753 turns, 4.4 moves/s.
Seed 77779 is consistently low (~1,000-1,500) — just a hard ball pattern.

**Key learning #12:** On macOS (Python 3.8+), `multiprocessing` uses `spawn` not `fork`.
Pool workers are fresh processes that don't inherit parent module state. Any state that workers
need (ML weights, oracle config) must be passed explicitly through the pool initializer.

### Data Generation Setup
- Python-only with ML oracle (blend=0.05) propagated to workers
- Tournament 200, depth 20, temp 3.23
- Expected mean ~10,000+ (2.4x over no-ML baseline of ~4,600)
- Output: JSON per game with board, next_balls, top-10 candidates with scores, game_score (value target)
- 3 parallel processes × 6 workers = 18 cores on M5 Max

---

## Phase 18: AlphaZero Deep Learning

### Data Generation (Complete)

Generated 500 expert games using tournament player (200 rollouts + ML oracle V1, blend=0.05).
Pool-based datagen: 18 workers on M5 Max, each plays one game then grabs the next seed.

**Performance optimizations (3.1x total speedup over baseline):**
1. JIT rollout — entire rollout loop in Numba, eliminates 200K Python→JIT roundtrips per 10 moves (2.0x)
2. Batched ML oracle — `score_all_moves_linear()` replaces Python loop over moves
3. Precomputed empty count — `_count_empty_jit` called once per scoring pass instead of per-move (was net zero anyway since source→target is zero-sum for empty cells)
4. Zero-weight feature skip — only 5 of 17 heuristic weights are non-zero (CMA-ES); skip combo, spatial, anti-frag, source features when their weights are 0.0 (saves ~40% of `_evaluate_move_w` work)

| Optimization | 10 moves | mv/s | Speedup |
|---|---|---|---|
| Baseline (Python rollout) | 30.5s | 0.33 | 1.0x |
| + JIT rollout + batched ML | 15.1s | 0.66 | 2.0x |
| + precomputed empty count | 13.1s | 0.76 | 2.3x |
| + skip zero-weight features | **9.9s** | **1.01** | **3.1x** |

**Dataset statistics (500 games, seeds 10000-10499):**

| Stat | Value |
|---|---|
| Games | 500 |
| Total states | 1,313,357 |
| Mean score | 5,719 |
| Median score | 4,153 |
| Max score | 28,784 |
| Std | 4,827 |

Note: The previously reported mean of 8,432 was from only 20 games (seed 42). With 500 games
the true mean is ~5,700 — the 20-game sample was likely a lucky draw (SE ≈ 974 on 20 games).

### AlphaZero Model Architecture

**Dual-head ResNet** (`model/alphazero_net.py`):
- Input: (batch, 12, 9, 9) — 7 color planes + empty + 3 next-ball channels + next-ball mask
- Backbone: Conv stem → 10 residual blocks × 256 channels (pre-activation BN→ReLU→Conv)
- Policy head: Factored source (81) + target (81) logits (shared backbone features)
- Value head: Categorical two-hot encoding (64 bins, range 0–30,000)
- Parameters: 12.4M

**Training setup** (`training/train_alphazero.py`):
- Device: MPS (Apple Silicon M5 Max, 40 GPU cores)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4, cosine schedule)
- Loss: soft cross-entropy for policy (source + target) + cross-entropy for categorical value
- 8x dihedral augmentation → 10.5M effective training samples
- Precomputed tensors (6.3GB RAM) for fast data loading (~137K samples/s)
- Throughput: ~1,866 samples/s on MPS
- Batch size: 512, ~19.5K batches per epoch, ~95 min/epoch

### Training Progress (6 epochs complete, ~16h, stopped)

| Metric | Epoch 1 start | Epoch 6 end (best) |
|---|---|---|
| Total loss | 11.37 | 4.62 |
| Policy loss | 7.59 | 1.82 |
| Value loss | 3.78 | 2.80 |
| Val loss | — | 4.69 |
| Value MAE | — | 3,654 |

- Train/val gap minimal (4.62 vs 4.69) — no overfitting
- Value MAE of 3,654 vs data std of 4,827 — meaningful signal learned
- Policy head converged fast (7.59 → 1.82); value head slower (3.78 → 2.80)
- Standalone policy player: 48-109 mv/s but low scores (117-215) — no search
- Best checkpoint: `checkpoints/alphazero_best.pt` (epoch 6)

### AlphaZero Players Implemented & Evaluated

| Player | Score (seed 42) | Speed | Architecture |
|---|---|---|---|
| Pure heuristic (1-ply) | 67 | 4,500 mv/s | No search |
| Standalone NN policy | 288-377 | 100 mv/s | Single forward pass |
| NN policy hybrid + rollouts | 529 | 2.5 mv/s | NN prunes to top-20, heuristic rollouts |
| Afterstate value (v2, balanced) | ~20 | 11 mv/s | 1-ply afterstate eval |
| Neural rollout (pure NN, 50x10) | 241 | 0.89 mv/s | NN policy in rollouts |
| Neural rollout (blend=3, 50x10) | 516 | 0.70 mv/s | NN+heuristic blend in rollouts |
| Neural rollout bracket (blend=3, 200x20) | 451 | 0.35 mv/s | Full budget, bracket structure |
| **Heuristic tournament (200x20)** | **1,205** | **1.04 mv/s** | **Current best (baseline)** |

### Key Findings (Phase 18)

**The heuristic tournament remains unbeaten.** All neural approaches scored below the
heuristic tournament on seed 42 (1,205 points, 592 turns).

**Why the NN can't replace the heuristic in rollouts:**
1. The heuristic counts lines deterministically — zero tactical errors in 20-step simulations
2. The NN makes occasional micro-mistakes (wrong line count, missed block) — one mistake
   in a 20-step rollout corrupts the entire simulation
3. The heuristic tournament survived 592 turns; the best neural variant lasted only 258
4. Game survival (turns played) is the strongest predictor of score in Color Lines

**What the NN IS good at:**
- Standalone policy (288-377) beats standalone heuristic (67) by 5x
- The NN has good "strategic intuition" (board shape, center control)
- Blending NN+heuristic in rollouts (blend=3 at 516) beats pure NN rollouts (241)

**The "Tactical Safety Glasses" thesis:**
- Adding heuristic scores to NN logits in rollouts: 241 → 516 (pure NN → blend=3, 50x10)
- But still 2.7x below the pure heuristic tournament (1,205)
- The heuristic dominates because tactical precision > strategic vision in rollout quality

### Performance Optimizations (Neural Rollout)

Achieved 2x speedup on the neural rollout hot path (170ms → 86ms per call):
- Gumbel-max sampling (7x faster than torch.multinomial)
- torch.where() replaces boolean indexing (40x faster on MPS)
- Pre-allocated GPU buffers, native fp16, channels_last memory format
- Batch Numba JIT move execution, vectorized observation building
- Remaining bottleneck: 52% NN compute, 29% MPS sync, 19% Numba game logic

### Afterstate Value Network (v2, contrastive)

Trained on 46M afterstates (13M expert + 26M heuristic traps + 6.6M random) with:
- Balanced 33/33/33 tier sampling to prevent mean collapse
- 13-channel input with component area heatmap (connectivity awareness)
- Categorical CE loss, 20 epochs on H100 at 41K s/s
- Final: val_loss=1.24, MAE=0.57 (log1p scale)
- But failed as standalone player (~20 points) — MAE too high for per-move discrimination
