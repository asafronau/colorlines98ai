# Color Lines 98 AI

An AlphaZero-inspired AI for [Color Lines 98](https://en.wikipedia.org/wiki/Lines_(video_game)), built from scratch as a deep learning project. The game is a single-player puzzle on a 9x9 board with 7 colors: move balls to form lines of 5+, while 3 new balls spawn each turn. It's a survival game: score is almost exactly a function of how long you stay alive (about 2 points per turn at every skill level), and with good enough play a game never has to end.

**This is an ongoing project.** I'm a SWE learning ML through building, and this repository documents the full journey: every experiment, every failure, and every discovery.

## Current results

The current model is a **3M-parameter ResNet (18 blocks × 96 channels)** that plays greedily: one forward pass per move, no search. Games are played in the C++ engine on fixed seed banks (2,600,000–2,603,999, 2,000–4,000 games) and stopped at 100,000 turns (~204,000 points). Because hazard is flat with game age (`~0.24–0.32` deaths per 100k turns across 0–5k, 5–20k, 20–50k, and 50–100k turns), models are judged by **deaths per 100,000 turns** (with 95% Poisson confidence intervals), **mean turns between failures (MTBF)**, the share of games surviving to the 100,000-turn cap, and lower-tail percentiles (`P1`, `P5`, `P10`, `P25`):

| Player | P1 | P5 | P10 | P25 | Median | Reach 100k turns | Deaths / 100k turns [95% CI] | MTBF (turns) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 18b96 e40 (old policy head) | 712 | 1,246 | 1,806 | 4,500 | 9,948 | 0.0% | ~10.0 | ~10,000 |
| A1: 18b96 + PAIR2 policy head + legal-move mask | 1,182 | 2,844 | 4,552 | 12,468 | 28,442 | 1.0% | ~3.5 | ~28,500 |
| A2: A1 + turn 1 (600-sim crisis corrections) | 1,924 | 8,426 | 16,085 | 43,272 | 102,938 | 25.3% | ~1.35 | ~74,000 |
| A3: A2 + turn 2 (600 sims) | 4,478 | 20,256 | 39,270 | 107,085 | at the cap | 55.4% | 0.59 [0.54, 0.64] | 170,000 |
| A4: A3 + turn 3 (600 sims) | 4,548 | 20,184 | 41,715 | 118,015 | at the cap | 61.1% | 0.49 [0.46, 0.53] | 202,343 |
| A6: A4 + turns 4–5 (800 sims) | 5,588 | 27,986 | 56,355 | 155,380 | at the cap | 69.1% | 0.37 [0.34, 0.40] | 269,759 |
| A8: A6 + turn 6 (1,600 sims, 4k games) | 7,127 | 35,522 | 69,193 | 180,295 | at the cap | 72.7% | 0.321 [0.302, 0.340] | 311,554 |
| Born-again A8 (from scratch on A8 8-view labels, 4k games) | 8,116 | 36,435 | 74,442 | 197,602 | at the cap | 72.8% | 0.318 [0.299, 0.337] | 314,485 |
| A9: Born-again A8 + 2,400-sim crisis turn (4k games) | 10,733 | 47,060 | 91,342 | at the cap | at the cap | 78.3% | 0.244 [0.228, 0.261] | 409,151 |
| **A10: A9 + 1,600-sim crisis turn (4k games)** | **11,746** | **50,149** | **103,231** | **at the cap** | **at the cap** | **80.6%** | **0.215 [0.201, 0.231]** | **464,080** |

Across 4,000 confirmation games (seeds 2,600,000–2,603,999, capped at 100,000 turns), **A10** (`ta_A9_e4_a0.5`) cuts total deaths from `1,094` (`A8`) to **`775` (`-33.0%` vs `A8`, `-11.8%` vs `A9`)**, raises MTBF to **464,080 turns (~940,000 points)**, and reaches the 100,000-turn cap in **80.6%** of games — matching `A8`'s 8-view symmetry ensemble (`0.210`) in a **single forward pass** and pushing **P25 to the 100k-turn cap** (~203,700 points) and **P10 past 100,000 points**.

Every evaluation is logged in [`alphatrain/EVAL.md`](alphatrain/EVAL.md), and every experiment in [`alphatrain/HISTORY.md`](alphatrain/HISTORY.md) (286 entries).

### The crisis-mining flywheel and why born-again resets unlock further turns

Each flywheel turn replays deaths from 15 and 30 moves earlier using PUCT MCTS (with a neural survival head as leaf evaluator), extracts the first 45 searched moves of each replay (`~200k–250k` states), fine-tunes the current actor for 4 epochs with BatchNorm running statistics frozen (`--freeze-bn`), and extrapolates along the task vector:

$$\theta_{\text{next}} = \theta_{\text{base}} + \alpha \cdot (\theta_{\text{ft}} - \theta_{\text{base}})$$

1. **Fixing the leaf evaluator (`A1 → A2 → A4`):** For two months the crisis-mining tool had loaded the survival head but searched with an older 27-feature heuristic value estimate. Fixing the C++ search to evaluate leaves with the neural survival head immediately unlocked the flywheel, taking the 100k-cap survival share from `1.0%` (`A1`) to `25.3%` (`A2`), `55.4%` (`A3`), and `61.1%` (`A4`, `0.49` deaths/100k turns).
2. **Scaling GPU batching and teacher simulations (`A4 → A8`):** Rewriting the C++ inference server to wait for all 256 worker threads (`--batch-wait-us 3000`) sped up crisis mining by `6.1×` (`~34,000` neural evaluations/s on one Mac). Stepping MCTS from 600 to 800 and 1,600 simulations drove the death rate down to `0.38` (`A5`), `0.37` (`A6`), and `0.321` (`A8`, `72.7%` capped).
3. **Why `A8` stalled — and how `Born-again A8 → A9 → A10` broke through (`0.321 → 0.244 → 0.215`):**
   - By `A8`, the actor sat at the end of **6 stacked task-arithmetic extrapolations** ($\sum \alpha = 7.0$) on top of `A1`'s frozen BatchNorm statistics (`56.5%` eval/train BN agreement on crisis boards). Applying a 7th task vector (`A8c`, 2,400 sims) or a 120-move slide-window turn (`A8s`) on top of `A8` produced zero gain (`0.32`).
   - Training a fresh 18×96 network from scratch (`born_again_A8_ep35`) on 12.9M states labeled by `A8`'s 8-symmetry ensemble reproduced `A8`'s single-pass death rate (`0.318` vs `0.321`), because the born-again corpus relabeled crisis states with search-free 8-view averages instead of MCTS search targets. However, it reset the weights to a clean $\alpha = 0$ checkpoint with fresh BatchNorm statistics (`84.2%` eval/train BN agreement).
   - Fine-tuning `born_again_A8_ep35` on the 2,400-sim `A8c` crisis corpus at $\alpha = 1.5$ (**`A9`**) dropped deaths per 100k turns from `0.321` to **`0.244` (`-24.0%`)**, and running the next 1,600-sim flywheel turn on `A9` at $\alpha = 0.5$ (**`A10`**) dropped deaths further to **`0.215` (`-33.0%` vs `A8`, `80.6%` capped)** across 4,000 games.

### What changed: the policy head

The small model had stalled at ~15k, and the working assumption was that 3M parameters weren't enough. They were enough; the policy head was the bottleneck. The old head produced, at every destination cell, 81 logits indexed by the *absolute* position of the source ball (a 1×1 convolution). "Which ball to move" was therefore 81 separate position-specific readouts that shared nothing across positions or rotations.

The **PAIR2** head scores each (source, destination) pair from shared per-cell embeddings:

```
logit(s → d) = u(s)·w(d)/√k + a(s) + b(d)
             + gated bilinear terms for pairs on a common row, column or diagonal within 4 cells,
               and for orthogonally adjacent pairs
```

It is exactly equivariant under the 8 board symmetries and still outputs 81 × 81 = 6,561 move logits, so the C++ engine didn't need changes. It was trained from scratch with the same recipe on the same 12.5M search-labeled positions, plus a loss that masks illegal moves. The result is **2.9× the old head's score** at the same size, on par with the 4× larger 256-channel model.

### Symmetry, capacity, and value-head lessons

- **8-view symmetry ensemble (`--tta 8`).** Averaging `A8`'s policy over the 8 $D_4$ board symmetries at inference cuts deaths by **38%** (`0.21` vs `0.34` deaths/100k turns over 25k turns) with no training. Earlier $D_4$ training augmentation rotated the board planes but not the 4 line-direction input channels (making 6 of 8 views wrong); fixing that LUT bug and adding $S_7$ color-permutation augmentation closed most of the gap.
- **Built-in equivariance (`p4m` and `c7`) vs. standard ResNet (`18×96`).**
  - **Color-equivariant slot networks (`alphatrain/model_c7.py`):** Maintaining 7 per-color feature slots with shared weights (DeepSets-style `slot_mean` + `slot_max` pooling) plus a color-free spatial stream dramatically outperforms plain CNNs in small-data/small-model regimes: a 4-block `c7` (`k12/s32 + smax`, `189k` params) achieves **`320` deaths/100k turns (`P50 = 513`)** after 3 epochs on 153k states, compared to **`1,214` deaths/100k (`P50 = 105`)** for a `446k`-param standard `4×72` CNN and **`1,110` (`P50 = 122`)** with exact $D_4 \times S_7$ input canonicalization (`--canon`). On the full 12.9M-state corpus with $8\times D_4 \times 5,040\times S_7$ augmentation, the plain `18×96` ResNet (`3.07M` params) overtakes `505k`-param `c7` after epoch 5 (`0.32` vs `0.49` at epoch 15–20) because spatial capacity binds.
  - **Rotation-equivariant group convolutions (`alphatrain/model_p4m.py`):** Exact $D_4$ group convolutions (`p4m`) learn more slowly per epoch than standard convolutions at equal compute (`0.77` vs `0.44` deaths/100k at epoch 7–8).
- **Anatomy of a death ("slides") and value-network unfreezing.**
  - Strong actors rarely blunder on a single turn; deaths are **60–120-move slides** starting around 40–45 balls where the board gradually loses partial lines of $\ge 3$ same-color balls (`AUC 0.19–0.22`) before filling up.
  - All value heads trained on the *frozen* policy backbone saturated at `~0.75–0.77` within-bin death AUC because the policy trunk discards survival-specific congestion features. Unfreezing the last 6 residual blocks (`--unfreeze-blocks 6 --cat-obs --augment`) jumps short-horizon ($H=25$) within-bin death AUC from **`0.773` to `0.845`** (`+18.7` pp in the `41–45` ball slide-onset bin), while $H=200$ AUC remains at `0.767` due to the intrinsic entropy of 200 turns (`600` random balls) of future spawns.
- **Two silent bugs caught along the way:**
  - BatchNorm running variances above 65,504 overflowed to `inf` in `fp16` TorchScript exports. Both `export_ts.py` and `export_policy_value.py` now rescale high-variance channels exactly before folding/exporting.
  - A zero-epoch warmup trained whole runs at 10% of the requested learning rate. Fixed, with a regression test.

## How it works

- **Game engines:** Python (Numba) in `game/` and C++ in `alphatrain/inference_cpp/`, checked against each other with golden tests.
- **Input:** 18 channels: 7 color planes, empty cells, next-ball positions, connected-component areas, and line potentials in 4 directions.
- **Models:** `alphatrain/model.py` (`PolicyNet` 18×96 ResNet + `PAIR2` policy head), `alphatrain/model_c7.py` ($S_7$ color-equivariant slot network), and `alphatrain/model_p4m.py` ($D_4$ group-equivariant network). A separate **survival head** (`ValueHead` or `SpatialValueHead` in `alphatrain/value_head.py`) predicts the probability of surviving the next $H \in \{25, 50, 100, 200\}$ turns for MCTS leaf evaluation.
- **Labels:** PUCT MCTS (`600–2,400` simulations, `c_puct = 1.5`, `q_weight = 2.0`) in C++ (`build/mcts_crisis`) using the survival head as leaf evaluator. Crisis mining probes thousands of games and rewinds 15 and 30 moves (or 120 moves for slide windows) before each death to search for an escape.
- **Flywheel turn** (`scripts/flywheel_turn.sh`):
  1. Export the actor to TorchScript (`export_ts.py`).
  2. Record 1,000 own games (capped at 100k turns) in C++ (`build/eval --record-dir ...`).
  3. Train the survival head (`train_value_head.py`) and export the fused policy+value module (`export_policy_value.py`).
  4. Mine crisis windows in C++ (`build/mcts_crisis` with 256 threads batching onto MPS `fp16`).
  5. Build the training tensor and run anomaly checks (`anomaly_checks.py`).
  6. Fine-tune for 4 epochs on MPS (`train_path_b.py --freeze-bn --lr 1e-4 --target-temperature 0.5`).
  7. Sweep task-arithmetic $\alpha$ (`scripts/ta_sweep.sh`) and gate-evaluate in C++ (`eval_log.py`).
- **Evaluation:** greedy single-pass policy in the C++ engine (`build/eval`, MPS `fp16` with folded BatchNorms), 2,000–4,000 games on fixed seed banks (`2,600,000–2,603,999`), capped at 100,000 turns.

## How we got here

| Stage | Metric | Key change |
|---|---:|---|
| Heuristic tournament (200 CPU rollouts per move) | 5,700 mean | Search over a hand-tuned heuristic |
| pillar2z | 7,460 mean | First NN-driven AlphaZero loop (V12 corpus) |
| pillar3a | 14,294 mean | Sharpened distillation targets (T = 0.25) |
| pillar3b | 18,865 mean | V13 self-play corpus |
| pillar3d_mC | 24,249 mean | Crisis-mining corrections |
| pillar3f | 31,617 mean | Crisis fine-tune merged by task arithmetic |
| pillar3k (11.9M params) | 43,390 mean | Decisiveness-weighted crisis distillation |
| 18b96 e40 (3M params) | 14,296 mean | Small model trained from scratch on all data (12.5M positions) |
| A1 (3M params) | 41,938 mean (1.0% cap) | PAIR2 policy head + legal-move mask |
| A2 (3M params) | median 102,938 (25.3% cap) | First flywheel turn with the fixed neural-leaf search |
| A3 (3M params) | 55.4% cap (0.59 / 100k) | Second flywheel turn (600 sims) |
| A4 (3M params) | 61.1% cap (0.49 / 100k) | Third flywheel turn (600 sims) |
| A5–A6 (3M params) | 69.1% cap (0.37 / 100k) | 256-thread GPU server batching (`6.1×` faster) + 800-sim teacher |
| A7–A8 (3M params) | 72.7% cap (0.321 / 100k) | 1,600-sim crisis turn + 120-move slide-window turn |
| Born-again A8 (3M params) | 72.8% cap (0.318 / 100k) | From-scratch reset on 12.9M 8-view-labeled states ($\alpha=0$, fresh BNs) |
| A9 (3M params) | 78.3% cap (0.244 / 100k) | 2,400-sim crisis task arithmetic ($\alpha=1.5$) on fresh Born-again A8 (`P25` capped) |
| **A10 (3M params)** | **80.6% cap (0.215 / 100k)** | **1,600-sim crisis flywheel turn ($\alpha=0.5$) on A9; matches A8's 8-view ensemble in 1 pass** |

All rows after the heuristic are greedy policy play without search. From A2 on, games are capped at 100,000 turns and compared by deaths per 100k turns, capped share, and lower-tail percentiles.

## Earlier discoveries

- **Score = survival.** Every value target we tried (board density, score, TD returns) reduced to turns survived. Score per turn is nearly constant (~2), so survival is the only thing to optimize.
- **The shared backbone conflict.** Training policy and value heads on one trunk was zero-sum. The value head never learned (SNR 0.03) and dragged the policy down; dropping it from training gave +85%.
- **The tipping point.** Death spirals begin around 41 empty squares, but 50 turns before death the boards look healthy by count (43.3 vs 42.8 empty). The difference is structural: connectivity, partitions, and multi-color clusters.
- **Crisis mining.** Greedy games are nearly free, so thousands of seeds can be probed for failures. Deep search then replays only the positions where the model dies.
- **Static search depth.** Cutting simulations on "confident" moves saves compute but confirms the model's confident mistakes. Label quality needs full search on every move.

## Project structure

```
game/                   Python game engine (Numba): board, pathfinding, line clears, spawns, RNG
alphatrain/
  model.py              ResNet trunk and policy heads (abs / pair / pair2)
  model_c7.py           S7 color-equivariant slot trunk + PAIR2 head
  model_p4m.py          D4 rotation-equivariant group-conv trunk + PAIR2 head
  model_variants.py     Unified loader for standard, c7, and p4m checkpoints
  value_head.py         Survival heads (GAP ValueHead and SpatialValueHead)
  train_path_b.py       Policy trainer: soft/hard CE, legal-move mask, D4 x S7 augmentation
  dataset.py            GPU-resident training tensors and LUT augmentation
  observation.py        18-channel observation builder
  canonical.py          Canonical form under board symmetries x color relabeling
  inference_cpp/        C++/LibTorch engine: greedy eval, MCTS self-play, crisis mining, relabeling
  scripts/              Corpus builders, value-head trainer, survival/hazard stats, eval_log.py
  tests/                Unit and equivariance tests
  EVAL.md, HISTORY.md   Evaluation log and experiment history (285 entries)
rust_engine/            Rust game engine (earlier data generation)
play_gui.py             Pygame GUI with AI hints and auto-play
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, and Numba. The C++ engine needs CMake and LibTorch. [`alphatrain/inference_cpp/README.md`](alphatrain/inference_cpp/README.md) covers the model export and build. A standard 2,000-game gate evaluation (or add `--tta 8` for 8-symmetry averaging):

```bash
python -m alphatrain.scripts.eval_log run --model alphatrain/data/ta_baA8c_e4_a1.5.pt --desc "A9 gate"
```

## Acknowledgments

Built with significant assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic) and Google's Gemini for code generation, experiment design, and analysis. The project demonstrates human-AI collaboration on a complex ML problem: human domain expertise (game intuition, experimental direction, engineering/performance insights) combined with AI capabilities (code implementation, data analysis, systematic diagnosis).
