# Color Lines 98 AI

An AlphaZero-inspired AI for [Color Lines 98](https://en.wikipedia.org/wiki/Lines_(video_game)), built from scratch as a deep learning project. The game is a single-player puzzle on a 9x9 board with 7 colors: move balls to form lines of 5+, while 3 new balls spawn each turn. It's a survival game: score is almost exactly a function of how long you stay alive (about 2 points per turn at every skill level), and with good enough play a game never has to end.

**This is an ongoing project.** I'm a SWE learning ML through building, and this repository documents the full journey: every experiment, every failure, and every discovery.

## Current results

The current model is a **3M-parameter ResNet (18 blocks × 96 channels)** that plays greedily: one forward pass per move, no search. Games run to natural death (no turn cap) on a fixed bank of 1,000 seeds (2,600,000–2,600,999):

| Player | Mean | Median | P10 | P95 | Games < 1,000 | Max |
|---|---:|---:|---:|---:|---:|---:|
| 18b96 e40 (old policy head) | 14,296 | 9,948 | 1,806 | 42,773 | 4.4% | 96,692 |
| 18b96 e40, averaged over the 8 board symmetries | 21,436 | 14,778 | 2,449 | 60,291 | 3.3% | 148,976 |
| **A1: 18b96 + PAIR2 policy head + legal-move mask (epoch 40)** | **41,938** | **28,442** | **4,552** | **134,502** | **0.9%** | **338,096** |
| A1 at epoch 33, averaged over the 8 board symmetries | 47,622 | 33,624 | 5,487 | 141,758 | 1.3% | 340,408 |
| *Reference:* pillar3k, 10 blocks × 256 channels, 11.9M params (5,000 other seeds) | 43,390 | 31,016 | 5,010 | 126,379 | 1.3% | 337,411 |

Every evaluation is logged in [`alphatrain/EVAL.md`](alphatrain/EVAL.md), and every experiment in [`alphatrain/HISTORY.md`](alphatrain/HISTORY.md) (245 entries).

### What changed: the policy head

The small model had stalled at ~15k, and the working assumption was that 3M parameters weren't enough. They were enough; the policy head was the bottleneck. The old head produced, at every destination cell, 81 logits indexed by the *absolute* position of the source ball (a 1×1 convolution). "Which ball to move" was therefore 81 separate position-specific readouts that shared nothing across positions or rotations.

The **PAIR2** head scores each (source, destination) pair from shared per-cell embeddings:

```
logit(s → d) = u(s)·w(d)/√k + a(s) + b(d)
             + gated bilinear terms for pairs on a common row, column or diagonal within 4 cells,
               and for orthogonally adjacent pairs
```

It is exactly equivariant under the 8 board symmetries and still outputs 81 × 81 = 6,561 move logits, so the C++ engine didn't need changes. It was trained from scratch with the same recipe on the same 12.5M search-labeled positions, plus a loss that masks illegal moves. The result is **2.9× the old head's score** at the same size, on par with the 4× larger 256-channel model.

### Other lessons from the latest round

- **Symmetry.** Averaging the old model over the 8 board symmetries at inference gave +50% with no training, so the model had not learned the symmetry. The training code's D4 augmentation rotated the board planes but not the 4 line-direction input channels, which made 6 of 8 views wrong. Fixed.
- **A student reaches its labels' strength.** Students trained on the symmetry-averaged e40's moves plateau at ~21k, which is that teacher's own level. The original search labels take the same student to 42k. The next gain has to come from stronger labels.
- **Fine-tuning a converged model is learning-rate drift.** Fine-tuning e40 on its *own* moves loses 9.6% at lr 5e-5, 5% at 2e-5, and nothing at 5e-6. Retraining from scratch works better.
- **Search overrides are noisy.** A single 400-simulation search sometimes overrides the policy; an independent re-search reproduces only 37% of those overrides. Distilling them one by one churns more good moves than it fixes.
- **Two silent bugs.**
  - BatchNorm running variances above 65,504 overflowed to `inf` in the fp16 export. The exporter now rescales them exactly.
  - A zero-epoch warmup trained whole runs at 10% of the requested learning rate. Fixed, with a regression test.

## How it works

- **Game engines:** Python (Numba) in `game/` and C++ in `alphatrain/inference_cpp/`, checked against each other with golden tests.
- **Input:** 18 channels: 7 color planes, empty cells, next-ball positions, connected-component areas, and line potentials in 4 directions.
- **Model:** a ResNet trunk plus the PAIR2 policy head. A separate **survival head** is trained on the frozen trunk to predict the probability of surviving the next H turns, at several horizons. Only the search uses it.
- **Labels:** PUCT MCTS with 400–600 simulations, c_puct 1.5, Q weight 2.0, and the survival head as leaf value. Two sources:
  - self-play games;
  - **crisis mining:** play the policy greedily, then at each death rewind 15 and 30 moves and replay with search to find an escape.
- **Training:** from scratch, on Colab.
  - Loss: hard cross-entropy on the search's move, with illegal moves masked.
  - D4 augmentation.
  - Batch 32,768; lr 3e-3 with 1 warmup epoch, then cosine; 40 epochs.
  - Command: `alphatrain/train_path_b.py --policy-head pair2 --legal-mask-loss`.
- **Evaluation:** greedy policy only, in the C++ engine (MPS, fp16), 1,000 uncapped games on the fixed seed bank. The policy plays alone; search is used only to make training labels.

**Next:** flywheel generation 1. A1 and its own survival head generate new labels (crisis mining on A1's deaths, plus self-play). A2 then trains from scratch on the old and new labels. The bar is +20% (≥ 50k).

## How we got here

| Stage | Mean | Key change |
|---|---:|---|
| Heuristic tournament (200 CPU rollouts per move) | 5,700 | Search over a hand-tuned heuristic |
| pillar2z | 7,460 | First NN-driven AlphaZero loop (V12 corpus) |
| pillar3a | 14,294 | Sharpened distillation targets (T = 0.25) |
| pillar3b | 18,865 | V13 self-play corpus |
| pillar3d_mC | 24,249 | Crisis-mining corrections |
| pillar3f | 31,617 | Crisis fine-tune merged by task arithmetic |
| pillar3k (11.9M params) | 43,390 | Decisiveness-weighted crisis distillation |
| 18b96 e40 (3M params) | 14,296 | Small model trained from scratch on all data (12.5M positions) |
| **A1 (3M params)** | **41,938** | **PAIR2 policy head + legal-move mask** |

All rows after the heuristic are greedy policy play without search. Sample sizes and seed banks differ by row (100–5,000 games); HISTORY.md has each one.

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
  value_head.py         Survival head (trained on a frozen trunk; used by search)
  train_path_b.py       Trainer: hard CE, legal-move mask, D4 augmentation
  dataset.py            GPU-resident training tensors and augmentation
  observation.py        18-channel observation builder
  canonical.py          Canonical form under board symmetries x color relabeling
  inference_cpp/        C++/LibTorch engine: greedy eval, MCTS self-play, crisis mining, relabeling
  scripts/              Corpus builders, diagnostics, eval_log.py
  tests/                Unit tests
  EVAL.md, HISTORY.md   Evaluation log and experiment history
rust_engine/            Rust game engine (earlier data generation)
play_gui.py             Pygame GUI with AI hints and auto-play
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, and Numba. The C++ engine needs CMake and LibTorch. [`alphatrain/inference_cpp/README.md`](alphatrain/inference_cpp/README.md) covers the model export and build. A gate evaluation, with `--tta 8` for symmetry averaging:

```bash
cd alphatrain/inference_cpp
./build/eval --model data/<model>_ts.pt --device mps --batch 500 --seed-start 2600000 --seed-end 2601000
```

## Acknowledgments

Built with significant assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic) for code generation, experiment design, and analysis. AI peer review from Google's Gemini informed several architectural decisions. The project demonstrates human-AI collaboration on a complex ML problem: human domain expertise (game intuition, experimental direction, engineering/performance insights) combined with AI capabilities (code implementation, data analysis, systematic diagnosis).
