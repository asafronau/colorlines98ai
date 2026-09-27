# Color Lines 98 AI

An AlphaZero-inspired AI for [Color Lines 98](https://en.wikipedia.org/wiki/Lines_(video_game)), built from scratch as a deep learning project. The game is a single-player puzzle on a 9x9 board with 7 colors: move balls to form lines of 5+, while 3 new balls spawn each turn. It's a survival game: score is almost exactly a function of how long you stay alive (about 2 points per turn at every skill level), and with good enough play a game never has to end.

**This is an ongoing project.** I'm a SWE learning ML through building, and this repository documents the full journey: every experiment, every failure, and every discovery.

## Current results

The current model is a **3M-parameter ResNet (18 blocks × 96 channels)** that plays greedily: one forward pass per move, no search. Games are played in the C++ engine on a fixed bank of 1,000 seeds (2,600,000–2,600,999) and stopped at 100,000 turns. Games are getting long, so results are judged by percentiles and by the share of games that survive to the cap, not by the mean:

| Player | P10 | P25 | Median | P75 | Reach 100k turns | Games < 1,000 |
|---|---:|---:|---:|---:|---:|---:|
| 18b96 e40 (old policy head) | 1,806 | 4,500 | 9,948 | 19,710 | 0.0% | 4.4% |
| 18b96 e40, averaged over the 8 board symmetries | 2,449 | 6,168 | 14,778 | 32,317 | 0.0% | 3.3% |
| A1: 18b96 + PAIR2 policy head + legal-move mask | 4,552 | 12,468 | 28,442 | 56,904 | 1.0% | 0.9% |
| A2: A1 + first flywheel turn (fixed-teacher crisis corrections) | 16,085 | 43,272 | 102,938 | 203,505 | 25.3% | 0.1% |
| **A3 (candidate): A2 + second flywheel turn** | **39,270** | **107,085** | **at the cap** | **at the cap** | **55.4%** | **0.2%** |

A2 held up on two more banks: seeds 3,000,000–3,000,999 (P10 15,884 / median 104,629 / 24.9% reaching 100k turns, against A1's 4,785 / 30,387 / 0.9%) and 3,400,000–3,400,999 (16,470 / 104,883 / 27.0%). A3 is the second turn's result on the gate bank; its fresh-bank check is running. More than half of A3's games survive 100,000 turns, so its median now sits at the cap and the share of games reaching the cap, P10 and P25 are the numbers to watch. For reference, the 4× larger 256-channel pillar3k model had a median of 31,016 (5,000 other seeds).

Every evaluation is logged in [`alphatrain/EVAL.md`](alphatrain/EVAL.md), and every experiment in [`alphatrain/HISTORY.md`](alphatrain/HISTORY.md) (252 entries).

### The first flywheel turn: A1 → A2

A1's own search (600 simulations, its survival head as leaf value) replayed each of A1's deaths from 15 and 30 moves earlier, and the first 45 searched moves of each replay became training labels: 194k positions. A1 was fine-tuned on them for 4 epochs with BatchNorm statistics frozen, and A2 moves 2.5× as far as that fine-tune did, **A2 = A1 + 2.5 · (fine-tuned − A1)**. The fine-tune points in the right direction but stops short; the gain keeps growing up to 2.5–3× before it flattens.

It only worked after finding a bug. For two months the crisis-mining tool had loaded the survival head but searched with an older 27-feature value estimate, while its output files claimed otherwise. Replayed from the same states with the same spawns, the buggy search escaped 60% of A1's deaths against 55% for A1 alone; the fixed search escaped 72%. Every way of training on the buggy labels (weight blending, continuous training, correction-only fine-tunes) came out even with A1. The same recipe on the fixed labels tripled the median.

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
- **Labels:** PUCT MCTS with 400–600 simulations, c_puct 1.5, Q weight 2.0, and the survival head as leaf value. The main source is **crisis mining**: play the policy greedily, then at each death rewind 15 and 30 moves and replay with search to find an escape.
- **A1 (base model):** trained from scratch on Colab on 12.5M search-labeled positions.
  - Loss: hard cross-entropy on the search's move, with illegal moves masked.
  - D4 augmentation.
  - Batch 32,768; lr 3e-3 with 1 warmup epoch, then cosine; 40 epochs.
  - Command: `alphatrain/train_path_b.py --policy-head pair2 --legal-mask-loss`.
- **Flywheel turn** (`scripts/flywheel_turn.sh`), a few hours on one Mac:
  1. Record the actor's own games.
  2. Train its survival head on the frozen trunk.
  3. Mine its deaths with search.
  4. Fine-tune on the crisis windows with BatchNorm frozen.
  5. Sweep how far to push along the fine-tune's direction.
  6. Gate the result.
- **Evaluation:** greedy policy only, in the C++ engine (MPS, fp16), 1,000 games on the fixed seed bank, capped at 100,000 turns and judged by percentiles and the share of games reaching the cap. The policy plays alone; search is used only to make training labels.

**Second turn (A2 → A3):** the same script from A2, with A2's own survival head. 2,240 deaths in 3,000 probes (760 probes survived 100k turns), 4,480 replays, then **A3 = A2 + 2 · (fine-tuned − A2)**: P10 39,270 and 55.4% of games reaching 100k turns, from A2's 16,085 and 25.3%.

**Next:** confirm A3 on fresh seeds, then the third turn. Deaths are getting rare, so mining will need more probe seeds or a longer probe cap, and evals a higher turn cap.

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
| A1 (3M params) | 41,938 | PAIR2 policy head + legal-move mask |
| A2 (3M params) | median 102,938 | First flywheel turn with the fixed search |
| **A3 candidate (3M params)** | **55.4% reach 100k turns** | **Second flywheel turn** |

All rows after the heuristic are greedy policy play without search. Sample sizes and seed banks differ by row (100–5,000 games); HISTORY.md has each one. From A2 on, games are capped at 100,000 turns and compared by percentiles.

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
./build/eval --model data/<model>_ts.pt --device mps --batch 500 --seed-start 2600000 --seed-end 2601000 --max-turns 100000
```

## Acknowledgments

Built with significant assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic) for code generation, experiment design, and analysis. AI peer review from Google's Gemini informed several architectural decisions. The project demonstrates human-AI collaboration on a complex ML problem: human domain expertise (game intuition, experimental direction, engineering/performance insights) combined with AI capabilities (code implementation, data analysis, systematic diagnosis).
