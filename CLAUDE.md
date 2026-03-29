# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## CRITICAL: Observability Requirements

**Every long-running process MUST report periodic progress. No exceptions.**

- **Game-level:** Print after every completed game (score, turns, elapsed, ETA)
- **Within-game:** Print every 500 turns (turn count, current score, elapsed time)
- **Always flush:** Use flush=True (Python) or eprintln! (Rust)
- **Checkpoints:** Save intermediate results every hour for crash recovery
- **Never run blind:** If a process produces no output for >5 minutes, something is wrong

## Project

An AI agent for Color Lines 98. Target: mean 15,000-20,000 points, deployed in browser.

**Current best: mean=8,432** (tournament 200 + ML oracle V1, blend=0.05). Max=20,875.

**Next phase: AlphaZero-style deep learning** — see TODO.md for the full plan.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Key Commands

```bash
# Evaluate players
python -m evaluation.evaluate --player tournament --simulations 200 --temperature 3.23 --games 20 --seed 42

# Train ML oracle (pairwise logistic regression on tournament data)
python -m training.train_linear_oracle --data data/rust_tournament/

# Generate tournament data with ML oracle
python -m training.gen_ml_tournament_data --games 30 --seed 10000 --num-workers 18

# Play the game (Pygame GUI with AI Hint)
python play_gui.py

# Benchmark
python benchmark.py
```

## Architecture

### Game engine (game/)
- board.py — 9x9 board, 7 colors, BFS pathfinding, line detection, ball spawning
- config.py — Constants
- fast_heuristic.py — Parameterized heuristic with 17 CMA-ES weights + ML oracle blend
- features.py — 30-feature spatial extractor for ML (0.8us/call, JIT compiled)

### Players (evaluation/)
- evaluate.py — CLI for running evaluations
- players.py — Tournament bracket (successive halving), policy players, beam search

### ML Training (training/)
- train_linear_oracle.py — Pairwise logistic regression on tournament data
- gen_ml_tournament_data.py — Generate (state, top-5 moves, scores) from tournament player
- train_dagger.py, train_fast.py, bc_data.py — BC/DAgger pipeline (legacy, produced checkpoints_soft/)
- tune_weights.py — CMA-ES heuristic weight optimization

### Models (model/)
- policy_net.py — Two-stage PolicyNet (source + target heads) with value head for PPO
- network.py — ValueNet with optional categorical head

### Assets
- checkpoints/linear_oracle.npz — ML oracle weights (30 features, pairwise-trained)
- checkpoints/tuned_weights.npz — CMA-ES optimized heuristic weights
- checkpoints_soft/dagger_best.pt — Best BC model (for distillation baseline)

### Data
- data/rust_tournament/ — 263 base tournament games (V1 oracle training data)
- data/gcp_run/ — 20 GCP-generated games
- data/ml_v2_a/ — ML-enhanced games (V2 attempt)
- data/human_games/ — Human play data from play_gui.py

### Other
- play_gui.py — Pygame GUI with AI Hint and auto-play
- benchmark.py — CPU/GPU benchmarks
- rust_engine/ — Rust game engine (verified identical heuristic, for future data gen)
- scripts/gcp_datagen.sh — GCP data generation script

## Key Design Decisions
- Tournament bracket (successive halving) is the best search architecture
- ML oracle at ROOT only (2-ply candidate selection), NOT in rollout inner loop
- Pairwise ranking with confidence weighting — only ML approach that consistently helps
- Rollouts use plain 5-weight heuristic (ML in rollouts is too slow or breaks temperature)
- Pathfinding must check ALL connected components (the break bug that doubled scores)
- No turn cap — games play to natural death
- V1 oracle trained on 263 Rust games (base heuristic). V2 on ML-enhanced games REGRESSED.
- Next: AlphaZero with dual-head ResNet replaces both oracle and rollouts
