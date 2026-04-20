# Color Lines 98 AI

An AlphaZero-inspired AI for Color Lines 98, built from scratch as a deep learning project. The game is a single-player puzzle on a 9x9 board with 7 colors: move balls to form lines of 5+, while 3 new balls spawn each turn. It's a survival game where score is purely a function of how long you stay alive (~2.1 points per turn at all skill levels).

**This is an ongoing project.** I'm a SWE learning ML through building, and this repository documents the full journey: every experiment, every failure, and every discovery.

## Results

| Player | Mean Score | Max Score | Search |
|--------|-----------|-----------|--------|
| Heuristic (tournament bracket, 200 rollouts) | 5,700 | 25,764 | CPU rollouts |
| **Neural policy (standalone, 0 sims)** | **1,763** | **9,061** | None |
| **Neural MCTS (400 sims)** | **5,436** | **30,544** | GPU MCTS |
| **Neural MCTS (1,600 sims)** | **8,552** | **30,330** | GPU MCTS |

The neural player at 1,600 simulations exceeds the heuristic by 50%. The standalone policy (single forward pass, no search) plays at expert human level.

## Architecture

- **Model:** 10-block x 256-channel ResNet (13.3M parameters)
- **Input:** 18-channel board representation (7 color planes, empty cells, next ball positions, connected component areas, line potentials in 4 directions)
- **Policy head:** 6,561 logits (81 source x 81 target positions)
- **Training:** Pure policy distillation from MCTS self-play visit distributions
- **Search:** PUCT MCTS with virtual loss batching, GPU inference server for parallel workers

## Key Discoveries

This project produced 78 documented lessons (see [`alphatrain/HISTORY.md`](alphatrain/HISTORY.md) for the complete experiment history). Some highlights:

**The shared backbone conflict.** Training both policy and value heads on a shared ResNet backbone is a zero-sum game. The value head never learned meaningful board discrimination (0.03 SNR), and increasing its loss weight progressively destroyed policy quality (1,244 -> 943 standalone score). Dropping the value head entirely and training pure policy gave +85% improvement.

**Color Lines may not need a value head.** A perfect player survives indefinitely, so the "value" of every healthy board is infinite. Value prediction is only meaningful for endgame positions. The policy can learn survival tactics directly from MCTS visit distributions.

**The tipping point.** Board death spirals begin at ~41 empty squares, but boards 50 turns before death look identical to healthy boards (43.3 vs 42.8 empty). The difference is structural: connectivity, partition risk, and multi-color cluster density. Human domain expertise ("danger is multi-colored clusters that rot fast") directly informed the diagnosis.

**Crisis mining.** Policy-only games (instant, ~1 second) probe thousands of seeds to find where the model fails. Then deep search (1,600+ sims) replays from those crisis positions, generating high-quality training data for the exact boards where the model needs help. 18x cheaper per crisis scenario than full static games.

**Dynamic sims as a confidence trap.** Reducing search depth on "confident" moves (P_max > 0.3) saves 2.5x compute but produces inferior training data. When the model is confidently wrong, shallow search confirms the error. Static search is necessary for quality.

## Project Structure

```
game/                  # Game engine (Python + Numba JIT)
  board.py             # 9x9 board, BFS pathfinding, line detection, ball spawning
  rng.py               # Deterministic SplitMix64 RNG (cross-language parity with Rust)
  config.py            # Constants

alphatrain/            # Neural MCTS training pipeline
  model.py             # Dual-head ResNet (policy + value, value currently unused)
  mcts.py              # PUCT MCTS with virtual loss batching, dynamic sims
  train.py             # Training loop (pure policy distillation)
  dataset.py           # GPU-resident dataset with dihedral augmentation
  observation.py       # 18-channel observation builder (Numba JIT)
  evaluate.py          # Model loading and evaluation
  inference_server.py  # Shared-memory GPU server for parallel MCTS workers
  HISTORY.md           # Complete experiment history (78 lessons, ~1,500 lines)

  scripts/
    selfplay.py        # Self-play game generation (local, CPU pool, GPU server modes)
    eval_parallel.py   # Parallel MCTS evaluation on multiple seeds
    crisis_mining.py   # Find failures with policy, solve with deep search
    build_expert_v2_tensor.py  # Build training tensors from game JSONs

rust_engine/           # Rust game engine (verified identical to Python, for fast data gen)
```

## Training Pipeline

1. **Self-play:** GPU inference server with N CPU workers playing MCTS games in parallel. Each game saves board states + MCTS visit distributions as JSON.

2. **Tensor building:** Convert game JSONs to GPU-resident training tensors with 8x dihedral augmentation, endgame oversampling, and pairwise ranking pairs.

3. **Training:** Pure policy distillation on H100/A100. The model learns to predict the MCTS search distribution from a single forward pass.

4. **Evaluation:** Parallel MCTS evaluation on 50 seeds, comparing policy-only and MCTS-guided play.

5. **Crisis mining:** Policy-only probes find seeds where the model dies, then deep search replays generate targeted training data for hard positions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, Numba.

## Acknowledgments

Built with significant assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic) for code generation, experiment design, and analysis. AI peer review from Google's Gemini 3.1 Pro informed several architectural decisions. The project demonstrates human-AI collaboration on a complex ML problem: human domain expertise (game intuition, experimental direction, engineering/performance insights) combined with AI capabilities (code implementation, data analysis, systematic diagnosis).
