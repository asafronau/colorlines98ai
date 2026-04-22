# Color Lines 98 AI

An AlphaZero-inspired AI for [Color Lines 98](https://en.wikipedia.org/wiki/Lines_(video_game)), built from scratch as a deep learning project. The game is a single-player puzzle on a 9x9 board with 7 colors: move balls to form lines of 5+, while 3 new balls spawn each turn. It's a survival game where score is purely a function of how long you stay alive (~2.1 points per turn at all skill levels).

**This is an ongoing project.** I'm a SWE learning ML through building, and this repository documents the full journey: every experiment, every failure, and every discovery.

## Results

| Player | Mean Score | Max Score | Search |
|--------|-----------|-----------|--------|
| Heuristic (tournament bracket, 200 rollouts) | 5,700 | 25,764 | CPU rollouts |
| **Neural policy (standalone, 0 sims)** | **2,972** | **21,260** | None |
| **Neural MCTS (400 sims)** | **6,016** | **30,544** | GPU MCTS |
| **Neural MCTS (1,600 sims)** | **8,552** | **30,330** | GPU MCTS |

The neural player at 1,600 simulations exceeds the heuristic by 50%. The standalone policy (single forward pass, no search) approaches expert human level. Score progression across training iterations:

| Iteration | Policy Mean | Key Change |
|-----------|------------|------------|
| 2U | 1,763 | Pure policy distillation (dropped value head from training) |
| 2V | 2,680 | Static 1600 sims + crisis mining |
| 2W | 2,972 | Opening book (2200 sims) + diverse crisis data |

## Architecture

- **Model:** 10-block x 256-channel ResNet (13.3M parameters)
- **Input:** 18-channel board representation (7 color planes, empty cells, next ball positions, connected component areas, line potentials in 4 directions)
- **Policy head:** 6,561 logits (81 source x 81 target positions)
- **Value head:** Exists but untrained (val_weight=0) — provides Q-diversity for MCTS exploration
- **Training:** Pure policy distillation from MCTS self-play visit distributions
- **Search:** PUCT MCTS with virtual loss batching, GPU inference server for parallel workers

## Key Discoveries

This project produced 86 documented lessons (see [`alphatrain/HISTORY.md`](alphatrain/HISTORY.md) for the complete experiment history). Some highlights:

**The shared backbone conflict.** Training both policy and value heads on a shared ResNet backbone is a zero-sum game. The value head never learned meaningful board discrimination (0.03 SNR), and increasing its loss weight progressively destroyed policy quality (1,244 → 943 standalone score). Dropping the value head from training and dedicating 100% of gradient to policy gave +85% improvement.

**The value head is dead for training but alive for search.** Attempting to remove the value head entirely broke MCTS — even garbage values provide essential Q-diversity for virtual loss and exploration. The untrained value head acts as exploration noise that prevents the search from degenerating.

**Color Lines may not need value prediction.** A perfect player survives indefinitely, so the "value" of every healthy board is infinite. The policy learns survival tactics directly from MCTS visit distributions without explicit value targets.

**The tipping point.** Board death spirals begin at ~41 empty squares, but boards 50 turns before death look identical to healthy boards (43.3 vs 42.8 empty). The difference is structural: connectivity, partition risk, and multi-color cluster density. Human domain expertise ("danger is multi-colored clusters that rot fast") directly informed the diagnosis.

**Crisis mining.** Policy-only games (instant, ~1 second) probe thousands of seeds to find where the model fails. Then deep search (1,600+ sims) replays from those crisis positions, generating high-quality training data for the exact boards where the model needs help. 18x cheaper per crisis scenario than full static games.

**Static sims are non-negotiable.** Reducing search depth on "confident" moves (dynamic sims) saves 2.5x compute but produces inferior training data. When the model is confidently wrong, shallow search confirms the error. Full static search on every move is required for quality.

**Diverse data prevents overfitting.** A mix of full games (mid-game backbone), openings at deeper search (first 200-500 turns at 2200 sims), and crisis replays (targeted recovery training) supported 8 training epochs without overfitting on 2.9M states — better than 6.5M homogeneous mid-game states which overfit by epoch 6.

## Project Structure

```
game/                  # Game engine (Python + Numba JIT)
  board.py             # 9x9 board, BFS pathfinding, line detection, ball spawning
  rng.py               # Deterministic SplitMix64 RNG (cross-language parity with Rust)
  config.py            # Constants

alphatrain/            # Neural MCTS training pipeline
  model.py             # ResNet with policy head + untrained value head (for MCTS)
  mcts.py              # PUCT MCTS with virtual loss batching
  train.py             # Training loop (pure policy distillation)
  dataset.py           # GPU-resident dataset with dihedral augmentation
  observation.py       # 18-channel observation builder (Numba JIT)
  evaluate.py          # Model loading and evaluation
  inference_server.py  # Shared-memory GPU server for parallel MCTS workers
  HISTORY.md           # Complete experiment history (86 lessons)

  scripts/
    selfplay.py        # Self-play game generation (local, CPU pool, GPU server modes)
    eval_parallel.py   # Parallel evaluation (policy-only + MCTS, GPU-accelerated)
    crisis_mining.py   # Find failures with policy, solve with deep search
    build_expert_v2_tensor.py  # Build training tensors from game JSONs

rust_engine/           # Rust game engine (verified identical to Python, for fast data gen)
```

## Training Pipeline

1. **Self-play:** GPU inference server with N CPU workers playing MCTS games in parallel. Static 1600 sims per move. Each game saves board states + MCTS visit distributions as JSON.

2. **Crisis mining:** Policy-only probes find seeds where the model dies (~1 second per game). Deep search (1600-2000 sims) replays from crisis positions generate targeted training data.

3. **Opening generation:** First 200-500 turns at 2200 sims for elite opening decisions.

4. **Tensor building:** Convert game JSONs to GPU-resident training tensors with 8x dihedral augmentation.

5. **Training:** Pure policy distillation on H100/G4. The model learns to predict the MCTS search distribution from a single forward pass.

6. **Evaluation:** GPU-accelerated parallel evaluation on 1000+ seeds with percentile breakdown.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, Numba.

## Acknowledgments

Built with significant assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic) for code generation, experiment design, and analysis. AI peer review from Google's Gemini informed several architectural decisions. The project demonstrates human-AI collaboration on a complex ML problem: human domain expertise (game intuition, experimental direction, engineering/performance insights) combined with AI capabilities (code implementation, data analysis, systematic diagnosis).
