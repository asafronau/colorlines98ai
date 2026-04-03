# AlphaTrain — TODO

## The Plan (4 Pillars)

### Pillar 1: Input Representation — DONE
Fix "CNN Blindness" by adding tactical features as input channels.

- [x] Add line potential channels (4 directions: H, V, D1, D2)
- [x] Add component area heatmap channel
- [x] Add max line length channel
- [x] Update observation builder (JIT single + batch versions)
- [x] Update model input channels (12 → 18)
- [x] Write unit tests (19 observation, 14 model, 13 dataset = 46 total)
- [x] Benchmark observation building (0.8µs single, 0.2ms batch of 200)
- [x] Train on Colab A100: 20 epochs, 1.3M states × 8x aug
- [x] Results: pol_loss=1.66, val_loss=2.00, MAE=2035
- [x] Standalone policy player: mean=265 (4.6x over 1-ply greedy)

### Pillar 1.5: Neural MCTS — BLOCKED (needs Pillar 2)
Attempted tree search with current model — value head not ready.

- [x] Implement MCTS with PUCT selection + value leaf evaluation
- [x] 12 unit tests passing
- [x] Diagnosis: value head rank-correlation with policy = 0.13
  - Value head learned "which game this came from" not "which move is better"
  - All states in a game share the same game_score target → no move-level signal
  - MAE=2035 is larger than per-move value differences (~100-500)
- [ ] **BLOCKED**: Need TD-learning value targets (Pillar 2) first

### Pillar 2: Value Training — DONE (Pillar 2f)
Asymmetric joint training solved the backbone war.
- [x] val_weight=0.001 prevents value gradients from corrupting policy features
- [x] Policy preserved (315 ≈ 314), MCTS improved (911 → 992)
- [x] Converged after 10 epochs on 1.3M expert data

### Pillar 3: Self-Play Iteration Loop ← CURRENT
Generate better data → train → repeat.

**Iteration 2 (in progress):**
- [ ] Generate 1000 self-play games with Pillar 2f model (seeds 500-1499, 400 sims)
- [ ] Build mixed tensor: expert data + sharpened self-play (T=0.1)
- [ ] Train Pillar 2g: asymmetric (val_weight=0.001), warm start from 2f
- [ ] Evaluate: policy should stay ~315, MCTS should improve beyond 992
- [ ] If improved, repeat with new model

**Future iterations:**
- [ ] Track MCTS mean across iterations (992 → ? → ? → target 5,700)
- [ ] Increase sims (400 → 800) when model is stronger
- [ ] Consider hybrid MCTS (NN policy + heuristic leaf eval) for 8,000+ data

### Known Issues

- [ ] **GPU server mode -14% quality gap**: 16-worker MCTS scores 335 vs 1-worker 389
  (250 games each, same seeds, p<0.01). Individual inference outputs are numerically
  identical (verified). Root cause unknown — may be related to batching latency
  interacting with MCTS virtual loss timing. Not blocking (value head quality is the
  primary bottleneck), but should be investigated before production self-play.

## Development Rules

Every change follows this process:
1. **Implement** — clean code in `alphatrain/`
2. **Test** — deterministic unit tests in `alphatrain/tests/`
3. **Benchmark** — standalone scripts in `alphatrain/benchmarks/`
4. **Review** — agent reviews for bugs + performance
5. **Profile** — verify no performance regressions
6. **Experiment** — only run after all above pass
7. **Scripts** — all ad-hoc analysis in `alphatrain/scripts/`, never `python3 -c`
