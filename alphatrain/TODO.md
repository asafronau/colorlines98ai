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

### Pillar 2: Value Targets (TD-Learning) ← CURRENT
Fix value head by training on discounted future reward, not raw game_score.

- [ ] Compute TD(λ) targets from existing 500 game trajectories
  - γ=0.99, λ=0.95, reward = score_delta per turn
  - Each state gets a different target based on its future trajectory
- [ ] Retrain value head on TD targets (same 18ch model)
- [ ] Verify: value head should now differentiate positions within a game
- [ ] Re-test value-based move ranking (target: correlation > 0.5 with policy)
- [ ] Re-test MCTS with TD-trained value head

### Pillar 3: Stochastic MCTS
Handle Color Lines' randomness properly in tree search.

- [ ] Current determinized MCTS breaks (different spawns → conflated tree nodes)
- [ ] Options: root-only evaluation, afterstate evaluation, or chance nodes
- [ ] Implement chosen approach after Pillar 2 value head works
- [ ] Test MCTS quality vs tournament bracket (target: beat 5,700 mean)

### Pillar 4: Self-Play Loop
The model teaches itself.

- [ ] MCTS generates games using current network
- [ ] Save MCTS visit counts as policy targets, TD values as value targets
- [ ] Train new network on self-play data
- [ ] Evaluate new vs old (100-game tournament)
- [ ] If new wins >55%, promote and repeat
- [ ] Track Elo progression across iterations

## Development Rules

Every change follows this process:
1. **Implement** — clean code in `alphatrain/`
2. **Test** — deterministic unit tests in `alphatrain/tests/`
3. **Benchmark** — standalone scripts in `alphatrain/benchmarks/`
4. **Review** — agent reviews for bugs + performance
5. **Profile** — verify no performance regressions
6. **Experiment** — only run after all above pass
7. **Scripts** — all ad-hoc analysis in `alphatrain/scripts/`, never `python3 -c`
