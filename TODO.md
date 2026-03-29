# Color Lines 98 — AlphaZero Phase TODO

## Target: Mean 15,000-20,000 points (expert human level)

## Current Status
- **Best player:** Tournament 200 + ML Oracle V1 (blend=0.05) = **mean ~5,700**
- **Neural (Pillar 1):** 18ch ResNet trained, standalone policy = mean 265
- **MCTS (Pillar 1.5):** Built but BLOCKED — value head doesn't rank moves (rho=0.13)
- **Next:** Pillar 2 (TD-learning value targets) to fix value head

---

## Phase 1: Data Generation — DONE
- [x] Generate 302 games (126 local + 176 GCP), 1.31M states
- [x] Precompute tensors: boards + sparse policy + two-hot value → alphatrain_v1.pt (624MB)

## Phase 2: Dual-Head ResNet Training — DONE (Pillar 1)
- [x] 18-channel observation (7 colors, empty, next balls, component area, line potentials, max line)
- [x] 10-block × 256ch ResNet, 12.1M params, flat 6561 joint policy
- [x] Trained 20 epochs on Colab A100 at 31K s/s
- [x] pol_loss=1.88, val_loss=2.00, MAE=2035 (best epoch=14)
- [x] 46 unit tests, 3 benchmarks, all passing

## Phase 3: Neural MCTS — BLOCKED
- [x] MCTS implementation: PUCT + value leaf eval + MuZero Q normalization
- [x] 12 unit tests passing
- [ ] **BLOCKED**: Value head rank-correlation = 0.13 (useless for move ranking)
- [ ] Root cause: game_score target is per-game, not per-position
- [ ] Fix: Pillar 2 (TD-learning) to give each position a different value target

## Next: Pillar 2 — TD-Learning Value Targets
- [ ] Compute TD(λ) targets: γ=0.99, λ=0.95, reward=score_delta per turn
- [ ] Each state gets target based on its discounted future trajectory
- [ ] Retrain value head, verify rank correlation > 0.5
- [ ] Re-test MCTS with improved value head
- [ ] Also consider: root-only evaluation (avoid stochastic tree issues)

## Phase 4: Self-Play Iteration
- [ ] MCTS generates games with improved network
- [ ] Train on self-play data (MCTS visit counts as policy, TD values as value)
- [ ] Evaluate → promote → repeat
- [ ] Target: mean 10,000+

## Phase 5: Browser Deployment
- [ ] Export ResNet to ONNX
- [ ] Rust/WASM search (Policy top-K → Value eval → pick best)
- [ ] Angular frontend with ONNX Runtime Web
- [ ] Target: <200ms/move, mean 12,000+

---

## Key Learnings (Do NOT repeat)
1. Dont mix data from different skill levels — contradictory labels
2. Dont iterate oracle on <100 games — echo chamber / confirmation bias
3. Dont inject ML into rollout inner loop — 36M evaluations = stuck
4. Always flush output — observability is critical
5. Pairwise training with confidence weighting — only ML approach that worked
6. center_dist in rollouts at -0.07 regressed — softmax temperature misalignment
7. Pathfinding bug fix was the biggest win — always check the basics
8. Turn cap was artificially limiting scores — dont cap games
9. V2 oracle regressed because of echo chamber — need volume + exploration diversity
10. GCP Intel cores are 2-3x slower per-core than M5 Max
11. game_score is a game-level label, not position-level — need TD-learning for value head
12. Determinized MCTS breaks in stochastic games — different spawns conflate tree nodes
13. Always validate value head with rank-correlation before building search on it
