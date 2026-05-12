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

### Phase 3R: Value Head Diagnostic & Retrain ← CURRENT

V11 NN value head trained (r=0.91-0.95 on K=64 calibration). Wired into the
inference server in server-mode (fused with policy backbone via
`_PolicyValueWrapper`). Initial 50-seed A/B vs linear evaluator at
q_weight=0.5 c_puct=2.5 400 sims trends near 2X2 quality — *worse than
linear evaluator*. Diagnose before retraining anything.

**Cheap fixes already validated against the code (do after current eval):**

- [ ] Local-mode terminal V=0 — `mcts.py:935` calls
  `_value_head_eval_single` on terminal boards; should be V=0 to match
  server semantics. One-line fix.
- [ ] Backbone compat warn at MCTS init — `value_head` ckpt stores
  `backbone_path`; warn if it doesn't match `--model`. One-liner.
- [ ] Cap-hit logging in `eval_parallel` (turns == max-turns counts).
- [ ] Persist `horizon_weights` in the value-head checkpoint instead of
  importing `DEFAULT_HORIZON_WEIGHTS` in the server. Defer until we
  actually tune blends.

**Diagnostic ladder (do not retrain until each level rules out a cause):**

1. [ ] **q_weight sweep** — head V ∈ [0, 2.55] vs linear evaluator's very
  different scale. q_weight=0.5 was tuned for the linear evaluator and
  is probably wrong. Sweep {0.1, 0.25, 0.5, 1.0, 2.0} on 10 seeds × 400
  sims server-mode. ~30 min total. Cheapest possible fix.

2. [ ] **fp16 parity smoke** — fixed-batch script: local fp32 V vs fused
  server fp16 V; report max/mean abs diff. If the fused-server pass
  drifts, force fp32 in `_PolicyValueWrapper`.

3. [ ] **Saturation diagnostic** — V11 has 92-98% positives across
  horizons. If the head outputs ~0.95 everywhere, V ≈ 2.55 constant and
  MCTS is effectively running policy-only. Script: play one game,
  every 50 turns dump (head logits, σ logits per horizon, scalar V) on
  current board AND on ~50 leaves MCTS visited. Healthy signal: V std
  ≥ 0.1, per-horizon σ spread across [0.3, 0.95].

4. [ ] **Correlation with linear evaluator** on the same set of visited
  leaves. r > 0.6 ⇒ noisy-but-similar signal, fixable by tuning. r < 0.3
  ⇒ wrong target.

**If diagnostic shows the head is the wrong target, retrain options:**

- [ ] **Retrain on score-to-go regression** (final_score - current_score,
  normalized). Continuous range, no censoring, matches what MCTS Q-norm
  cares about. Same frozen backbone, same V11 corpus. Survival labels
  with 92%+ positives don't carry enough information for MCTS leaf eval.
- [ ] If saturation is the issue but signal direction is right: retrain
  with `pos_weight` or focal loss to force the head off the
  always-survive prior.
- [ ] Crisis-state oversampling — V11 trajectories are mostly mid-game.
  States MCTS spends sims on at game end (≤10 empty squares) are
  underrepresented. Re-weight or oversample those during training.

**Methodological caveats from ChatGPT review (validated):**

- Best-ckpt selection in `train_value_head.py:323` uses `inner_val_loss`
  (trajectory BCE), not K-rollout calibration. Coincided this run
  (epoch 3 best on both), but won't always.
- Training labels measure survival under recorded V11 trajectories
  (MCTS+oracle); validation rollouts are policy-only greedy. Different
  policies — diagnostic mismatch baked in.

### Phase 4.5: Distillation-objective ablations on V12 (BEFORE iter 2)

Test cheap objective/loss-shape changes on the **existing V12 tensor** before
committing to V13 self-play generation (which costs ~5 days wall on M5).
Each variant trains 20 epochs from pillar2y2 warm-start. ~6h H100 each
(~$15-20). If a variant improves pol-only mean ≥5%, fold into iter 2.

**Variants (only 2 per user request):**

- [ ] **2za — hard/soft blend.** Train with `--blend-alpha 0.7`:
  `loss = 0.7·CE(soft_visits) + 0.3·CE(argmax)`. Addresses the
  "policy matches distribution shape but picks 2nd-best at argmax"
  failure mode by directly penalizing top-1 disagreement.
- [ ] **2zb — target sharpening.** Train with `--target-temperature 0.5`:
  `targets**(1/T)` renormalized before CE. Peakier teacher targets force
  the student to commit. Risk: amplifies teacher noise on close calls.

**Skipped variant — top-K=15 visits:**
Would require regenerating V12 game data — selfplay.py + crisis_mining.py
hardcode top_k_save=5 in JSON output. For V13 generation, bump
top_k_save to 15 in both scripts so this variant becomes testable later
without further regen.

**Code already in place:**
- `train.py` has `--blend-alpha` and `--target-temperature` flags
  (defaults 1.0/1.0 = V12 baseline). `distillation_loss()` implements
  both. Tests pass.

**Eval plan per variant:**
- 1000-game policy-only eval on M5 (~1h)
- 50-game MCTS @ q=2.0, 400 sims, 10K cap on M5 (~3h)
- Compare to pillar2z baseline (pol mean 7,460 / MCTS mean 15,465)

### Phase 4.6: H100 throughput speedup (optional, after 4.5)

Current H100 throughput is ~75K samples/sec @ batch 32768 → ~17 min/epoch.
That's ~11% of theoretical peak (94 TFLOPs/step × 2.3 steps/sec = 107 TFLOPS
achieved vs H100's 990 TFLOPS peak). Bottleneck is small spatial dim (9×9)
inefficient for cuDNN convolutions + per-batch observation building in
collate.

**Speedup options (ordered by expected ROI):**

- [ ] **Batch 65536 on H100** — should fit (80GB VRAM). 2× batch
  amortizes kernel-launch overhead. Expected: 1.3-1.7× throughput.
  Cost: ~1 line change. May need lr re-tune (sqrt scaling → 4.2e-4).
- [ ] **Precompute observations** in `build_expert_v2_tensor.py` — store
  18-channel obs directly. Eliminates per-batch obs building (the 20-iter
  component-labeling loop in `dataset.py:298`). Tensor grows from 5.8GB
  to ~28GB (fp16) or ~57GB (fp32). Expected: 30-50% throughput gain.
- [ ] **Explicit bf16 autocast** — confirm `--amp` on H100 picks bf16.
  Add `dtype=torch.bfloat16` to autocast call if not. ~5% gain.
- [ ] **Profile** with `torch.profiler` to confirm bottleneck before
  doing the precompute work.

### Known Issues

- [ ] **GPU server mode -14% quality gap with multi-worker**: 16-worker MCTS scores
  ~14% lower than 1-worker local mode (confirmed over 250 games, p<0.01).
  Investigation findings:
  - Individual inference outputs are numerically identical (zero diff verified)
  - 1-worker server mode matches local mode (787 vs 736, within noise)
  - Gap appears ONLY with multiple concurrent workers
  - MPS batch-size non-determinism found (policy logits differ by 0.75 at different
    batch sizes) — fixed with per-request processing, but gap persists
  - Root cause likely MPS driver behavior with concurrent GPU access from multiple
    processes. Not fixable from Python.
  - Workaround: use 1-worker local mode for quality-critical evaluation. Multi-worker
    is acceptable for self-play data generation.

## Development Rules

Every change follows this process:
1. **Implement** — clean code in `alphatrain/`
2. **Test** — deterministic unit tests in `alphatrain/tests/`
3. **Benchmark** — standalone scripts in `alphatrain/benchmarks/`
4. **Review** — agent reviews for bugs + performance
5. **Profile** — verify no performance regressions
6. **Experiment** — only run after all above pass
7. **Scripts** — all ad-hoc analysis in `alphatrain/scripts/`, never `python3 -c`
