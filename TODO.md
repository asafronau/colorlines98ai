# Color Lines 98 — TODO

**Target**: 30K policy-only median (browser/WASM deployment).
**Project goal**: Achievable through 2-3 more iterations of sharpened V-corpus distillation.

---

## Current state (2026-05-19)

| metric | value | notes |
|---|---|---|
| **sharp_50_epoch_12 policy** (500 seeds) | **mean 12,622** | new SOA. V12 + `--target-temperature 0.5` + `--color-augment` (HISTORY 153-156) |
| sharp_50 P50 / P75 / P95 (policy) | 8,848 / 17,356 / **37,071** | individual game record: **89,508** |
| sharp_50 + MCTS@100, cap=20K | **mean 14,602** | +24% over policy. MCTS regime restored. P10/P50/floor all dramatically better. |
| sharp_75_epoch_12 policy (500 seeds) | mean 8,817 (+10%) | T=0.75 — mild sharpening, mild gain. Confirms dose-response. |
| sharp_25 (T=0.25) | training | answers: more sharpening helps or saturates? |
| (older) pillar2z policy_only | 7,460 | superseded |
| (older) pillar2z + MCTS@400 | 15,465 | matched at 1/4 sim cost by sharp_50+MCTS@100 |
| **Project target** | **30K policy-only median** | sharp_50 P50 = 8,848. ~3.4× needed on median. |

**Current diagnosis**: V12 visit-distribution targets were too soft (top1=0.26 mean). Training on them with cross-entropy + 8× dihedral + color augmentation produced a model with legal-top1 mean only 0.046 — barely above uniform. Sharpening targets at training time (T=0.5) forced policy commitment. The under-commitment was the bottleneck that the whole oracle / value-head detour was a symptom of. Sharpening is the unlock; iteration on a sharpened pipeline is the path to target.

---

## Active work

1. **sharp_25 training** — testing T=0.25 (Colab G4, ~8h). Decides if T=0.5 saturated the gain or we have more headroom.
2. **V13 corpus generation** — once sharp_X chosen as pillar3 baseline, generate self-play with MCTS@200 or @400, cap ≥ 25K (sharp_50 already routinely scores > 20K).
3. **Pillar4 distillation** — train next student on V13 with target-temperature sharpening built in. Decision criterion: ≥+30% over pillar3.

---

## Closed paths (kept for reference)

- **Oracle mining (Path B v1)** — soft KL on top-6, ran A/B/C/D × 12 epochs. Oracle harmful at convergence (C/D ep12 < B ep12). K=128 audit confirmed K=32 labels were sampling noise. (HISTORY 143-145, 151)
- **Oracle mining (Path B v2)** — abandoned at calibration. K=128 on 100 fresh B-distribution anchors gave only 4% stable separable pairs at Δcap≥0.25 (gate was ≥10%). Structural infeasibility, not implementation bug. (HISTORY 152)
- **Pillar 3a-v3 SpatialValueHead** — all q-values below pure-prior. Pre-current-effort dead branch.
- **DAgger fine-tunes** — 9 attempts on pillar2z all regressed (HISTORY late-pillar2z era).

---

## Original active path (historical — kept for context): TEACHER TOURNAMENT

**Status 2026-05-15 night**: 9 DAgger fine-tune attempts on pillar2z all regressed
(best v9 at 4,435 vs baseline 6,952). The K=32 rollout oracle is real signal
(Phase 0: 14.8% policy_top_1 wrong with +0.44 cap_rate margin) but small
fine-tunes can't extract it without damaging compound play.

### Tournament v1 result (2026-05-15 night, post-DAgger autopsy)

| source | n | 2z Δcap | v9 Δcap | 2z Δturns | v9 Δturns |
|---|---|---|---|---|---|
| selfplay | 200 | 0.040 | 0.039 | 5.4 | 5.1 |
| **crisis** | 200 | **0.045** | **0.052** | 8.7 | 10.1 |
| oracle_disagree | 105 | 0.075 | 0.078 | 17.6 | 19.4 |

**v9 measurably worse on crisis states** (~2 SE). Selfplay tied. Coherent causal chain:
- Healthy/shared states: v9 ≈ 2z
- Crisis states: v9 worse
- Gameplay: v9 much worse (−36%)

**KEY LESSON (ChatGPT)**: near-ties in this game are NOT harmless. They are near-ties
because the model is uncertain in hard states, not because the actions are
interchangeable. Broad KL / global top1 preservation is INSUFFICIENT — need
**crisis-preservation constraints**.

### Decisions taken

- ❌ Do NOT iterate DAgger from v9 (it's already broken on crisis).
- ❌ Do NOT use v9 as a reference policy.
- ❌ Global KL trust regions are insufficient for this game's distribution.

### Source C (first-divergence) judging in progress

300 paired games (2z vs v9 same seeds). Median divergence at turn 4. Judging
now to determine if v9's first-divergence pick is already worse or if damage
appears downstream.

### Tournament v2 results (2026-05-16 morning) — DECISIVE

| teacher | crisis Δcap | crisis Δturns | selfplay Δcap |
|---|---|---|---|
| 2y2 + v11 MCTS | **0.0441** | **8.45** | 0.0419 |
| 2z policy raw | 0.0452 | 8.67 | 0.0405 |
| 2z + v12targets MCTS | 0.0494 | 8.90 | 0.0391 |
| v9 policy | 0.0520 | 10.11 | 0.0392 |
| Rollout oracle (K=32) | **0.0000** | 0.67 | 0.0000 |

Head-to-head on crisis:
- **2z policy beats 2z+v12targets MCTS 33-21** — current MCTS teacher is HARMFUL on crisis
- 2y2+v11 MCTS beats 2z+v12targets MCTS 31-19
- All teachers vs oracle: 0W on crisis

Conclusions per ChatGPT's decision table:
- ❌ Stop using 2z + v12targets MCTS as a teacher (worse than raw 2z policy)
- ✅ Rollout oracle is the best teacher by 4-8pp cap_rate margin
- ✅ 2y2 + v11 MCTS is a viable secondary teacher (tiny edge over raw 2z)
- ❌ DAgger from v9 path stays rejected (worse than 2z on crisis)

### Pending — Tournament v2 (MCTS teachers)

### Tournament spec

Anchor sources (200 each, 800 total):
- A. Normal 2z policy states (sample from selfplay_v12)
- B. 2z crisis/failure states (sample from crisis_v12)
- C. **First-divergence states**: same-seed games with 2z vs v9, find first turn moves differ
- D. High-margin rollout-oracle disagreement states (filtered from phase1_oracle.pt, 105 already known)

Teachers (each picks one move per anchor):
1. 2z raw policy top-1
2. v9 raw policy top-1
3. 2z + current MCTS (100 sims, v11-targets value head, q=2.0)
4. 2y2 + old strong MCTS (`pillar2y2_epoch_40.pt`, v11-targets head, q=2.0)
5. Rollout oracle over top-K (K=32 common-RNG, H=300, pick highest cap_rate)

Judge: K=32 common-RNG rollouts at H=300 on each UNIQUE candidate move
selected by teachers. Outcomes: cap_rate, mean_turns, score_gain. Apply
split-half stability filter (drop unstable judgments).

Primary metric: **regret = oracle_best_metric − teacher_selected_metric** per (anchor_source, teacher).

### Decision table (ChatGPT 2026-05-15)

| finding | conclusion |
|---|---|
| v9 loses to 2z mainly on first-divergence states | fine-tune update directly harmful; don't iterate DAgger from v9 |
| v9 OK on 2z states, bad on v9-on-policy states | distribution shift real; DAgger loop may help w/ very small updates + on-policy labels |
| 2z raw beats 2z+MCTS on hard states | current MCTS/value teacher is harmful; stop using it |
| 2y2+old MCTS beats 2z+MCTS | use old teacher or hybrid for hard labels; latest ≠ best |
| Rollout oracle beats all | use rollout oracle for hard-state correction; mix into FULL training (not fine-tune) |
| No teacher wins stably | labels too noisy; scaling data won't fix |

### Estimated compute

- Dev: ~3-4h to build tournament infrastructure
- Compute: ~2-3h on M5 MAX (16 workers)
- 4 sources × 200 anchors × 5 teachers × ~K candidates × K=32 rollouts × H=300

### After tournament — revised paths (post-v1 result)

Path A (DAgger iteration from v9) is **rejected** — tournament v1 showed v9 is
worse than 2z on crisis states. Iterating from a known-broken policy would
compound the damage.

Remaining paths:

- **Path B — Train fresh/full-corpus with hard-state corrections**: rerun
  pillar2z's training recipe with V12 self-play + heavily oversampled oracle
  data. No fine-tune perturbation; the new policy learns crisis corrections
  from the start. Cost: 6-12h Colab H100.

- **Path E — Constrained auxiliary reranker**: train a separate small model
  that adjusts policy ONLY at crisis states (detected by entropy/empty
  count/margin). Pillar2z untouched. Reranker is trained on oracle data on
  crisis anchors only. Cost: ~1-2 days dev + train.

- **Path F — Crisis-preservation training objective**: add a "preserve
  pillar2z's argmax on healthy states" loss with HIGH weight, plus oracle
  corrections on crisis states only. Train from scratch or fine-tune. Avoids
  the global-trust-region failure mode of v1-v9.

After Source C result, ChatGPT suggested Tournament v2 with MCTS teachers
(2z+v12targets MCTS, 2y2+v11 MCTS, rollout oracle) to decide which signal
to use in B/E/F. The 2y2+v11 = 13,476 historical best is a particularly
interesting comparator since pillar2z + same head bimodally regressed.

---

## Architecture / training ideas (after DAgger iter 1 result)

These are *deferred* until we know DAgger works. In rough order of expected ROI:

### High ROI, cheap
- [ ] **Color permutation augmentation**. Game has full 7-color symmetry; we only do 8× dihedral. Adding 4-8× color perms = 32-64× total augmentation. ~30 LOC in `dataset.py`. Expected: +2-5% policy quality, faster convergence.
- [ ] **Auxiliary heads during training**. Predict empty count, future cap_rate, line potentials. Adds gradient signal, no inference cost (heads dropped at deploy). Expected: better feature learning.
- [ ] **Sample more states from on-policy games**. iter-1 oracle labels were 95% crisis-heavy (Stage 1 corpus); next round should use π₁'s own play trajectories.

### Medium ROI, moderate cost
- [ ] **Architecture: 12 blocks × 320 ch** (modest bump from 10×256, +30% params, ~15M total). Diagnostic at iter 2-3.
- [ ] **Self-attention layers** at the end of the conv stack (2-3 layers, 8 heads). Handles global topology / connected-region reasoning naturally. Worth testing if conv-only ResNet plateaus.
- [ ] **Bigger oracle: shallow MCTS as continuation** instead of raw policy. Each rollout step becomes policy + 20-sim MCTS. Stronger teacher = bigger DAgger gain.

### Big ROI, big cost
- [ ] **Transformer backbone** (ViT-style over 81 board cells + color/feature tokens). Native global reasoning. Months of work, fundamental redesign. Only if conv plateaus before 15K.
- [ ] **Decoupled value backbone**. Separate ResNet for value head. ~2× inference at MCTS time. Useful only after oracle proves stable value signal exists. Defer until DAgger plateaus.

---

## Already-built ideas worth keeping

- [x] `alphatrain/scripts/pilot_clean_pairwise.py` — stable-label miner (88.8% pilot, 84.4% production inter-half agreement). Now with common-RNG seed fix + score-tracking fix.
- [x] `alphatrain/scripts/phase1_oracle_label.py` — Phase 1 oracle labeler (top-6, K=32, full per-move stats).
- [x] `alphatrain/scripts/analyze_pairwise_head.py` — per-metric/source val accuracy + head output diagnostics.
- [x] `alphatrain/scripts/phase0_oracle_screen.py` — policy-rank-vs-oracle pre-screening.

---

## Abandoned / negative-result threads

These don't need re-investigation unless circumstances change:

- ❌ **Multi-horizon survival classification value head** (Pillar 2y survival 25/50/100/200): saturates at strong-policy regime — head can predict 99% survival → loss minimized at constant.
- ❌ **Q-bootstrap value head from stronger MCTS** (Pillar 2g/h): self-referential, search inherits head's saturation.
- ❌ **SpatialValueHead with pairwise BPR ranking** (Pillar 3a-v3): val acc 67% on cap_rate is real, but outputs are too compressed (V_w−V_l std=0.084) and miscalibrate terminal; harmful in MCTS at all q values.
- ❌ **Decoupled value/policy nets** (Pillar 2c era, -56%): caused gradient conflict, abandoned.
- ❌ **Spatial rollout injection**: regressed -46% due to temperature misalignment.
- ❌ **ML oracle V2** (echo-chamber retraining on ML-enhanced games): -27% regression.
- ❌ **DAgger fine-tunes of pillar2z (Phase 2 v1–v9, 2026-05-15)**: 9 attempts, all regressed to 2K-5K vs baseline 6,952. Tried: no distill → catastrophic. KL distill (forward, reverse, symmetric) → all regress. Margin filter (105 high-margin anchors) → still regress. Hard CE distill → top1 collapse. Frozen backbone → same drift as full network. Lower lr → identical drift (turned out to be BN running-stats bug masquerading as gradient drift). After fixing BN bug, v9 had **0% flips on high-confidence broad states** but still regressed 36% in games — distribution shift during compound play. **Lesson**: fine-tuning a strong policy with small targeted dataset is fundamentally hard; need either much more on-policy data (DAgger loop) or fresh training (not fine-tune).
  - Pivotal bug found: `net.train(True)` updates BatchNorm running stats regardless of `requires_grad`. Always set `net.train(False)` for fine-tune.

---

## Deployment (long-term, after policy hits 20K+ median)

- [ ] Export PolicyNet to ONNX
- [ ] Rust/WASM inference for browser (no MCTS, no value head — single forward pass)
- [ ] Web UI: existing pygame `play_gui.py` ported to Angular + ONNX Runtime Web
- [ ] Target: <50ms per move, mean ≥20K policy-only

---

## Project history references

- `alphatrain/HISTORY.md` — chronological training run log (Pillars 1-3a)
- `CLAUDE.md` — project conventions and key design decisions
- `data/rust_tournament/` — original tournament-player dataset (263 base games, source of ML oracle V1)
- `data/crisis_v12/`, `data/selfplay_v12/` — pillar2z's training corpus (V12)
