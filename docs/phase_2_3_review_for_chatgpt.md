# Phase 2 results + Phase 3 plan — for ChatGPT peer review

**Date:** 2026-05-24
**Branch:** `experiment/stationary-risk-head`
**Where we are:** Phase 2 mining complete. Phase 3 distillation design needs your sign-off before committing ~12h Colab compute per training variant.

## Recap of the arc

Goal: lift pillar3b's **policy-only floor** for browser/WASM deployment. Pillar3b OOS eval: mean 17,255 (good), P5 ≈ 1,576 (bad), <1000 rate 2.5% (bad). Target: P5 ≥ 2,500.

Diagnostics so far:
- **Phase A**: 19 confirmed floor failures all show density-spiral pattern, clear rate ~34% (just below 37.5% equilibrium).
- **Phase B**: at late-danger points in failure trajectories, top-2..5 don't beat top-1 (28.7% vs 28.2% in success controls). NO-GO on local-action correction at those states.
- **Phase C** (your direction): real divergence happens at turn 100, far before the late danger points. Score rate 1.85 vs 2.04 (failing vs surviving) diverges first.
- **Stationary-window analysis**: V13 MCTS targets are nearly uniform in the stationary regime (top1 ≈ 0.13, gap ≈ 0.003). Teacher gives weak signal where it matters for floor.

Your guidance: build a counterfactual teacher that produces floor-aware labels at stationary boundary states.

## Phase 1 (done)

Trained an auxiliary head on frozen pillar3b backbone to predict forward 100-turn outcomes. New sustained-fragmentation labels (per your spec) dominated:

| label | GAP AUC | Spatial AUC |
|---|---|---|
| min_lec<10 | 0.744 | 0.748 |
| min_lec<15 | 0.723 | 0.728 |
| min_empty<30 | 0.834 | 0.842 |
| min_empty<25 | 0.894 | 0.900 |
| lec_delta<=-10 | 0.783 | 0.789 |
| **lec_under_10_frac > 0.2** | **0.959** | **0.961** |
| **lec_under_15_frac > 0.5** | **0.982** | **0.983** |
| **lec_shortfall > 3** | **0.982** | **0.985** |

Backbone clearly carries forward-risk information. Sustained-fragmentation labels are highly learnable (0.96-0.98). The binary min thresholds (0.72-0.74) were noisy; the integral/fraction labels are clean — your "transient vs sustained" framing was decisive.

Spatial head (1×1 conv → flatten → MLP) adds +0.005-0.01 over GAP. Marginal but consistent.

**Phase 1 PASSED.**

## Phase 2 (done) — the headline

Counterfactual rollout mining: 1000 stationary-boundary anchors sampled from V13 selfplay (empty 32-50, turn ≥ 100, stratified by LEC bucket × n_components bucket). For each anchor, top-10 policy candidates × R=24 common-RNG rollouts × H=100 turns. Lex floor objective: die_rate → leave_band_rate (P(min_empty<30)) → −P10(score) → −mean(score).

### Floor-winner rank distribution

| rank | n | % of anchors |
|---|---|---|
| **top-1** | 180 | **18.0%** |
| top-2 | 128 | 12.8% |
| top-3 | 99 | 9.9% |
| top-4 | 88 | 8.8% |
| top-5 | 98 | 9.8% |
| top-6 | 98 | 9.8% |
| top-7 | 77 | 7.7% |
| top-8 | 87 | 8.7% |
| top-9 | 70 | 7.0% |
| top-10 | 75 | 7.5% |

**Top-1 wins floor in only 18% of stationary boundary states; top-2..10 wins in 82%.** Distribution past top-1 is roughly uniform.

### Compared to Phase B

| analysis | source states | top-1 floor wins |
|---|---|---|
| Phase B | late danger points (failure traj) | 28.7% |
| Phase B | matched success-traj DPs | 28.2% |
| **Phase 2** | **stationary boundary states** | **18.0%** |

Your prediction that Phase B sampled the wrong regime was validated. The actionable signal lives at stationary boundary states (precursor window), not at late danger points (already lost).

### Two clear failure modes for top-1

Inspection of individual anchors:
- **Primary**: top-1 has nonzero die_rate; some rank 2-10 candidate has die_rate=0 (saved by lex primary)
- **Secondary**: top-1 already safe (die_rate=0) but a rank 2-10 candidate has higher P10 score (won by lex secondary)

### Files
- `alphatrain/data/stationary_counterfactuals_v1.pt` (1.8 MB) — 1000 records with per-candidate (die_rate, leave_band_rate, mean/P10 score, mean/P10 min_empty, mean/P10 min_lec)
- Code: `scripts/mine_stationary_counterfactuals.py`

## Phase 3 design (proposed) — needs your review

Distill Phase 2 labels into pillar3c as an auxiliary loss during V14 corpus training.

### Loss options (ranked by my preference)

**A. Soft-target KL on candidates** (preferred):
```
weights[rank] = softmax(−lex_penalty[rank] / τ)
where lex_penalty = die_rate*10 + leave_band_rate*1 + (max_mean − mean)/100
aux_loss_per_anchor = −sum_k weights[k] * log_softmax(logits)[move_k]
total_loss = soft_CE_V14_main + λ * aux_loss
```
- τ controls sharpness (τ=0.5 → moderate); λ controls aux contribution
- Carries lex-ranking information beyond just top-1

**B. Hard CE on floor_winner only**:
```
aux_loss = CE(logits, floor_winner_move_idx)
```
- Simpler but discards rank-2..10 signal; overfit risk on N=1000

**C. Pairwise ranking margin**:
```
aux_loss = max(0, margin − (logit[floor_winner] − logit[top_1_policy]))
```
- Decent gradient signal, scale-invariant

### Proposed defaults
- Option A
- λ ∈ {0.05, 0.10, 0.20} sweep — pick one to start (probably 0.10)
- τ = 0.5
- Filter anchors with weak floor-best advantage (e.g., only train where die_rate[top1] − die_rate[winner] > 0.05 OR p10_score[winner] − p10_score[top1] > 20)

### Training pipeline
1. Add counterfactual loader to `train_path_b.py`: small batch (~256 anchors) per main step
2. Compute aux loss alongside main V14 distillation loss
3. Train pillar3c 17 epochs, warm-start from pillar3b_epoch_20, target_temperature=0.5
4. Save per-epoch checkpoints

### Decision gate (1000 OOS seeds 777000-777999)

| outcome | criterion |
|---|---|
| Strong win | P5 ≥ 2500 AND <1000 ≤ 1.0% AND mean within 5% of pillar3b (≥16,400) |
| Acceptable | P5 ≥ 2200 (vs 1576) AND <1000 ≤ 1.5% (vs 2.5%) AND mean within 5% |
| No-go | mean drops > 10% OR floor doesn't improve |

### Risks I'm aware of

1. **N=1000 is small** vs 9M+ V14 main-distillation states. Aux signal could get drowned. Mitigation: sample 256 anchors per main batch (effective oversampling).
2. **Labels are R=24 H=100 rollouts** — some "floor-best" advantages are within sampling noise. Mitigation: filter (above).
3. **Aux could pull mean down** (Path B oracle attempts stalled at λ=0.05/0.10 partly because the oracle labels were noisy). Mitigation: λ sweep + decision gate.
4. **Bias toward "safe boring" play** — risk-averse pillar3c could regress on long games. Mitigation: aux loss only fires on stationary states; main loss preserves overall play.

## 5 specific questions for you

1. **Loss formulation**: Option A (soft KL with lex weights) vs Option C (pairwise margin) at this N=1000 scale. I lean A. What's your read?

2. **Filter threshold**: would you filter the 1000 anchors to only those with a "clean" floor-best advantage? Trade-off: noise vs N. My instinct is to filter — at R=24 a ±5% die_rate difference is within noise.

3. **λ range**: my proposed 0.05/0.10/0.20 — too low/high? Path B (different loss but similar idea) stalled at λ=0.05/0.10 with noisier labels. Should we go higher this time given cleaner labels?

4. **Source states**: stationary boundary only (empty 32-50), or also include the LATER mid-decline states (empty 25-35, turn 200+)? My instinct: only boundary — late states are pillar3b-poisoned per Phase B.

5. **τ for soft KL**: τ=0.5 my default. τ=0.2 makes target nearly one-hot, τ=1.0 makes it very soft. Which?

## Open additional ideas (not yet committed)

- Generate a second batch of 1000 anchors from V14 selfplay (different distribution) to grow N if needed.
- Add the auxiliary head from Phase 1 (predicts forward risk) as a MTL head during pillar3c training — keeps the model's risk-awareness sharp even with the main distillation loss.

Specific ask: green-light Option A + your preferred λ/τ/filter, OR specify a different design before we burn ~12h Colab.

## Files for reference

- `docs/phase2_results.md` — Phase 2 numbers in detail
- `docs/phase3_distillation_plan.md` — full Phase 3 plan with risks
- `alphatrain/data/stationary_counterfactuals_v1.pt` — the dataset
- `scripts/mine_stationary_counterfactuals.py` — how labels were produced
- `scripts/build_stationary_risk_dataset.py`, `scripts/train_stationary_risk_head.py` — Phase 1 artifacts
