# Phase 3 Plan: Distill Counterfactual Labels into Pillar3c — for ChatGPT Review

**Date:** 2026-05-24
**Status:** PROPOSED — pending ChatGPT review before implementation

## Context

Phase 2 produced `stationary_counterfactuals_v1.pt`: 1000 stationary boundary anchors, each with 10 candidate moves and rollout-derived per-candidate stats (die_rate, leave_band_rate, P10 score, mean score, min_empty/lec stats). 82% of anchors have a floor-best move at rank 2-10 — meaning pillar3b's policy is genuinely floor-suboptimal at these states.

Phase 3 distills these labels into pillar3c. The big risk: with only 1000 anchors and 9M+ states in V14, the aux signal is tiny in raw count. We need the right loss structure to make it bite without overfitting / regressing the main policy.

## Proposed loss

Three options ranked by my preference (chose this on ChatGPT review):

### Option A (preferred): Soft-target KL on counterfactual states

For each counterfactual anchor, build a soft target distribution over the 10 candidate moves based on lex-ranking:

```
weights[rank] = softmax(−lex_penalty[rank] / τ)
where lex_penalty = die_rate * 10 + leave_band_rate * 1 + (mean_score_max - mean_score) / 100
```

Each anchor contributes a soft-CE loss: `−sum_k weights[k] * log_softmax(logits)[move_k]`. The model learns "shift mass toward the floor-best candidate" without specifying a single hard target.

Auxiliary loss combined with V14 distillation:
```
total_loss = soft_CE_V14_main  +  λ * soft_CE_counterfactual
```

Where λ ~ 0.05-0.20 (similar to Path B oracle λ that we tried earlier, but with cleaner labels).

τ controls sharpness of the target. τ=1.0 = relatively soft; τ=0.2 = nearly one-hot on floor-best.

### Option B: Hard target on floor-best move

For each anchor, cross-entropy toward floor_winner only:
```
aux_loss = CE(logits, floor_winner_move_idx)
total_loss = soft_CE_V14_main + λ * aux_loss
```

Simpler, but ignores the lex-ranking signal beyond top1. Likely overfit risk — only 1000 examples.

### Option C: Pairwise ranking

For each anchor, the floor-winner should have higher logit than top-1 (the policy's current pick):
```
aux_loss = max(0, margin - (logit[floor_winner] - logit[top1]))
```

Margin-based ranking. Avoids absolute scale issues. But limited gradient signal compared to soft KL.

## Hyperparameters to choose

- **λ**: 0.05, 0.10, 0.20 — three settings to sweep (each ~12h Colab training)
- **τ** (Option A only): 0.2, 0.5, 1.0 — start with 0.5
- **augmentation**: full dihedral 8× + color permutation 7! same as main training. Counterfactual labels carry over symmetrically (same recoded move).

## Training pipeline

1. **Add counterfactual loader** to `train_path_b.py`: reads `stationary_counterfactuals_v1.pt`, for each batch step also samples a small batch of counterfactual anchors (size ~256, every iteration). Returns (obs, candidate_move_indices_1024 (10 per anchor), soft_target_weights).
2. **Compute aux loss** alongside main V14 distillation loss.
3. **Train pillar3c** on V14 + aux for 17 epochs, warm-start from pillar3b_epoch_20.
4. Save per-epoch checkpoints; eval each at 500 seeds OOS.

Standard pillar3a→3b recipe otherwise: target_temperature=0.5 (or 0.3 per earlier note), batch_size=32K, lr=3e-4, etc.

## Validation / decision gate

Eval candidate pillar3c checkpoints on 1000 OOS seeds (777000-777999, the same seeds used for pillar3b eval). Targets:

| outcome | metric |
|---|---|
| Strong win | P5 ≥ 2500 AND <1000 rate ≤ 1.0% AND mean within 5% of pillar3b (≥16,400) |
| Acceptable | P5 ≥ 2200 (vs 1576) AND <1000 ≤ 1.5% (vs 2.5%) AND mean within 5% |
| No-go | mean drops >10% OR floor doesn't improve |

If a checkpoint hits "acceptable" → merge to main. If only one λ value works → that's our default; sweep others as ablation later.

## Risk register

1. **Only 1000 anchors is small**. The aux signal might get drowned out by main loss. Mitigation: ratio the aux batch size to anchor count (256 anchors per main batch = re-cycle every 4 main steps). Effective oversampling.

2. **Counterfactual labels are noisy** (R=24 rollouts, H=100). Some "floor-best" moves are within rollout sampling noise of top-1. Mitigation: filter records where the floor-best advantage is small (e.g., only train on records where die_rate_top1 - die_rate_winner > 0.1 OR p10_score_winner - p10_score_top1 > 30).

3. **Aux loss might regress the main policy**. The 9M-state V14 distillation is what produces strong mean; aux loss could pull the model toward floor-only behavior at the cost of mean. Mitigation: λ sweep with strong cap on aux contribution.

4. **Pillar3c might just reproduce pillar3b** if the labels are inconsistent with what's learnable. We'd see this in per-epoch val loss not changing materially. Decision gate catches it.

## Open questions for ChatGPT

1. **Loss formulation**: Option A (soft KL on candidates) vs Option C (pairwise ranking) — which would you prioritize for this small-N setting?
2. **Filter threshold**: should we filter the 1000 anchors to only those with a "clear" floor-best (significant advantage)? Risk: cuts data to ~500-600. Reward: cleaner labels.
3. **λ range**: 0.05/0.10/0.20 — too low/high? Path B oracle used 0.05/0.10 and stalled.
4. **Stationary band only**, or also include the LATER dangerous states (turn 200+ with empty 25-35) where the position is mid-decline? My instinct says stationary boundary only — late danger points are already pillar3b-poisoned (no available floor-best as Phase B showed).
5. **Augmentation**: counterfactual moves must be transformed consistently under dihedral. Standard 8× symmetric. Color permutation maps same-color move chains correctly.

## Files for review

- `alphatrain/data/stationary_counterfactuals_v1.pt` (1.8 MB) — 1000 anchors
- `docs/phase2_results.md` — Phase 2 headline numbers
- `scripts/mine_stationary_counterfactuals.py` — how labels were produced

## My recommendation

Go with **Option A, soft KL, λ=0.10, τ=0.5, filter for advantage > noise**. Train ONE pillar3c variant with these settings. Evaluate against pillar3b. If acceptable, sweep λ as ablation. If no-go, iterate on filter / τ / loss formulation.

Cost: ~12h Colab compute per training run. Two evaluations (500-seed and 1000-seed OOS) per checkpoint ≈ 2h on M5 each.
