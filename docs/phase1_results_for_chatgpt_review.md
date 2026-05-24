# Phase 1 Results: Stationary-Risk Head — for ChatGPT Review

**Date:** 2026-05-23
**Branch:** `experiment/stationary-risk-head`
**Question:** Did the Phase 1 prototype pass your decision gate? Proceed to Phase 2?

## Recap

Built `build_stationary_risk_dataset.py` and `train_stationary_risk_head.py` per your spec. Dataset of ~190K windows from V13 selfplay + V14 selfplay + V13 crisis, multi-output regression of forward-100-turn outcomes on a frozen pillar3b backbone.

## Dataset

| corpus | min_start_turn | windows |
|---|---|---|
| selfplay_v13 | 100 | 116,816 |
| selfplay_v14 (partial) | 100 | 18,849 |
| crisis_v13 | 0 | 54,159 |
| **total** | — | **189,824** |

Labels (per state, forward H=100):
- `min_empty_H`, `min_lec_H` (regression; cleaner-than-binary)
- `empty_delta_H`, `lec_delta_H`
- `score_rate_H`, `clear_rate_H`

Label statistics (computed at dataset build):
- `min_empty<30`: 23.9%, `min_empty<25`: 8.2%
- `min_lec<15`: 69.1%, `min_lec<10`: 22.1%
- `lec_delta<=-10`: 23.9%

## Head architecture

Pillar3b backbone (10 blocks × 256 channels) **frozen**. Backbone outputs (B, 256, 9, 9). Then:
- GAP → (B, 256)
- FC 256 → 128 → 128 → 6 (multi-output regression)
- Total head params: **50,182**

Training: 5 epochs, AdamW (lr 1e-3, wd 1e-4), batch 4096, on M5 MPS. Total wall time **~65 seconds** for all 5 epochs (frozen backbone is cheap; only the small head trains).

Game-level train/val split (10% val, no seed leakage):
- Train games: 3,015 → 171,172 windows
- Val games: 335 → 18,652 windows

## Results — final epoch (5)

| metric | AUC | gate (>0.75)? |
|---|---|---|
| **min_empty<25** | **0.895** | ✓✓ strongly above |
| min_empty<30 | 0.835 | ✓ above |
| lec_delta<=-10 | 0.785 | ✓ above |
| min_lec<10 | 0.746 | ~ just below (-0.004) |
| min_lec<15 | 0.725 | ~ slightly below (-0.025) |

Val MSE per epoch: 0.707 → 0.683 → 0.666 → 0.661 → 0.656 (still descending, hasn't plateaued).

## Interpretation

**Strong signal on empty-count drift.** AUC 0.83-0.90 means the backbone genuinely "sees" how board occupancy will evolve over the next 100 turns. This is the cleaner derivable from features.

**Moderate signal on LEC-based metrics.** AUC 0.72-0.78. The model partly extracts topology info from spatial features but loses precision under GAP. LEC depends on graph connectivity, which compresses badly under global average pooling.

**My read:** Phase 1's hypothesis ("backbone features carry stationary risk") is **confirmed** for the cleanest signals. LEC predictability is marginal but non-trivial — clearly above chance, just not at the 0.75 threshold you set.

## Why the LEC AUCs are weak — diagnosis

GAP discards spatial structure. LEC requires BFS over the empty-cell graph; the network would need to learn an implicit "is this configuration connected" feature inside the backbone. Two reasons it's not great:

1. The backbone was never trained on a connectivity objective; it learned to predict MCTS visit distributions, which favor *immediate-move* features over *global-structure* features.
2. GAP averaging across 9×9 hides structural details — a board with 30 empties in one component looks ~the same average-feature-wise as 30 empties in 6 fragments.

If we want stronger LEC prediction, the head needs spatial structure. Options:
- **Replace GAP with a 1×1 conv → flatten**: head sees per-cell features. ~256K params (5× larger).
- **Add a conv layer before GAP** (e.g., 3×3 conv → ReLU → GAP): preserves local structure. Cheap.
- **Larger MLP hidden** (128 → 256): probably won't help; the info-loss is at GAP, not in the MLP.

## Decision points for you (ChatGPT)

### Q1: Is Phase 1 passed?

Strong path: 3 of 5 binary derivatives well above 0.75. Empty-based signals are very predictable. LEC is marginal.

Strict path: 2 of 5 below threshold. Not "all signals at AUC > 0.75."

**Which read?** Specifically — is `min_empty<25` AUC 0.895 alone enough to confirm signal, or does the floor case require LEC prediction to be tight too?

### Q2: Improve head first, or proceed to Phase 2?

Option A: **Proceed now** — empty-count signal is strong enough for Phase 2 mining to use as a primary risk target. LEC stays as a secondary feature.

Option B: **Improve head first** (~30 min M5 work): swap GAP for a 1×1 conv head, retrain, see if LEC AUCs clear the gate. Confirms the latent signal isn't just empty count.

Option C: **Different labels** — drop LEC targets, focus only on empty/score targets. Reduces the head from 6 outputs to 4, simpler signal.

My instinct: Option B (improve head). Doesn't cost much, and if LEC AUC clears 0.78+, Phase 2 has access to both density and topology signals. If LEC stays stuck at 0.72-0.75 even with spatial head, the conclusion is "LEC isn't well-encoded in backbone features" — informative for Phase 2's mining design (use LEC as a hand-computed feature alongside model predictions).

### Q3: For Phase 2 mining specifically

Given the AUC profile (empty signal strong, LEC moderate), should Phase 2:
- Sample stationary boundary states using **hand-computed LEC** as the stratification criterion (cheap, exact), OR
- Use the trained head's predictions to guide sampling (faster but model-dependent)?

I'd argue hand-computed LEC: it's free, exact, and not subject to the head's calibration limits.

### Q4: Validation set diagnostics

Should I report:
- AUC per **bucket** (empty bucket, LEC bucket) to see where the head fails?
- Calibration plots (Brier score, reliability curves)?

The aggregate 0.74 for min_lec<10 might hide a wide range — high accuracy in some regimes and 0.5 chance in others.

## Bottom line

I think we passed but with one weak spot. The conservative read calls for Q2-B (improve head); the pragmatic read calls for Q2-A (proceed). Want your call before committing to ~2 days of Phase 2 compute.

## Files for reference

- `scripts/build_stationary_risk_dataset.py` — dataset builder
- `scripts/train_stationary_risk_head.py` — head training
- `alphatrain/data/stationary_risk_v1.pt` — 189K-window dataset (26 MB)
- `alphatrain/data/stationary_risk_head_v1.pt` — trained head (best epoch 5)
- `logs/train_stationary_risk_v1.log` — training log
