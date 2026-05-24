# Phase 2 Results: Stationary-Boundary Counterfactual Mining

**Date:** 2026-05-24
**Branch:** `experiment/stationary-risk-head`
**Status:** complete; ready for Phase 3 design review

## Summary

Mined counterfactual rollout labels at 1000 stationary boundary states from V13 selfplay. For each anchor, evaluated pillar3b's top-10 candidate moves via common-RNG rollouts (R=24, H=100) and identified the floor-best move by lex objective (die_rate → leave_band_rate → P10 score → mean score).

**Headline result: pillar3b's top-1 wins the floor in only 18% of stationary boundary states. The floor-best move is rank 2-10 in 82% of cases.**

This is the actionable signal Phase B missed (Phase B sampled late-danger trajectories where the position was already lost). At stationary boundary states, the policy is genuinely floor-suboptimal — and the suboptimality is broadly distributed across the top-10.

## Run config

- Source corpus: `data/selfplay_v13` (893 games)
- Anchor sampling: 1000 states stratified by (LEC bucket × n_components bucket), empty ∈ [32, 50], turn ≥ 100, stride 50
- Top-K candidates per anchor: 10 (policy's top-10 by logit)
- Rollouts per (anchor, candidate): R = 24 common-RNG continuations
- Horizon: H = 100 turns
- Floor objective (lex): die_rate → leave_band_rate (P(min_empty<30)) → −P10(score) → −mean(score)
- Compute: 12 workers, CPU, ~7h wall time

## Floor-winner rank distribution

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

The distribution past top-1 is roughly uniform across ranks 2-10. Consistent with pillar3b's flat policy in this regime (top1_p ≈ 0.13): MCTS visit margins are too narrow to consistently identify the floor-best move, so it lands almost anywhere in the top-10.

## Sample anchors (illustrative)

```
[100] empty=33, lec=14, n_comp=8:
        top-1 die=0.08 p10=196
        floor_winner #8 die=0.00 p10=200   # rank-8 move avoids deaths

[500] empty=40, lec=28, n_comp=4:
        top-1 die=0.00 p10=198
        floor_winner #7 die=0.00 p10=204   # both safe; rank-7 has higher P10

[999] empty=38, lec=19, n_comp=10:
        top-1 die=0.08 p10=192
        floor_winner #2 die=0.00 p10=194
```

Two failure modes for top-1 visible:
- **Primary**: top-1 has nonzero die_rate; some lower-rank candidate has die_rate=0 (~50% of non-top-1-wins)
- **Secondary**: top-1 has die_rate=0 already; a lower-rank candidate has higher P10 score (the remainder)

## Comparison to Phase B

| analysis | source states | top-1 floor wins |
|---|---|---|
| Phase B | late danger points in failure trajectories | 28.7% |
| Phase B | matched success-trajectory DPs | 28.2% |
| **Phase 2** | **stationary boundary states (empty 32-50)** | **18.0%** |

Phase B's null result (failure vs success matched at ~28%) was misleading — it was sampling the wrong regime. At stationary boundary states (the precursor window before deterioration), top-1 wins LESS often and the gap to floor-best is exploitable.

## Files

- `alphatrain/data/stationary_counterfactuals_v1.pt` (1.8 MB) — 1000 records
- `logs/overnight_phase2/mine_counterfactuals.log` — full mining log

## Implications for Phase 3

The data supports building a distillation auxiliary loss that teaches pillar3c to prefer floor-best moves at stationary boundary states. The label exists for 82% of these anchors. Now we need the right loss design — see `phase3_distillation_plan.md`.
