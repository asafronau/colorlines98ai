# Current State Review for ChatGPT

**Date:** 2026-05-23
**Question:** What's the best next move to raise policy-only P1-P25 for pillar3b?

## Reframe: the deployment model

**Policy-only is the deployment target.** MCTS has never been planned for deployment — it's:
1. The teacher in selfplay corpus generation (produces visit distributions used as distillation targets)
2. A measurement tool during eval (shows what the model COULD do with search, as a ceiling reference)

So when we ask "does MCTS help the floor," we're asking it as an evaluation question, not as "should we ship MCTS." The relevant question for deployment is: **how do we lift policy-only P1-P25?**

Survival-based training (value head's `survival_H` target) is conceptually fine — a perfect player in Color Lines never dies, so optimizing for long survival is consistent with optimizing for long-game score.

## Current state numbers

### Pillar3b policy-only (1000 OOS seeds 777000-777999, HISTORY 160)
- mean 17,255
- P5 ≈ 1,576, P10 ≈ 2,476, P25 = 5,483, P50 = 12,567, P75 = 22,486
- floor (<1000) = 2.5%

### Pillar3b policy-only (150 seeds 10000-10149, max-turns=10000)
- P1=742, P5=1,618, P10=2,340, P25=7,042, P50=12,770
- floor (<1000) = 2.7% [CI 0.7%, 6.7%]

### Pillar3b + MCTS@200 + value_head_v14, q=2.0 (same 150 seeds)
- P1=1,051, P5=2,564, P10=3,690, P25=6,464, P50=14,132 (cap-bound)
- floor (<1000) = 1.3% [CI 0.2%, 4.7%]
- MCTS lift on tail: P5 +58%, P10 +58% — **but** P25 regresses -8%, and floor difference is NOT statistically distinguishable at N=150 (CIs overlap)

### Target
- Min > 1000 (reframed: aspirational, accept some floor)
- **P5 > 2,500** (currently 1,618 policy-only)

## What we've ruled out (with evidence)

1. **MCTS sim-budget scaling**: paired 50-seed test, 200 vs 400 vs 800 sims on the same crisis seeds — survival-turn medians all 0, win/tie/loss roughly even. Doubling sims past 200 buys nothing. This is the **plateau finding**.

2. **Risk-blind hypothesis (Phase B)**: top-2..5 alternative moves do NOT win the floor at late danger points. 28% top-1 wins in failures matches 28% in successes. NO failure-vs-success differential. The local choice isn't the load-bearing variable at these late states.

3. **q_weight beyond 2.0**: tested 1.0, 1.5, 2.0, 2.5 on pillar3a. q=2.5 has 4× higher floor failure rate. q=1.5 trades mean for P25 lift. q=2.0 is the "balanced floor" choice.

## Phase C finding (the actionable signal)

Comparing failure trajectories to success trajectories on rolling-window metrics, the divergence emerges **at turn 100** — much earlier than the "late danger points" we sampled in Phase B:

| metric | divergence turn | failure value | success value |
|---|---|---|---|
| score_rate_50 | 100 | 1.854 pts/turn | 2.038 pts/turn (+9%) |
| clear_rate_50 | 100 | 34% | 37% |
| empties_slope | 125 | −0.16/turn | +0.02/turn |
| top1_p_mean | 150 | 0.314 (peakier) | 0.258 |

By turn 100, failure trajectories are already scoring slightly less per turn. Board density actively deteriorates from turn 125. Policy uncertainty *increases* at turn 150 — model belatedly "realizes" the position is bad. By turn 175-200, state-level metrics (empties, lec) have clearly diverged.

**Implication**: floor failures are caused by **slow accumulation of slightly worse density management** over many mid-game turns (100-200), not by one bad decision near death. By the time we got to turn 250+ "late danger points," it was already too late.

## Value head status

`value_head_v14` is currently being A/B tested against `value_head_sharp25_ep12` (the pillar3a head, mis-paired to pillar3b backbone). The 16-seed 200-sim 1M-cap test showed MCTS gave +19% mean over Pol with value_head_v14 — modest, but in line with HISTORY 158's pillar3a pattern.

Value head training target: V11-style **survival_H** (turns-until-death over horizons {25, 50, 100, 200}). NOT score-based. (We considered score-aware head training but cancelled when Phase B refuted risk-blindness.)

## Live questions

### Q1. Is the MCTS plateau caused by policy flatness, or by value head limits?

If pillar3b's policy were sharper (top1_p ≥ 0.5 instead of ≈0.13), would MCTS at the same sim count provide more lift? Or does the value head fail to distinguish branches regardless of policy sharpness?

This matters because:
- If policy flatness is the cause → sharpen pillar3c via lower target_temperature
- If value head is the cause → retrain value head with sharper labels (multi-horizon survival is already there; what else?)

### Q2. Which intervention has the highest expected P1-P25 lift?

Candidates (your earlier guidance was V14 corpus + curriculum):

**A. V14 corpus generation with pillar3b teacher + sharper distillation targets**
- Sims: MCTS@200 selfplay + crisis (per plateau finding)
- target_temperature: 0.3 or 0.25 (sharper than V13's 0.5) to address policy flatness
- Effort: 2 days compute (corpus gen + training + eval)
- Expected: incremental — pillar3b → pillar3c lift similar to pillar3a → pillar3b (+15-20%)

**B. Dense-state curriculum / oversampling**
- During training, upweight mid-game states from low-tail trajectories
- Specifically: states from games scoring <P25, sampled at turns 80-200 (Phase C divergence window)
- Effort: trivial — modify training data loader to apply sample weights based on (final_score, turn) per state
- Composable with A
- Expected: if Phase C signal is real, this directly targets the cause

**C. Score-rate auxiliary head**
- Train an auxiliary head that predicts `score_gained over next 50 turns`
- Use it (alongside survival head) at MCTS time → cleaner score-vs-survival tradeoff
- Effort: medium — design + train + integrate
- Risk: similar to score-aware head idea you previously cautioned against — but with hybrid target as you suggested

**D. Larger model**
- ResNet 14 blocks 320 channels (vs 10 blocks 256ch)
- Effort: 4-7 days for retrain + new corpus
- Expected: probably needed eventually but heavy lift

### Q3. Should we re-run Phase B with strict DP filter or longer horizon?

Phase B used H=100 turn horizon and "any-of" DP filter. We could:
- Tighten filter to "all-of conditions met" → fewer DPs but ALL high-confidence danger points
- Run with H=200 to capture longer-horizon consequences
- Run with MCTS continuation instead of policy-only (you previously argued against this; sticking with policy-only for deployment-relevant signal)

Costs: ~2-3h additional M5 compute per Phase B variant.

### Q4. Does the Phase C "policy gets peakier under stress" signal mean anything?

top1_p_mean rose from 0.258 (success) to 0.314 (failure) by turn 150. The model commits MORE confidently when the position is already bad. This is the opposite of what we'd want — under stress, the model SHOULD probably explore more (or pick safer moves).

Is this a fixable training-objective issue? Or is it just the natural consequence of dense boards having fewer legal moves (forcing top1 higher mechanically)?

## My current best-guess plan

1. **V14 corpus generation** with pillar3b + value_head_v14 + MCTS@200 (per plateau). Selfplay 200 games, crisis 50 seeds.
2. **Train pillar3c with target_temperature=0.3** (sharper than V13's 0.5) — directly attacks policy flatness.
3. **Add dense-state oversampling** for pillar3c: upweight states with low LEC or empties<30, especially from low-final-score trajectories. Composable with (2).
4. **Skip MCTS sim variation** in this iteration — we've confirmed the plateau.
5. **Don't retrain value head** for V14 — keep value_head_v14, since survival-based is conceptually fine per "perfect player never dies."

This avoids the bigger-lift options (larger model, score-rate aux) but is the cheapest path forward.

**Decision request for you (ChatGPT):** does this plan look right, or would you prioritize differently? Especially: is sharper target_temperature (0.3) safe, or is 0.5 a sweet spot we shouldn't cross?
