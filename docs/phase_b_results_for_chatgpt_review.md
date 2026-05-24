# Phase B Counterfactual Results — for ChatGPT review

**Date:** 2026-05-23
**Status:** Diagnostic complete. Verdict: NO-GO on the risk-blind hypothesis.

## Recap of the hypothesis

The 19 confirmed pillar3b_epoch_20 floor failures (seeds 800000-800999) all show "clear rate ~28-35% vs equilibrium 37.5%" — board fills slowly, dies at empties 1-3 over 100-500 turns. User intuition: the policy chooses EV-optimal moves that bet on cooperative RNG; when RNG punishes the bet, position is too constrained to recover.

Hypothesis: at "risk decision points" in failure trajectories, the policy's **top-1 move** maximizes mean but has a fat left tail; **top-2..5 alternatives** would have lower mean but better **floor** (survival, P10 score).

## Methodology (per your earlier guidance)

- **Failure pool**: 28 seeds Phase-1 flagged with score <1000 → 26/28 confirmed in full-length replay. Re-instrumented eval_parallel run captured full per-turn state (board, next_balls, top1_p, top1_top2_gap, empties, largest empty component, n_components).
- **Success pool**: 24 random non-failure seeds matched by density bucket to failure DPs.
- **Decision-point filter** (any-of, no `top1_p` hard cut — recorded as analysis field):
  - empties < 30 AND empties[t] < empties[t-5]
  - largest_empty_component[t] < lec[t-5]
  - turns_since_clear > 15
  - AND t < final_turn - 20
- **DPs**: 150 failure + 149 matched success.
- **Counterfactuals**: top-5 candidate moves × K=64 common-RNG rollouts × H=100 turns × policy-only continuation.
- **Lex floor metric** (per your guidance): survival_100 → P10 score → mean score.
- **Workers**: 12 CPU processes (multiprocessing.Pool), ~2.9h on M5.

eval_parallel inference determinism: I verified the same-seed same-args reproduces identical results (7274==7274). What varies game outcomes across runs is the *seeds-list composition* — batched fp16 ordering shifts with worker batch assignments.

## Results

### Floor-winner distribution by candidate rank

| rank | failure DPs (N=150) | success DPs (N=149) |
|---|---|---|
| top-1 | 28.7% | 28.2% |
| top-2 | 22.7% | 15.4% |
| top-3 | 18.0% | 17.4% |
| top-4 | 10.7% | 19.5% |
| top-5 | 20.0% | 19.5% |

### Floor gap when winner ≠ top-1

| | failure | success |
|---|---|---|
| median P10 score gap (floor-winner − top-1) | **+2** | **+2** |
| median survival_100 gap | **+0.0pp** | **+0.0pp** |

Both medians are essentially zero. Means are tiny too (low single-digit pts on a 100-turn horizon producing ~150-300 pts of total score).

### Differential

- Failures lose **−0.5pp** more often (top-2..5 wins 71.3% in failures vs 71.8% in successes — within noise)
- P10 gap: identical at +2 pts
- Survival_100 gap: identical at +0.0pp

## Interpretation

1. **Top-1 wins floor only ~28%** in BOTH failure and success states. The policy is essentially indifferent between its top-5 candidates at these decision points — not a flat-policy *artifact specific to failures*.

2. **No failure-vs-success differential.** The diagnostic was specifically designed to detect "is top-2..5 winning floor MORE in dangerous states?" Answer: no. The pattern is symmetric, which means the policy's choice at these DPs is not the load-bearing variable.

3. **Median P10 gap = +2** out of ~hundreds of points score over 100 turns. There's no exploitable choice-time signal: the "floor-winning move" is statistically indistinguishable from top-1 on score.

4. **Median survival_100 gap = +0.0pp.** Whichever move the policy picks at these DPs, the survival-to-100 rate is identical. **Future RNG dominates the outcome, not the move choice.**

Together these say: the model's flat ranking at these states isn't a mistake to be corrected — it's the **correct** assessment that all top-5 candidates lead to essentially equivalent floors at this strength. The "failures" weren't policy errors; they were unlucky trajectories where the position was already too constrained for any top-5 candidate to recover.

## What this kills

- Risk-blind hypothesis (no signal)
- Correction-dataset fine-tune (no consistent "right move" to label)
- Score-aware value head retrain (head can't see what isn't there)
- CVaR/quantile value head with same selfplay (same data, same conclusions)

## What this leaves

Floor failures at pillar3b's strength are **RNG-bound at this model capacity**. To materially raise the floor:

1. **Larger model**: more parameters → potentially less flat policy → genuinely better top-1 picks. Cost: 2-5× training compute + new architecture decisions. High effort, uncertain payoff.

2. **MCTS at inference**: already +57% mean lift over policy alone (HISTORY 158). Lifts floor too. Out of scope per "policy-only deployment" goal — but if floor is the priority, MCTS is the proven lever.

3. **Curriculum / data enrichment** on high-density borderline states: oversample states like our 150 failure DPs in pillar3c training. Forces the model to spend more gradient on these specific configurations.

4. **CVaR-aware selfplay-corpus generation**: change the V14 selfplay MCTS to optimize CVaR (low-percentile score) instead of mean visit-count. New corpus would reflect floor-aware preferences → next-gen policy distills that taste. This is option (2) from your earlier writeup repurposed for the corpus generation step rather than the value head training step. The current Phase-B finding doesn't argue against this — it just says we can't FIND the labels by rollout judgment from the current policy. Generating fresh corpus with a different MCTS objective could still help.

## Open questions / where I might be wrong

1. **Sample size**: N=150 per side. Power to detect a 5pp survival gap at α=0.05 is roughly 80%. A 2pp gap would slip through. Is 5pp the threshold we should accept, or do we need bigger N?

2. **H=100 horizon**: maybe the consequences of a "safe vs risky" move at turn t materialize beyond H=100? Most failures die <500 turns from game start so at decision-points ~turn 250+, H=100 covers the relevant window. But I haven't checked H=200 directly. Should I escalate top-10 ambiguous DPs to H=200/K=128?

3. **DP filter**: I used "any of" empties<30 OR empties-trending-down OR lec-trending-down OR tslc>15. Maybe these are too lenient and dilute the signal. Tightening to "AND" would give fewer but more dangerous DPs. The 19 failures had ~5-15 such turns each — restricting to "all 4 conditions met" might give the genuinely critical 1-3 turns per game.

4. **Policy-only continuation** for rollouts: I used pillar3b itself as the rollout player. If pillar3b is uniformly risk-blind (not just at the choice point), all branches inherit that risk-blindness downstream. Switching to MCTS@100 as the rollout player might reveal that "top-2 leads to a state where MCTS finds a good continuation but pillar3b doesn't" — but compute cost ~10×.

5. **The 19 trajectories might not be representative**: they're the bottom 2.8% of 1000 seeds. Maybe scaling to 10K seeds would reveal a richer failure-mode taxonomy I haven't bucketed yet.

6. **Lex floor metric**: I prioritize survival_100 over P10 score. Maybe the right ordering is the reverse (P10 first, survival second)? Curious if your guidance on this changes given the empirical result.

## Specific question for you

The data says: **policy choice at our flagged DPs has near-zero influence on 100-turn floor outcomes.** Future RNG dominates.

If this is genuinely the case, the right intervention isn't "fix the policy's choices" — it's "make the policy face fewer dangerous states in the first place." That points to:

- **Different selfplay-corpus generation** (option 4) to shape the policy's *long-horizon* behavior so it doesn't drift into dangerous density profiles
- OR **accept current floor** and focus elsewhere (mean, MCTS-deployment, larger model)

What's your read? Specifically:
- Is the NO-GO verdict defensible at this sample size and methodology?
- Among the alternative intervention paths, which would you prioritize for raising P5 / P10 of policy-only outcomes?
- Is there a smarter "Phase C" diagnostic I should run before committing to a training plan?
