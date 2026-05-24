# Phase B: Counterfactual Diagnostic for Risk-Blind Move Selection

**Status:** APPROVED — methodology revised per ChatGPT review 2026-05-22
**Date:** 2026-05-22
**Context:** Phase A's failure-mining produced 19 confirmed pillar3b_epoch_20 floor failures. User insight: insufficient clearing is the *symptom*; **risk-blind move selection is the cause**. The policy chooses moves with high expected value that bet on cooperative RNG; when RNG punishes the bet, the position is already too constrained to recover.

This document proposes Phase B: a counterfactual rollout judge with statistically defensible sampling, matched success controls, and a lexicographic floor metric.

## The hypothesis

At choice points in failing trajectories, the policy's **top-1 move** maximizes expected score but has a fat left tail. A **top-2 through top-5 alternative** would have lower mean but better survival / floor. If true:
- Identify mis-priced choice points programmatically
- Build a correction dataset for pillar3c training
- Risk-aware training intervention is justified

If false: floor is RNG-bound, different lever needed (larger model, curriculum, MCTS at inference, etc.)

## Methodology (revised per ChatGPT)

### Inputs

- **Model**: `alphatrain/data/pillar3b_epoch_20.pt`
- **Failure trajectories**: 19 confirmed deaths from `logs/instrumented_failures/`
- **Failure seeds list**: the 25-seed list used for the instrumented run (preserves the 19 failures via reproducible eval_parallel batching)
- **Success seeds list**: 25 new seeds sampled from the same 1000-seed range, **excluding** known failures and recoveries (gives ~20-25 successful trajectories at varied score levels)

### Step 1 — Re-run with full per-turn state dumps (~60 min)

Modify `_policy_server_worker` instrumentation to additionally save per turn:
- `board` (9×9 int8 → JSON list)
- `next_balls` (list of `((r, c), color)`)
- `rng_state` (game.rng's internal state — needed only if rollouts want to *resume* the original spawn sequence; we use independent rollout seeds so we can skip this)

(All current per-turn metrics — empties, n_legal_top30, top1_p, top1_top2_gap, cleared, score_delta — stay. Additional bookkeeping for filtering: `lec`, `n_components`, `turns_since_clear`, `legal_move_count`. `available_clears` left uncomputed in this pass; expensive and only needed post-hoc on flagged decision points.)

Re-run twice with the same composition rules:
- **Run F**: 25 failure-seeds list → 19 failure trajectories with full state
- **Run S**: 25 success-seeds list (sampled outside failure set) → ~20-25 success trajectories

### Step 2 — Identify decision points (no hard filter on top1_p)

**ChatGPT correction**: don't require `top1_p > 0.3`. The policy is often FLAT (top1_p ~ 0.13), and floor mistakes can happen at low confidence. Record `top1_p` and `top1_top2_gap` as **analysis fields**, not filters. Post-hoc bucket into low/medium/high confidence.

Decision point filter (any-of):
- `empties < 30` AND `empties[t] < empties[t-5]` (board getting denser)
- `largest_empty_component[t] < largest_empty_component[t-5]` (component fragmenting)
- `turns_since_clear > 15` (haven't cleared in a while)
- AND `t < final_turn - 20` (≥20 turns before death — enough horizon to study consequences)

**Sample sizes**:
- 100-200 **failure** decision points (from 19 trajectories)
- 100-200 **matched-success** decision points (from success trajectories, matched by density profile: `(empties_bucket, lec_bucket, turns_since_clear_bucket)`)

If a failure trajectory has only ~5 valid DPs satisfying the filter, we accept fewer than 100 per side and report sample-size honestly. Goal is statistical decisiveness, not arbitrary count.

### Step 3 — Counterfactual rollouts (top-5 candidates, K=64, H=100)

**ChatGPT correction**: top-3 too narrow (policy is flat → safer move may be rank 4-8). K=32 too thin for stable P10 (effectively 3rd-worst sample). H=50 may miss consequences.

For each decision point `(seed, turn_t, board_state, next_balls)`:

1. Load saved game state.
2. Run forward, extract **top-5 legal moves** by policy.
3. For each top-k move (k ∈ {1..5}):
   - For each `rollout_seed` ∈ {0..K-1} (K=64):
     - Reset game to snapshot state
     - Seed game RNG with `rollout_seed` (common-RNG across branches: same `rollout_seed` means same future spawn sequence regardless of which top-k branch you picked)
     - Apply move `(sr, sc) → (tr, tc)`
     - Play policy-only for up to H=100 turns
     - Record: `score_gained`, `turns_survived`, `died (bool)`, `survived_to_50 (bool)`, `survived_to_100 (bool)`, `final_empties`
4. Per top-k aggregate over K=64 rollouts:
   - `survival_rate_50`, `survival_rate_100`
   - `mean_score_gained`
   - `P10_score_gained`, `P25_score_gained`
   - `median_turns_survived`

**Compute estimate**: 200 DPs × 5 candidates × 64 rollouts × ~1s/rollout (100 turns × ~10ms) / 16 workers ≈ **~1.1h**.

### Step 4 — Aggregate analysis (lexicographic floor metric)

**ChatGPT correction**: don't use P10 alone. A defensive move that survives more often but scores slightly less is a floor win that pure score-P10 would miss. Use a **lexicographic** ranking:

For each decision point, the "floor winner" among top-1..top-5 is decided by:
1. **Primary**: `survival_rate_100` (highest wins)
2. **Secondary** (tie or near-tie on survival): `P10_score_gained` (highest wins)
3. **Tertiary** (tie on both): `mean_score_gained`

Tie tolerance: ±2pp on survival, ±50 pts on P10.

### Step 5 — Cross-tabulate with success controls

| Win-rank in failure DPs | Count | % | | Win-rank in success DPs | Count | % |
|---|---|---|---|---|---|---|
| top-1 | … | … | | top-1 | … | … |
| top-2 | … | … | | top-2 | … | … |
| top-3 | … | … | | top-3 | … | … |
| top-4 | … | … | | top-4 | … | … |
| top-5 | … | … | | top-5 | … | … |

**Bucket by**:
- `empty_count` (deciles or fixed: <10, 10-20, 20-30, 30+)
- `largest_empty_component`
- `turns_since_clear`
- `top1_top2_gap` (low/medium/high confidence) — *post-hoc* analysis field

## Decision rules (revised)

ChatGPT's new rules:

**Go** if at FAILURE decision points, top-2..top-5 beats top-1 by:
- `survival_rate_100` ≥ +10pp, OR
- median `turns_survived` ≥ +20, OR
- `P10_score_gained` ≥ +100 (or +200 in dense states)

**AND** this effect is materially stronger than in MATCHED-SUCCESS controls.

If top-2..5 wins by similar margin in success controls → not specific to failures → the policy isn't risk-blind, top-2 is just sometimes better in expectation. No-go for risk-aware training.

If top-2..5 wins much more in failures than successes → specific to dangerous states → risk-blind hypothesis confirmed → build correction dataset.

## If hypothesis confirmed → training fix

ChatGPT confirmed all three earlier options remain valid:

1. **Direct fine-tune on correction dataset** (cheapest): at flagged failure DPs, replace policy's pick with the floor-winning move from rollouts. Supervised fine-tune. ~6h Colab.

2. **CVaR/quantile value head retrain** (ChatGPT's original recommendation, principled): retrain value head with hybrid target — `survival_H_primary + small_λ × score_rate_H_or_clear_tempo`. Pure score targets risk overfitting to greedy clears; pure survival misses tempo. Both signals needed. ~12h Colab.

3. **Risk-shaped distillation** (cheapest, no extra data): re-weight training samples by `1 - P(trajectory survived)`. Failure-prone states get higher gradient weight.

## Compute estimate

| step | wall time |
|---|---|
| 1. Re-run instrumented (F + S, 50 games) | ~60 min M5 |
| 2. Identify decision points | negligible |
| 3. Counterfactual rollouts (200 DPs × 5 × 64 × 100t) | ~1.1h M5 |
| 4. Analysis + bucketing | negligible |
| **Total** | **~2-2.5h** |

Escalation if signal is weak/disputed:
- K=128 on disputed DPs only
- H=200 on close-call DPs only

## Worst-case rejection cases

1. **All branches die similarly within H=100**: position was already lost. Filter to DPs where ≥1 candidate has survival_rate_100 > 0.
2. **Top-1 wins on ALL three lex tiers in majority**: policy is correct; floor is RNG-bound. Abandon hypothesis; consider stronger model or MCTS at inference.
3. **Failure win-rate ≈ success win-rate**: top-2..5 sometimes wins in expectation everywhere, not specifically at risk decision points. Not actionable for floor.
4. **High variance, no clear winner**: escalate to K=128 on top 30 disputed DPs.

## Files this will produce

| file | purpose |
|---|---|
| `scripts/phase_b_counterfactual.py` | Main script (state-load + rollouts + analysis) |
| `logs/instrumented_failures_full/` | Re-run with full state dumps (board + next_balls per turn) |
| `logs/instrumented_success/` | Success-trajectory dumps |
| `logs/phase_b/per_dp.csv` | Each (seed, turn) studied: top-5 stats |
| `logs/phase_b/summary_failures.txt` | Failure DP cross-tabulation |
| `logs/phase_b/summary_success.txt` | Success-control cross-tabulation |
| `logs/phase_b/comparison.txt` | Failure-vs-success gap analysis + go/no-go verdict |
| `logs/phase_b/correction_dataset.json` (if confirmed) | Training data: `{state, original_pick, floor_optimal_move}` triples |

## Implementation order

1. ✓ Plan approved by ChatGPT
2. Modify `_policy_server_worker` to save full state (board, next_balls) per turn
3. Re-run eval_parallel TWICE with same composition rules:
   - Run F (25 failure seeds) → 19 failure trajectories
   - Run S (25 success seeds) → ~25 success trajectories
4. Identify decision points + match by density profile
5. Implement counterfactual script + analysis
6. Run + analyze
7. Verdict per decision rules
