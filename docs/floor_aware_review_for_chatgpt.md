# Review request: floor-aware (catastrophe-minimizing) policy improvement for Color Lines 98

We'd like a critical review of the plan below — methodology, feasibility, and
whether we're missing a better approach. Context is self-contained.

## The project in one paragraph

AlphaZero-style agent for Color Lines 98 (9×9, 7 colors; move a ball along a
free path each turn, then 3 balls spawn; clear lines of 5+). Deployment is
**policy-only** (a 10-block×256ch ResNet, no search at inference). Current best,
pillar3b, scores a mean of ~12–17k. **Score is purely a function of survival
time** — ~2.1 points/turn at *all* skill levels (measured and stable; a 140k
game is ~67k turns). So "score" and "how long you survive" are the same
objective, and the only thing that varies game-to-game is when you die.

## The problem we're attacking: the floor, not the mean

Some games die early ("catastrophes," e.g. a density spiral kills the board by
turn ~150). We want to lift the **floor** (P5/P10, %<1000 over a large seed
set), not the mean — because the mean rewards lucky-tail gambling and is fooled
by score=survival.

A prior autopsy established: **every scalar value target we tried saturates**
(survival, cumulative score, TD, density). For a strong player all healthy
boards are equivalent (P(survive)→1), so the value head gives no gradient in the
safe ~75% of the game → MCTS visit targets are flat there → the distilled policy
picks among "equivalent" moves arbitrarily, including locally-wasteful ones.

## The key insight we want vetted

**Tail risk does not saturate where the mean does.** We took a specific stationary
position and compared two candidate moves by running the policy forward under
many common-RNG seeds to natural death, then looked at the *distribution* of
final scores. Two moves had **identical median/mean survival** (the saturating
part) but measurably **different catastrophe rates** (early-death probability).
So the downside/tail of the outcome distribution discriminates moves where the
expectation cannot. This signal is (a) not a hand-coded heuristic — it's the
policy's own rollout outcomes; (b) not a saturating scalar — it's a tail
statistic; (c) demonstrably move-discriminating.

## What we actually measured (and a cautionary result)

We picked a recorded worst-game (died turn 141, score 215), took an early frame,
and compared the policy's move (call it A) against a human-proposed alternative
(B) by resuming policy-only under **paired common-RNG seeds** to death, fp32
(deterministic). We verified the harness reproduces the exact recorded death
(215@141) by injecting the original RNG state.

- At n=500 seeds, B looked ~20% better on the floor (P1/P5/P10).
- At **n=10,000 paired seeds, the difference is not significant** on any metric
  (all bootstrap CIs span 0). Paired sign test: B wins 50.9% of seeds (z=1.76,
  p≈0.08). Catastrophe ≤1000 score: A 3.2% vs B 2.8%. So there is at most a
  **faint (~1-2%) edge** to B, far below the "A is a bad move" we suspected.
- A *single* real-RNG counterfactual (same exact RNG that killed A at 215) gave
  B→20,526 vs A→215 — a 100× gap that is a **true but unrepresentative** sample
  (relocating the ball shifts which cells are empty, so the same RNG draws map to
  different spawns — confounded). The 10k distribution is the truth; the single
  sample wildly overstates the effect.

**Two honest conclusions:** (i) this *opening* move washes out — a turn-10 move's
board footprint is gone within ~30 turns, long before catastrophes strike ~turn
150+; (ii) crucially, **comparing to one alternative and finding a tie does NOT
prove the policy's move is good** — we never searched the top-K for the
floor-*best* move. A substantially better move could exist.

## The plan (what we want reviewed)

**Objective / north star:** minimize the catastrophe rate / lift P5/P10 over a
2,000+-seed eval. (Not the mean.)

**Three routes to a floor-aware policy:**
- **A. Per-move floor mining → soft-reweight distillation.** Mine positions;
  score the **top-K** candidate moves by rollout floor (catastrophe rate over
  R≈200–500 common-RNG rollouts, floor-capped); down-weight gambling moves in a
  small auxiliary distillation loss. (A prior attempt failed using R=24 and a
  100-turn horizon — pure label noise; the signal is in the floor, which needs
  many long rollouts.)
- **B. Catastrophe-probability value head.** Train a head to predict P(die
  within H turns | board), labeled directly from existing self-play (binary
  "did this state's game die within H?" — free). The one scalar we have evidence
  *won't* saturate. Use it as a risk term in search.
- **C. Risk-averse self-play.** Back up a lower quantile (CVaR/P10) of rollout
  outcomes in MCTS instead of the mean, so the visit-distribution targets prefer
  floor-safe moves. Attacks the saturation root cause head-on.

**Finding the bad moves without infinite compute (the feasibility crux):**
brute-forcing every position is infeasible (~weeks). Our intended trick is
**rewind-from-death**: scan for early-death games (cheap), then walk *backward*
from the death — at each rewind depth d, floor-evaluate the top-K candidate
moves over R seeds, and record whether the policy's move is floor-dominated.
The catastrophes self-identify, so rollouts are spent only on death trajectories.
Plotting "is the policy's move floor-dominated?" vs rewind depth d gives an
**avoidability curve**: *are catastrophes move-avoidable, and at what horizon?*
We expect shallow rewinds (near death) to be "sealed" (no move helps) and the
signal, if any, at the density-onset region (d≈20–50).

**Cost:** labeling one position ≈ K×R×(rollout length)/throughput ≈ 5–20 min at
our ~3,100 turns/s. A few hundred targeted positions = hours-to-days. Feasible
given the rewind-from-death targeting; infeasible untargeted.

**Gates (each kills the idea cheaply):** G0 done (frame-10 near-tie, above).
G1 = is it systematic? — rewind-from-death over several death games; go only if
an avoidable fork consistently exists. G2 = does a small floor-aware fine-tune
lift the 2,000-seed floor? — the only real test of whether it compounds.

## Specific questions for you

1. **Is the tail-risk premise sound?** Is "catastrophe probability discriminates
   moves where mean survival saturates" a correct and useful reframing, or are we
   fooling ourselves (e.g., is the tiny per-move signal just irreducible RNG)?
2. **The continuation confound.** Rollout labels use the current policy as the
   continuation, so "floor-best" is bounded by the policy's own downstream skill,
   and labels go stale after a retrain. Is iterate-and-re-mine the right answer,
   or is there a cleaner way (e.g., route B/C amortization)?
3. **Accumulation.** A single move's catastrophe edge is ~1pp and washes out by
   the median. The whole bet is that fixing the *tendency* across many moves
   compounds into a lower game-level floor. Is there a way to test/estimate this
   *before* committing to a full mining+retrain loop?
4. **Of routes A/B/C, which would you prioritize, and why?** Is the
   catastrophe-probability value head (B) likely to saturate too, or does the
   tail-risk argument genuinely save it? How would you sanity-check its target
   distribution before training?
5. **Risk-averse backup (C):** any pitfalls with CVaR/quantile backup in MCTS for
   a single-player stochastic game (variance, exploration collapse, etc.)?
6. **Are we missing a better approach** to lifting the floor of a strong
   policy-only agent that we haven't considered?
