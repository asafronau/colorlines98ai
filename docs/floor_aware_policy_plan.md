# Floor-Aware Policy Improvement — Plan & Feasibility

Status: 2026-05-30. **GATE 0 ran and came back a near-tie** (see §0.5). Numbers
are sourced from `logs/resume_28_vs_74_n{500,2000,10000}*` + the GPU throughput
line (~3,100 fp32 rollout-turns/s, MPS, 16 workers). Verify before re-quoting.

## 0.5 GATE 0 result (frame-10 (2,8) vs (7,4), paired, fp32)

n=10,000 paired seeds: **no metric is statistically significant** (every
bootstrap CI spans 0). Paired sign test: (7,4) wins **50.9%** of seeds
(z=1.76, p≈0.08) — a faint, sub-significant lean. Catastrophe ≤1000: (2,8)
3.2% vs (7,4) 2.8% (−0.4pp). The n=500 "~20% better floor" was sampling noise
(shrank to ~2% at 10k).

**Interpretation — three things, precisely (don't overstate any):**
1. An early move like (2,8) is **recoverable** unless RNG is extremely unlucky;
   that's why the bulk of the distribution is tied. The difference lives only in
   the unlucky tail (catastrophe), where (7,4) is **marginally** more robust.
   Objective = minimize catastrophe, so even this tiny edge is the right *kind*
   of signal.
2. **This does NOT establish (2,8) is fine.** We compared to ONE hand-picked
   alternative. We did NOT search the top-K for the floor-best move. A
   substantially better move may exist; "(2,8) ≈ (7,4)" ≠ "(2,8) is good." The
   proper experiment ranks all top-K candidates by floor.
3. The single real-RNG counterfactual ((7,4)→20,526 vs (2,8)→215) is a *true*
   point but **unrepresentative** in magnitude (one lucky spawn-shift draw,
   confounded). The ~1-2% distributional edge is the real number.

**Scope:** frame 10 is the *opening* (rewind d=131 from the turn-141 death), so
this confirms the autopsy's washout prediction *for opening moves*. It does NOT
test the density-onset region (rewind d≈20–50) — the only place a sharp
avoidable fork could still live. → GATE 1 (§3.5 rewind-from-death).

---

## 0. What we just established

- **score = survival**, ~2.1 pts/turn at all skill levels (README; our probe 2.0).
  Every *mean/median* value target saturates for a strong player (graveyard).
- The deployed policy (pillar3b_epoch_20) plays moves a human flags as "dead"
  (e.g. seed-835 frame-10 `(0,0)->(2,8)`: blocks a blue line, no red future).
- **n=500 finding (not yet significant):** vs relocating the *same* ball to
  (7,4), (2,8) has a ~20% lower floor (P1/P5/P10) and more catastrophes at every
  threshold (≤1000: 2.8% vs 1.8%); **median is tied**. Direction unanimous,
  underpowered. Paired n=2000 run pending.

## 0.7 ChatGPT review (2026-05-30) — adopted refinements

Route priority revised → **B′ → A (as data-gen) → C (later)**.

- **B′ — action-conditioned risk teacher, NOT state-only.** Predict downside per
  (state, candidate move) / afterstate: P(score<1000), P(die≤H), P(leave
  stationary band), P(min_LEC low), + score-rate. A state-only P(die|board) head
  detects danger but cannot teach *which move* reduces risk. This is the new top
  route; it amortizes rollouts into a learned teacher.
- **Don't repeat the aux-loss surgery (A).** High-R rollout labels train **B′
  first** (validate on held-out counterfactuals); only then distill to policy
  with conservative soft targets + policy-prior blending. Never inject sparse
  argmax-flips as a tiny aux loss against the big distillation corpus — that is
  what sank pillar3c (+ the BN-concat-batch contamination lesson).
- **C — risk-averse search = root/top-K reranking only, later.** Full-tree
  CVaR/quantile backup is noisy, not Bellman-consistent, and can collapse
  exploration toward locally-safe/globally-bad moves. Only after B′ is calibrated,
  rerank the root top-K by sampled downside.
- **The real guard is the WINNER'S CURSE, not a healthy control group.**
  (Correction 2026-05-30: "matched healthy controls" was wrong — a healthy board
  stays healthy under good play, catastrophe≈0 for every move, so it has no fork
  to find; healthy controls are a vacuous null.) The actual trap: scoring K
  candidates at R seeds and keeping the floor-*best* biases it to beat the policy
  by a few pp **even when all K are truly equal** (selecting the luckiest of K
  noisy estimates) — exactly the (2,8) magnitude. **Guard = held-out
  re-evaluation:** re-run each flagged floor-best + the policy's move on a FRESH
  independent seed set; keep the fork only if it still wins with paired CI
  excluding 0 and is ≠ the policy's move. Lightweight (re-eval survivors only).
  A *density*-matched control (empties/LEC/components, NOT "healthy") is an
  optional later question — "does death-selection harbor avoidable errors beyond
  just being dense?" — relevant to the teacher, not to validating a fork.
- **Score-rate is a constraint/tiebreaker, not dropped.** Globally score=survival,
  but *locally* clear-cadence keeps the board stationary; a floor-only objective
  can learn low-tempo turtling that worsens density. Keep score-rate as a guard.
- **Explicit Gate-1 threshold.** "Avoidable fork" := floor-best candidate beats
  stored top-1 by **≥~3pp paired catastrophe rate, and SURVIVES held-out
  re-evaluation** (re-run on fresh seeds, paired CI excludes 0), is ≠ the
  policy's move, and recurs across games/depth buckets.

### Revised Gate 1 (teacher-feasibility, not policy-training)
Rewind sweep d∈[5,50] (skip the sealed cliff; cap at the density-onset region),
top-5/8 moves, R≥500 paired fp32, floor-cap ~1000 turns → **discovery**. For each
flagged floor-best, **held-out re-eval** on fresh seeds → keep real forks only
(winner's-curse guard). THEN train an action-conditioned risk head on the
surviving labels; proceed to policy distillation only if held-out action-ranking
accuracy is clearly > chance. (Replaces the earlier fixed-depth / aux-loss /
healthy-control framing.)

## 1. The key insight — tail risk is the non-saturating signal

Mean survival saturates (all safe boards → P(survive)≈1), which is why every
scalar value target died. But **the catastrophe rate does not saturate where the
mean does**: two moves with identical median outcome had *different* early-death
probabilities (2.8% vs 1.8%). The thing that discriminates moves in the safe
~75% of the game is the **downside of the outcome distribution**, not its
expectation. This signal is (a) not a heuristic — it's the policy's own
rollouts; (b) not a saturating scalar — it's a tail statistic; (c) demonstrably
discriminates a move the policy gets wrong. All three constraints we've been
stuck on, satisfied at once.

## 2. The proposed loop (your idea, formalized)

1. **Find** stationary positions where the policy's top move is floor-suboptimal.
2. **Label** by running N common-RNG rollouts to death (floor-capped) over the
   **top-K candidate moves** and ranking by floor (P10 / catastrophe rate) — NOT
   by mean. (This answers "is (7,4) best?": we search top-K and take the actual
   floor-best, which may beat (7,4). The hand-picked alternative was just a probe.)
3. **Fine-tune** the policy toward the floor-best continuations.

Two forms for step 3: **(a) hard target** — replace the bad move's target with
the floor-best move (risk: pillar3c's argmax-flip regression); **(b) soft
re-weight** — down-weight gambling moves / up-weight floor-safe across the
top-K (gentler, preferred). Or skip per-move labels entirely — see §3B/C.

## 3. Three ways to generate the signal

**A. Per-move floor mining (most direct; your literal proposal).**
Label top-K moves per position by rollout floor. Direct, testable on the current
policy without regenerating self-play. Cost-bound by floor-cap + common-RNG
ranking. This is "pillar3c done right" — pillar3c failed because it used R=24
and a 100-turn horizon, which is pure noise; the signal lives in the floor,
which needs many *long* rollouts.

**B. Catastrophe-probability value head (amortized).**
Train a head to predict P(die within H turns | board), labeled directly from
existing V11–V13 self-play (binary: did this state's game die within H? — free,
no new rollouts). If it has signal (must blob-check the target dist per
`feedback_sanity_check_data`), use it as a risk term. Amortizes the rollout cost
across self-play we already have. The one scalar we have evidence *won't*
saturate.

**C. Risk-averse self-play backup (the principled fix).**
Change MCTS to back up a lower quantile (CVaR / P10) of rollout outcomes instead
of the mean. Visit-distribution targets then inherently prefer floor-safe moves;
the distilled policy stops gambling. Attacks the autopsy root cause (flat
survival targets → arbitrary picks) head-on. Biggest investment (regenerate
self-play), highest ceiling.

## 3.5 The concrete miner: rewind-from-death (solves the pre-filter)

Instead of sampling random stationary positions (most are already floor-optimal
→ wasted rollouts) or hand-building a structural detector, **let the
catastrophes self-identify**: scan for early-death seeds (cheap,
`find_worst_game.py`), then for each dead game walk BACKWARD from the death:

- At rewind depth d (board d moves before death), enumerate the top-K policy
  candidate moves and floor-evaluate each over R≈200 fresh common-RNG rollouts
  to death (floor-capped). Record the policy move's catastrophe rate vs the
  floor-best candidate's. Sweep d = 1, 2, 3, …

This is exactly what `resume_eval_parallel.py` already does (resume from a frame,
try a move, N seeds to death) — the rewind sweep is just running it at frames
[death−1, death−2, …] over top-K candidates. **The infra is already built.**

- **Why it's the right targeting:** the pre-filter problem (don't burn rollouts
  where the policy is already optimal) is solved for free — death trajectories
  ARE the targeted positions. Catastrophes self-select; no detector needed.
- **The avoidability curve (scientific payoff):** plotting "is the policy's move
  floor-dominated?" vs rewind depth d directly answers *are catastrophes
  move-avoidable, and at what horizon?* — the question the autopsy left open.
- **Honest expectation, from the autopsy:** spirals are sealed ~turn 110–120 and
  the policy plays the best move AT the cliff. So expect **shallow rewinds
  (d≈1–15) to be "sealed"** — every candidate dies, no signal. The signal, if
  any, is at **deeper rewinds (d≈20–50)**, the density-onset region, where the
  per-move effect is the small (~1pp) tail thing. (Seed-835 frame-10 is rewind
  d=131 — the *opening* — which is why it showed only ~1pp; the sweep probes the
  density-onset region we have NOT tested, where effects may be larger.) The
  sweep efficiently finds the depth where avoidability turns on, if it does.
- **Cost management:** coarse-to-fine — stride the rewind (every 5th move) at low
  R to locate the avoidable window, then go dense + high R only there. Stop
  rewinding once the floor is uniformly safe (recoverable) or uniformly dead
  (sealed). Don't full-R every depth.
- **Clean, not confounded:** each candidate's floor is over fresh random seeds,
  so comparing candidates' catastrophe rates is the validated distributional
  test — not the confounded single-RNG paired sample.

## 4. Feasibility — honest cost math

Throughput ≈ 3,100 rollout-turns/s (fp32/MPS/16w). Floor-cap H_cap≈1,500 turns
captures all catastrophes (early deaths producing ≤2,000 score happen by
~turn 950); most rollouts hit the cap (median natural death ~6,300 ≫ 1,500), so
~1,450 turns/rollout avg. Common-RNG pairing across candidates lets a modest R
*rank* candidates even if it can't pin exact rates.

Cost to floor-label ONE position = K × R × 1,450 / 3,100:
- K=5, R=128  → ~5 min/position
- K=10, R=256 → ~20 min/position

| Scope | Positions | Time (Mac) | Verdict |
|---|---|---|---|
| Gate sample | 30–50 | 2.5–4 h | **feasible now** |
| Targeted fine-tune set | 300–1,000 | 1–3.5 days | feasible (overnight/cloud) |
| Untargeted (no pre-filter) | 10k+ | weeks | **infeasible on Mac** |

**The feasibility wall is the pre-filter — and §3.5 (rewind-from-death) is the
answer.** Brute-forcing every position is infeasible; we must *cheaply flag*
likely-bad positions. The cleanest flag is **the deaths themselves**: scan for
early-death seeds, then rewind. Catastrophes self-select, so no separate
detector is needed, and the coarse-to-fine rewind concentrates R only on the
depths where avoidability might turn on. (Other pre-filters considered —
structural dead-block detector, low top-1 confidence — are weaker and the
detector is heuristic-adjacent; rewind-from-death dominates them.) Route B
sidesteps per-move rollouts entirely (labels from self-play outcomes).

**Two layers of feasibility, be explicit:**
1. *Generating labels* — feasible at hundreds-of-positions scale (hours–days),
   given a pre-filter and/or cloud. Quantified above.
2. *Whether fine-tuning actually lifts the floor* — **UNKNOWN, the real risk.**
   The per-move signal is small (~1pp); value only materializes if the policy
   makes such moves *often* and fixing the *tendency* compounds (the accumulation
   hypothesis — still untested; single-move swaps can't test it because the move
   washes out by the median). pillar3c regressed on (noisy) move-preference
   labels; clean labels may or may not change that.

## 5. Open risks & gates

- **Systematic or one-off?** We'd have one position. GATE 1 below.
- **Continuation confound.** Rollout labels use the current policy as
  continuation, so "floor-best" is bounded by the policy's own downstream quality
  and labels go stale after a retrain → needs iteration (mine→train→re-mine).
- **Accumulation unproven.** GATE 2 below is the only real test.
- **Label form.** Hard-target argmax-flip is what sank pillar3c; prefer soft
  risk-aware re-weighting + small λ + warmup.

## 6. Recommended staged plan (each gate kills the project cheaply if it fails)

- **GATE 0 (running):** paired n=2000 confirms (2,8) is floor-worse than (7,4).
- **GATE 1 — Systematic? (~half a day), via rewind-from-death (§3.5).** Take a
  handful of early-death games; for each, sweep rewind depth d (coarse-to-fine)
  and floor-label the top-K candidates at each depth (reuse
  `resume_eval_parallel.py` + a top-K driver). Produce the **avoidability curve**
  per game. Go iff there is, *consistently across games*, a rewind window where
  the policy's move is floor-dominated by an alternative (i.e. an avoidable fork
  exists). If every death is "sealed at the cliff with no earlier fork," stop —
  the floor is genuinely RNG, not move-fixable.
- **GATE 2 — Learnable / does it compound? (~2–3 days).** Build a few-hundred-
  position floor-labeled set, soft-re-weight fine-tune pillar3b, eval the
  **floor** (below) on 2,000 seeds. Go iff the floor lifts without tanking mean.
- **THEN scale:** either expand per-move mining (Route A) or commit to the
  catastrophe-prob head + risk-averse self-play (Routes B/C) for the real loop.

## 7. Success metric (the north star shift)

Stop optimizing **mean** (rewards lucky-tail gambling, fooled by score=survival).
Optimize the **floor**: P5 / P10 and **%<1000** over a 2,000-seed fp32 eval.
Crisp, non-saturating, and exactly what the anti-gambling reframe targets.

---

### Bottom line
Generating the labels is feasible at the scale needed for a first fine-tune
(hundreds of targeted positions, hours–days) **given a pre-filter** — finding
that pre-filter is the gating sub-problem. Whether the fine-tune actually lifts
the floor is the genuine unknown and is settled only by GATE 2. Recommend
running GATE 1 the moment GATE 0 validates; it's cheap and decides everything
downstream.
