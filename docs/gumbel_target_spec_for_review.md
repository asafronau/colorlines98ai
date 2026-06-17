# Peer review #2: the completed-Q distillation target (Color Lines self-play trunk)

*Follow-up to `selfplay_distill_recipe_for_review.md`. You (Gemini + ChatGPT) diagnosed
the pillar3g regression as **prior-domination** and prescribed a **Gumbel/completed-Q
improvement target** instead of visit-distillation. We acted on it: re-recorded the
self-play corpus with the clean pre-Dirichlet prior + root Q + visit counts per
candidate, then **measured the Q signal before building the trainer.** The measurement
changed the spec in one important way and surfaced ~6 design questions. This brief asks
you to sanity-check the refined target before we commit a multi-hour H100 run.*

---

## 0. One-paragraph recap of the game (why value saturates)

Color Lines 98: 9×9, place-and-clear, **stationary survival** game. Score ≈ 2.05 ×
turns-survived (we measured this — score and survival are the same signal; per-turn
score rate is rock-constant ~2.0). There is **no endgame**; you die when the board
fills. The value head therefore saturates: on the safe ~75–95% of a game *every*
reasonable move survives, so the afterstate value is ~identical (~2.55) across
candidates. Deaths are late density spirals; the decisive states are the crisis onset.
The current policy (pillar3f) is strong — survives to the 10k-move cap on most seeds.

## 1. What we built since last time

- **Recorded, per root, the clean label-side ingredients** (pre-Dirichlet, so exploration
  noise never enters the label): `cand_prior` (log clean prior), `cand_q` (root
  Q = Σ value_sum / Σ visit_count), `cand_visits`, `root_value`, `q_min`, `q_max`,
  top-10 candidates. Dirichlet/temperature stay **on** for the state distribution only.
- **V15 corpus**: 5,345,342 states (legacy V14 re-searched clean@400 sims no-Dirichlet,
  stride-4, + fresh native-Q self-play). MCTS = 400 sims, c_puct 2.5, leaf value = a
  linear feature-value head, q_weight 2.0.

## 2. The measurement that changed the spec (300k sampled states)

**(a) Raw `cand_q`-argmax is a NOISE TRAP, not a signal.**
`argmax(cand_q) ≠ argmax(prior)` **96.9%** of states — *below* random-chance agreement.
Cause: at high-value-spread states the Q-argmax move has a **median of 0 visits** (mean
10.7, vs **109** visits for the prior's move), and is the *least-visited* candidate
**91%** of the time (≤5 visits 85%). I.e. raw Q-argmax systematically selects the
barely-explored candidate whose `value_sum/visit_count` is one high-variance leaf eval.
The most-visited (prior) move's Q regresses to the mean, so it almost never *is* the
argmax. **A naive `softmax(prior + raw_Q)` target would pull the policy toward
unexplored garbage at every decisive state — worse than the visit-distillation we're
replacing.**

**(b) The real signal is sparse but exactly where we want it.** Restricting to
*adequately-visited* candidates (the only trustworthy Q):

| visit floor | disagree(Q,prior) among well-visited | q_spread P50 / P90 | **trustworthy corrections** (≥2 well-visited cands, disagree AND spread≥0.05) | mean \|q_gap\| |
|---|---|---|---|---|
| ≥10 | 83.5% | 0.0039 / 0.036 | **4.9%** | 0.045 |
| ≥20 | 82.5% | 0.0035 / 0.031 | **3.7%** | 0.051 |
| ≥30 | 81.1% | 0.0031 / 0.027 | **3.0%** | 0.056 |

- Median spread even among well-visited candidates ≈ 0.004 → value head **tied/saturated
  on ~95% of states**; the prior is correct there and must be left alone.
- The 3–5% of genuine corrections **concentrate at crisis onset**: dying-game last-decile
  q_spread 0.098 (4× the global mean) vs survived/capped 0.025.
- With disagreement-weight γ≈10 the 4% carries ≈31% of the gradient.

## 3. Refined target spec (what we plan to train)

For each state, over the top-10 recorded candidates:

```
# 1. completed, denoised Q  (the fix for §2a)
N(a)      = cand_visits[a]
Qraw(a)   = cand_q[a]                       # Σvalue_sum/Σvisit_count
completed_Q(a) = Qraw(a)              if N(a) >= VISIT_FLOOR   (~10–20)
               = root_value           otherwise               # no boost: ≈ batch mean

# 2. improvement target
adv(a)    = completed_Q(a) - root_value      # center; raw offset cancels in softmax anyway
target(a) ∝ exp( cand_prior[a] + adv(a) / TAU )      # TAU tuned to the ~0.05 spread
prior(a)  ∝ exp( cand_prior[a] )                     # anchor reference

# 3. per-state disagreement weight (so the 4% isn't drowned)
w = 1 + GAMMA * 1[ argmax_a(completed_Q) != argmax_a(cand_prior) ]   # GAMMA≈10

# 4. loss
L = w * CE(student_logits, target)  +  BETA * KL(student || prior)   # NO sharpening
```

- **No T=0.5 sharpening** (you flagged it; we're dropping it).
- Implementation: scatter target + prior to dense 6561 vectors, reuse the existing
  dihedral augmentation (`policy[mask]=policy[mask][:,pol_lut[t]]`), so index alignment
  is identical to the current trainer. Warm-start TBD (pillar3b base vs pillar3f merge;
  last time warm-start didn't affect the regression).

## 4. Questions for you (where we're unsure)

1. **σ / Q-scaling.** The Gumbel paper uses `σ(q̂)=(c_visit+max_b N_b)·c_scale·q̂`,
   designed for the *few-sim, Gumbel-root-action-selection* regime. We have a **fixed
   400 sims over a fixed top-10 set**. Is the visit-count-adaptive `(c_visit+maxN)`
   scaling still appropriate, or — given fixed sims — is a plain fixed temperature on
   centered advantage (`adv/TAU`, as in §3) cleaner and less fiddly? If σ, what c_scale
   given maxN≈100–200 and the meaningful spread is ~0.05?

2. **Completed-Q denoising.** For *visited* candidates, raw `value_sum/visit_count` is
   still noisy at moderate visit counts. Worth shrinking toward `root_value` by visit
   count (e.g. `(Σvalue_sum + κ·root_value)/(Σvisits + κ)`)? Or is a hard VISIT_FLOOR
   enough? What floor — 10, 20, 30? (Tradeoff: higher floor = cleaner Q but fewer
   candidates comparable; at ≥30, mean well-visited cands ≈ 7.2.)

3. **Is 4% × gap-0.05 enough** to move a 5.3M-state corpus, given the value head is
   saturated (gap is 0.05 in ~2.55-scale survival units)? Or is completed-Q over a
   *saturated* value head fundamentally signal-starved, and we should fold in the
   deeper/widened search (our 4800-sim "firefighting") at the decisive states to get a
   crisper Q before distilling? (We'd prefer the 400-sim trunk to stand alone — 4800 is
   impractical full-game — but want your read on whether the signal is too thin.)

4. **Candidate-restricted CE vs full-softmax CE.** Target lives on the top-10 (≈ the
   prior's support). CE over the 10 candidate logits (sampled-softmax style), or
   full-softmax over 6561 with zero target off-candidate (stricter — also penalizes mass
   the student puts on non-candidate moves)? The prior concentrates on candidates so we
   lean full-softmax; agree?

5. **Disagreement-weight × KL-anchor: redundant or complementary?** Is `w·CE(target) +
   β·KL(prior)` everywhere right, or a **hard partition** cleaner — Gumbel-CE only on the
   ~4% disagree states, KL-anchor only on the ~96% agree states? Worry: applying both
   everywhere, the CE-toward-target and KL-toward-prior partly cancel on agree states
   (where target≈prior anyway, so maybe harmless).

6. **One-shot vs iterate.** This is one generation (V15 from pillar3f). Do you expect a
   single completed-Q distillation pass to give a measurable floor lift, or is the
   per-generation step now so small (4% sparse) that we should batch several
   self-play→relabel→distill cycles before evaluating?

## 5. What we are NOT asking

The prior-domination diagnosis is settled and we agree with it. We are not revisiting
visit-distillation, sharpening, or whether to record Q (done). This is purely: *given the
measured Q signal (sparse, saturated, low-visit-noisy), is the §3 completed-Q target the
right way to extract it, and which knobs matter?*
