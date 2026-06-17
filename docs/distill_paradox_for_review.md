# Peer review #3: a 2×-better teacher, yet distilling it REGRESSES the policy

*Color Lines 98, AlphaZero-style trunk. Two prior reviews diagnosed prior-domination and
designed a completed-Q target. We built it, validated it offline, trained it — and it
regressed hard in gameplay. The autopsy found we distilled the wrong thing. We want your
read on the remaining paradox and the fix.*

## The setup
- **Stationary survival game** (score ≈ 2.05 × turns survived; no endgame; death = board fills).
- **pillar3f**: strong base policy. Raw greedy policy (no search), 200 held-out seeds, cap
  20000 turns: **mean 23042, P50 20959, P10 3355, P5 2479, <1000 0.5%**.
- **Teacher = MCTS@400(pillar3f)** (400 sims, c_puct 2.5, value head, q_weight 2.0; plays
  `argmax visits`). Its self-play floor (capped at 10k moves, so means are floored):
  **P10 7127, P5 3508, <1000 0.9%** — i.e. its P10 floor is ~2× the raw policy's. The
  teacher is genuinely a stronger player.

## What we tried, both REGRESSED (raw policy, same eval, same seeds)
| model | mean | P50 | P10 | <1000 |
|---|---|---|---|---|
| pillar3f (base) | 23042 | 20959 | 3355 | 0.5% |
| **pillar3g** — distill soft VISIT distribution (+ T=0.5 sharpening, uniform CE) | regressed (~18.6k median, floor worse) | | | |
| **pillar3g2** — distill completed-Q Gumbel target (τ=0.02, κ=15, candidate-restricted CE, γ=10, warm-start pillar3f) | 11411 (ep1) → 7684 (ep2) | 7939→6076 | 1690→1275 | 5.5%→8.0% |

pillar3g2 **halved** the policy in one epoch, monotonic with training; the floor we were
fixing got **worse** (sub-1000 deaths 1→16 of 200). Offline signals were ALL green
(corr_match rose, target audit clean, a 4800-sim calibration "confirmed" 94% of corrections).

## The smoking gun (why pillar3g2 failed)
Our completed-Q target peaks at `q_arg` = the highest-value **well-visited** candidate. But
the teacher *plays* `argmax visits`. On the correction states (where we intervened):
- teacher's actually-played move == our target `q_arg`: **2.4%**
- teacher's actually-played move == the prior's pick `p_arg`: **84.3%**
- `q_arg` got a median **42/400** visits; the move the teacher played got **100/400**.

So **we distilled toward moves the search looked at and declined to play.** The value head
rated `q_arg` higher, but the *search* (integrating prior + value + visits) overrides that
and plays `p_arg`. The 4800 calibration's "deep visits prefer q only 8.3%" was telling us
exactly this — we misread a *rejected* move as a correction. The teacher was never the
problem; our target was anti-teacher.

## The remaining paradox (what we want your help on)
A teacher that plays the floor ~2× better cannot, in principle, be the source of a worse
student — so the fix should be to distill the teacher's ACTUAL policy (its `chosen_move` /
its visit distribution), not Q. **But pillar3g already distilled the visit distribution and
ALSO regressed.** So:

1. **Is clean hard behavioral-cloning on `chosen_move`** (no Q, no soft visits, no
   sharpening, no aux, warm-start pillar3f) **the right next experiment?** Our hypothesis:
   pillar3g failed because of T=0.5 sharpening + uniform soft-CE over noisy visit tails, not
   because the target was wrong. Hard BC on the argmax removes that noise. Agree, or is soft
   (unsharpened) visit-distillation better than hard BC here?

2. **Is the teacher's edge partly UN-distillable into a raw prior?** MCTS@400's 2× floor
   comes from 400 sims of lookahead *per move at play time*. A raw greedy policy (single
   forward, no search) may be unable to reproduce that no matter how we distill — the
   improvement may only show up when the distilled policy is itself run *with* search next
   iteration. If so, evaluating the raw distilled policy vs raw pillar3f is the wrong test —
   we should compare *policy+MCTS vs policy+MCTS*. How would you measure "did distillation
   help" correctly? (Note: a regressed *raw* prior is still bad — distillation shouldn't make
   the prior worse.)

3. **The gap is real (NOT exhausted) — it's concentrated on the floor.** Range-level
   (distribution, not per-seed — RNG forks games so per-seed is invalid), the MCTS@400 teacher
   beats the raw policy on the floor: P10 7127 vs 3355 (~2×), P5 3508 vs 2479. Yet the teacher
   plays the SAME move as the raw policy ~84% of the time (visit-argmax==prior-argmax). So the
   teacher's edge is real but CONCENTRATED in ~16% of (floor-decisive) moves — which we never
   distilled (we distilled q_arg, the rejected tail). The question: what's the best way to
   distill a teacher whose advantage is a sparse set of crisis-move corrections, without the
   broad spillover (18.7%) that γ-weighting caused? Hard chosen_move BC on the 16%? Soft
   unsharpened visits everywhere with a trust region? (Separately, a 4800-sim widened teacher
   + task-arithmetic DID give +36% — escalating sims is a known-good lever if 400 proves too thin.)

4. **Methodology autopsy (done — rules some things out, surfaces another):** on 20k states,
   ep1's argmax vs pillar3f's: differs on **18.7%** (we only intended to change ~2.5%
   corrections), and ep1's argmax lands inside pillar3f's top-10 **99.9%** of the time. So:
   off-candidate drift is **NOT** happening (candidate-restricted CE didn't leak mass to
   garbage moves), but the intervention was **far broader than intended** — γ=10 ×
   near-one-hot (τ=0.02) targets on a shared backbone spilled the (anti-teacher) corrections
   onto 7× more states than the 2.5% we targeted. So the regression = wrong target (q_arg,
   rejected) × broad spillover. Does this argue for (a) a much gentler intervention (small γ,
   soft target, low lr) on top of (b) the correct target (chosen_move), or (c) is broad
   argmax change unavoidable when γ-weighting a shared net and we should instead train a
   separate correction head / use a frozen-backbone adapter?

## What we're NOT asking
We accept the completed-Q/Gumbel target was wrong (anti-teacher). The question is the right
way to distill the teacher's *actual* decisions, why visit-distillation also failed, and
whether a 400-sim teacher even has a distillable gap at pillar3f's strength.
