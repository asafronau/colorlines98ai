# Distilling a GROWING set of crisis corrections into a fixed policy — why does more data hurt?

We distill MCTS "crisis corrections" into a fixed policy net. We just hit a counterintuitive result:
**more (genuinely diverse) corrections made the policy worse.** We think this is a distillation-scaling
bug, not a data problem, and want your read on the diagnosis + the fix.

## Setup
- Policy net: 11.9M params (10 ResBlocks × 256ch), warm-started from pillar3b (distilled from 400-sim
  self-play, ~17.5k mean).
- Each main batch (V13 self-play tensor, 32768) is paired with a **256-sample aux batch** of crisis
  corrections: `loss = main_CE + λ·aux_softCE`. Recipe (tuned, "mC"): λ=0.01, aux_T=0.5 (target
  sharpening), aux-warmup 0.5 ep, aux-weighted (by margin), aux-holdout 0.15, **6 epochs**, peak ep2.
- Corrections: rewind policy *death* games to a crisis band, run widened MCTS@4800, soft visit dist
  is the target. "Decisive" corpus = keep only margin (mcts_top_share − policy_share) ≥ 0.05.

## The counterintuitive data point (5k held-out, same seeds, median is the metric)
| corpus | games | decisive corrections | median | mean |
|---|---|---|---|---|
| mC | 1,837 | 13.8k | **16,504** | **23,434** |
| mC (same recipe) | 2,593 | 19.6k | 15,385 (ep2) | 21,957 |

ep3 of the 2,593 run is trending *worse* than ep2 — so it's not just "peak shifted later." **+756 games
/ +5.8k decisive corrections REGRESSED the median ~7%.** (This echoes an earlier regression we already
diagnosed-and-fixed for the *wrong* reason: λ was too high then; it's λ=0.01 now, the tuned value.)

## Our thesis (the human's, and we think it's right)
These are **NOT "more of the same."** Each correction is a *different* failure position (different
death game/seed, a state where MCTS@4800 clearly out-plays the policy). We've covered **<5%** of the
positions the model handles badly. So by *coverage* logic, more diverse "play X here instead"
corrections should monotonically help. They don't → **the bug is in how we distill, not the data.**

## Our diagnosis (please confirm/correct)
**The aux budget is fixed, so a bigger corpus is distilled less per correction.** 256 aux/step × 2,126
steps × 6 epochs ≈ **3.3M aux samples total**, spread over the corpus:
- 13.8k corrections → ~240 exposures/correction
- 19.6k corrections → ~168 exposures/correction

So the bigger corpus gets *under-distilled per correction*, not more coverage. And "more epochs" to add
exposure backfires: ep3 regressed because more epochs = (a) the base V13 re-distill drifts the policy
further off, and (b) corrections seen early overfit. So epoch count is the wrong exposure knob.

Secondary suspect: **capacity / interference** — a fixed 11.9M net distilling a growing *diverse* set,
where corrections start to conflict (fixing position A degrades position B) and the net settles on a
worse compromise.

## Candidate fixes — rank / critique / extend
1. **Scale aux-batch-size with the corpus** (256 → ~384 for this 1.4× corpus): keep ~constant
   per-correction exposure at *fixed* epochs, decoupling exposure from epoch count (avoids the
   overfit/drift confound). Per-*epoch* per-anchor exposure scales with batch size even though per-step
   gradient magnitude doesn't. Right lever, or are we fooling ourselves (it also raises the aux's share
   of the gradient per step → re-tune λ)?
2. **More epochs + lower lr** (5e-5→3e-5, 8–10 ep): more total exposure via gentler steps that overfit
   slower. But the base re-distill keeps drifting — is that fatal?
3. **Anti-interference:** replace the V13 base re-distill with a **KL-anchor to the current policy**, or
   EWC, so distilling more diverse corrections doesn't erode general play. (We deferred KL-anchor
   earlier; is *now* when it becomes necessary, as the correction set outgrows the net's slack?)
4. **Capacity:** the 11.9M net may be the bottleneck for a large diverse correction set — bigger net?
5. **Curriculum / dedup:** order or cluster corrections (easy→hard, or dedup near-duplicate positions)
   so the budget isn't wasted.

## Questions
- (a) Is "fixed aux budget spread thinner over more anchors" the right primary diagnosis, or is it
  capacity/interference (the net can't hold all the corrections at once)?
- (b) For distilling a **continuously growing, diverse** correction set into a **fixed** policy so that
  more corrections *monotonically* help: what's the principled recipe? (batch-scaling vs epoch-scaling
  vs anti-forgetting vs capacity — and how do they interact with the λ-balance the grad-audit showed,
  where the converged base has tiny gradient |g_main|≈0.23 and the OOD aux dominates?)
- (c) The human is mining toward 3.5k → 5k games (more useful corrections incoming). We need the recipe
  to *scale* with the corpus, not regress. What's the smallest change that makes "+games ⇒ +median"
  hold?
- (d) Are we wrong that these corrections are genuinely additive — could a fixed-policy distillation
  have a real saturation point regardless of recipe, where new corrections necessarily trade off old
  ones?
