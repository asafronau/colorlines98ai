# Crisis-corrections aux distillation — which hyperparameters to tweak? (review request)

We distill 4800-sim MCTS "crisis corrections" into a policy net. We've found the aux-strength
sweet spot (λ); now we want your read on the OTHER knobs, and on a specific thesis vs a catch.

## Setup
- **Policy** (pillar3b, ~17.5k mean) was distilled from **400-sim** MCTS self-play.
- **Aux corrections**: rewind policy *death* games to a crisis band (D-15..D-85 before death); at
  each state run **widened 4800-sim MCTS** (top_k=300 = all legal + Dirichlet, 3 determinizations,
  linear-feature leaf value); the **soft visit distribution** (top-20, renormalized) is the target,
  margin-weighted (margin = mcts_top_share − policy_share).
- **Training**: warm-start pillar3b; per main batch (V13 = the 400-sim self-play tensor, 32768) also
  do a 256-sample aux batch: `loss = main_CE + λ·aux_softCE`. Current recipe: lr 5e-5, warmup 1 ep,
  main target-temp 0.5, **aux-lambda 0.01, aux-target-temperature 0.5, aux-warmup-epochs 2.0,
  aux-batch-size 256, aux-weighted**, aux-holdout 0.15, ~6 epochs.
- **Target metric pivoted to MEDIAN** (+ mean); the bottom/floor is a no-regress guardrail, not the
  objective (it's RNG-chaotic — a better move can draw a worse spawn).

## What we've established
- **Grad-audit:** at λ=0.03 the aux is NOT gentle — effective share `λ|g_aux|/|g_main| = 2.09`
  (aux drives training 2:1 over the base; |g_main|≈0.23 because pillar3b is converged & re-distilling
  its own data; |g_aux|≈15.8 because crisis states are OOD). cos(main,aux)≈0 (orthogonal). At λ=0.01
  the share is ~0.7. Sweet spot ≈ λ=0.01: λ=0.03 over-indexes (regressed), λ=0.003 too gentle.
- **Best so far:** decisive-corpus (margin-filtered, 13.8k) + λ=0.01 @ep2: median 15,447 / mean
  22,203 on 5k (+~2%/+5% vs the v2.1 baseline; floor tied). Confirmed at 5k (not a 2k mirage).
- **"Use them all" (full 31.8k corpus) @ λ=0.01:** ep2 *below* decisive (median 14,874 / mean 20,895)
  — BUT confounded: 2-epoch λ-warmup means ep2 is at half-λ, and the bigger corpus gets fewer
  replays/anchor (fixed 256/step), so its peak is likely ep4–5; only ep2 evaluated so far.

## The thesis vs the catch (please adjudicate)
- **Thesis (we lean toward agreeing): "use them all."** The policy learned from 400-sim self-play;
  every 4800-sim label is strictly better supervision — even where the argmax agrees, the
  probabilities are better-calibrated. Filtering by margin discards good 12×-deeper-search data.
- **The catch:** we distill the soft *visit distribution*, and the teacher is *widened* (top_k=300 +
  Dirichlet) precisely so it can find buried fixes — so at low-margin states the visit distribution
  is **near-uniform** (exploration spread, not move-quality). "Use them all" as soft targets imports
  that flatness → teaches a *less decisive* policy. The margin filter wasn't keeping "better labels,"
  it incidentally kept the *peaked* distributions and dropped flat ones.
- **Our hypothesis:** the fix for "use them all" is **sharper aux targets** (aux-target-temperature
  0.5 → ~0.3) — keep every label but extract move-quality, suppress exploration flatness — rather
  than the margin filter. Right, or are we wrong about the soft-target-flatness mechanism?

## Knobs we're considering — please prioritize / correct / extend
1. **aux-target-temperature** (sharpen, 0.5→0.3) — the move above. Or is sharpening a hand-tuned
   band-aid; should the aux target be the **top-k visits**, the **argmax (hard)**, or **Q-values**
   instead of the full soft visit distribution?
2. **aux-batch-size** (256 → 512/1024) — with the full 31.8k corpus, a bigger aux batch covers more
   of the corpus per epoch (less per-anchor replay/imprinting). Worth it, or does it just rescale the
   effective share (which we'd re-tune via λ)?
3. **aux-weighted** on/off — if "use them all," should all 4800-sim labels be equal, or keep
   margin-weighting (decisive ones dominate)? Does weighting partly *recreate* the filter inside the
   full corpus (the better answer than hard-filtering)?
4. **aux-warmup-epochs** (2.0 → 0.5/1.0) — 2 epochs of ramp wastes a third of a 6-epoch run.
5. **Base distillation:** re-distilling the 400-sim V13 tensor barely contributes gradient
   (|g_main|≈0.23) and the main loss *creeps up* as the aux drives — replace base-CE with a
   **KL-to-pillar3b anchor** (regularize toward the start model instead of re-teaching 400-sim data)?
6. **Epochs / lr / lr-warmup** — anything in the 5e-5 / 1-warmup / 6-epoch schedule.
7. Anything we're missing for distilling a *stronger-but-widened-search* teacher into a policy while
   lifting the median.
