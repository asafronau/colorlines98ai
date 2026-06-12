# Crisis-corrections distillation — a seed-list confound just undermined our "more data regressed" premise

Follow-up to the earlier "why does more crisis data hurt" brief. We ran the λ-scaling experiment you
suggested, and in checking the result we found the comparison was contaminated by an **eval seed-list
mismatch**. We want your read on (a) whether our diagnosis of the confound is right, (b) what the λ
result actually tells us now, and (c) the next experiment.

## Recap of the setup (unchanged)
- Fixed policy net (11.9M params, distilled from 400-sim self-play, ~17.5k mean).
- We mine the policy's **death games**, rewind to a crisis band, run **widened MCTS@4800** (top_k=300 +
  Dirichlet, 3 determinizations, linear-feature leaf value), and distill the **soft visit distribution**
  as an auxiliary target: `loss = main_CE(V13 self-play) + λ·aux_softCE(corrections)`.
- "mC" keeper recipe: warm-start, lr 5e-5, **decisive corpus** (margin ≥ 0.05), **λ=0.01**, aux_T=0.5,
  aux-warmup 0.5 ep, 6 epochs, peak ep2. Metric = **MEDIAN** of 5,000 held-out games (mean secondary;
  floor = no-regress guardrail).

## What we believed going in
At a FIXED λ, the per-correction optimization weight ≈ `(steps/N)·λ`, so a bigger corpus (N↑) is
distilled *less per correction*. We saw the median drop when we grew the corpus 1,837→2,593 games
(13.8k→19.6k decisive corrections): **16,504 → 15,385**. Hypothesis: under-distillation → fix by scaling
λ with N (0.01 → 0.012 / 0.014, the ×1.42 corpus growth).

## The λ-scaling result (mF λ0.012, mG λ0.014, on the 19.6k corpus, ep2)
| run | λ | median | mean | <1000 |
|---|---|---|---|---|
| mC (19.6k) | 0.010 | 15,385 | 21,957 | — |
| mF | 0.012 | 15,433 | 21,340 | 2.1% |
| mG | 0.014 | 15,397 | 22,254 | 2.4% |

**Median is dead flat** (within ~50, well under the ~290 median SE). Mean ticked up slightly with λ.
So in the moderate regime, **λ is NOT the median lever** — scaling per-correction weight did nothing.
(These three are all on the same seed list, so this sub-conclusion is clean.)

## The confound we then found
The "deployable best" **16,504 was evaluated on seeds 775000–779999**. Everything since — the 19.6k mC
(15,385), mF, mG — was evaluated on **777000–781999** (a stale default in our notebook generator). Our
eval is **batched fp16**, and we have prior evidence that the *same seed in a different list composition
produces a different game* (batch-ordering changes the fp16 rounding path). So:
- The mF/mG-vs-mC(19.6k) λ comparison is valid (same list). ✔
- But **"more data regressed 16,504→15,385" compares 13.8k-on-775k against 19.6k-on-777k — cross-list.**
  The entire premise that drove this investigation may be partly or wholly a seed-list artifact. We do
  not currently know the 756 extra games hurt at all.

## The fix we're applying
1. Switch all evals to a **single-process batched policy player with a constant batch (256)** on the
   **canonical 775000–779999** list. fp16 is sufficient: scores are reproducible for a fixed
   (seed-list, batch), so models compared on the same (list, batch) are directly comparable; the per-game
   fp16 noise averages out over 5k games. The bug was the LIST mismatch, not the precision — so we just
   pin both (list and batch), not the precision.
2. Re-establish the bar: run the **13.8k mC checkpoint** (the one that scored 16,504) through this eval on
   775000–779999, and re-run the 19.6k / mF / mG checkpoints the same way. Only then do we know whether
   "more data" regressed.

## A separate correction that reopened "use them all"
Earlier we'd concluded high λ is intrinsically harmful, from a grad-audit on an OLD small corpus ("v2.2"):
at λ=0.03 the effective aux share `λ|g_aux|/|g_main|` = 2.09 (aux-dominated), and that run regressed. The
human pushed back: **v2.2 was data-starved** — 26k corrections from relatively few games = low-diversity,
highly-correlated death trajectories. At high λ that overfits a few patterns and drifts from general play.
With 2,593 *diverse* games, the same aux force is spread over genuinely different positions and should
distill rather than memorize. So "share≈2 regresses" was a property of v2.2's **data diversity**, not a
law about λ. That dissolves our "aux-domination wall," and means the **full corpus at properly-scaled λ**
is worth testing (we'd avoided it as "the v2.2 regime").

## Next experiment (queued, fp32 / canonical seeds)
- **mH = full corpus (~44.8k, min_margin 0) at λ=0.03**, mI at λ=0.02 — same mC recipe otherwise. λ0.03
  matches the full corpus's per-correction weight to mC's known-good 13.8k@λ0.01 (0.01·44.8/13.8≈0.03).
  This is the un-confounded "use them all" test, now that we won't reject it on the stale v2.2 share #.

## Questions
1. Do you agree the cross-list eval invalidates the "more data regressed" premise (not just the
   mF/mG-vs-bar comparison)? Is fp16 eval on one fixed (list, batch) the right and sufficient fix, or is
   there a subtler comparability trap (e.g. the policy player itself being argmax-greedy)?
2. Given λ is empirically flat for the median in [0.01, 0.014] on a fixed corpus, what does that imply
   about the mechanism? If under-distillation were the story, a 40% per-correction-weight bump should
   have moved *something* in the median — it didn't. Does that point at interference/capacity, or just
   that the median is insensitive to crisis corrections (which mostly touch the lower/floor games)?
3. If the re-baselined eval shows NO regression (13.8k and 19.6k tie on one list), is "median" simply the
   wrong metric to detect what these corrections buy — should we be tracking a lower quantile (P10/P25)
   or the conditional mean of the bottom decile instead, where crisis fixes would actually show up?
4. The human is mining toward 3.5k/5k games. Assuming mH (full @ scaled λ) doesn't beat the bar: rank
   KL-anchor-to-current-policy vs bigger net vs balanced correction-replay as the structural next step,
   given the grad facts (|g_main|≈0.23 converged base, |g_aux|≈15.8 OOD, cos≈0 orthogonal).
