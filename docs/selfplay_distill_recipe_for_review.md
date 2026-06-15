# Peer review: why does 400-sim self-play visit-distillation FAIL to improve an already-strong policy, and what's the right recipe?

We have a working AlphaZero-style pipeline for a single-player game. A **task-arithmetic crisis-correction** trick gave a big one-off jump (+73%). Now we want the **generational self-play loop** to work — distill a stronger search player back into the policy — and it **regresses**. We've misdiagnosed it twice; we want an outside read on the recipe. Please be blunt.

## The game (matters for the recipe)
Color Lines 98, 9×9 board, 7 colors. Move a ball along a free path; lines of 5+ clear and score; else 3 balls spawn at uniform-random empty cells. **Stationary survival game**: score ≈ 2.05 × turns, no endgame, no win/loss — you play until accumulated mistakes drive an unrecoverable density spiral. A strong policy holds a low-density equilibrium for tens of thousands of turns; game length is bimodal (rare early RNG death, or survive to the turn-cap). "Strength" = lower death-hazard per turn.

## Policy & lineage
- Net: 11.9M params (10 ResBlocks × 256ch), 18-channel 9×9 obs → policy over 6561 moves. **Deployed greedy (argmax, no search).**
- `pillar3b`: base, distilled from V13 self-play. 1k eval: mean 18,865, median 13,421, P10 2,409.
- `pillar3f` = **pillar3b + 0.5·(crisis task-vector)** via task arithmetic — **current best**. 5k eval: **mean 31,617, median 22,181, P10 3,961, P25 9,729, <1000 1.6%, max 301k**.

## What worked (firefighting — NOT the question here)
Mine the policy's death games, rewind to the crisis band, label states with **widened MCTS@4800** (decisive forks only), fine-tune a copy ONLY on those corrections (frozen-BN), then **merge weights**: θ = θ_base + α·(θ_crisis − θ_base), sweep α. +73% mean over base. **Important: this is firefighting** — 4800 sims is impractical as a general player (we can't 4800-sim every move of a 20k-turn game), and it only patches specific crisis states. It is NOT the generational improvement loop.

## What we WANT (the generational trunk — the actual question)
Standard AlphaZero policy improvement: self-play with a **practical** search budget (MCTS@400) generates targets, distill them into the policy, the greedy policy gets stronger and **plays stably on its own**. 400-sim self-play is a genuinely stronger player and is "good enough" as the next-stage teacher — we just need the distillation to transfer its strength.

## The teacher IS much stronger (so this is not a data problem)
V14 self-play = pillar3f + MCTS@400 + Dirichlet/temperature, 9.2M states. The MCTS@400 player vs the greedy pillar3f policy, on the floor:

| quantile | greedy pillar3f | MCTS@400 self-play |
|---|---|---|
| P5 | ~2,110 | 4,956 |
| P10 | 3,961 | 10,348 |
| P25 | 9,729 | 20,291 (turn-capped) |

The 10th-percentile self-play game survives ~2.6× longer. So the search makes materially better *decisions*. (Move-argmax agreement between MCTS@400 and greedy is ~85% on hard states, higher on easy ones — the strength is concentrated in a decisive minority of states.)

## What we tried (pillar3g) and where it fell short
Distill V14 into the policy: warm-start, **target-temperature 0.5 sharpening** (this sharpening gave +57% in an earlier weak-policy generation), lr 3e-4, batch 32768, 20 epochs, 8× dihedral + color aug. Tried warm-start from **both** pillar3b and pillar3f.

**Identical failure from both warm-starts:**
- Train loss flat (~2.294 → 2.289 over epochs — barely moves).
- Val loss (CE to held-out V14 targets) **rises** every epoch after ep1: 2.337 → 2.350 → 2.352…
- Gameplay (1k): **mean 24,589 / median 18,612 / P10 3,115 / P25 7,749 / <1000 2.1%** — *below* pillar3f's ~22k 1k median, floor regressed toward base. The first (lr-warmup) epoch is the "best" by val and is ≈ the warm-start; every subsequent epoch is worse.

Two different strong warm-starts → bit-identical curves ⇒ the warm-start is irrelevant; it's the data × recipe.

## Our current diagnosis (please confirm/correct)
**Soft visit-distribution targets don't transfer the search's decisive moves to an already-strong policy, because the policy IS the MCTS prior.** At 400 sims with a strong, confident prior, PUCT exploration is prior-dominated, so the visit distribution barely departs from the policy's own distribution — except on the decisive minority, where it shifts only modestly (e.g., 45/40 between the good move and the policy's greedy pick). Distilling that (even sharpened to ~56/44) is a weak CE pull against a confidently-wrong argmax → the argmax doesn't flip. Meanwhile the policy already fits the soft targets (flat train loss), so **sharpening just hardens its existing argmaxes — including the wrong ones — making it peakier (val rises) and more overconfident → gameplay regresses.** This is the *same* failure mode we hit when we tried to distill the 4800 crisis corrections as a soft auxiliary loss (soft visit targets can't flip decisive argmaxes); task arithmetic fixed *that* by sidestepping the soft-target distillation entirely.

The val curve is also partly a sharpening artifact: training fits **sharpened** targets while val measures **unsharpened** CE, so getting peakier raises val by construction. The earlier generation's val *dropped* only because it warm-started from a much weaker policy with lots of genuine learning to offset the sharpening rise. Now there's little to learn, so the rise shows.

## Questions
1. **Is the core issue "policy improvement collapses when the prior is strong relative to the sim count"?** The policy is the MCTS prior; at 400 sims it barely out-paces itself, so the visit distribution is a near-zero improvement signal except on rare decisive states. Is that the right mental model?
2. **What target should we distill instead of soft visit counts**, to transfer the search's *decision* rather than its (prior-anchored) exploration? Rank/critique: (a) hard argmax (one-hot MCTS move); (b) **Gumbel-AlphaZero-style "completed-Q" improved-policy target** (we now record root Q per move); (c) advantage-softmax over Q; (d) lower temperature still; (e) weight the loss toward high-disagreement (MCTS-argmax ≠ policy-argmax) states so the decisive minority isn't drowned by 85% agreement; (f) a KL trust-region to the prior + improvement term.
3. **Or is 400 sims simply too few** to generate an improvement target for a policy this strong, and the real fix is more sims / Gumbel's guaranteed-improvement-at-low-sims construction? (We can't go very high: at 600 sims games run effectively forever — ~1.4 s/turn, 20k+ turns.)
4. **Does the stationary-survival structure** (no terminal reward, score=survival, ~2/turn everywhere) change what a good distillation target is? (Our value targets are all saturated — every scalar value signal is a near-constant proxy for "turns survived," so MCTS leaf values come from a hand-tuned linear survival proxy, not a learned value head.)
5. **Sharpening**: is T=0.5 actively harmful now (hardening wrong argmaxes), and should the generational step drop it / use plain KL, reserving sharpening for the weak-policy regime?
6. Constraint to respect: the answer must be a **scalable trunk recipe** (≤~400-sim self-play, distillable, yielding a stable greedy player), NOT 4800-sim per-state firefighting.

## Failed/!= options (don't re-suggest)
- "Use the 4800 crisis corrections instead" — that's the firefighting tool, already works via task arithmetic; not the generational loop. We need the 400-sim trunk to work.
- "Warm-start matters" — refuted (pillar3b and pillar3f give identical curves).
- "It's a data problem / ctrl0" — refuted; the teacher is genuinely ~2× stronger on the floor.
