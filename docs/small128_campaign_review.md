# Review brief #5: the small-model campaign — six failed rounds, one coherent question

You have reviewed this project four times (single-move distillation closure, DAgger design, the dagger1 postmortem, the R2 arms). This is the full-campaign stocktake. The owner's directive under review: **"the 256ch model was progressing fine — put the smaller model on the same track."** Your job: audit that hypothesis against the complete record, then design iteration-4 or tell us why not to run it.

## The two tracks

**The 256ch track (WORKED, repeatedly):** pillar3f → pillar3k-v1 (+18% mean) → v2 (+5.4%/+8.2%). Recipe (HISTORY 173-174): corpus = 2.26M states, 70% crisis / 30% selfplay, mined by MCTS @1600-2400 sims with a neural value head at q=2.0; targets = raw visit distributions; **decisiveness weighting dw=3** (per-state CE weight = visit-top-share³, mean-normalized: decisive escapes ~6.6×, flat quiet states ~0.15×); **T=0.7**; warm-start; lr 3e-4 / batch 32768; NO hard-CE blend; NO rehearsal mix (the 30% selfplay was the anchor); pick epoch by GAMEPLAY FLOOR. Its documented lessons: deeper sims → flatter distributions; **no temperature sharpens a flat corpus — decisiveness is the lever, T is not** (174 lesson 2); selfplay is 98% flat / crisis 15-22% decisive (lesson 3); val is doubly unreliable (lesson 4).

**The small-line track (won ONCE, then 0-for-6):** gate-3 recipe = 55k own-crisis micro-corpus + 3:1 pillar3k-labeled rehearsal, blend 0.5 hard-CE, **T=1.0, dw=0**, lr 1e-4 / batch 4096, warm-start, seeded. Produced vh1 (+4.6% median, the only win). Every round since — different corpora, labels, losses, taxes — kept dw=0/T=1.0/blend and failed.

## Current bars

Student small128_vh1 (10b×128ch, 3.0M): 5k = mean 13,080 / P50 9,323 / P5 1,222 / <1000 3.5%. Frozen teacher pillar3k (10b×256ch): 43,390 / 31,016. Constraints: 128ch only (no width changes, hard directive), master frozen, 5k seeds decide (500-screens are 0-for-5 on close calls), training now fully local (MPS, ~25 min/round).

## The complete (signal → outcome) ledger

| round | corpus / labels | recipe | judge signal | 5k P50 vs vh1 |
|---|---|---|---|---|
| gate-3 → **vh1** | 55k own-crisis, MCTS@600-1200 visits (old head) | 3:1, dw0, T1, blend.5 | **+4.5pp, 27-29% gen** | **+4.6% WON** (over ep87) |
| iter-2 (3 methods) | 2.5M own-crisis, own-MCTS ≤4800 | grids over γ/lr/blend/steps | ≤+0.3pp | −11..−15% |
| dagger R1 | 67k on-policy, pillar3k top-5 | 3:1, dw0, blend.5 | +1.69pp (row-valid +2.4-2.8 at gap≥1) | −4% |
| dagger R2 (3 geometries) | 25k gap≥1.0 rows | task-vector / legalmax+KL | same labels | NO-GO gates / −6..−10% |
| iter-3 | 86k vh5x-head crisis @1200/800 | 3:1, dw0, T1, blend.5 | **+3.37pp [+1.17,+5.73], 22%/6%** | −8.0% |
| tax arms (same corpus) | 〃 | **6:1**, dw0 | 〃 | **−2.4%** (best post-vh1); 10:1 −4.6%; lr/2 −6.8% |
| iter-3b | 86k deep @2400/1600 | 6:1, dw0, T1, blend.5 | **+3.88pp [+2.37,+5.45], 0% phantoms** | −5.7% |

Key micro-measurements along the way: warm-start rounds churn 4-8% of quiet-state argmaxes (measured fp16 and fp32, both holdout and rehearsal distributions); churn is directionless (not toward the teacher); rehearsal ratio monotonically buys churn down (the "tax"); the vh5x head (13× data) beats the old head at every survival horizon and its judge arm is stronger and cleaner.

## The two anomalies any theory must explain

1. **Same recipe, opposite outcomes**: gate-3's dw0/T1/blend recipe won (+4.6%) on the old-head @600-1200 corpus, then the *same* recipe lost (−8%) on a *better-judged* corpus (+3.37 vs +4.5 is weaker, but row-validated). The threshold model (signal must exceed a ~2-6% training tax) explains this pair.
2. **Deep corpus < shallow corpus at the same tax point**: +3.88pp/0-phantom @2400 trained to −5.7% while +3.37pp @1200 trained to −2.4%. The threshold model does NOT explain this. HISTORY 174 lesson 2 does: deeper search → flatter visit targets → de-peaks a greedy student, and dw=0 leaves that undefended. (We have not yet measured the two corpora's top-share distributions — flagged as the first verification below.)

## The owner's hypothesis, operationalized

Iteration-4 = the 256ch track at 128ch: mine LARGE (0.5-1M states, now ~1-2 days local at @1600-2400 with the vh5x head), 70/30 crisis/selfplay, train with **dw=3, T=0.7, NO blend, NO rehearsal-mix** (selfplay fraction as the anchor), lr/batch scaled for MPS, pick by floor at 5k. The claim: the small line's failures are recipe-family failures — flat targets trained un-weighted at toy corpus scale — and the fix is the track that already works, not another knob on the gate-3 family.

Facts FOR: dw was invented for exactly the flatness we just hit; the 256ch line paid no visible tax at scale (v1→v2 net +5-8% — signal ≫ tax with 2.26M states and 6.6×/0.15× gradient shaping); every small-line failure kept dw=0; the corpus type we already mine is the input dw expects. Facts AGAINST / open: the 3b-recipe was tried once on the small line (iter-1, regressed — but on the later-falsified old-head corpus, heavily confounded); 128ch warm-starts from a *distillate* whose policy fabric is pillar3k-shaped, unlike the lineage-native 256ch warm-starts; corpus scale 86k→1M is a 10× mining spend on an unproven transfer; and vh1's single win came from the OTHER recipe, which a dw3 pivot abandons.

## Questions

1. **Verify-first list**: we propose (a) measuring top-share distributions of the @1200/800 vs @2400/1600 corpora vs the corpora that won on each line (cheap, decisive for the flatness story); (b) a dw∈{1.5, 3} × existing-deep-corpus arm (25 min each) BEFORE any large mine — a positive slope there de-risks the 10× mining spend. Right order? What else must be measured first?
2. **Recipe transfer**: which elements of the 256ch recipe are load-bearing for a 4×-smaller warm-started distillate — dw exponent, T=0.7, no-blend, selfplay-as-anchor vs rehearsal-mix, lr/batch? Where would you deviate and why?
3. **Corpus scale**: is 0.5-1M states necessary for dw to work (gradient shaping needs a population of flat states to down-weight), or can dw+6:1-rehearsal substitute at ~100k scale?
4. **The tax within this frame**: is the measured 4-8% quiet churn just what dw=0 training of flat targets looks like (i.e., the tax IS the flatness disease), or a separate warm-start pathology that dw won't cure?
5. **Anything we're not seeing**: six rounds of judge-positive signal failing to train in — is there a fundamentally different integration you'd try before/instead of the dw pivot (given: no width changes, master frozen, everything else on the table)?
