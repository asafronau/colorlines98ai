# Review brief #4: dagger1 postmortem — judge-validated corrections regressed when trained

You are reviewing a training failure. The per-move signal was adversarially validated before training; training on it still regressed. Your job: identify the mechanism from the measurements below and design the minimal round-2 fix. Be adversarial to our stories, not to the data.

## Recap (context from brief #3, which you reviewed)

Color Lines 98; score ≈ 2.03 × turns survived; 5k fixed-seed evals decide (500-seed screens flip vs 5k). Student **small128_vh1** (10b×128ch, 3.0M params; 5k: mean 13,080 / P50 9,323 / P5 1,222 / <1000 3.5%), a distillate of the frozen teacher **pillar3k** (10b×256ch; 5k mean 43,390 greedy) plus one successful self-improvement round (gate-3: +4.6% median over the raw distillate ep87 = 12,895 mean). Master is frozen by project directive; all evolution on the small line.

## What was validated BEFORE training (the redesigned Phase 0b you specified)

2,135 true on-policy disagreement anchors (full-legal argmax both policies), 2 arms × 64 paired seeds, 7 continuation conditions, seed-cluster bootstrap:

- Student continuation (primary): **+1.69pp [+1.32, +2.09]** died-within-300 uplift for pillar3k's move; genuine:phantom 18%:9%.
- Concentration: gap≥1.0 **+3.73pp [+3.03, +4.42]**; gap<0.5 +0.34pp ≈ 0; recovery +2.25 vs prevention +1.06.
- Burst ladder L=1..16 and teacher continuation: FLAT (+1.30..+1.81) — advantage cashes at the single move, continuation-robust.

## The corpus and recipe (per your GO recommendation + the concentration finding)

2,000 recorded vh1 greedy games (on-policy sanity: mean 12,860 ≈ vh1's bar) → 703,788 candidates → teacher selection → **66,917 states**: recovery 23,139 / prevention 31,992 / broad 11,786; **82% disagreements** (band gap≥0.5 all + 25% of gap<0.5; broad gap≥1.0 only), agreements downsampled to 18%. Labels: pillar3k top-5 legal softmax + argmax (identical convention as the rehearsal corpus, which is itself pillar3k-relabeled; label top-share P50 = 0.41). Mix: 3:1 rehearsal → 267,668 states, exactly 25% new signal. Recipe = the gate-3 winner: warm-start vh1, blend 0.5 hard-CE, T=1.0, dw 0, lr 1e-4, warmup 1 epoch (~523 steps), batch 4096, aug 8, seed 42, 3 epochs, checkpoints every 100 steps.

## Result: regression at EVERY point of the step grid

500-seed screens of all 15 checkpoints (steps 100→1,570): no catastrophe, but uniformly soft. 5k evals of the floor-first shortlist:

| | mean | P50 | P5 | <1000 |
|---|---|---|---|---|
| vh1 bar | 13,080 | 9,323 | 1,222 | 3.5% |
| e2_s200 | 12,312 (−5.9%) | 8,554 (−8.2%) | 1,021 | 4.9% |
| epoch_2 | 12,678 (−3.1%) | 8,816 (−5.4%) | 1,124 | 4.2% |
| e3_s400 | 12,610 (−3.6%) | 8,917 (−4.4%) | 1,170 | 3.8% |

Also: epoch_2's val loss 1.9003 IMPROVED over vh1's 2.0465 while gameplay regressed (val = mimicry-CE; known trap, confirming again).

## Diagnostics (all fp32, full-legal argmax)

**1. Absorption vs drift** (47,308 confident correction states = training rows with hard-CE argmax labels; holdout = 18,321 on-policy quiet states NOT in corpus; rehearsal = 20k sample):

| model | absorb% (plays 3k move) | keep_vh1% | third% | drift_hold% | drift_reh% |
|---|---|---|---|---|---|
| vh1 | 10.8 | 80.2 | 8.9 | 0 | 0 |
| e1_s100 (warmup LR ≈ 1.9e-5!) | 17.1 | 64.5 | 18.3 | 7.9 | 7.4 |
| epoch_2 | 18.0 | 66.3 | 15.7 | 7.1 | 6.7 |
| e3_s400 | 18.1 | 66.3 | 15.6 | 6.8 | 6.7 |

Notes: (a) vh1's 10.8% "absorb" and 80.2% keep on its OWN recorded moves reveal near-tie argmax instability (recorded moves came from fp16 batched play; fp32 recompute flips ~20% on these contested states). (b) Absorption after ~24 augmented touches of every correction row: +7pp. (c) The drift and third-move corruption are nearly fully formed at step 100, INSIDE warmup at LR ~2e-5, and partially heal with more steps.

**2. Drift direction — mimicry-pull hypothesis REFUTED.** Match-to-pillar3k argmax:

| model | rehearsal→3k% | holdout→3k% |
|---|---|---|
| ep87 (raw distillate) | 73.04 | 73.24 |
| vh1 | 72.97 | 73.18 |
| e3_s400 | 72.86 | 72.80 |
| e1_s100 | 72.39 | 72.41 |

The trained models moved slightly AWAY from the teacher globally. The 7% churn is not "re-distillation erasing gate-3's delta"; it's incoherent.

## The puzzle, sharply

The judge proved the swaps help (+1.69pp, 7 conditions, paired seeds). Training on those exact rows: (i) failed to install them (18% absorption), (ii) churned ~7% of argmaxes globally in a direction that is neither toward the teacher nor toward anything we can name, (iii) cost −3 to −6% at 5k. Meanwhile the structurally identical gate-3 recipe (same warm-start, same LR/blend/T, same 3:1 rehearsal on the SAME rehearsal tensor, similar corpus size) WON (+4.6%). Known deltas between the winner and this failure:

| | gate-3 (won) | dagger1 (lost) |
|---|---|---|
| new-signal labels | MCTS-visit distributions from search on vh1's own net | pillar3k policy top-5 (top-share 0.41, soft) |
| correction density | natural (replay states, majority agree with vh1) | 82% disagreements by construction |
| correction character | decisive escapes (judge +4.5pp on its mined set) | mixed gaps (median 0.65; includes vh1 near-ties) |
| label⇄policy relation | labels from vh1's own search = close to vh1's function | labels from a different (bigger) function's fine structure |

## Questions

1. **Mechanism**: what explains barely-absorbed + incoherent-global-churn + regression, given the drift is NOT toward the teacher? Candidates we see: (a) 82%-contradiction rows with soft top-share-0.41 labels = large per-row CE gradients against the current policy on near-tie states → churn inside each state's top-5 rather than clean flips; (b) BN running-stats shift from the death-band-heavy input distribution (we have prior evidence BN stats matter: concat-batch contamination incident); (c) hard-CE argmax-flip aggression (the pillar3c failure signature). What measurement would separate (a)/(b)/(c)? (BN check idea: re-estimate BN stats on rehearsal data post-training and re-eval — cheap.)
2. **Near-tie contamination**: vh1 keeps only 80% of its own recorded moves under fp32 recompute — so part of the "confident disagreement" set is vh1 tie-flips, not real preference conflicts. Should round 2 require the STUDENT to be decisively wrong (student's own top-2 logit margin ≥ ε on its chosen move) in addition to teacher gap ≥ 0.5? The judge's +3.73pp gap≥1.0 stratum presumably survives this filter, but it shrinks the corpus — how would you re-size?
3. **Recipe for absorbing confident corrections without churn** — rank these candidate round-2 arms (cheap to run, we have step-checkpointing): (i) corrections-only labels hardened to argmax one-hots (no soft top-5 on correction rows; soft labels only on agreement/rehearsal rows); (ii) gap≥1.0-only corpus (drop the +0.34pp dead weight, ~14k correction rows); (iii) density dilution to ~15-20% disagreements (gate-3-like) at the same total size; (iv) γ-weighting via the existing disagree_mask instead of composition change; (v) LR 3e-5 with longer horizon. Which 2 arms would you run first, and what early-abort measurement (absorption% at step 100?) would you pre-register?
4. **Or is single-state supervised flipping just the wrong tool** for installing search-validated preferences into a warm policy (the pillar3c + dagger1 pattern), and round 2 should instead: judge-filter the corrections through the TRAINED-model re-judge loop (train → re-judge → keep only survivors → retrain), or use advantage-weighted soft targets (shift probability mass by the measured +pp, not to the teacher's full distribution)?
5. Sanity: anything in the diagnostics that contradicts our reading? What additional cheap measurement would you demand before any round-2 compute?

Constraints (unchanged): 5k decides; master frozen (pillar3k = label source only); training on Colab; step-count gating; rehearsal > regularization; val untrustworthy.
