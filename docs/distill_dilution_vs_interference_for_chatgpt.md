# 2× more (provably same-distribution) corrections regressed gameplay at IDENTICAL held-out match — dilution or interference?

Follow-up with much cleaner data. We fixed the eval confounds (one box, one tool, one batch, one
fixed 1k seed list; median SE ≈ 600) and ran the missing controls. Please critique the diagnosis
and the two decisive experiments at the end.

## Setup (unchanged)
Policy 11.9M params (10×256 ResNet). Recipe "mC": warm-start the converged pillar3b, re-distill its
400-sim self-play tensor (9.16M states, main CE, batch 32768, lr 5e-5, T=0.5) + auxiliary 256-sample
batch/step of crisis corrections: `loss = main_CE + 0.01·aux_softCE(T=0.5, margin-weighted)`.
Corrections: rewind each death game's final D-15..D-85 band, widened MCTS@4800 (top_k=300+Dirichlet,
3 determinizations), soft visit dist = target. Peak is ALWAYS epoch 2. Eval = greedy policy, 1k games.

## New, clean results (all same config — directly comparable)
| model | median | mean | P10 | P25 | <1000 |
|---|---|---|---|---|---|
| pillar3b base | 13,421 | 18,865 | 2,409 | 6,283 | 2.6% |
| ctrl0 = recipe with λ=0 (re-distill only) | 13,316 | 18,207 | 2,402 | 5,835 | 2.5% |
| mC = recipe + 13.8k corrections (1,837 games) | **16,738** | **24,249** | **3,196** | **7,801** | **1.6%** |
| mC27k = recipe + 27.6k corrections (3,676 games) | 14,943 | 22,370 | 2,211 | 6,263 | 2.0% |

Facts these establish:
1. **ctrl0 ≈ base**: the gentle re-distill contributes ~ZERO. The +25% median of mC is corrections,
   entirely. The channel transmits.
2. **2× corpus regressed −1.8k median (~3 SE), mean and P10 agree** (P10 even below base). Genuine.
3. **Held-out (by-seed) match is IDENTICAL ~0.215 for both corpora** (and ≈ the 13.8k model's match
   on never-seen corrections: 0.214–0.224). Per-state argmax absorption did NOT change.
4. **Composition is ruled out**: chronological mining slices are statistically identical (source-game
   median score 12.7k/13.0k/11.3k/12.5k; corrections/game 17.2–17.7; decisive margin P50 0.110–0.113;
   median band depth 54 in all slices). The new half is the same kind of data.
5. Earlier (dirtier-config) data points: λ ×1.2–1.4 on a 19.6k corpus was flat; λ=0.03 on an OLD
   low-diversity corpus regressed; a trust-region variant (teacher-primary + KL-anchor to pillar3b,
   no V13 re-distill) lost to mC by ~2k at every anchor strength (held match 0.27 — higher than
   mC's 0.215 — yet played worse). Held match does NOT predict gameplay.
6. Manual review of corrections (GUI): genuinely better moves (blunder fixes + multi-step
   "grandmaster" setups + right-line-wrong-ball cascade unblocks); heterogeneous; good across the
   whole band including last-effort escapes at D-25.

## The puzzle
Same recipe, same data distribution, 2× samples → worse play at identical held match. Aux exposure
per correction by ep2: ~93 (13.8k) vs ~46 (27.6k) — halved. Match-rate saturates early, so it can't
see exposure depth; gameplay apparently can.

## Hypotheses left standing
- **H1 dilution**: per-correction weight = (steps/N)·λ halved; the corrections are encoded
  shallowly (right argmax, flabby distribution/confidence) and shallow encoding loses to the
  policy's strong priors at play time. Fix: λ ∝ N (λ=0.02 for 27.6k). The old "λ flat" evidence
  only tested ×1.2–1.4, never ×2, and on a confounded eval.
- **H2 interference/capacity**: 2× diverse local preferences in a fixed 11.9M net at a fixed drift
  budget → degraded compromise encoding even where argmax survives. Fix: capacity, or smaller
  curated corpora, or sequential/curriculum absorption.

## The decisive experiments (queued; ~2h A100 each; please critique/extend)
- **E1**: random 13.8k SUBSAMPLE of the 27.6k corpus, λ=0.01 — same size, same diverse source.
  ≈16.7k bar ⇒ size is the whole story. Regresses ⇒ deeper problem.
- **E2**: full 27.6k, λ=0.02 (exact per-correction-weight match). Recovers bar ⇒ H1; corpus scaling
  works with λ∝N. Flat ⇒ H2.

## UPDATE: your margin diagnostics, run on 1,160 common-unseen corrections (all models, local)
| model | match | p(top1) | mass top20 | softCE | logit margin | KL vs pillar3b (corr states) |
|---|---|---|---|---|---|---|
| base | 0.010 | 0.1775 | 0.959 | 3.204 | −1.49 | 0 |
| ctrl0 (λ=0) | 0.066 | 0.1775 | 0.960 | 3.211 | −1.50 | 0.004 |
| mC 13.8k | 0.230 | 0.1973 | 0.944 | 3.124 | −1.35 | 0.172 |
| mC 19.6k | 0.230 | 0.1967 | 0.941 | 3.104 | −1.30 | 0.185 |
| mC 27.6k | 0.199 | 0.1936 | 0.936 | **3.059** | **−1.24** | 0.181 |
| mF 19.6k λ.012 | 0.245 | 0.2017 | 0.936 | 3.105 | −1.28 | 0.232 |
| mG 19.6k λ.014 | **0.258** | **0.2051** | 0.936 | 3.133 | −1.31 | 0.269 |

The naive dilution signature is ABSENT: 27.6k has the BEST softCE and margin but the WORST match/p(top1)
— better calibrated to the soft target, less committed to the argmax. λ (mF/mG) raises commitment
(match 0.230→0.258) at proportional drift cost (KL 0.185→0.269). Refined H1: halved per-anchor exposure
under-SHARPENS rather than under-encodes; decisiveness — not CE — is the gameplay currency here
(target sharpening was historically a +57% lever). Next: clean-config gameplay evals of the 19.6k/mF/mG
checkpoints (all exist locally; their old "λ flat" result came from the confounded list) — if mG > 19.6k-λ.01
on clean eval, H1-as-undersharpening is confirmed and λ∝N is the recipe.

Also: your "if E1/E2 fail, switch to trust-region (teacher-primary + KL-anchor)" suggestion is the exact
experiment reported in fact 5 — it LOST to mC by ~2k at every anchor strength. Please strike it from the
fallback list; the live fallbacks are "wrong distillation object" (→ Q-softmax targets; our miner now
records root Q) and "globally inconsistent labels".

## Questions
1. Does the (identical held match, worse gameplay) signature discriminate H1 vs H2 by itself? Is
   "encoding depth" (sharpness/confidence on corrected states) the right mental model, and is there
   a cheap LOCAL metric for it (e.g., teacher-target probability mass on held-out corrections,
   margin between top1 and top2 logits) we should add to our eval-free screening toolkit?
2. If E2 recovers the bar: how far does λ∝N scale before the old aux-domination failure returns
   (grad-audit says effective aux share ≈ 70×λ; λ=0.05 @ 69k corpus → share ~3.5)? Is there a
   principled λ(N) schedule, or a better invariant to hold than per-correction weight?
3. If E1 regresses AND E2 is flat (both hypotheses fail): what else explains a clean 2×-data
   regression at constant per-state absorption?
4. The user's endgame: pick the best corrected model, then start a NEW self-play generation from it
   (V14), keeping corrections "properly absorbed". Given 400-sim self-play search may not re-find
   4800-sim moves, would you keep the crisis aux term during V14 training, or trust the corrected
   prior to bias the search? Any literature-grade precedent for distill-then-iterate retention?
