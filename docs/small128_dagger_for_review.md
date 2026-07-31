# Review brief #3: DAgger-from-strong-teacher for the small model

You are reviewing a training-plan proposal for a hobby research project. Be adversarial: your job is to find the flaw in the rationale, the gates, or the recipe BEFORE compute is spent. Data first, theories labeled as such.

## Setup (one paragraph)

Color Lines 98: 9×9 board, 7 colors, 3 balls spawn per turn, clear lines of 5+. Measured: score ≈ 2.03 × turns survived (score-rate is rock-constant; every scalar value target is a proxy for "turns survived"). The game is effectively infinite — goal is monotonic improvement, especially the floor (early deaths). Eval discipline: 5,000 fixed seeds (775000-779999) is the gold standard; 500-seed screens have flipped vs the 5k THREE times; we never compare per-seed across players (butterfly effect), only distributions.

## The two models

**Teacher: pillar3k_r3_dw3_T0.7_epoch_22** (10 blocks × 256ch ResNet, 11.9M params).
5k eval, **greedy policy-only** (argmax, no search at eval): mean 43,390 / P50 31,016 / P10 5,010 / <1000 1.3% / max 337,411.

**Student: small128_vh1** (10 blocks × 128ch, 3.0M params, ~4× smaller).
5k eval, same protocol: mean 13,080 / P50 9,323 / P10 1,889 / P5 1,222 / <1000 3.5%.

Student provenance: from-scratch distillation of the teacher's top-5 legal-softmax policy over 3,846,619 teacher-visited states (56% broad selfplay / 44% crisis escapes), recipe 0.5·soft-CE(T=0.5) + 0.5·hard-CE(teacher argmax), batch 4096, lr 1e-3, cosine, 100ep → ep87 (500-seed mean 12,987). Then one successful self-improvement iteration (+4.6% median, 5k-confirmed) → vh1.

**Key fidelity measurement:** on the training corpus the student matches the teacher at ~72% argmax / 95% top-3, and this was FLAT while gameplay climbed +35% — mimicry and play strength are decoupled. Diagnosed residual gap (13k vs 43k): **distribution shift + death tail** (student P10 1,889 vs teacher P10 5,010), not capacity (two independent reviewers concurred 3M is plenty for a 9×9 local-feature board).

## What we falsified since (receipts) — read carefully, it scopes the proposal

The self-improvement loop used **MCTS on the student's own net** (with a survival value head trained on the student's backbone, validated by calibration + judge gates) as the teacher. It produced exactly ONE win (vh1: micro-corpus of 55k relabeled own-crisis states + 3:1 rehearsal, gentle warm-start, +4.6% median at 5k). Then iteration 2 failed by EVERY lever, and the cause was measured:

| Lever tried on iteration 2 | Result |
|---|---|
| Corpus volume ×45 (2.5M own-crisis states) | −11% at 5k |
| Grid: disagreement-γ {0,2,6} × lr {1e-4,3e-4} × blend, step-gated | best arm −15% at 5k |
| Fresh value head retrained on vh1's own games (calibration r 0.76-0.84, fine) | corrections −0.1pp at the judge |
| Deeper search: 600 → 1200 → 4800 sims | +0.2pp → +0.3pp |
| Wider search: top-k 30 → 300 at 4800 sims | +0.3pp, 2% genuine / 97% tie |

Judge = swap in the teacher's move at a mined crisis state, then continue with the student's greedy policy; measure died-within-horizon over common-seed rollouts (|Δ| > 0.08 → genuine/phantom). Calibration of the bars: the gate that predicted the ONE win read **+4.5pp, 27-29% genuine**; every failure read **≤ +0.3pp**.

**The mechanism (measured, not theorized):** MCTS-on-vh1 escapes 84% of vh1's greedy deaths when it searches EVERY move (rolling), yet its root-move corrections are survival-neutral under vh1-greedy continuation. The search edge exists but is not per-move cashable — it lives in sustained sequences. Single-move distillation from the student's OWN search teacher is closed.

## The proposal: DAgger with pillar3k as the label source

Relabel **student-visited** states (especially its death-band states) with **pillar3k's policy** and train the student on them (proven hardce channel + rehearsal + gentle warm-start).

Why this signal is not the one we falsified (hypothesis, gated below):

1. **Greedy executability.** pillar3k's 43k mean is achieved by argmax play — no search at eval. Its per-move preferences are, by construction, the choices of a greedy policy that survives 3.3× longer. The falsified teacher's edge existed only under rolling search; pillar3k's edge exists under greedy execution — the same execution mode the student uses.
2. **Label type.** pillar3k labels = policy top-5 soft + argmax (the channel that built the student in the first place), NOT MCTS visit distributions (which failed separately via prior-domination).
3. **It attacks the diagnosed cause.** The original corpus is teacher-visited states. The student's own trajectories — especially its death spirals — are off that distribution, and that's where its P10/P5 collapse lives. Classic DAgger closes exactly this gap.

**The known risk we must falsify first:** pillar3k's move at a student-visited state may only be better under PILLAR3K's continuation (a softer form of the same rolling-horizon trap). The student executes its own continuation. Phase 0b is designed to measure exactly this before any training.

## Pre-registered gates (no Colab spend before they pass)

- **Phase 0a — distribution-shift audit: ALREADY RAN, signature ABSENT.** Fixed diag, 20k sampled states per set: teacher↔student argmax agreement **72.5%** on the teacher-visited corpus vs **70.4%** on the student's OWN death-band states (a 2.5M-state tensor mined from its real deaths); top-3 containment 95-97% in BOTH directions on BOTH sets. The pre-registered signature (a clear drop on the death-band set) did not appear: the student is NOT garbage-picking off-distribution in its death band — it matches the teacher there nearly as well as in-distribution. (Caveat: the corpus set is 56% broad / 44% crisis, so not a perfectly type-matched control.) Implication: the strong DAgger premise is weakened. The teacher's greedy edge lives either in the ~30% disagreements (~750k death-band states — mineable IF survival-relevant) or in fine-margin/sequence structure (the same shape that closed the own-teacher arc). Phase 0b on the disagreement set is now the decisive measurement, and per pre-registration we PAUSED here for this review before running it.
- **Phase 0b — the judge gate (local, hours):** on death-band states where pillar3k top-1 ≠ vh1 top-1: play pillar3k's move vs vh1's move, both continued by vh1-greedy, died-within-H over common seeds (identical protocol and thresholds as the calibrated gate above). **GO bar:** uplift and genuine% in the vicinity of the win-predicting reference (+4.5pp / 27-29%). **NO-GO:** ≤ +0.3pp (the consumed signature).
- **Phase 1 (GO only):** relabel ~2.5M already-mined student-death-band states with pillar3k (states exist; only labels change) + 3:1 rehearsal from the original corpus; gentle warm-start on vh1 (blend 0.5, lr 1e-4, seeded); pre-registered gate: **epoch 1 must beat vh1** on the 500-seed screen (healthy runs beat base by ep1 — verified repeatedly in this project); confirm at 5k. Training on Colab only.

## Questions for you

0. **Interpret the 0a result first.** No agreement drop on the student's own death-band states (72.5% vs 70.4%, top-3 ~95% both ways). Does the 13k-vs-43k gap remain attackable per-move through the 30% disagreements, or does this result already predict 0b ties (the teacher's edge being fine-margin/sequential, like the closed own-teacher arc)? Would you still run 0b, and would you redesign it — e.g., condition on the teacher's disagreement MARGIN (its top-1 prob minus its prob on the student's move), restrict to contested/decisive states, or measure something else entirely?
1. Is the greedy-executability rationale sound, and is 0b the right falsifier for it? Any confound in judging the teacher's single move under student-greedy continuation (e.g., could a genuinely better transfer signal still read as ties at 0b, or vice versa)?
2. Corpus composition: death-band-only vs the full on-policy state distribution (classic DAgger uses everything the student visits)? Our history: a 55k micro-corpus won; 45× volume did nothing or hurt. How would you size and compose the DAgger corpus?
3. Labels: pillar3k policy top-5, or pillar3k+MCTS (stronger teacher, much slower, and possibly reintroduces the rolling-horizon signal we falsified)?
4. Recipe: gentle warm-start on vh1 vs from-scratch re-distillation on original-corpus + DAgger-corpus combined? (History: warm+gentle produced the only win; but DAgger literature typically aggregates and retrains.)
5. **Self-improvement:** after a successful transfer round, we'd re-run the own-MCTS judge gate at the student's NEW level — the per-move gap that was consumed at vh1's level may reopen for a different policy with different blunder modes. Does the alternation "teacher-transfer round ⇄ self-play round, each gated by the cheap judge" make sense as a sustainable improvement scheme? A separate ceiling-raiser exists: the teacher itself has a queued self-improvement iteration (its own crisis loop, previously +18% then +5-8% per round), and any teacher gain flows down the transfer channel.
6. If 0b reads ties: is sequence/multi-step distillation (distilling the teacher's escape SEQUENCES rather than single moves) the right next door, and what's the minimal viable version you'd design?
7. Critique the pre-registered bars and sample sizes (0b uses a few hundred disagreement states; the |Δ|>0.08 threshold and the +4.5pp/+0.3pp calibration points come from the gates that correctly predicted the one win and all failures).

Constraints and hard-won lessons you should respect in your answer: 5k-seed evals decide, 500-seed screens are noisy; healthy runs beat base by epoch 1; val loss is untrustworthy under weighted/soft objectives (it has rewarded flat collapse); rehearsal/replay beats regularization (KL-anchor lost head-to-head); BN stats are contaminated by concat-batch mixing; from-scratch convergence is gated by optimizer steps, not epochs.
