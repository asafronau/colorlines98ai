# 128-channel policy-improvement flywheel

Status: working plan, updated 2026-08-14.  The premise is that a 128-cost-class
model has enough capacity to match the 256-channel model and plausibly sustain
indefinite play.  Width is therefore not an allowed explanation until the
data, teacher, objective, and optimization hypotheses below have been tested.

## What the evidence currently says

- `scratch128_lr3e3` has plateaued **within its completed 40-epoch cosine
  run**, but that is not yet an architectural-capacity verdict.  On 5,000
  games epoch 36 versus epoch 40 is a distributional wash (mean 9,942 versus
  10,136; median 6,978 versus 6,950; P10 1,482 versus 1,455; `<1000` 5.24%
  versus 5.32%).  Their fixed-state actions agree on 97.22% of rows and legal
  NLL only changes 0.635 to 0.632 while the cosine reaches 3e-5.  This proves
  that more tail epochs under the same decayed schedule and historical R2
  corpus are low value.  It does not show that a learning-rate restart or,
  more importantly, a fresh on-policy search target cannot restart progress.
- The complete width control demonstrates a materially higher achievable
  envelope for the fixed 10-block recipe.  `scratch192_lr3e3` epoch 39 reaches
  mean 18,500 / median 13,040 / P10 2,532 / P90 41,478 / `<1000` 2.72% over
  5,000 games.  Against `small128_vh3`, its independent-bootstrap gains are
  mean +4,641 [+4,035,+5,261], median +3,424 [+2,878,+4,136], P10 +564
  [+344,+764], and `>10000` +10.88pp [+8.98,+12.82]; all staged survival
  horizons improve significantly.  Epoch 39 also improves decisively from
  epoch 20, so the larger model has not plateaued.  Width is a real crisis-
  reliability and sustainable-play lever under this frozen-corpus recipe,
  although the owner's approximately 2x production inference cost still makes
  192 primarily a reference and possible selective teacher.
- The near-equal-latency architectural alternative has succeeded on the
  frozen-corpus test.  `scratch18b96_lr3e3` epoch 30 is distributionally tied
  with `small128_vh3`; epoch 40 reaches mean 14,766 / median 10,384 / P10
  1,873 / P90 33,982 / `<1000` 4.16% over 5,000 games.  Versus vh3 it improves
  mean +907 [+363,+1,450], median +766 [+264,+1,257], P90 +2,669
  [+563,+4,424], and 5k/10k survival by +2.90/+2.94pp, while P10 and 1k
  survival remain unresolved.  It is broadly tied with 192e20 except for a
  slightly weaker crisis floor.  Native FP16 latency at observed MCTS batch
  55 is only about 5% slower than 10b128.  This makes 18b96 the smallest
  credible actor and proves the old 10b128 stall was architecture/recipe
  dependent; it does not yet prove that 18b96 can self-improve or drive early
  catastrophe hazard toward zero.
- The 13.4M-state Pillar3k-relabeled strong-corpus arm succeeded rather than
  failed: its best region was epochs 30--32, hidden by the initially inspected
  epoch-29/39 endpoints.  Epoch 32 at 5,000 games is mean 9,430 / median 6,524 /
  P10 1,409 / `<1000` 5.72%, distributionally indistinguishable from old
  scratch192 epoch 19.  Its abrupt epoch-32-to-33 regression makes frequent
  gameplay checkpoint gating part of the recipe; the final checkpoint cannot
  stand in for a training trajectory.
- The corrected-D4 scratch discriminator has now satisfied its predeclared
  broad-strength condition: 10b128 flattened, 18b96 caught and exceeded vh3,
  and 10b192 continued to a substantially higher envelope.  Call this a
  capacity signal for **particular architectures under historical R2**, not a
  verdict that a 128-cost-class policy cannot self-improve.  The decisive next
  experiment is closed loop: generate clean search corrections on the current
  actor's own states, show the corrections are causally better, and test whether
  18b96 can absorb them without broad regression.  Search-win/student-fail on
  18b96 while 192 absorbs the same corrections would be evidence for a real
  actor-capacity floor; frozen-corpus curves alone cannot establish one.
- The completed high-LR width ladder separates sample efficiency from system
  efficiency.  Early on, 192 learned the same frozen targets faster per epoch;
  late, it also reached a higher gameplay envelope than 10b128 and 18b96.
  Because inference dominates flywheel cost, however, a larger actor must
  improve the production FP16 strength/throughput frontier, not merely learn
  faster or score higher per checkpoint.  The current measurements put 18b96
  on that frontier and 192 above it in strength but behind it in throughput.

- Search is a real improvement operator.  The uncapped search games in the
  local bank have a median around 90k turns/score, far beyond the 128-channel
  greedy policy.  On fixed high-confidence crisis states, common-random-number
  rollouts also show a causal search-action advantage: at R32/H1000, death rate
  improved by 5.04 percentage points and lifetime by 45.6 turns.  Broad-state
  disagreements did not show a measurable benefit.
- The old training objective is not a faithful policy-improvement step.
  Current hard-CE corpora agree with the base legal argmax roughly 90% of the
  time, so most rows sharpen the policy's own decision.  The resulting endpoint
  is overconfident and often weaker; weight interpolation accidentally supplied
  a trust region and produced `small128_vh3`.
- The historical "undercommitment" magnitude was an analysis artifact.  Some
  scripts softmaxed twice.  Corrected all-legal measurements show that old
  sharpening wins came mainly from changing margins, with only modest ranking
  adoption.  The current model is already much sharper, so that recipe should
  not be copied mechanically.
- Behavior and labels are mixed in the historical JSON schema.  The native
  generator now runs an optional second noise-free root, stores its candidates
  and `teacher_move`, and keeps the exploratory `behavior_move` separate.  A
  real FP16 smoke produced different behavior and teacher actions while
  advancing the game with behavior, as intended.
- The present value scalar is nearly saturated.  It is a weighted sum of
  survival predictions at horizons 25/50/100/200 with maximum 2.55.  Logged Q
  gaps are consequently tiny and unstable on healthy states.  This is not a
  suitable long-horizon value for an indefinite-play objective.
- The current vh3 survival head also lacks an independent K-rollout
  calibration set: its checkpoint records `val_data=None`, even though the
  head later supplied MCTS values and the continuous danger weights used in
  round 2.  Its single-trajectory inner validation loss is not a probability
  calibration result.  Before retraining, audit the deployed FP16 scalar
  separately from the raw four logits; FP16 sigmoid saturation may discard
  useful ordering information.  Any alternate scalar is only a search-sweep
  hypothesis until fixed-state ranking and 5k gameplay validate it.
- BatchNorm running statistics are a large, independently destructive update
  channel.  They must remain frozen during small corrective steps unless a
  separately controlled deployment-distribution recalibration passes an
  evaluation.
- The current `r2_bulk` artifact is composition-confounded: an inherited 10k
  per-game default truncated old uncapped games despite the intended all-data
  design.  Future builds now pass the cap explicitly, but this tensor cannot be
  used to infer a clean scaling law.
- The lineage-pure flywheel artifacts now preserve game/seed/turn/source
  provenance and split by original source-game seed.  They contain 3,098,789
  current-vh3 target rows plus all 19,610,085 states from 20,000 current-vh3
  greedy games: 22,708,874 rows total, with no 256-channel data in the primary
  manifest.
- The scratch width discriminator is not showing a clear capacity wall.  At
  approximately equal wall time, 128ch epoch 16 scored mean 3,174 / median
  2,379 over 5,000 games; 192ch epoch 7 scored 3,641 / 2,674.  On 50,000 fixed
  training states they were effectively identical: legal hard-label match
  73.52% vs 73.28% and legal NLL 0.776 vs 0.784, including the danger tail.
  Continue the predeclared 128-epoch-40 versus 192-epoch-16 wall-clock curve;
  the current modest gameplay difference is not a recipe verdict.
- The scratch discriminator itself is not a tuned small-model recipe.  It uses
  the hall warm-start geometry (`bs=32768`, `lr=3e-4`) for both widths.  The
  historical best 128ch from-scratch distillation used `bs=4096`, `lr=1e-3`,
  a long cosine consolidation, and explicitly found large batches
  step-starved scratch learning.  If the current 128 curve plateaus, an
  update-rich 128-only arm on lineage-pure capped-game data is required before
  interpreting the plateau as capacity.
- The optimizer-step accounting makes this quantitative.  The current
  scratch128 run gets about 2,903 updates/epoch and 116k updates by epoch 40.
  The historical successful 128ch run got about 7,137 updates/epoch and 714k
  total.  At roughly 136k updates (historical epoch 19) it still scored only
  about 6.2k; it reached 10.9k at epoch 49 and 13.0k at epoch 87 while top-1
  teacher match was already flat.  Thus scratch128 epoch 40 versus scratch192
  epoch 16 is a useful equal-wall-clock width comparison, but neither endpoint
  is an update-matched capacity test against vh3.
- Three corrected-code 128ch recipe controls now make the step-starvation
  question identifiable.  On the 95% training split, one augmented corpus pass
  is 95,098,760 examples.  `bs=2048/lr=3e-4` gives 46,435 updates per epoch and
  reaches 743k updates after 16 epochs; `bs=32768/lr=3e-3` keeps the original
  2,903 updates per epoch and tests update magnitude rather than consolidation;
  `bs=4096/lr=1e-3` gives 23,218 updates per epoch.  The last arm is the closest
  historical-geometry bridge: 31 epochs are 720k updates / 2.95B examples,
  versus 714k / 2.92B for the old launch pad, while its one-epoch warmup is
  23.2k steps versus the historical three-epoch 21.4k-step warmup.  These runs
  are retrospective mixed-era `r2_bulk` recipe diagnostics, not primary
  lineage candidates.  Because the stopped 29-epoch and ongoing width runs
  used corrupt D4 views, a corrected `bs=32768/lr=3e-4` baseline is required
  before attributing any new-run delta specifically to batch size or LR.
- Visit truncation was a hidden target transform.  At 400 simulations, actions
  16--30 carried 24.6% of opening-root visits and 32.9% over 1,969 trajectory
  states.  Keeping only 15 inflated median teacher share from about 0.19 to
  0.29.  New generation and builds retain all 30 searched root candidates.
- Raw visit counts are not a deployment policy.  After exploratory turns, a
  clean 400-sim teacher differed from the clean policy-prior argmax on only
  1.53% of 850 broad trajectory states, while its visit top-share was only
  0.195 despite prior top-share 0.647.  The rare disagreements had positive
  teacher-vs-prior Q gaps in all 13 sampled cases.  Counts mainly encode PUCT
  allocation; the hard clean winner is the primary retrospective target.
- The historical uncapped search games establish teacher strength, not a
  corpus-length prescription.  This game has no chess-like late-game phase:
  a strong player quickly enters a continuing sustainable regime.  Many
  independent games capped around 1,000 turns provide more new boards and
  source-game diversity per unit of search compute than a few enormous games.
  This is measured in the 1,500-game vh3 bank: 98.93% reached turn 999; empty
  count, color count, adjacency structure, visit top-share, and visit entropy
  were essentially flat from turn 200 through 1,000; every sampled full state
  in every turn band was unique.
- Symmetry is learned rather than represented.  The policy head emits 81
  fixed absolute-source channels at each destination cell, and ordinary
  convolutions are not D4-equivariant or color-permutation invariant.  A fixed
  10k-state FP16 audit measures only about 80% nonidentity-D4 action agreement
  for vh3 and both compute-matched scratch widths; just 56--57% of states agree
  across all eight views.  Color-relabel agreement is about 91% for vh3 and
  87% for the scratch models.  Averaging the eight mapped probabilities raises
  scratch recorded-target match by 4.3--4.5 points, but this fixed-state proxy
  is not gameplay evidence.  The tax is nevertheless material enough that
  canonicalized states or a relational source/destination head are a
  capacity-efficient 128ch recipe axis.
- Historical D4 augmentation was not an exact game symmetry at the feature
  level.  It rotated the pixels in all 18 planes but failed to permute the
  horizontal, vertical, and two diagonal line-potential planes (channels
  13--16).  Six of the eight sampled transforms therefore attached incorrect
  directional features to the rotated board.  This dates to the original
  loader and confounds every augmented historical recipe, including the
  current scratch128/scratch192 discriminator.  The width comparison remains
  controlled because both widths saw the same error, but their absolute
  learning curve is not a clean capacity test.  The loader and symmetry audit
  now use an exact direction-channel permutation, verified for all eight views
  against observations rebuilt from transformed boards.  Corrected-D4 versus
  no-D4 is now a required small-model recipe ablation before blaming width.
- Every historical policy CE also normalized over all 6,561 source/destination
  pairs even though deployment computes an exact legal mask and can never
  choose the overwhelming majority of them.  Mature checkpoints conceal the
  optimization cost (their full argmax is already legal about 99.8% of the
  time, and full versus legal NLL differs only modestly), but from scratch the
  small network must learn occupancy and path reachability merely to remove
  denominator mass.  A legal-support CE/KL option now uses the same exact mask
  as deployment.  Treat it as a controlled recipe axis, not an assumed win;
  it changes early gradients and is not comparable to R1/R2 unless held fixed.
- Historical task-vector geometry must be read by parameter class.  The
  previously quoted cosine near 0.95 between round-2 bulk epoch 3 and the
  successful round-1 vector came from concatenating the full state and was
  dominated by BatchNorm running buffers.  The corrected cosines are about
  0.67 for convolution/head weights, 0.94 for BN affine parameters, and 0.96
  for BN buffers.  This still shows some shared direction, but does not justify
  calling the learned weight updates essentially identical.  Any norm-scaled
  merge must report these groups separately.

## Measurement rules

1. Whole-policy gameplay is compared as independent score distributions.  A
   matching numeric seed does not make two stochastic trajectories a paired
   experimental unit after their actions diverge.  Use at least 5,000 games,
   independent bootstrap confidence intervals, and report mean, median, P10,
   P90, `<1000`, and long-survival/cap rate.
2. Pairing is allowed only for fixed-state counterfactual rollouts with common
   random numbers, where the state and candidate first actions are held fixed.
   Resample by source-state cluster, not by individual rollout.
3. Seeds 1,100,000--1,119,999 remain untouched until final confirmation.
4. Screens are debugging tools, not promotion evidence.  A 999/1,000-game
   screen cannot promote a checkpoint.
5. Every slow local command runs under `caffeinate -i -s`.
6. Use staged capped reliability gates.  A 5,000-game `--max-turns 1000` run
   directly estimates the early catastrophe rate and is much cheaper than
   letting every already-sustainable game continue.  Only survivors of that
   gate need longer 2k/5k/10k horizons.  Selfplay and reanalysis likewise use
   many independent games capped near 1,000 turns; uncapped play is neither
   required nor a scalable source of board diversity.
7. Scratch learning on a frozen corpus cannot establish self-improvement.  A
   candidate actor passes the closed-loop test only when (a) its own frozen
   search produces causally better actions on fixed-state common-random-number
   rollouts, (b) the same architecture absorbs those edits without broad
   regression, (c) a 5,000-game independent distribution improves over its
   frozen base, and (d) the promoted policy repeats the result for another
   iteration.  If search wins but the student does not, investigate
   capacity/objective/optimization.  If search itself does not win, changing
   student width cannot repair the improvement operator.

## The improvement operator

For current policy `pi_k`, build a frozen base distribution on every training
state.  Let `rho_k` be the declared teacher object.  Initially it is a one-hot
clean-search winner; raw PUCT counts remain a diagnostic until a calibrated
prior/Q-corrected target beats this control.  The update is applied on
**millions of states**, not only an extracted disagreement tensor:

```
L(s) = KL(pi_k(.|s) || pi(.|s))
       + eta(s) * CE(rho_k(.|s), pi(.|s)).
```

In unconstrained probability space the target is exactly

```
(pi_k + eta * rho_k) / (1 + eta).
```

Thus `eta` is an explicit policy-mass budget.  Every target row remains in the
corpus, but agreement rows can receive a low nonzero dose while stable crisis
disagreements receive more.  Every broad anchor receives the frozen-base KL.
This is the function-space version of the trust region that weight
interpolation found accidentally, and it avoids relying on linear checkpoint
geometry or BatchNorm compensation.

Initial policy experiment:

- base: `small128_vh3`;
- states: all 22.7M current-lineage rows, with source/danger/turn/frontier
  tracked as strata rather than substituted for the full corpus;
- primary target: the recorded search winner, mixed by a small predeclared
  dose into the frozen base distribution; behavior and teacher are distinct in
  new data;
- retrospective arm: existing top-15/noisy-visit files can test hard-winner
  optimization, but cannot validate soft targets or the final teacher protocol;
- canonical arm: newly generated, noise-separated clean reanalysis labels;
- BatchNorm running buffers frozen; FP16 forward with FP32 loss/KL;
- small `eta` ladder, with frequent checkpoints and functional audits;
- audit gates: target-distribution movement, broad/base argmax retention,
  held-out full-distribution KL, and no BatchNorm-buffer drift;
- only audit-passing checkpoints receive a 5,000-game distribution evaluation.

The 43,281 `r2_frontier` argmax disagreements (7,371 after the first confidence
gate) are only a diagnostic slice.  They are too small and too narrow to be the
main learning corpus.  Their value is to measure causal action quality and
collateral drift.  The full update still consumes millions of labelled states;
the slice is an edit-quality instrument, not a corpus.  Existing frontier
visits are also not the final teacher because they were produced with root
noise and truncated support.

## Immediate retrospective matrix

All arms use the same 22.7M current-vh3 rows, group-held-out audits, frozen BN,
FP16, and frequent step checkpoints.  No big-model games enter these arms.

| arm | purpose | target/update |
|---|---|---|
| R0a legacy-hard/frozen | Separate target-mixture effects from BN | hard teacher CE on 3.1M search rows + hard recorded action CE on 19.6M anchors, frozen BN |
| R0b legacy-hard/historical-BN | Faithful retry of the old vector-generator channel | same pooled row-level mixture, train-mode BN, then predeclared full-state interpolation doses |
| R1 bounded-hard | Test whether the accidental interpolation trust region can be made explicit | base KL on every row + `eta=0.1` hard teacher dose, uniform sources |
| R2 bounded-selective | Put capacity on crisis edits without throwing broad data away | R1, source weights `1 / 2 / 0.25` for crisis600/deep2400/selfplay400 and base-agree/disagree weights `0.25 / 1` |
| R2p bounded-selective/pooled | Remove optimizer-moment confounding | R2, but target and anchor rows are mixed within every Adam batch instead of alternating source-pure correction/pullback batches |
| R3 bounded-blend | Diagnostic only if R1/R2 fail | 50/50 hard winner and historical top-15 visits; never interpreted as a clean soft-target test |

Functional gates are frozen-base KL, base argmax retention, teacher match before
and after, adoption on base/teacher disagreements, agreement-row preservation,
and exact legal metrics.  A small screen can only reject broken checkpoints.
At most two or three audit survivors get a 5,000-game independent-distribution
evaluation; the reserved final bank remains untouched.

R1 completed on 2026-08-08.  Across step 250 through the endpoint, exact-legal
base retention stayed near 96%; only about 13--14% of base/teacher
disagreements were adopted, while losses on agreement rows made net teacher
match worse.  The cosine tail did not repair the imbalance.  It therefore
failed the proposed functional promotion gate, but that gate is not yet
validated strongly enough to substitute for gameplay: historical 128ch
gameplay often improved after top-1 match flattened, and vh3 itself came from
a losing endpoint.  Keep the lowest-drift R1 checkpoint as one diagnostic 5k
arm if the final shortlist has room; never promote it from proxy metrics.

R1 and the first R2 run also used source-pure batches: a target correction
batch followed, on average, by roughly six anchor-only KL pullback batches.
That has the correct mean objective under SGD but different and much noisier
Adam moments than a random pooled corpus.  Future runs use row-pooled batches;
the first R2 remains a useful source-pure control.  R2p is therefore the direct
optimization-corrected follow-up, while R2 still tests less pressure on
agreement rows, more on crisis/deep disagreements, and no augmented-view
disagreement confound.

R2's lowest-drift step-250 checkpoint received the required 5k gameplay test
on development seeds 805000--809999 at FP16 with a fixed 100k cap.  Despite
its poor functional proxy (only 15.4% edit adoption and 3.8% broad-anchor
argmax churn), gameplay was a wash against vh3 under independent bootstrap:
mean -108 [-634,+414], median -8 [-644,+555], P10 -16 [-222,+182], `<1000`
+0.16pp [-0.62,+0.92], and mean turns -55 [-312,+198]; neither run capped.
Therefore adoption/retention can reject gross breakage but cannot substitute
for gameplay.  R2 is not a promotion, but the result keeps the pooled R2p
optimization control scientifically live.

## Iteration 1 pilot now active

The first canonical clean pilot is frozen in
`flywheel/vh3_iteration_1_pilot.json`; exact restart and audit commands are in
`flywheel/VH3_ITERATION_1_PILOT.md`.  It targets 3.5--4.5M fresh current-vh3
search states (3,000 exploit games, 750 behavior/teacher-separated explore
games, and 500 crisis probes) plus a deterministic 1M current-lineage anchor
sample during training.  This is deliberately enough to expose the mechanism
without immediately rebuilding the full 22.7M-row retrospective corpus.

The 8k end-to-end canary passed full-record, clean-label, legality, corpus,
sidecar, aggregate-loss, and exact mid-epoch-resume checks.  Exact fp16 base
annotation found only 333/8,000 (4.16%) teacher disagreements.  Therefore the
plastic arms keep all rows but apply a 16x dose to immutable original-view
disagreements, making them roughly 40% of effective loss at the declared
target/anchor mix.  Online disagreement on randomly D4-transformed views is
about 17% because vh3 is not equivariant; it is logged separately and never
used to decide which rows receive the search-edit dose.

Three matched students use the identical clean tensor: bounded warm control,
aggregate warm, and aggregate same-architecture scratch.  `aggregate` means
unconstrained hard clean-winner CE on search trajectories plus full base-policy
KL on replay anchors, avoiding legacy one-hot sharpening of anchor actions.
Warm and scratch plastic arms share objective, BN mode, augmentation, legal
support, optimizer, update budget, and inference architecture, so their
difference identifies initialization/basin effects.  The full exploit stream
started at roughly 20.5k MCTS leaf evaluations/s (effective batch 55).

The completed pilot contains 3,903,611 audited rows: 2,928,729 exploit,
724,807 explore, and 250,075 crisis.  All 4,554 files have clean/full records
and no invalid targets.  The 500-probe crisis tranche produced 402 failed
probes, 98 capped probes, and 804 recovery/prevention replays.  Exact vh3 edits
are 4.14% overall; crisis rows are enriched to 6.24% and failed crisis tails
to 8.48%.  The full MPS-fp16 sidecar and 50k-row target audit passed.

A matched 18b96e30 absorption branch is active on the same corpus.  Its own
deployment-fp16 disagreement rate is 19.42%, but that is not the clean search-
correction stratum: of 757,937 18b96/teacher disagreements, 654,802 (86.4%)
are ordinary rows where the teacher equals the vh3 actor and only 18b96
differs.  The full overlap is 103,135 edits shared by both bases, 58,504 vh3-
only edits, 654,802 18b96-only differences, and 3,087,170 rows where neither
base differs from the teacher.

The primary architecture test therefore gives both models the identical vh3
actor-root edit mask, identical 1x/16x row weights, source weights, anchors,
and full optimizer schedule.  The 18b96 sidecar still proves that the frozen
base checkpoint and corpus match; a separate vh3 sidecar supplies only the
immutable training stratum.  This asks which architecture can absorb the same
causal search intervention.  Candidate-specific mass matching (18b96 weights
1.1880338285875751/3.5102420706199298, for the same total mass and 41.579%
nominal edit dose) remains a secondary imitation diagnostic, not the primary
capacity comparison.

### Retrospective verdict and actor handoff (2026-08-14)

The vh3 bounded-target matrix is complete enough to stop tuning stale labels.
Low-dose hard edits failed to approach their exact bounded optimum; raising
edit exposure fixed much of that optimizer problem but produced a 5k gameplay
wash. Dense visit targets were then calibrated against the exact target rather
than their raw loss. The best mixed target (`alpha=0.5`, `eta=0.1`, 8x edit
exposure) reduced all-row, edit, high-confidence, and crisis target residuals
by roughly 8--10% while retaining 98.61% of anchor actions. Nevertheless its
independent 5k cap-rate delta was -0.70 percentage points
[-1.90,+0.50] and its mean delta was -5 [-16,+7]. This separates target
absorption from policy improvement: the old vh3 search object is not a useful
source for another optimizer sweep.

The flywheel actor is therefore now 18b96 epoch 40. It is tied with vh3 on the
1,000-turn reliability distribution but significantly stronger in long play
at approximately the same production inference cost. Its on-policy canary
passed 11,219/11,219 rows with full search records and no invalid actions. Of
six greedy crisis deaths, 500-turn recovery survived once and prevention
survived five times, giving a direct reason to prioritize prevention
trajectories. The actor-native anchor bank is complete: 5,000 capped games and
1,798,692 recorded states, split by source game.

Full target generation is declared in
`flywheel/18b96e40_iteration_1_pilot.json`; operational details and live status
are in `flywheel/18B96E40_ITERATION_1_PILOT.md`. This iteration never mixes vh3
or big-model labels. Its two training hypotheses are a crisis-focused bounded
update and aggregate rolling-search distillation from the same actor. Capacity
is implicated only if a validated on-policy search class wins but 18b96 cannot
absorb it without regression; failure of the search target itself is a recipe
or improvement-operator failure.

### Iteration-1 corpus and first student result (2026-08-16)

Generation and strict audit are complete. The actor-native corpus contains
3,903,888 clean full-root search states: 2,945,600 exploit, 723,704 explore,
and 234,584 crisis. It is backed by 1,798,692 preservation states from 5,000
independent actor games. No foreign-lineage games enter either tensor. Rolling
400-simulation exploit search capped 95.03% of 3,000 games versus 89.28% for
the 5,000-game greedy anchor distribution, an independent +5.75pp
[+4.59,+6.91] search-operator gain. Prevention replay also survived 500 turns
in 335/389 crisis cases versus 104/389 for recovery, identifying prevention
as the highest-value crisis trajectory class without pretending those are
paired single-action interventions.

Four small objective smokes separated update mechanics before using the whole
corpus. The bounded crisis arm was safe but adopted only about 5% of edits
against an analytic target near 53%, so it has not yet tested the crisis-label
hypothesis. Removing color augmentation increased policy drift in both
objectives. Aggregate hard-teacher distillation with color augmentation was
the only checkpoint advanced to gameplay.

Its epoch-1 student, trained on 200k target plus 50k anchor rows, appeared to
win the 5,000-game cap-1k development distribution: mean +14 [+2,+25] and
`<1000` -0.86pp [-1.60,-0.12]. That result did **not** replicate. A previously
unused 5k bank was mean +7 [-4,+18], a second deterministic training subsample
was mean -5 [-16,+5], and both a 4.7M-row dose-matched run and its lower-dose
checkpoint were washes. The initial interval was conditional on checkpoint
selection and cannot be treated as promotion evidence.

An edit-only bounded arm removed hard CE from the 97.6% agreement rows. Its
higher-LR checkpoint again won the development bank (mean +15 [+3,+26],
`<1000` -1.02pp [-1.74,-0.28]) and again failed a fresh 5k bank (mean -3
[-14,+8], cap -0.74pp [-1.88,+0.42]). Therefore the failure is not simply
aggregate one-hot sharpening. No student is promoted and the final
1,100,000--1,119,999 bank remains untouched.

Fixed-state mining identified the missing discriminator. In a 924-state CRN
pilot, successful and failed-broad exploit edits were neutral. A fresh-seed
64-repetition rejudge of 353 crisis edits confirmed death-within-200 uplift
+1.24pp [+0.68,+1.84] and +2.18 turns [+1.25,+3.21]; recovery edits alone
were +2.02pp [+1.06,+3.04] and +3.81 turns [+2.07,+5.74]. Explore's small
pilot signal disappeared on fresh rollout seeds. These paired estimates are
valid fixed-state interventions and are not confused with paired gameplay.

The initial mined mask contained only 9,603 crisis edits. The predeclared
diversity expansion is now complete: 2,000 disjoint same-actor probes added
905,645 states and 36,717 edits. Combined with the pilot, the immutable crisis
corpus has 1,140,229 states, 46,320 edits, and 1,867 independent death seeds.
A fresh 600-state, 64-repetition CRN judge replicated the causal class almost
exactly: death-within-200 uplift +1.31pp [+0.82,+1.81] and +2.26 turns
[+1.46,+3.12], with both prevention and recovery positive.

More diversity did not repair sparse one-hot edit distillation. A full-data
18b96 pass reduced bounded-target KL on training edits by only 1.8% (3.7% on
the new tranche), while held-out edit KL worsened at every archived dose; the
endpoint moved 0.00381 to 0.00430. A 192ch positive-control base on the same
46,320 immutable edit mask fit training edits more readily (0.00921 to
0.00875) but also worsened every held-out checkpoint (best 0.00841 to 0.00869,
endpoint 0.00915). Width therefore buys in-sample plasticity but does not
solve this supervision/objective. Neither student advances to gameplay, and
this is not evidence for an 18b96 capacity wall.

The next target recipe on the **local-edit track** must exploit causal
magnitude rather than label every MCTS argmax disagreement equally. In the
replicated judge, 89.3% of edits are rollout ties, 8.5% clearly genuine, and
2.2% phantom. Mine several thousand states with cheaper CRN repetitions,
retain/weight actions by held-out rollout advantage, and train a
teacher-vs-base pairwise logit margin (plus frozen-base KL anchors) or an
advantage/value head. Predeclare and freshly validate any top-share or recovery
weighting before training; the current 600-state sample suggests confidence
may help but is not a threshold-selection set.

The immediate primary run remains the distinct aggregate-trajectory test. The
original exploit/explore data and both crisis tranches form 4,809,533 unique
current-18b96 search states, backed by 1,798,692 current-policy anchors. The
resumable `train_flywheel_18b96e40_i1_colab.ipynb` warm-starts epoch 40 and
trains every trajectory with source weights `1/1/2/2`; it does not duplicate
the pilot crisis rows or import foreign-lineage games. This arm asks whether a
coherent rolling-search policy can be compressed into one forward. The
fixed-state judge correctly rejected indiscriminate isolated edits, but cannot
reject coordinated multi-step trajectory distillation, whose evidence unit is
whole-policy gameplay.

Aggregate epoch 1 has now supplied that first gameplay point. After 1,533
updates it reached legal KL 0.0680 and 96.2% retention, then correctly stopped
at the 0.05 operational gate. A fresh 5,000-game cap-1k distribution was a
wash versus the frozen actor: mean +3 [-9,+14], `<1000` -0.30pp
[-1.04,+0.44], and cap rate +0.02pp [-1.16,+1.20]. This rules out promotion
but not a later coordinated-distillation dose. Any resume now saves every 250
steps; do not make another full-epoch jump without intermediate models.

## Clean play/reanalyse/train loop

Each promoted iteration performs the following steps.

1. **Play.** Run many independent capped games from fresh environment seeds.
   Use a cap long enough to reach and sample the sustainable strong-play
   regime (1,000 turns is already sufficient), then spend additional compute
   on more games rather than longer continuations.  Retain every sampled broad
   state plus dense pre-death/crisis windows and their provenance.
2. **Explore separately.** Search may use temperature or root noise to find new
   trajectories, but record its action as `behavior_move`, never as the label.
3. **Clean reanalysis.** On retained states, run a noise-free, temperature-zero
   teacher at a declared model/value/search budget.  Store all 30 actions in the
   searched root support, visits, clean priors, Q estimates, `teacher_move`, and
   `base_move`.  Repeat a stratified subset with new search RNG/budgets to
   measure winner stability.
4. **Judge the edit class.** Use fixed-state common-random-number rollouts to
   establish that each selected *class* of disagreement improves survival.
   Increase simulations for ambiguous roots; retain unstable/near-tie rows as
   anchors but reduce their teacher dose.  Do not manufacture high-weight broad
   corrections while their measured uplift is zero.
5. **Train with the track-matched unit.** Relabel preservation anchors from
   `pi_k`. On the local-edit track, apply bounded corrections only to the
   validated class. On the aggregate track, distill complete rolling-search
   trajectories and judge the coordinated policy as a whole. Split
   train/validation by source game and archive the update trajectory; proxy
   loss never replaces gameplay.
6. **Evaluate and promote.** Compare 5,000-game score distributions.  Promote
   only when the mean/central distribution improves without an unacceptable
   floor regression.  Refresh all labels and anchors from the promoted policy;
   do not let stale hard targets become permanent truth.

Every state artifact should carry at least `source_game_id`, environment seed,
turn, source stratum, behavior-policy hash, teacher-policy/value hashes, search
configuration, `behavior_move`, `base_move`, `teacher_move`, and split group.
The build manifest should contain immutable input paths/hashes and explicit
caps; environment-variable defaults must not silently define a corpus.

The play stream has two declared modes.  An exploratory stream may use root
noise/temperature but needs a distinct clean label.  An exploit stream uses
zero root noise and temperature zero, so its behavior tree's visit winner is
already a clean label and no duplicate tree is needed.  Exploit games run to
the declared cap; independent environment seeds provide board diversity.
Both streams retain raw visit counts, clean log-priors, candidate Q values,
root value/range, and all 30 root candidates when recorded; normalized visits
alone are no longer considered adequate provenance.

The fixed-state judge answers whether one action is immediately cashable when
the old policy controls the continuation.  It does **not** falsify a coherent
multi-step search policy.  This distinction was already measured in the vh1
arc: isolated search substitutions were nearly neutral after the first
iteration, while rolling search still escaped most greedy deaths.  Therefore
the flywheel keeps two training tracks:

- a bounded local-edit track, where fixed-state causal validation is the right
  gate; and
- an aggregate trajectory track, where the student imitates every action and
  every state visited by rolling search across many capped games.  Its unit of
  evidence is whole-policy gameplay after coordinated distillation, not the
  single-action judge.

For the update-rich 128ch aggregate arm, consume every current-lineage target
and anchor but weight rather than discard.  The 19.6M greedy anchors should not
outvote the 3.1M rolling-search states merely because they are cheap: an
initial legacy-hard scratch control uses all anchors at 0.1 weight, all search
trajectories at their declared source weights, pooled row-level batches,
train-mode BN, corrected D4 and color symmetries, deployment-legal CE,
`bs=4096`, `lr=1e-3`, and roughly 700k optimizer updates with a long cosine
tail.  With the current group split, one unexpanded pass is 5,258 updates, so
`augment_factor=1` and about 133 epochs provide the historical update budget
without allocating an enormous 8x row permutation; each pass still samples a
fresh D4/color view.  This is a recipe hypothesis,
not a claim that the historical recipe was optimal.  New capped generation
should increase the search-trajectory share before this becomes the main
iteration trainer.

The aggregate experiment should include a matched no-D4 arm for enough updates
to measure sample efficiency.  Color permutation remains enabled in both.  If
corrected D4 is better, it replaces the historical corrupt augmentation.  If
no-D4 wins early, do not restore the corrupt transform: use canonicalization,
symmetry consistency regularization, or an equivariant/relational head instead.

Before the long arm, use a matched 2x2 mechanism audit (corrected D4 versus no
D4; full-softmax versus legal-support loss), with color permutation, corpus,
batch size, and optimizer updates fixed.  This is an optimization/symmetry
audit, not a 1k-game promotion tournament.  Carry at least the best full-loss
control and best legal-loss arm far enough into the long cosine regime, because
historical gameplay continued improving long after label match flattened.

If corrected augmentation still leaves a large symmetry tax, the next
small-capacity representation arm should quotient the symmetry instead of
adding width.  The low-risk first step is deterministic color canonicalization:
relabel colors by first occurrence in the board/preview before building the
observation; moves are color-invariant, so no policy remap is needed.  A joint
D4+color canonical form can then enumerate the eight spatial views, canonicalize
colors within each, choose a deterministic representative of the full state
(board plus ordered preview), and map source/destination actions back after
inference.  Symmetric-tie cases require stabilizer averaging rather than an
orientation-dependent tie break.  Python tensor building and C++ play/search
must share golden fixtures before this arm trains.  This offers exact symmetry
at one policy forward, whereas an eight-view ensemble is only a slow diagnostic.

## Value/search track

The policy loop can start with validated search actions, but the value target
must be formulated as a **continuing task**, not as a chess-like march toward a
distant terminal outcome.  Many capped trajectories sample the sustainable
state distribution efficiently.  A cap bootstraps from the current value (or
is treated as right-censoring); it is never a death label.

The existing short-horizon survival head is useful because it recognizes local
crises, but its arbitrary four-output weighted sum saturates.  Replace it with
two declared pieces: a calibrated short/medium failure-risk curve for escaping
danger, and a bootstrapped continuing/differential value for returning to and
maintaining the safe recurrent regime.  MCTS then acts as receding-horizon
control and replans on the next turn; it does not need a unique "late game" or
an uncapped demonstration to support indefinite play.

Before integrating a new value head, require all of the following on fixed
states:

- nontrivial output variance on both broad and crisis leaves;
- stable action ranking across independent rollout halves;
- positive rank correlation with fixed-horizon failure and bootstrapped
  continuing-return outcomes;
- a Q-weight sweep whose winner repeats on a fresh state set;
- gameplay improvement as a 5,000-game distribution.

Capped games contribute survival likelihood through their cap and bootstrap
the continuing target; they are not terminal wins or losses.  Evaluation
should report the cap-reaching fraction and restricted mean score/turns at a
fixed cap, alongside the ordinary score distribution.

## Diagnostic decision tree

- **Edits are not adopted:** optimization/sampling is too weak; increase edit
  exposure or duration while holding the functional KL budget fixed.
- **Edits are adopted but broad behavior drifts:** preservation coverage,
  BatchNorm, or optimizer coupling is failing; do not blame label quality yet.
- **Edits are adopted and preserved but gameplay is flat:** the selected
  teacher disagreement class is not causally useful enough; improve clean
  reanalysis and rollout gating.
- **Clean search wins fixed-state rollouts but cannot be distilled under a
  bounded update:** investigate representation/optimization and only then test
  architectural capacity.
- **Clean search itself stops winning:** improve the value target/search budget;
  more imitation data cannot create an improvement operator that is absent.

The first decisive experiment is therefore a controlled all-data comparison:
historical hard-CE/vector generation versus an explicit bounded hard-teacher
update, with the frontier used to validate and weight edit classes.  If the
bounded update installs validated edits while preserving broad behavior but
gameplay remains flat, the next bottleneck is the search/value target—not an
unsupported declaration that 128 channels are exhausted.
