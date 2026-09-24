# 18b96 epoch-40 iteration-1 pilot

Status: corpus complete; no student promoted. Search is stronger, but broad
search disagreements failed fresh gameplay validation. Rollout mining isolated
crisis edits as the repeatable class; more independent crisis data is the next
step, 2026-08-16. The reserved final seed bank remains untouched.

This is the first closed-loop pilot whose actor, search priors, training
targets, preservation anchors, and eventual student all belong to the same
18-block x 96-channel lineage. No vh3, 192/256-channel, Pillar3f, or Pillar3k
games are labels in this iteration.

The actor is `scratch18b96_lr3e3_ckpts_epoch_40.pt`. On independent 5,000-game
distributions it is unresolved against vh3 at a 1,000-turn cap, but is clearly
stronger in long play: mean +907 [approximately +358,+1456], median +766
[+275,+1246], P90 +2669 [+522,+4423], and `>10000` +3.06 percentage points
[+1.12,+4.98]. Its native FP16 search throughput is in the 128-cost class.
That makes it the current smallest credible actor, not yet a promoted
self-improving lineage.

The immutable manifest is
`alphatrain/flywheel/18b96e40_iteration_1_pilot.json`. The launcher verifies
the checkpoint, TorchScript policy, and feature-value hashes and refuses a
reserved final-evaluation seed overlap. Every slow command runs under
`caffeinate -i -s`; all policy inference is MPS FP16.

## Completed canary

The disjoint canary used four exploit games, four explore games, and eight
10,000-turn crisis probes. Its strict audit covered 20 files and 11,219 rows:

- all JSON, actions, full root records, and clean-label declarations passed;
- explore behavior/teacher agreement was 96.53%, proving label separation was
  active rather than merely configured;
- six greedy probes died and two reached the 10,000-turn probe cap;
- only one of six recovery replays survived 500 turns, while five of six
  prevention replays survived 500 turns;
- failed crisis tails had 93.55% teacher/prior agreement, or about 6.45%
  search corrections.

The prevention result is a causal signal for the crisis trajectory class. It
does not prove that every broad root edit is useful.

## Completed actor-native anchors

Seeds 2,205,000--2,209,999 produced 5,000 independent greedy games capped at
1,000 turns. The final distribution was mean 1,934, mean turns 958.9, and
4,464/5,000 (89.28%) capped. Recording kept every fourth broad state and the
complete final 160-turn window.

The audited tensor is
`alphatrain/data/flywheel_18b96e40_i1_anchors.pt`:

- 1,798,692 states from exactly 5,000 source games;
- 89,763 validation rows (4.99%), split by whole source-game seed;
- 359/370/370 rows per game at P10/P50/P90;
- native final score, turn count, and cap/death metadata retained.

The builder accepts this architecture only because the manifest explicitly
declares `lineage_family=small_policy` and the exact generator prefix. It still
rejects foreign-model sources, and verifies the base checkpoint hash.

## Completed target generation

The streams ran sequentially so only one MPS search workload was active. Each
native generator wrote one atomic JSON per game and skipped completed seeds on
restart. `run_config.json` prevents changing a recipe inside an existing
directory. The exact replay command was:

```bash
cd alphatrain

PYTHONPATH=.. NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  caffeinate -i -s ../.venv/bin/python \
  -m alphatrain.scripts.run_flywheel_stream \
  --manifest alphatrain/flywheel/18b96e40_iteration_1_pilot.json \
  --source 18b96e40_i1_exploit_400 \
  --source 18b96e40_i1_explore_200x400 \
  --source 18b96e40_i1_crisis_600_1600
```

The declared ranges are 3,000 exploit games at 400 simulations, 750 explore
games with 200-simulation noisy behavior plus a separate 400-simulation clean
label, and 500 crisis probes with 600-simulation recovery and 1,600-simulation
prevention. Games are capped; no uncapped demonstrations are needed.

### Generation result

The strict combined audit accepted all 4,528 files and 3,903,888 rows with no
invalid targets, incomplete records, mixed settings, or provenance failures:

- 2,945,600 full-record clean states and 3,000 unique source seeds;
- 2,851/3,000 games (95.03%) reached the 1,000-turn cap;
- search versus independent greedy anchors improved cap rate by 5.75
  percentage points, with an independent Bernoulli-bootstrap 95% interval of
  [+4.59,+6.91];
- the search winner changed the actor-prior top action on 2.32% of all rows,
  2.66% of failed-game rows, and 6.04% of failed final-20 rows;
- only 5.58% of search winners were maximum-Q among the visited candidates, so
  raw Q argmax is still not a justified teacher target.
- explore contributed 723,704 rows from 750 games, capped 91.2%, and kept
  noisy behavior separate from the clean teacher; the two agreed on 96.51%
  of rows;
- 500 crisis probes produced 389 deaths and 111 10,000-turn caps. Their 778
  recovery/prevention replays contributed 234,584 rows. Recovery survived its
  500-turn replay in 104/389 cases (26.7%), while prevention survived in
  335/389 (86.1%). Because the replay starting states differ, this is strong
  trajectory-class evidence rather than a paired single-action causal claim.

This establishes rolling 400-simulation search as a materially stronger
on-policy actor. It does not establish that isolated broad edits are causal;
the aggregate-distillation and crisis-local hypotheses remain distinct.

## Completed tensor and base annotation

The resumable build produced
`data/flywheel_18b96e40_i1_pilot_targets.pt`: 3,903,888 rows with a
group-held-out validation split of 196,290 rows (5.03%). Failed tails retain a
0.25 target weight; every row remains represented. The exact MPS-FP16 base
sidecar is `data/flywheel_18b96e40_i1_pilot_base_policy.npz`. It agrees with
the recorded actor-root action on 99.88% of rows; the independent batched
deployment recomputation differs from the teacher on 99,044 rows (2.54%).

The corpus composition is:

- exploit: 2,945,600 rows, 68,318 actor/search edits (2.32%);
- explore: 723,704 rows, 16,570 edits (2.29%);
- crisis: 234,584 rows, 9,603 edits (4.09%);
- prevention supplies 174,090 crisis rows and 7,437 edits; recovery supplies
  60,494 rows and 2,166 edits.

The commands below are retained as the reproducible rebuild procedure:

```bash
PYTHONPATH=. NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  caffeinate -i -s .venv/bin/python \
  -m alphatrain.scripts.audit_flywheel_generation \
  --manifest alphatrain/flywheel/18b96e40_iteration_1_pilot.json \
  --require-complete --sample-rows 200000 \
  --output alphatrain/data/flywheel_18b96e40_i1_pilot/generation_audit.json

PYTHONPATH=. NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  caffeinate -i -s .venv/bin/python \
  -m alphatrain.scripts.build_flywheel_corpus \
  --manifest alphatrain/flywheel/18b96e40_iteration_1_pilot.json \
  --kind target \
  --resume-dir alphatrain/data/flywheel_18b96e40_i1_pilot/target_build_state \
  --checkpoint-every-files 25
```

Reject the corpus if any stream has missing seeds/markers, mixed run settings,
invalid actions, incomplete root records, noisy exploit labels, or an
unexpected model/hash. Preserve all 30 root candidates, visits, priors, Qs,
source/game/seed/turn provenance, and the group-held-out split.

## Training and first gameplay result

Do not reuse the vh3 labels. The vh3 experiments showed that target fitting is
not sufficient: a mixed visit target with 8x edit exposure reduced exact
target residuals by 8--10% yet regressed the independent cap rate by 0.70
percentage points, with a confidence interval spanning a small win. More vh3
optimizer tuning is therefore not the next experiment.

The first aggregate smoke produced a nominal development-bank win, including
mean +14 [+2,+25] and `<1000` -0.86pp [-1.60,-0.12]. It did not replicate:
the previously unused 5k bank was mean +7 [-4,+18], and a second deterministic
training subsample was mean -5 [-16,+5]. A 4.7M-row dose-matched run and its
lower-dose checkpoint were also washes. The original result was selection
noise, not a promoted flywheel step.

An edit-only bounded arm removed hard CE from the 97.6% teacher-agreement rows.
At higher LR it too won the development bank (mean +15 [+3,+26], `<1000`
-1.02pp [-1.74,-0.28]) and failed a new 5k bank (mean -3 [-14,+8], cap
-0.74pp [-1.88,+0.42]). Thus neither broad aggregate trajectories nor all
search disagreements define a generalizable training target. No student is
promoted, and the reserved final bank remains untouched.

The native CRN judge then tested 924 held-out edits by source class. Broad
exploit edits were neutral. A fresh-seed, 64-repetition rejudge confirmed the
combined crisis class: death-within-200 uplift +1.24pp [+0.68,+1.84] and
teacher-turn delta +2.18 [+1.25,+3.21]. Recovery was strongest locally at
+2.02pp [+1.06,+3.04] and +3.81 turns [+2.07,+5.74]. Prevention retained a
positive turn delta, consistent with its much stronger complete-trajectory
replay, while explore edits became neutral on fresh rollout seeds.

The immutable mined mask therefore contains only 9,603 crisis edits (7,437
prevention and 2,166 recovery). Training diagnostics now reserve every one of
the 9,094 training-split edits before adding preservation rows. Low-LR 18b96
reduces exact target residual on training edits from 0.01264 to 0.01119, but
does not improve held-out residual (0.01169 to 0.01175); high LR is worse even
in-sample. This shows limited generalization from a small, trajectory-correlated
edit set, not an inability to fit any correction and not yet a channel-capacity
proof.

The next data action is to expand actor-native crisis mining on new seeds until
there are tens of thousands of independent recovery/prevention edits. Keep
neutral exploit/explore rows only as frozen-policy KL coverage. Re-run the
train/held-out exact-target audit before gameplay; capacity becomes implicated
only if a substantially more diverse validated edit corpus retains the same
gap, preferably with a larger architecture as a positive control. Matching
game seeds continue to be analysed as independent distributions; CRN pairing
is used only for fixed-state causal action judging.

## Crisis expansion and capacity control

The expansion completed on seeds 2,210,000--2,211,999 under MPS FP16 and
`caffeinate`: 2,000 probes produced 1,478 deaths, 2,956 recovery/prevention
games, 905,645 clean full-record states, and 36,717 actor/search edits. Strict
audit found zero provenance or target errors. The combined pilot+expansion
artifact has 1,140,229 states, 46,320 edits, 1,867 death-seed groups, and a
leak-free 5.01% held-out split; recovery and prevention siblings always share
the same split.

A fresh expansion-only CRN judge sampled 300 prevention and 300 recovery
edits, with 64 repetitions per arm and horizon 200. It independently repeated
the old result: overall death uplift +1.31pp [+0.82,+1.81] and teacher-turn
delta +2.26 [+1.46,+3.12]. Prevention was +0.84pp [+0.35,+1.36]; recovery was
+1.77pp [+0.94,+2.64]. Thus the search-action class is causal and the new
corpus is not merely more unvalidated MCTS output.

Full-data bounded training nevertheless failed the held-out target gate.
18b96 trained on all 1.14M target states plus all 1.80M anchors, with
`eta=0.05`, 8x declared-edit exposure, LR 1e-5, frozen BN, legal-support loss,
and no D4 augmentation. Its endpoint reduced training-edit target KL from
0.00399 to 0.00392, but worsened held-out KL from 0.00381 to 0.00430; every
archived dose was worse than the untouched base. No gameplay was run.

The larger positive control used scratch192 epoch 39 as its own frozen base
and the identical immutable 18b96 edit mask. It reduced training-edit target
KL from 0.00921 to 0.00875, but its best held-out checkpoint was 0.00869 versus
base 0.00841 and the endpoint was 0.00915. The bases differ, so this is an
absorption control rather than a lineage or gameplay comparison. Extra width
improves in-sample plasticity but does not make sparse search edits
generalizable; a simple 18b96 capacity explanation is rejected for this
objective.

The next **local-edit** iteration should not apply equal one-hot CE to all
disagreements. In the 600-state causal judge, 89.3% were rollout ties, 8.5%
clearly genuine, and 2.2% phantom. A later local mine should judge several
thousand edits cheaply, derive rollout-advantage weights, and train a pairwise
teacher/base ranking loss or an advantage/value head under the same anchor KL.
Top-share bins are a candidate routing feature, but thresholds require a fresh
predeclared test.

That diagnostic branch is not a replacement for the planned aggregate
trajectory arm. The non-overlapping exploit, explore, pilot-crisis, and
expansion-crisis streams have now been combined into
`data/flywheel_18b96e40_i1_mixed_targets.pt`: 4,809,533 unique same-actor
search states with 239,905 validation rows, plus all 1,798,692 actor-native
anchors. The primary next run is the resumable warm-start Colab notebook
`train_flywheel_18b96e40_i1_colab.ipynb`, using full hard clean-search
trajectory CE, frozen-base anchor KL, and source weights `1/1/2/2`. This tests
whether coordinated rolling-search behavior can be distilled even though
isolated broad edits are neutral. Sparse-edit failure cannot answer that
whole-policy question. The reserved final seed bank remains untouched.

The first full Colab epoch completed 1,533 optimizer updates before the
operational gate stopped it at legal KL 0.0680 versus the conservative 0.05
limit. Retention remained 96.2%, so this was a usable high-dose checkpoint,
not a corruption. On a fresh independent 5,000-game cap-1k distribution it
was a wash against e40: mean +3 [-9,+14], median +2 [+0,+4], `<1000`
-0.30pp [-1.04,+0.44], and cap rate +0.02pp [-1.16,+1.20]. It is not
promoted. Subsequent resumed epochs must archive every 250 steps so the next
dose is observed within an epoch rather than only after another 1,533-update
jump.
