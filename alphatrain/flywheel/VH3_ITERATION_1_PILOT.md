# vh3 iteration-1 clean pilot

This pilot freezes one 128-channel champion and one terminal-fixed teacher
configuration. It is designed to produce roughly 3.5--4.5M fresh labelled
states plus a deterministic 1M-state sample from the existing lineage-pure
vh3 replay anchors. All long stages are atomic/resumable and run under
`caffeinate -i -s`. No 192/256-channel games or labels enter this pilot.

Run Python commands from the repository root (`colorlines98`) with the project
venv. The generation launcher checks checkpoint/TS/evaluator SHA-256 values and
the reserved final seed range before starting. Native output is one atomic JSON
per completed game; rerunning the same command skips it. `run_config.json`
prevents a directory from accepting a changed search recipe.

## 0. Canary

Before the full ranges, run four exploit and four explore games into directories
which are deliberately outside the training manifest:

```bash
cd alphatrain/inference_cpp
caffeinate -i -s ./build/mcts_selfplay \
  --run-id vh3_i1_canary_exploit_s400_v1 \
  --model data/vh3_policy_ts.pt --device mps \
  --out-dir ../data/flywheel_vh3_i1_canary/exploit \
  --seed-start 2099900 --seed-end 2099904 \
  --sims 400 --batch-size 8 --top-k 30 --c-puct 2.5 --q-weight 1 \
  --temperature-moves 0 --dirichlet-alpha 0.3 --dirichlet-weight 0 \
  --max-turns 1000 --threads 4 --full-record

caffeinate -i -s ./build/mcts_selfplay \
  --run-id vh3_i1_canary_explore_b200_l400_v1 \
  --model data/vh3_policy_ts.pt --device mps \
  --out-dir ../data/flywheel_vh3_i1_canary/explore \
  --seed-start 2099910 --seed-end 2099914 \
  --sims 200 --clean-label-sims 400 --batch-size 8 --top-k 30 \
  --c-puct 2.5 --q-weight 1 --temperature-moves 15 \
  --dirichlet-alpha 0.3 --dirichlet-weight 0.25 \
  --max-turns 1000 --threads 4 --full-record
```

Check that all eight files parse, have complete Q/prior/root records, use fp16,
and that explore contains behavior/teacher disagreements. These canary states
are not later mixed into the corpus.

Before the first full crisis range, run and audit its separate eight-probe
canary. This exercises both bulk greedy probing and deep recovery/prevention
replay; it remains outside the training manifest:

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  .venv/bin/python -m alphatrain.scripts.run_flywheel_stream \
  --manifest alphatrain/flywheel/vh3_iteration_1_canary.json \
  --source canary_crisis_600_1600

NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.audit_flywheel_generation \
  --manifest alphatrain/flywheel/vh3_iteration_1_canary.json \
  --source canary_crisis_600_1600 --require-complete --sample-rows 50000 \
  --output alphatrain/data/flywheel_vh3_i1_canary/crisis_generation_audit.json
```

## 1. Generate the frozen pilot streams

The three commands may run sequentially or on separate machines. Each is safe
to interrupt and rerun.

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  .venv/bin/python -m alphatrain.scripts.run_flywheel_stream \
  --manifest alphatrain/flywheel/vh3_iteration_1_pilot.json \
  --source vh3_i1_exploit_400

NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  .venv/bin/python -m alphatrain.scripts.run_flywheel_stream \
  --manifest alphatrain/flywheel/vh3_iteration_1_pilot.json \
  --source vh3_i1_explore_200x400

NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  .venv/bin/python -m alphatrain.scripts.run_flywheel_stream \
  --manifest alphatrain/flywheel/vh3_iteration_1_pilot.json \
  --source vh3_i1_crisis_600_1600
```

The launcher itself wraps the native process in `caffeinate -i -s`.

## 2. Validate generation and build the target tensor

The first command is a streaming audit. `--require-complete` rejects missing
seeds, missing completion markers, mixed run IDs/search settings, corrupt JSON,
or incomplete full-search records. It also records per-stream game lengths,
cap rate, behavior/teacher agreement, visit entropy/margins, prior rank, Q
margin, and turn coverage from a deterministic row reservoir.
During staged generation, add `--source <manifest-source-name>` to audit a
completed stream before later streams exist; the final audit still covers all
three sources together.

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.audit_flywheel_generation \
  --manifest alphatrain/flywheel/vh3_iteration_1_pilot.json \
  --require-complete --sample-rows 200000 \
  --output alphatrain/data/flywheel_vh3_i1_pilot/generation_audit.json

NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.build_flywheel_corpus \
  --manifest alphatrain/flywheel/vh3_iteration_1_pilot.json --kind target \
  --resume-dir alphatrain/data/flywheel_vh3_i1_pilot/target_build_state \
  --checkpoint-every-files 25
```

The builder uses disk-backed arrays and checkpoints only after flushing them.
Rerunning resumes at the next source file. The final tensor is also written by
temporary-file rename.

## 3. Annotate exact base disagreements

This is diagnostic data and a fixed-stratum audit sidecar; training recomputes
the base legal action on each augmented view. Annotation resumes at the last
flushed inference batch and defaults to production fp16.

Full generator records also preserve `base_move`, the exact actor-root legal
clean-prior argmax. The sidecar stores both that behavior-time edit mask and a
separate batch-2048 deployment recomputation. Pilot training weights immutable
original-view edits with `recorded_disagree`; the recomputation remains an
inference-parity diagnostic.

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.add_fulllegal_mask \
  --tensor alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --base alphatrain/data/small128_vh3.pt --device mps --batch-size 2048 \
  --output alphatrain/data/flywheel_vh3_i1_pilot_base_policy.npz

NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.audit_flywheel_targets \
  --targets alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --base alphatrain/data/small128_vh3.pt --device mps --n-samples 50000
```

Do not start full training if clean/full-record coverage is below 100%, an
exploit file reports behavior noise/temperature, or target actions are illegal.

## 4. Three matched student arms

All arms use every fresh target row and the same deterministic 1M-row replay
sample. Source weights follow manifest order: exploit=1, explore=1, crisis=2.
Hard clean winners are primary because raw PUCT visits are not a deployment
policy. Corrected D4 and color augmentation are enabled, and CE/KL is
deployment-legal. The canary measured only 4.16% base/teacher disagreements;
at the declared 4:1 target/anchor row mix and anchor weight 0.1, a 16x
disagreement multiplier makes those edits about 40% of effective loss rather
than letting them disappear. Agreement rows and all anchors remain present.

First run a plumbing smoke by adding
`--max-target-states 200000 --max-anchor-states 50000 --epochs 3` to a copy of
each command and using a distinct save directory. These are not candidates.

Conservative warm control:

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.train_flywheel \
  --targets alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --anchors alphatrain/data/flywheel_vh3_anchors.pt \
  --base alphatrain/data/small128_vh3.pt \
  --base-policy-sidecar alphatrain/data/flywheel_vh3_i1_pilot_base_policy.npz \
  --sidecar-disagree-key recorded_disagree \
  --objective bounded --eta 0.2 --soft-alpha 0 \
  --source-weights 1 1 2 --base-agree-weight 0.25 \
  --base-disagree-weight 1 --max-anchor-states 1000000 \
  --epochs 40 --batch-size 4096 --lr 0.0001 --legal-support-loss \
  --color-augment --precision fp16 --resume-every-steps 250 \
  --save-every-steps 2500 --archive-every-epochs 2 \
  --save-dir alphatrain/checkpoints/vh3_i1_bounded_warm
```

Plastic warm student:

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.train_flywheel \
  --targets alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --anchors alphatrain/data/flywheel_vh3_anchors.pt \
  --base alphatrain/data/small128_vh3.pt \
  --base-policy-sidecar alphatrain/data/flywheel_vh3_i1_pilot_base_policy.npz \
  --sidecar-disagree-key recorded_disagree \
  --objective aggregate --soft-alpha 0 --anchor-weight 0.1 --update-bn \
  --source-weights 1 1 2 --base-agree-weight 1 \
  --base-disagree-weight 16 --max-anchor-states 1000000 \
  --epochs 550 --batch-size 4096 --lr 0.001 --legal-support-loss \
  --color-augment --precision fp16 --warmup-fraction 0.03 \
  --resume-every-steps 250 --save-every-steps 5000 \
  --archive-every-epochs 5 \
  --save-dir alphatrain/checkpoints/vh3_i1_aggregate_warm
```

From-scratch student: use the identical plastic command and add
`--from-scratch`, changing only `--save-dir` to
`alphatrain/checkpoints/vh3_i1_aggregate_scratch`. This isolates initialization
without changing 128-channel inference cost, corpus, loss, optimizer, or BN.

### Near-latency 18b96 absorption branch

Annotate the 18b96 epoch-30 base separately in deployment FP16. This sidecar
proves base-checkpoint and corpus provenance and supplies candidate-relative
diagnostics:

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.add_fulllegal_mask \
  --tensor alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --base alphatrain/data/scratch18b96_lr3e3_ckpts_epoch_30.pt \
  --device mps --batch-size 2048 \
  --output alphatrain/data/flywheel_vh3_i1_pilot_18b96e30_base_policy.npz
```

The teacher differs from vh3 on 4.14% of rows but from 18b96e30 on 19.42%.
These are not interchangeable definitions of a search edit. The exact overlap
is:

- both bases differ from teacher: 103,135 rows (2.642%);
- vh3 only differs: 58,504 rows (1.499%);
- 18b96 only differs: 654,802 rows (16.774%);
- neither differs: 3,087,170 rows (79.085%).

Thus 86.4% of the candidate-relative disagreement stratum is merely an
ordinary vh3 action that 18b96 has not imitated, not a vh3 search correction.
The primary capacity test must expose both architectures to the exact same
intervention. It uses the candidate sidecar to validate the 18b96 base but the
vh3 actor-root sidecar to choose the same 1x/16x weighted rows:

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.train_flywheel \
  --targets alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --anchors alphatrain/data/flywheel_vh3_anchors.pt \
  --base alphatrain/data/scratch18b96_lr3e3_ckpts_epoch_30.pt \
  --base-policy-sidecar \
    alphatrain/data/flywheel_vh3_i1_pilot_18b96e30_base_policy.npz \
  --sidecar-disagree-key disagree \
  --edit-weight-sidecar \
    alphatrain/data/flywheel_vh3_i1_pilot_base_policy.npz \
  --edit-weight-key recorded_disagree \
  --objective aggregate --soft-alpha 0 --anchor-weight 0.1 --update-bn \
  --source-weights 1 1 2 --base-agree-weight 1 \
  --base-disagree-weight 16 --max-anchor-states 1000000 \
  --epochs 550 --batch-size 4096 --lr 0.001 --legal-support-loss \
  --color-augment --precision fp16 --warmup-fraction 0.03 \
  --resume-every-steps 250 --save-every-steps 5000 \
  --archive-every-epochs 5 \
  --save-dir alphatrain/checkpoints/18b96_i1_aggregate_vh3edit_warm
```

Candidate-specific mass matching remains a secondary diagnostic: it asks how
well 18b96 can move toward the full vh3 teacher, not how well it absorbs the
same search edits. Derive that secondary arm's weights with:

```bash
.venv/bin/python -m alphatrain.scripts.derive_matched_edit_weights \
  --state-dir alphatrain/data/flywheel_vh3_i1_pilot/target_build_state \
  --reference-sidecar alphatrain/data/flywheel_vh3_i1_pilot_base_policy.npz \
  --candidate-sidecar \
    alphatrain/data/flywheel_vh3_i1_pilot_18b96e30_base_policy.npz \
  --reference-key recorded_disagree --candidate-key disagree \
  --source-weights 1 1 2 --reference-agree-weight 1 \
  --reference-disagree-weight 16 \
  --output alphatrain/data/flywheel_vh3_i1_pilot_18b96e30_matched_dose.json
```

For this corpus the exact secondary-arm weights are agreement
`1.1880338285875751` and disagreement `3.5102420706199298`; it then matches the
vh3 arm's total target mass and assigns 41.579% of mass to its own much broader
candidate-relative stratum. Do not interpret that arm as the clean capacity
comparison.

For long local runs, `--stop-after-epoch N` is an operational gate: the
scheduler still uses the full `--epochs` budget, a resumable `latest.pt` is
saved, and the process exits cleanly after epoch N. The flag may change on
resume and therefore does not alter the optimization problem.

Resume any arm by rerunning its exact command with
`--resume <save-dir>/latest.pt`. Mid-epoch resume restores optimizer, scheduler,
CPU/device RNG, deterministic row order, next batch, and partial diagnostics.

## 5. Checkpoint gates

Run the fixed held-out functional audit before gameplay:

```bash
NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache caffeinate -i -s \
  .venv/bin/python -m alphatrain.scripts.audit_flywheel_checkpoints \
  --targets alphatrain/data/flywheel_vh3_i1_pilot_targets.pt \
  --anchors alphatrain/data/flywheel_vh3_anchors.pt \
  --base-policy-sidecar alphatrain/data/flywheel_vh3_i1_pilot_base_policy.npz \
  --sidecar-disagree-key recorded_disagree \
  --base alphatrain/data/small128_vh3.pt --device mps --batch-size 1024 \
  --rows-per-source 50000 --models <checkpoint...>
```

It reports each source plus exact base-agree, disagreement, and confident-
disagreement strata. Select by adoption/KL/retention trajectories only to reject
broken checkpoints. Screen survivors on 999 games and promote only from a
5,000-game independent-distribution comparison against vh3. Never use seedwise
score deltas; report independent-bootstrap mean/median/tail intervals and the
turn-survival curve.

For the primary 18b96 arm, retain the 18b96 sidecar as
`--base-policy-sidecar` and add the vh3 sidecar as `--strata-sidecar` with
`--strata-key recorded_disagree`. This reports both models on the identical
search-edit subset while still checking each frozen base's SHA-256 provenance.
