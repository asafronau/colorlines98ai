# Peer review request: crisis-fork distillation for floor improvement (Color Lines 98)

## TL;DR for the reviewer

We tried to lift the **worst-case floor** of a policy-only NN by distilling
"avoid this move" lessons mined from near-death positions. We found and fixed a
BatchNorm bug along the way. The clean result: **the signal overfits and does
NOT move the floor** (epoch-1 floor flat vs baseline; epoch-2 broadly regresses).
We want your read on whether this approach is worth continuing (more forks +
regularization) or has a structural low ceiling, and whether a better teaching
signal exists.

## System context

- **Game:** Color Lines 98 (9×9, 7 colors). Place a ball each turn; lines of ≥5
  clear and score. Spawns are i.i.d.-uniform. Game ends when the board fills.
- **Agent:** policy-only ResNet (10 blocks × 256 ch, 11.9M params, 18-ch board
  obs → 6561 flat move logits). Deployed as argmax/sampled policy (browser/WASM
  target) — **no inference-time search or second model.**
- **Baseline `pillar3b`** (2000 seeds 777000–778999, played to natural death, no
  turn cap): **mean 17,386; P5 1,430; P10 2,348; P50 12,186; max 134,364;
  `<1000` 2.9%.** Trained by distillation from MCTS visit targets with target
  sharpening (T=0.5).
- **Key established facts** (from extensive prior autopsy): score ≈ survival
  (~2 pts/turn at all skill levels), so the floor = **early density-spiral
  deaths**. Of those, roughly **1/3 are RNG-sealed** (unavoidable given the
  spawn draws) and **~2/3 have an avoidable "crisis fork"** — a single move,
  ~15–45 plies before death, where the policy's choice is materially riskier
  than an alternative (validated held-out and by a human in a GUI).

## What we built

1. **Mining (rewind-from-death).** Play a game to natural death; rewind to each
   depth d in [15,45]; for each candidate move (policy top-K ∪ a feature-value
   survival-evaluator's top-K), roll out R times under the policy and measure
   **catastrophe = "died within H=300 plies of the rewind point"**. An adaptive
   band keeps only depths where the policy move's catastrophe ∈ [15,85]%
   (recoverable crisis, not safe and not sealed).
2. **Screen → confirm (winner's-curse guard).** R=100 screen flags a fork if
   some candidate beats the policy move by ≥10pp catastrophe; R=500 fresh seeds
   + paired bootstrap **confirm** only forks whose 95% CI excludes 0.
3. **Corpus.** 110 games → **292 confirmed forks** (2.65/game, 89% of games
   yield ≥1). Each fork = (board, next_balls, policy_move, confirmed_safe_move,
   other clearly-worse moves).
4. **Teaching (listwise margin aux loss).** Warm-start `pillar3b`, continue the
   V12 distillation (target-temp 0.5), and add an auxiliary hinge that pushes
   `logit[safe] > logit[policy_move] + margin` and above clean losers, on a
   **separate forward** over the fork corpus. Each fork weighted by its
   confirmed catastrophe gap (anti-dilution). λ warmup 0→0.15 over 2 epochs.
   This is the same machinery as our earlier failed "pillar3c" run, but fed
   R=500 CI-confirmed forks instead of R=24 noisy labels.
5. **Held-out fork metric (by SEED split).** We watch, on forks from games never
   trained on: `flip` (fraction where logit[safe] > logit[policy_move]), the
   logit margin, and concordance vs clean losers. We trust this over validation
   CE (a prior run regressed while val CE looked fine).

## The BatchNorm bug we found and fixed

The aux's separate train-mode forward over **OOD crisis boards** (dense
late-game) was **poisoning BN running mean/var** — and inference uses those
running stats. Splitting the forward fixed the *loss* contamination but not the
running-stat *update*. Evidence (val on V12 targets, 3 ways):

| checkpoint | eval-BN (running stats) | train-BN (batch stats) | BN recomputed on clean data |
|---|---|---|---|
| baseline / no-aux control | 2.18 | 2.18 | 2.18 |
| aux (buggy) | **3.18** | **2.19** | **2.19** |

So the weights were fine; only the running stats were poisoned (recomputing them
fully recovers val). **Fix:** freeze BN (eval-mode, no stat update) during the
aux forward, routed through the uncompiled module (the BN toggle is unreliable
under `torch.compile`). This very likely contributed to our earlier pillar3c
regression too, which we'd blamed on label noise.

## Clean results (post-fix)

**Subsample verify (held-out fork trajectory):** train forks → **100% flip,
margin +3.5** by epoch 2 (memorized); **held-out flip peaks at epoch 1 (~0.20,
margin −2.99→−1.28) then DECAYS back to baseline by epoch 5.** Classic
overfitting on 233 train forks at λ=0.15. Val stays ~2.2 (fix confirmed).

**Full-corpus run, 2000-seed floor eval (paired, natural death):**

| metric | pillar3b | epoch 1 | epoch 2 |
|---|---|---|---|
| mean | 17,386 | 17,974 (+3.4%) | 14,154 (−18.6%) |
| P5 | 1,430 | 1,440 (flat) | 1,144 (−20%) |
| P10 | 2,348 | 2,312 (flat) | 1,880 (−20%) |
| P50 | 12,186 | 13,007 (+6.7%) | 10,233 |
| `<1000` | 2.9% | 2.6% (flat, within noise) | 4.0% (worse) |

- **Epoch 1:** floor **statistically flat** (`<1000` SE ≈0.4pp at n=2000; P5/P10
  unchanged). Small mean/median uptick, but not cleanly attributable to the aux
  vs simply one more epoch of sharpened distillation (no full-corpus no-aux
  epoch-1 control yet).
- **Epoch 2:** clear broad regression (mean −19%, ≈8σ) — the overfit damaging
  the whole policy.

## Fork-count learning curve (the decision gate)

We measured peak held-out flip/margin vs number of training forks K (fixed 59
held-out forks, same recipe + BN fix):

```
[FILL IN: K vs peak held-out flip table from scripts/fork_learning_curve.py]
```

Interpretation we'll draw: rising at K=233 → mining more forks could push
held-out generalization high enough to move the floor; flat by K~150 → forks
aren't the lever.

## Questions for you

1. **Is this worth continuing?** Held-out generalization peaks at ~20% flip
   (margin −2.99→−1.28) on 59 held-out forks and the gameplay floor didn't move.
   Given ~1/3 of floor deaths are RNG-sealed and the avoidable forks are a
   subset of crisis states, is there a plausible path where per-move fork
   distillation meaningfully moves a 2.9%-`<1000` floor, or is the ceiling
   structurally low?
2. **Overfit vs generalization.** Train flips to 100% while held-out peaks then
   decays. Is "more forks + lower λ + early-stop" the right read, or does the
   peak-then-decay pattern indicate the forks are too idiosyncratic to
   generalize (each crisis board is nearly unique)?
3. **Better teaching signal?** We use a listwise hinge on argmax order. Would a
   **soft target** (e.g., distill a catastrophe-temperature-weighted move
   distribution from the R=500 rollouts) generalize better than a hard
   margin push? Or a small **risk head** consumed as an auxiliary feature rather
   than reshaping the policy logits directly?
4. **Wrong target?** Prior autopsy suggested the real lever may be **global
   board-health / openness drift** in the *safe* region (which saturates and
   isn't per-move-fixable), not the last-hope crisis forks. Should we redirect
   to a board-openness objective instead of crisis forks?
5. **Mining economics.** ~2.65 confirmed forks/game; reaching ~1000 forks needs
   ~400 games (hours of GPU). Worth it conditional on the learning-curve slope?

## Artifacts (for reference)
- Mining: `scripts/mine_crisis_sweep.py`, `scripts/batched_rollout.py`
- Training: `alphatrain/train_path_b.py --aux-crisis-corpus`,
  `alphatrain/counterfactual.py` (`build_crisis_corpus`, `listwise_margin_loss`,
  `frozen_bn`)
- Eval: `scripts/eval_fork_ranking.py`, `scripts/fork_learning_curve.py`,
  `scripts/check_bn_bug.py`
