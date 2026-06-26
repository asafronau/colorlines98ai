# Small-model policy distillation is too slow/weak — peer review (v2, full context)

**Domain:** Color Lines 98 (9×9 board, 7 colors, stochastic i.i.d.-uniform ball spawns). Single-player survival game; score ≈ turns survived; no turn cap; effectively infinite for a strong policy. Deployed model is **policy-only, greedy at inference** (argmax over legal moves), no search.

## Goal
Compress the best policy **pillar3k** (PreAct-ResNet, **10 blocks × 256ch, ~11.9M params**, mean ≈ **43,000** over 5k seeds) into a **4× smaller student (10 blocks × 128ch, ~3.0M params)** for browser deploy + faster generation. A few-% score loss is fine. (Param counts: 256ch=11.9M, 192ch=6.7M, 128ch=3.0M.)

## Method (current)
- **Corpus:** 3.85M board states (broad self-play "normal" play + crisis/near-death states), each **relabeled with pillar3k's top-5 legal-move policy** (softmax over legal moves; top-5 indices+probs stored). Targets are *deterministic* (teacher is fixed).
- **Train (`train_path_b`):** student matches the teacher's policy via soft cross-entropy (sparse top-5 target scattered into 6561-logit space, then soft-CE). From scratch. `batch=4096, lr=1e-3, 3-epoch warmup + cosine, color+dihedral aug (×8), target_temperature=0.5`. Arch: stem(conv→BN→relu) → 10×[BN→relu→conv→BN→relu→conv +residual] → BN→relu → policy head(conv 256→128, BN, conv 128→81) → 6561 logits.

## The symptom
The student **learns but slowly, and underfits the TRAINING distribution**, converging far short of the teacher. Best run (batch 4096, T=0.5, cosine):

| epoch | argmax-match to teacher | top-3 match | gameplay mean |
|---|---|---|---|
| 5  | 18.9% | 39.9% | 1,835 |
| 10 | 22.8% | 44.6% | 4,692 |
| 15 | 23.9% | 45.9% | 7,414 |

- Gameplay climbs (~linear early) but is ~**10–17% of the teacher's** (43k). argmax-match **decelerates and seems to plateau ~24%**, measured **on the training states themselves** (so this is underfitting, not just distribution shift).
- Teacher's policy is **soft** (mean top-move prob ≈ 0.34, many near-ties), so exact-argmax may understate the student — but a 24% match + 4–7k gameplay after many epochs is the core concern.
- The student's output is **much flatter** than the teacher's (logit-std across states ≈ 1.5–3 vs teacher's 15.9 when healthy).

## What we've ALREADY tried and RULED OUT (please don't re-suggest)
1. **Batch / optimizer-step starvation — FIXED.** Big batch (32768/65536) gave ~470 steps/epoch and barely learned; from-scratch convergence is gated by # steps. Dropped to **batch 4096** (~7,300 steps/epoch, the project's proven from-scratch batch) → the climb above. (Confirmed: bigger batch is worse here.)
2. **Target temperature — NON-ISSUE.** T=1.0 and T=0.5 give ~identical early gameplay (ep8: T=1.0→3,030, T=0.5→~3,100). Temperature is not the lever.
3. **Flat-LR (Hinton-KD-style "soften + hold LR flat 70%") — CATASTROPHIC.** With T=1.0 + lr held flat at 1e-3 for ~62 epochs, the model was healthy at ep8 (mean 3,030) but **collapsed by ep86**: val_loss 1.647 ("best") yet gameplay mean **1**, logit-std 1.5 (flat/degenerate). A from-scratch net held at high LR with no decay drifts into a flat minimum. (Lesson: **val_loss / KL is a TRAP here** — a flat policy minimizes soft-CE but plays dead; track gameplay.) Reverted to plain cosine.
4. **Truncation warm-start (copy teacher's first-128 of 256 channels + √2 scale reduced-input convs + BN recalibration) — FAILED.** Verified *before* training: argmax-match to teacher = **0.2%, identical to random init**. Trained-ResNet channels aren't importance-ordered, so a front-slice isn't a working sub-network. (No cheap warm-start available — widths differ.)
5. **MCTS-visit targets** — deprioritized (this is compression, not improvement; teacher is policy-only/greedy anyway).
6. **Capacity** — a prior reviewer argued 3.0M is plenty for a 9×9 local-feature board (not Go). We tentatively agree (gameplay still climbing, not flat-plateaued), but a 192ch (6.7M) control is cheap to run if you think capacity is the wall.

## The core question
**Why does from-scratch distillation of a strong greedy policy into a 4×-smaller net underfit the training distribution (~24% argmax / 4–7k vs 43k gameplay) and converge so slowly — and what is the highest-leverage fix?** We can't warm-start (width mismatch) and step-count is already maxed (batch 4096). Specific candidates we'd like graded:

1. **Feature/hint distillation (FitNets-style):** also match the teacher's *intermediate activations* (with a learned 128→256 projection), not just the output policy. Is this the standard cure for slow small-model distillation here, and where should the hint losses attach (after each block? backbone output?)?
2. **DAgger / on-policy relabel:** the student errs → reaches states the teacher's training corpus never covered → no signal there. Relabel the *student's own* game states with the teacher and add them. Worth it given the underfit is partly on-distribution?
3. **Just train far longer:** is ~24% match at ep15 actually on track to 70–85% by ep150–300, i.e. is this just the expected slow tail of from-scratch compression, and we should be patient (it's still climbing)?
4. **Architecture:** for a fixed ~3M budget, is 10×128 the wrong shape — would deeper-narrower, wider-shallower, or a different head help a small net mimic a deep teacher?
5. **Loss:** pure soft-CE on top-5. Would full-distribution KL (all 6561, temperature-scaled), or adding a hard-CE-on-teacher-argmax term, materially help the *argmax* fidelity that greedy play needs?
6. **Is 4× simply too aggressive** for this policy, such that 192ch (6.7M, 1.77× smaller) is the realistic floor?

## Data we can produce on request
Per-epoch match+gameplay curves; KL/val curves; a 192ch control; FitNets feature-distillation run; DAgger run; full-KL vs top-5 ablation; the teacher and a few student checkpoints.
