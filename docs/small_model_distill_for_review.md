# Small-model policy distillation underperforming — peer review

**Domain:** Color Lines 98 (9×9, 7 colors, stochastic i.i.d.-uniform ball spawns). Single-player survival game; score ≈ turns survived; no turn cap; effectively infinite for a strong policy. The deployed model is **policy-only, greedy at inference** (argmax over legal moves), no search.

## Goal
Compress the best policy **pillar3k** (ResNet, 10 blocks × 256 ch, ~11.9M params, mean ≈ 43,000 over 5k seeds) into a **4× smaller** student (10 blocks × 128 ch, ~3.0M params) for browser deployment + faster self-play generation. A few-% score loss is acceptable.

## Method (policy distillation, from scratch)
- **Corpus:** 3.85M board states (broad self-play "normal" play + crisis/near-death states), each **relabeled with pillar3k's top-5 legal-move policy** (softmax over legal moves, top-5 indices + probs stored).
- **Train:** student matches the teacher's policy via soft cross-entropy. `batch=4096, lr=1e-3, 3-epoch warmup + cosine decay over 40 epochs, color+dihedral augmentation (×8), target_temperature=0.5 (sharpens the teacher target).` From scratch (no warm-start — architecture differs from any 256-ch checkpoint, so teacher weights can't be copied).

## Symptom: it learns, but slowly, and seems to be converging well short of the teacher
Per-epoch, **argmax-match to teacher** (student's top legal move == teacher's top legal move, measured on the training states) and **greedy gameplay mean**:

| epoch | argmax-match | top-3 match | gameplay mean |
|---|---|---|---|
| 5  | 18.9% | 39.9% | 1,835 |
| 10 | 22.8% | 44.6% | 4,692 |
| 15 | 23.9% | 45.9% | 7,414 |

- Gameplay is **still climbing** (~linear, +2,700/5 epochs; max single game 40,818) — not a flat plateau.
- But argmax-match is **decelerating hard** (+3.9 then +1.1) and the absolute match (~24%) is low for a 4× distillation. Note: the teacher's policy is **soft** (mean top-move probability ≈ 0.34 — many genuine near-ties), so exact-argmax match may understate the student.
- This is **underfitting on the training distribution itself** (not just distribution shift): ~24% argmax-match after ~113k optimizer steps.

## Prior mistakes already ruled out (so you can skip them)
- It is **not** step-starvation: an earlier run at batch 32768 gave ~470 steps/epoch and barely learned (10% match); dropping to batch 4096 (~7,500 steps/epoch, the project's proven from-scratch recipe) is what produced the climb above.
- It is **not** a load/arch bug: eval auto-detects the 128-ch arch and the student plays real games.
- Temperature was tested at T=1.0 and T=0.5 **only while step-starved**, so neither is a clean result.

## Our read
We suspect it is **NOT capacity** (knowledge distillation at 4× usually transfers well; the teacher itself is still improving via its own iteration loop; gameplay is still climbing). We suspect the **recipe/approach** is suboptimal for from-scratch distillation.

## Questions
1. **Target temperature (primary):** We *sharpen* the teacher target (T=0.5). Classic KD *softens* (T≥1) to transfer dark knowledge. For a **from-scratch** student that plays **greedy** at inference, which wins — sharper targets (better argmax commitment) or softer targets (richer learning signal)? Is sharpening actively hurting here?
2. **From-scratch vs warm-start:** We can't copy teacher weights (256→128 width mismatch). Is there a better init/curriculum for compressing a *converged, lineage-built* policy into a smaller width — e.g. teacher-assistant distillation (256→192→128), progressive width pruning + fine-tune, or learning a width-projection? Or is from-scratch genuinely just slow and we should train far longer (80–120 epochs)?
3. **LR schedule:** Cosine-to-zero over 40 epochs decays LR while gameplay is still climbing linearly. Extend epochs / use a higher LR floor / one-cycle? How would you set it for a model clearly not yet converged?
4. **Target signal:** We distill the teacher's *greedy policy* (top-5). Would distilling the *MCTS visit distributions* (the richer search-improved targets that originally trained the lineage) be a materially better learning signal for the small student, despite being more work?
5. **Is the 24% argmax-match a red herring** given the teacher's soft policy (top-share 0.34)? Is greedy gameplay the only metric we should optimize/track, and if so how do we predict the plateau without full evals?
6. **Capacity sanity check:** For a 9×9 board policy, is 3.0M params (10×128) plausibly enough to match an 11.9M (10×256) teacher to within a few % of gameplay, or is ~4× compression of a strong policy known to be lossy here?

## Data we can produce on request
Per-epoch match + gameplay curves; a clean T=0.5-vs-1.0 A/B at batch 4096; a 192-ch (6.7M) capacity-control run; DAgger (student-state relabel) results; MCTS-visit-target variant.
