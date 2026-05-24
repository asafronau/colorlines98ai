# Image-3 Diagnosis: Resolved — model was right, no bug

**Resolution (Phase 1 rollout judge, 2026-05-22):** the image-3 "obvious clear miss" is NOT a bug. Common-RNG paired rollouts (K=256, H=500, pillar3b_epoch_20) show the green-clear and pink-setup moves are statistically tied on score-over-horizon. Pink-setup wins paired comparisons 54.7% vs A's 43.8% — pillar3b correctly reads that the patient setup leads to bigger clears that recoup the missed +5pts. N=1 diagnosis was overreach.

Title kept for grep continuity. Original narrative below; corrected findings appended at "Phase 1 results."

---

# Image-3 Diagnosis: Value Head is the Bottleneck (ORIGINAL — refuted by Phase 1)

**Date:** 2026-05-22
**Models tested:** pillar3a (`sharp_25_epoch_12.pt`) and pillar3b (`pillar3b_epoch_20.pt`)
**Value head used in MCTS:** `value_head_sharp25_ep12.pt` (= value_head_pillar3a)
**Scripts:** `scripts/debug_image3.py`, `scripts/debug_image3_mcts.py`

## TL;DR

On a concrete state where the obvious correct move is to complete a 4-greens-in-a-row line by moving the only reachable green ball, both pillar3a and pillar3b systematically prefer a marginal pink-extension move. Sweeping MCTS sims (100 → 3200), q_weight (0 → 3), and c_puct (1.5 → 6.0) does **not** flip the choice. At ≥800 sims MCTS converges to **~equal visit counts (12.5-12.7% each) across the top 3 candidates** — it cannot distinguish post-clear from post-pink-setup states.

**This is a value-head / leaf-evaluation problem, not a search problem.** The bias is shared by the policy prior AND the value head. More search just refines the equivalence. Different q_weight, c_puct, Dirichlet — none of the search-side levers fix it.

The likely root cause: **the value head was trained on survival targets**, not score-per-state. The +5pts immediate reward and the +1 free turn (clears suppress next-turn spawn) from a line clear don't show up in the survival objective.

## The state (image 3, Score 35 / Turn 27)

```
    0 1 2 3 4 5 6 7 8
  0 Y B . . . . . T .       Y=yellow, B=blue, T=tan/brown
  1 P T B . . . . . .       P=pink, C=cyan, G=green, R=red
  2 . C . . . . . . .
  3 . G G G G . . R .   ← 4 greens (cols 1-4), one cell from a 5-line at (3,5) or (3,0)
  4 . C T . . P R . .       pink at (4,5) is the model's choice of source
  5 B . . . . . R B .
  6 T . . R . . G . .       green at (6,6) is the ONLY ball that can complete the line
  7 . . . . . P P P .       3 pinks at (7,5),(7,6),(7,7)
  8 P . . . . . B . Y
```

Next balls (spawn next turn): `brown @ (3,5)`, `brown @ (2,3)`, `green @ (6,5)`.

**Critical**: `(3,5)` will be **brown** next turn — completing the green line by moving green→(3,5) is only possible THIS turn. After this turn, the row-3 green opportunity is gone forever (one needs (3,0) which is unreachable by green(6,6)).

### What the player chose

Pillar3a policy-only: `pink (4,5) → (7,4)` at 24.51% probability. The green-clear move `green (6,6) → (3,5)` was ranked #3 at 19.29%.

### What the only clearing move does

`green (6,6) → (3,5)`: completes a 5-green line, clears 5 balls, +5 pts. Per Color Lines rules, when a line clears, **the spawn for that turn is also suppressed** — so the move also saves 3 incoming balls.

## Sweep methodology

For each (sims, q_weight, c_puct, Dirichlet) combination, ran `mcts.search(state, return_policy=True)` once and recorded the top-3 moves by normalized visit count.

- Sims swept: 100, 200, 400, 800, 1200, 1600, 3200
- q_weight swept: 0.0, 1.0, 2.0, 3.0 (sims=400)
- c_puct swept: 1.5, 2.5, 4.0, 6.0 (sims=400)
- Dirichlet (α, w) swept: (0.0, 0.0), (0.3, 0.25), (1.0, 0.5)

`temperature=0` (argmax of visits) for action selection — same as inference/eval.

## Results

### Pillar3a (sharp_25_epoch_12) sim sweep

| sims | top1 | top1 prob | green-clear rank | clear prob | argmax flipped to clear? |
|---|---|---|---|---|---|
| 100 | pink (4,5)→(7,4) | 15.0% | #2 | 14.0% | no |
| 200 | pink (4,5)→(7,4) | 13.5% | #2 | 13.0% | no |
| 400 | pink (4,5)→(7,4) | 13.0% | #2 | 12.7% | no |
| 800 | pink (4,5)→(7,4) | 12.7% | #2 | 12.6% | no |
| 1200 | pink (4,5)→(7,4) | 12.7% | #2 | 12.6% | no |
| 1600 | pink (4,5)→(7,4) | 12.6% | #2 | 12.6% | no |
| **3200** | pink (4,5)→(7,4) | 12.6% | #2 | 12.5% | **no** |

At sims ≥ 1600, top-3 are all within ±0.05pp. MCTS has decided these moves are equivalent.

### Pillar3b (epoch 20) sim sweep — same picture, slightly worse

| sims | top1 | top1 prob | green-clear rank | clear prob | argmax flipped to clear? |
|---|---|---|---|---|---|
| 100 | pink (4,5)→(7,8) | 14.0% | not in top-3 | 12.0% | no |
| 400 | pink (4,5)→(7,8) | 12.7% | not in top-3 | 12.3% | no |
| **3200** | pink (4,5)→(7,8) | 12.5% | not in top-3 (tied) | 12.5% | **no** |

Pillar3b is slightly *less* aware of the clear in MCTS visits: a third pink move (pink (8,0)→(7,4) at 12.4%) now displaces green-clear from top-3 in most rows.

**Caveat:** pillar3b was tested with `value_head_sharp25_ep12.pt` (pillar3a's value head). HISTORY 138 says value heads must be retrained when the backbone moves, so the value head input distribution is mismatched. Pure-prior results below are unaffected.

### Pillar3a q_weight sweep (sims=400, c_puct=2.5)

| q_weight | top1 | top1 prob | clear rank | clear prob | flipped? |
|---|---|---|---|---|---|
| **0.0** (no value head) | pink (4,5)→(7,4) | 25.5% | **#3** | **20.0%** | no |
| 1.0 | pink (4,5)→(7,4) | 13.0% | #3 | 12.7% | no |
| 2.0 | pink (4,5)→(7,4) | 13.0% | #2 | 12.7% | no |
| 3.0 | pink (4,5)→(7,4) | 13.0% | #2 | 12.7% | no |

**Key row: q=0.** Setting q_weight=0 makes the search pure-prior — no value head signal at all. The policy alone has clear at #3 with 20.0% behind pink-setup at 25.5%. **So even WITHOUT the value head, the policy already prefers pink.** The bias is baked into the prior. Adding the value head just doesn't help anchor toward the obvious clear.

### Pillar3a c_puct sweep (sims=400, q=2.0)

| c_puct | top1 | clear rank | argmax flipped? |
|---|---|---|---|
| 1.5 | pink (4,5)→(7,4) | #2 | no |
| 2.5 | pink (4,5)→(7,4) | #2 | no |
| 4.0 | pink (4,5)→(7,4) | #2 | no |
| 6.0 | pink (4,5)→(7,4) | #2 | no |

No combination of exploration coefficient unlocks the clear move.

### Pillar3a Dirichlet noise (sims=400, q=2.0)

| α | w | argmax flipped? | notes |
|---|---|---|---|
| 0.0 | 0.0 | no | identical to baseline |
| 0.3 | 0.25 | **YES, by tiebreak** | all top 3 at 12.7% — noise randomized the argmax |
| 1.0 | 0.5 | **YES, by tiebreak** | all top 3 at 12.3-12.7% — same |

The "wins" under Dirichlet noise are arbitrary — they only happen because noise scrambles a near-tie. MCTS still cannot distinguish the moves on the merits.

## Diagnosis

Three independent observations converge:

1. **Pure prior bias (q=0).** Pillar3a's policy alone has pink #1 at 25.5%, green-clear #3 at 20.0%. Pillar3b's policy alone has the same ordering with pink #1 at 22.5%, clear tied #3 at 20.7%. Both backbones genuinely think pink-extend is marginally better than clearing. *The prior is biased.*

2. **Sim budget converges to equivalence.** At 3200 sims, top-3 moves are within 0.1pp visit share. MCTS has explored these branches deeply and the leaf-value evaluator (value head + final state value) returns *equally good* values for "post-clear state" and "post-pink-setup state". *The value head cannot break the tie.*

3. **Robustness across both pillars.** Pillar3a (trained on V12 corpus, target_temperature=0.25) and pillar3b (trained on V13 corpus, target_temperature=0.5, +30% mean over pillar3a) have nearly identical priors on this state. *The bias survived an entire iteration with a different teacher and different sharpening.*

These together mean:

**The value head trained on survival targets cannot represent the value of immediate score gain + free turn.** Both backbones learn similar priors because both were distilled from MCTS visit distributions where the same value head misvalued the post-clear state. The bias is self-perpetuating through the corpus.

## What this rules out (with evidence)

- **More sims in next selfplay corpus.** Disproved: 3200-sim MCTS on this state still picks pink. Increasing sims for V14 selfplay would not fix this class of error.
- **q_weight tuning.** Disproved: q ∈ {0, 1, 2, 3} all pick pink.
- **c_puct exploration.** Disproved: c_puct ∈ {1.5, 2.5, 4.0, 6.0} all pick pink.
- **Dirichlet noise as a corrective.** Disproved: it works only by randomizing among tied moves — not a fix in deployment (no noise) or in eval.
- **Target sharpening (the T=0.5 lever).** Argmax is invariant to monotonic transformations, so sharpening cannot change the model's chosen move. Reaffirmed not the culprit.
- **Better policy distillation with current value head.** Disproved by transitivity: pillar3b is a different teacher result with the same value head structure (survival-trained), and it shows the same bias. Iterating the same pipeline won't escape this.
- **"Force the model to clear" hard supervision.** Rejected per user (2026-05-22): we want the model to learn *when* to clear, not always clear.

## What likely fixes it

### Hypothesis A (most likely): the value head's training objective is wrong

Currently: value_head trained on V11/V12 trajectories with **survival targets** (turns-until-death). This signal does not differentiate "I cleared a 5-line and got +5 + free turn" from "I extended a pink line that might complete next turn." A strong policy survives roughly equally in both.

Fix candidates:
1. **Score-based ranking head.** Targets = score gained over next H turns (e.g., H=20). The post-clear state has +5 immediate plus skipped-spawn (saves ~2pts of average future damage), so it should rank measurably higher than the pink-setup state under this objective.
2. **Density / score-per-turn head.** Targets = future score / future turns. Cleaner normalization.
3. **Hybrid survival + score targets.** Multi-task value head; combine both signals.

Cost: ~3-5h to relabel V13 corpus with score targets + retrain head. Backbone untouched.

### Hypothesis B: the backbone's feature space doesn't separate post-clear states

If the pillar3a/3b backbone activations are similar for the two candidate post-states, no value head trained on those features can rank them. We'd need a backbone that's been trained to discriminate these.

Diagnostic: take the post-clear board and the post-pink-setup board, compute pillar3b backbone features for both, and look at L2 distance + cosine similarity. If similar, B is implicated; if very different, A is the only fix needed.

Cost: ~30 min.

### Hypothesis C: the corpus itself lacks decisive examples

If the MCTS that generated V12/V13 also mis-valued clears on the corpus, the policy can only learn what the corpus shows. Generating a fresh corpus with a score-aware value head (after Hypothesis A is implemented) would propagate the corrected taste.

Cost: A → generate V14 with score-aware head + MCTS@400 → pillar3c training. ~36h total compute.

## Next steps proposed

1. **Run Hypothesis B diagnostic** (cheap): compare pillar3b backbone features for post-clear vs post-pink-setup states on image 3. ~30 min.
2. **If B is clear**: implement Hypothesis A (score-based value head training). Retrain on pillar3b backbone, then re-run this sweep. ~5h.
3. **If A fixes image 3**: train pillar3c on a corpus generated with the new score-aware MCTS. The new corpus should naturally weight clears higher in visit distributions.

## Engine mechanics verified (ChatGPT review 2026-05-22)

- **`calculate_score(n) = n * (n - 4)`** — super-linear in line size:

  | clear size n | points |
  |---|---|
  | 5 | 5 |
  | 6 | 12 |
  | 7 | 21 |
  | 8 | 32 |
  | 9 | 45 |

  A 5-clear is worth only 5 pts. A 6-clear is worth 12 (2.4×). A 7-clear is 21 (4.2×). **Patient setup for a bigger clear is non-trivially rewarded by the scoring.** This shifts the analysis: pink-extend may genuinely be the right call IF it sets up a ≥6-clear next turn.

- **Spawn suppression on clear is confirmed.** `ColorLinesGame.move()` only spawns balls in the `else` branch (cleared == 0). On any clearing move, the spawn is skipped. The clear's true value = +score + ~equivalent_of_free_turn.

So the rules question is settled. **What's NOT settled is whether the image-3 clear is actually better than the pink-setup under our current policy's play.** This is the real diagnostic.

## Revised diagnostic protocol (post-ChatGPT review)

The original diagnosis overreached from N=1. The q=0 row showed this is a *policy-prior* issue first (clear ranks #3 even with no value-head signal), which means a value-head retrain won't fix the bias unless we also regenerate corpus data with the corrected head. And the pink-setup might *genuinely* lead to a bigger eventual clear — visual intuition doesn't settle it.

**Phase 1 — Image-3 rollout judge** (cheap, decisive on this one state):

For move ∈ {A=green-clear, B=pink-setup}:
1. Apply the move to the image-3 state.
2. From the post-move state, run K=256-512 policy-only continuations.
3. Use **common RNG**: branch A and branch B both seed their RNG with the same `seed` for each replicate, so spawn sequences are aligned.
4. Measure per branch:
   - Score gained over horizon H=300, H=500
   - Turns survived (or capped at H)
   - Final empty count
   - Whether the branch had a bigger clear (≥6) over the horizon

If A wins on score-over-H AND survival under our current policy, the image-3 clear is genuinely the better move — the model is wrong. If B wins, the model's preference is justified and the "obvious clear" intuition was misleading.

**Phase 2 — Aggregate diagnostic** (only if Phase 1 shows the model is wrong on image 3):

Sample 100-300 corpus states where ≥1 immediate clear is legal. For each:
1. Identify policy top1 move.
2. Identify best immediate clear by score (largest n × (n-4)).
3. If they differ, run common-RNG continuations from both and compare score-over-H + survival.
4. Bucket by:
   - Clear size (5, 6, 7+)
   - Whether next_balls would block the clear-target cell
   - Available-clear count (1 vs multiple)

Only if immediate clears systematically win the rollout comparison should we change training/value targets.

**Phase 3 — Score-aware value head retrain** (only if Phase 2 confirms systematic miss):

Use a **hybrid target**, not pure score:

```
V = survival_H_primary  +  small_λ · score_rate_H_or_clear_tempo
```

Pure score targets risk overfitting to greedy clears. Pure survival misses tempo. Both signals needed. Backbone untouched; new head only.

After head retrain: re-run Phase 1 + Phase 2. If clears now win MCTS argmax on the same states, the hypothesis is confirmed and we propagate to pillar3c via a fresh corpus generated with the new head.

## Pillar3b eval reference (context for "is pillar3b really better?")

Pillar3b is materially stronger than pillar3a in aggregate gameplay even though it doesn't fix the image-3 specific failure:

500-seed eval, seeds 0..499:

| ckpt | mean | P25 | P50 | P75 | <1000 | >10K |
|---|---|---|---|---|---|---|
| pillar3b_epoch_5 | 13,417 | 4,767 | 9,652 | 18,406 | 3.6% | 49% |
| pillar3b_epoch_10 | 17,489 | 5,488 | 13,199 | 24,190 | 5.6% | 59% |
| pillar3b_epoch_15 | 16,776 | 5,600 | 12,057 | 23,252 | 2.0% | 57% |
| **pillar3b_epoch_20** | **18,863** | 5,815 | 14,232 | 26,442 | 2.4% | 60% |

1000-seed eval on out-of-sample seeds 777000..777999 (the seeds 0-499 used during pillar3a tuning are now overfit-suspect):

| ckpt | mean | P25 | P50 | P75 | <1000 | >10K |
|---|---|---|---|---|---|---|
| sharp_25_epoch_12 (pillar3a) | 15,002 | 4,076 | 9,596 | 21,196 | 4.8% | 49% |
| **pillar3b_epoch_20** | **17,255** | **5,483** | **12,567** | **22,486** | **2.5%** | **58%** |

Pillar3b gains: +15% mean, +35% P25, +31% P50, +6% P75, **floor halved** (4.8% → 2.5%), +9pp on >10K rate.

The image-3 failure mode persists in pillar3b but it doesn't undo the broader policy improvements — pillar3b is the new best model, decision gate cleared.

## Phase 1 results — model was right (2026-05-22)

Common-RNG paired rollouts from the image-3 state. K=256 seeds, H=500 turns, policy player = pillar3b_epoch_20.

| branch | init pts | mean Δscore over H | mean turns | die rate | mean #clears | %bigclr ≥6 over H |
|---|---|---|---|---|---|---|
| A: green clear (6,6)→(3,5) | +5 | 1004.9 | 496.3 | **2.0%** | 182.80 | 100% |
| B: pink setup (4,5)→(7,4) | 0 | **1005.4** | 494.9 | 3.1% | 183.24 | 100% |
| B2: pink setup (4,5)→(7,8) | 0 | 1005.0 | 495.0 | 3.1% | 183.29 | 100% |

**Paired A vs B (same RNG seed for both branches):**

| metric | value | interpretation |
|---|---|---|
| mean Δscore (A − B) | −0.5 ± 6.6 (SE) | tied; CI crosses zero |
| median Δscore | −5 | B's median 1 small-clear ahead — noise |
| A wins / B wins | 43.8% / 54.7% | B wins 10.9pp more (≈3.5σ on win-rate) |
| die rate diff | 2.0% vs 3.1% (1.1pp) | favors A but only 1.1σ — inconclusive |

The +5 from the immediate clear is recouped over horizon by bigger clears in the patient branch. Pillar3b's policy is *reading the position correctly* — the pink-setup is marginally better on score-over-H. The visual intuition that "the obvious clear is the right play" was confirmation bias on a near-tie.

### What this refutes (from the original write-up above)

- ❌ "Value head is the bottleneck on this state" — value head was correct.
- ❌ "Policy systematically misses clears" — pillar3b's prior is actually *correctly* ranking these candidates as nearly equivalent.
- ❌ "Score-aware value head retrain will fix this" — there's nothing to fix on this state.

### What remains valid

- ✓ Engine mechanics: `calculate_score(n) = n × (n − 4)`, clears suppress spawn.
- ✓ Methodology of the MCTS sweep (it correctly converged to ~uniform — because the moves *are* nearly equivalent).
- ✓ ChatGPT's review judgment: do not approve compute on the basis of N=1.

### Implications for future work

1. **No score-aware value head retrain.** The hypothesis that motivated it was wrong.
2. **No "force the clears" supervision.** Would actively hurt — the patient choice was right.
3. **Phase 2 (aggregate diagnostic) still worth running** with a corrected framing: "is the policy systematically off from rollout-judge across clear-available states, or just on this one near-tie?" If aggregate confirms the policy is well-calibrated, look for floor leaks elsewhere (e.g., late-game density patterns, specific spawn configurations).
4. **General methodology lesson**: visual analysis on a single state ≠ correct play. Common-RNG rollouts are the source of truth for "is this move better than that one." Always run them before redesigning training objectives.

## Files

- `scripts/debug_image3.py` — policy-only top-K analysis on the state
- `scripts/debug_image3_mcts.py` — MCTS sweep across sims, q_weight, c_puct, Dirichlet
- `scripts/image3_rollout_judge.py` — Phase 1 rollout judge (definitive)
- `logs/image3_mcts_pillar3b.log` — MCTS sweep raw output
- `logs/image3_rollout_judge.log` — rollout judge raw output
