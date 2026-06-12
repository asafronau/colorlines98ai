# Peer review request: distilling deep-MCTS crisis corrections into a policy net — noise, saturation, or wrong recipe?

You are reviewing a months-long effort to improve a strong game policy using expensive search-derived
corrections. We have clean (finally) experimental data, three competing interpretations, and a limited
compute budget. We want your critique of the statistics, the experimental design, and above all the
**training recipe** — the human's leading hypothesis is that the recipe is the bottleneck.

## 1. The game and the policy

Color Lines 98: 9×9 board, 7 colors. Each turn: move one ball along a free path; lines of 5+ clear and
score; otherwise 3 new balls spawn at uniform-random empty cells (positions+colors of the next 3 are
shown). Game ends when the board fills. Empirically score ≈ 2.1 × turns survived for ALL our models
(score≡survival; we proved every scalar value target — score, density, TD, survival — is the same
saturated signal). **All games end in death**; long games and short games die the same way: RNG spawns
steer the board into a configuration the policy plays badly.

Policy: AlphaZero-style distillation lineage. Current base = **pillar3b_epoch_20**: 11.9M params
(10 residual blocks × 256ch, 18-channel 9×9 observation), trained by soft-CE distillation of V13 =
9.16M states of 400-sim-MCTS self-play (visit-distribution targets, target-sharpening T=0.5, dihedral
+ color-permutation augmentation). Greedy-argmax policy play (no search at deploy time).

## 2. The crisis-corrections pipeline (the "teacher")

1. Record the policy's death games (greedy argmax, fp16).
2. Rewind each game to the crisis band **D-15..D-85** (15–85 moves before death).
3. At each band state run **widened MCTS@4800**: top_k=300 (all legal moves), Dirichlet noise
   (α=0.3, w=0.25), 3 determinizations (the spawn RNG is hidden state), c_puct 2.5, q_weight 2.0,
   leaf evaluation = a linear feature value function (a survival proxy; known to be saturated — it's
   the best scalar evaluator we have, all learned value heads failed).
4. Keep states where the MCTS top move ≠ the policy's move. Record the **soft visit distribution**
   (top-20, renormalized) and, since 2026-06-09, **per-candidate root Q** (q, q_min, q_max).
5. Corpus build: margin = mcts_top_share − pol_share; per-correction weight = max(margin, 0)
   normalized to mean 1; "decisive" filter = margin ≥ 0.05.

Yield: ~17.3 corrections/game, ~7.5 decisive/game. **3,690 games mined so far** — all with identical
parameters, model, script, machine. Provenance verified: chronological slices are statistically
identical (source-game median score 12.7k/13.0k/11.3k/12.5k across slices; margin P50 0.110–0.113;
median depth 54). ~4s/state on an M5 Max; the corpus is genuinely expensive.

**Label quality is human-validated**: the user reviewed corrections in the GUI. Taxonomy: (a) blunder
fixes (policy immediately blocks its own prospective clearance), (b) "grandmaster moves" — elaborate
multi-step setups raising near-future clearance chances, (c) right-line-wrong-ball — policy completes
a line with a ball whose removal doesn't help; teacher completes it with a different ball that
unblocks a cascade. Good corrections across the whole band, including "last-effort escapes" at D-25.
Heterogeneous (no single fixable mistake type). This review also surfaced a real engine bug (fixed).

## 3. The training recipe under review ("mC")

Warm-start pillar3b_epoch_20 and train with TWO terms:
```
loss = CE(student, V13 sharpened T=0.5)            # main: re-distill the base corpus
     + λ · softCE(student, corrections T=0.5)      # aux: 256-sample batch per main step
```
Main batch 32768, lr 5e-5 (AdamW wd 1e-4, 1-epoch warmup then cosine), augmentation ×8 → 2,126
steps/epoch. Aux: margin-weighted, λ=0.01 with 0.5-epoch warmup, frozen-BN forward on the aux batch
(crisis states are OOD; train-mode BN poisons running stats — we learned this the hard way), by-seed
15% holdout. **Gameplay peak is ALWAYS epoch 2** regardless of any knob; ep3+ degrades.

Knob bracketing already done (each a separate run): λ ∈ {0 (control), 0.003, 0.01, 0.012, 0.014,
0.03}, aux_T ∈ {0.3, 0.5}, weighted on/off, margin filter ∈ {none, 0.05, 0.10}, warmup 2.0→0.5.

## 4. Eval protocol (the part we fixed mid-investigation)

Early comparisons were contaminated: different seed *lists* give different games (batched fp16
forward; same seed in a different batch composition diverges), and two different eval tools were in
use. Everything below is the CLEAN protocol: `scripts/eval_policy.py` = single-process batched
greedy-argmax player, fp16, batch 256, fixed seed list 775000–775999 (1k games), one machine (M5 Max).
Median SE @1k ≈ 600. Per-seed scores now saved (`logs/eval_scores/*.json`); paired comparisons via
`scripts/compare_evals.py` (same seed list ⇒ per-seed deltas cancel seed luck). 5k list
(775000–779999) reserved for finalists.

## 5. Clean results (all directly comparable; median / mean, 1k games)

| run | corpus | λ | median | mean | P10 | <1000 |
|---|---|---|---|---|---|---|
| pillar3b base | — | — | 13,421 | 18,865 | 2,409 | 2.6% |
| ctrl0 (λ=0 control, ep2) | — | 0 | 13,316 | 18,207 | 2,402 | 2.5% |
| mC ep2 ("the bar") | 13.8k dec (1,837 games) | .010 | **16,738** | **24,249** | 3,196 | 1.6% |
| mC ep2 | 19.6k dec (2,593 games) | .010 | 15,924 | 21,948 | 2,413 | 3.1% |
| mF ep2 | 19.6k dec | .012 | 14,782 | 20,387 | 2,510 | 2.1% |
| mG ep2 | 19.6k dec | .014 | 16,008 | 22,431 | 2,588 | 2.7% |
| mC27k ep2 | 27.6k dec (3,676 games) | .010 | 14,943 | 22,370 | 2,211 | 2.0% |

Established facts:
- **ctrl0 ≈ base**: the gentle re-distill alone contributes ZERO. mC's +25% median is 100% from
  corrections. The channel transmits.
- **Held-out (by-seed) correction match ≈ 0.21–0.22 for every corrected model** regardless of corpus
  size (and the base's match is ~0.01 by construction — corrections are exactly the states where the
  base disagreed with the teacher).
- A **trust-region alternative failed decisively** (pillar3e): teacher-primary CE on corrections +
  β·KL(frozen pillar3b ‖ student) on broad V13 states (4096/step), NO V13 re-distill. All β ∈
  {5,15,45} lost to mC by ~2k (best 14,824); β=45 @ep120 collapsed to 8,787 (with cosine lr decay,
  Adam favors the consistent anchor gradient over the large-but-noisy teacher gradient and un-learns).
  Its held-out match reached 0.27 — HIGHER than mC's 0.215 — while playing 2k WORSE. Conclusions:
  held match does not predict gameplay; full-coverage rehearsal (~70M augmented V13 samples/epoch)
  beats KL-regularization on sampled anchors (classic replay > EWC/LwF).

## 6. Encoding diagnostics (margin metrics on 1,160 common-unseen corrections)

| model | match | p(top1) | mass top20 | softCE | logit margin | KL vs base (corr states) |
|---|---|---|---|---|---|---|
| base | 0.010 | 0.1775 | 0.959 | 3.204 | −1.49 | 0 |
| ctrl0 | 0.066 | 0.1775 | 0.960 | 3.211 | −1.50 | 0.004 |
| mC 13.8k | 0.230 | 0.1973 | 0.944 | 3.124 | −1.35 | 0.172 |
| mC 19.6k | 0.230 | 0.1967 | 0.941 | 3.104 | −1.30 | 0.185 |
| mC 27.6k | 0.199 | 0.1936 | 0.936 | **3.059** | **−1.24** | 0.181 |
| mF λ.012 | 0.245 | 0.2017 | 0.936 | 3.105 | −1.28 | 0.232 |
| mG λ.014 | **0.258** | **0.2051** | 0.936 | 3.133 | −1.31 | 0.269 |

The bigger corpus produced a model **better calibrated to the teacher distribution (best softCE,
best margin) but less committed to the argmax (worst match)**. More λ raises commitment AND drift
proportionally — and the gameplay table shows the net is a wash (mG ≈ mC-19.6k). "Under-sharpening"
as the regression mechanism was falsified by the mG gameplay result.

## 7. The noise problem (current crux)

mF vs mG differ by λ=0.002 — a sub-noise config change, i.e. effectively replicates — yet their
medians differ by **1,226 on the same seed list** (paired). The trainer sets no torch seed; each run
has its own aux sampling and augmentation draws. So **training-run σ is ~1k**, and the entire spread
of the five corrected runs (14.8k–16.7k) is consistent with a SINGLE true value ≈15.5–16k plus noise.
Under that reading: corrections give a robust +2.5k; corpus size, λ, and margin thresholds all do
nothing measurable; the bar (16,738) is the luckiest draw, not the best recipe.

We have ZERO true replicate runs of any config. Compute budget: ~2h A100 per training run (hobby
budget, ~a few runs/week); 1k eval = 12 min local.

## 8. Three competing interpretations

- **(I1) Noise/saturation**: the aux channel transmits a fixed ~+2.5k worth of crisis competence
  (~21% held-match) and is insensitive to everything else. Next gains must come from structural moves
  (see §10), not from this channel.
- **(I2) Size/dilution**: smaller corpora genuinely encode better per-correction (fixed aux budget).
  Weakened by §6 (no shallow-encoding signature) and by flat λ, but not dead until replicated runs.
- **(I3) THE HUMAN'S LEAD HYPOTHESIS — recipe problem**: the corrections are human-validated gems
  from the strongest (and slowest, ~4s/state) teacher we've ever had; a 21% held-match ceiling and
  +2.5k flat transfer means the *recipe* (tiny-λ aux soft-CE bolted onto a re-distill of a converged
  net) is a weak channel. There should exist a recipe where MORE validated corrections ⇒ MORE
  gameplay, monotonically. We have not found it: λ-scaling failed, trust-region/KL-anchor failed,
  sharpening variants failed, margin filtering failed.

## 9. Experiment in flight

**E1**: random 13.8k subsample of the 27.6k corpus (same size as the bar's corpus, drawn from all
3,676 games — deterministic seed 0), trained with the exact bar recipe. Purpose: does ANY random
13.8k reproduce the bar's 16,738, or was the bar a lucky run? (One run can't fully separate these —
we know — but combined with the existing five runs it tightens the picture; it doubles as a
near-replicate for σ_train.)

## 10. Questions for you

1. **Statistics**: Do you agree with the σ_train≈1k reading from the mF/mG pair? Given ~4 runs/week
   and paired per-seed evals, what is the most information-efficient design to (a) estimate σ_train
   properly, (b) decide I1 vs I2 vs I3? Sequential/bandit designs welcome.
2. **The recipe (the core ask)**: If you take I3 seriously — what training scheme extracts
   monotonically-more gameplay from a growing set of strong, sparse, OOD, human-validated
   state→distribution corrections into an 11.9M warm-started policy? Specifically critique/rank:
   (a) folding corrections INTO the main dataset (as extra V13-style rows, with oversampling weight —
   why keep two loss terms at all?); (b) LoRA-style low-rank adapter trained on corrections only,
   merged or kept as a side-branch; (c) a separate "crisis head" / gating; (d) fine-tune on
   corrections then MERGE weights with the base (model soup / task-arithmetic style, interpolation
   coefficient tuned on eval); (e) iterative small-λ passes (absorb 13.8k, re-mine, absorb next batch)
   vs one-shot big-corpus; (f) anything else with literature precedent for "sparse expert corrections
   into a dense policy".
3. **A control we haven't run**: scrambled-label control — same correction states, targets shuffled
   between them (or replaced with the policy's own argmax). If gameplay STILL gains ~+2.5k, the gain
   is a regularization/perturbation effect and the content of the corrections barely matters; if it
   collapses to ctrl0, content matters and the channel is just narrow. Worth one of our precious runs?
   Predictions?
4. **Q targets**: per-candidate root Q is now recorded (visits-Q correlation ≈ 0.79 at the root).
   Best way to build targets from (visits, Q) for this distillation — Q-softmax? completed-Q à la
   Gumbel MuZero? advantage-vs-policy-top1 softmax? How to set the temperature given Q is in the
   units of a saturated linear survival proxy (per-state q_min/q_max recorded)?
5. **ep2 universality**: gameplay always peaks at epoch 2 (~4.2k steps) across every knob setting,
   then degrades while the aux keeps fitting. ctrl0 (λ=0) shows ~no gameplay change at ep2. What does
   this signature suggest about the dynamics (main-loss noise-floor drift vs aux overfitting), and
   does it change the recipe recommendation?
6. **The endgame**: the plan after this is settled — seed a NEW self-play generation (V14: self-play
   data generated with the best corrected policy as MCTS prior at 400 sims) and keep the corrections
   from being washed out (the 400-sim search won't re-find 4800-sim moves; retention via continued
   low-λ crisis aux during V14 training, or a correction replay buffer, or re-mining against the new
   policy). Critique this plan; is there a better iteration loop given the mining cost asymmetry
   (4s/state teacher vs 400-sim self-play)?

## 11. File inventory (repo: colorlines98)

- **Trainers**: `alphatrain/train_path_b.py` (main+aux recipe, all λ/T/weight knobs, `--grad-audit`);
  `alphatrain/train_trust_region.py` (the failed pillar3e: teacher-primary + KL anchor, grad-share
  instrumentation).
- **Miner**: `scripts/gen_corrections_parallel.py` (GPU inference server + N CPU MCTS workers;
  records visits + root Q); `alphatrain/mcts.py` (the search).
- **Corpora**: `crisis/corrections/corr_*.json` (3,690 per-game files);
  `scripts/build_corrections_corpus.py` → `crisis/corrections_corpus.pt` (full 63,765),
  `crisis/corrections_corpus_mm05.pt` (decisive 27,605), `crisis/corrections_corpus_sub13k.pt`
  (E1 subsample 13,800); `scripts/subsample_corpus.py`.
- **Eval**: `scripts/eval_policy.py` (clean protocol; per-seed JSON to `logs/eval_scores/`);
  `scripts/compare_evals.py` (paired test).
- **Diagnostics**: `scripts/fingerprint_corpus_membership.py` (which corpus trained a checkpoint —
  behavioral membership test via mtime slices); `scripts/margin_diagnostics.py` (the §6 table);
  `scripts/diag_corpus_slices.py` (the provenance table).
- **Checkpoints** (`alphatrain/data/`): `pillar3b_epoch_20.pt` (base);
  `pillar3d_ctrl0_epoch_2.pt`; `pillar3d_mC_dec_T05_epoch_2.pt` (the bar, 13.8k);
  `pillar3d_mC_dec_T05_2500_epoch_2.pt` (19.6k); `pillar3d_mC27k_epoch_2.pt`;
  `pillar3d_mF_dec_lam0.012_epoch_2.pt`, `pillar3d_mG_dec_lam0.014_epoch_2.pt`;
  `pillar3d_mC_dec_T05_m10_epoch_2.pt` (margin≥0.10 strongest-only 10.9k — unevaluated on clean
  config).
- **Colab notebooks** (`alphatrain/`): `train_pillar3d_mE1sub13k_colab.ipynb` (E1, in flight),
  `train_pillar3d_mE2lam02_colab.ipynb` (27.6k @ λ0.02, on hold), `train_pillar3d_mC27k_colab.ipynb`,
  `train_pillar3d_ctrl0_colab.ipynb`, `train_pillar3e_trA_b{5,15,45}_colab.ipynb`,
  `eval_policy_colab.ipynb`.
- **Prior briefs (ChatGPT)**: `docs/distill_scaling_for_chatgpt.md`,
  `docs/distill_seed_confound_for_chatgpt.md`, `docs/distill_dilution_vs_interference_for_chatgpt.md`
  (has the §5/§6 tables), `docs/aux_distill_params_for_chatgpt.md`.

Please be blunt. We've had two rounds of confident-but-wrong diagnoses already (a seed-list eval
confound, and an "under-sharpening" mechanism falsified within a day). The human's instinct — that
the corrections deserve a better recipe than a λ=0.01 side-channel — has not been refuted by
anything above; it just hasn't been confirmed by anything either.
