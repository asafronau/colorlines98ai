# Crisis-correction paradox: genuine corrections, yet distillation degrades a strong policy

**Audience:** Gemini + ChatGPT peer review. **Domain:** Color Lines 98 (9×9, 7 colors,
stochastic ball spawns), AlphaZero-style policy. Score = turns survived (every game
eventually dies; no turn cap). Policy-only greedy play (argmax legal move), no search at
deploy. Evals are distribution over a fixed seed range, no cap, fp16.

## The proven recipe (worked once, +37%)

Task arithmetic (Ilharco et al.):
1. Mine the base policy's death games; rewind the crisis band (depths 15–85 before death).
2. At each band state run **widened MCTS@4800** (3 determinizations, q_weight 2.0, a
   **linear feature-value (FV) value net** — NOT a neural value head). Keep states where
   MCTS-top ≠ policy move; target = soft MCTS visit distribution. Weight = margin
   (mcts_top_share − pol_share), mean-normalized.
3. `train_crisis_ft`: fine-tune a COPY of the base ONLY on the corrections (soft-CE,
   T=0.5 sharpening, **frozen BN** so running stats stay == base), ~15 epochs.
4. `merge`: θ(α) = θ_base + α·(θ_ft − θ_base), sweep α, eval gameplay.

**Result on pillar3b (weak base, ~18.8k mean):** broad plateau α∈[0.4,0.7], **+34–37%
median/mean over the bar**, equal floor. A scrambled-label control destroyed the policy
(fine-tune couldn't fit random labels), proving the gain was real 4800-sim knowledge.
The deployed policy **pillar3f = pillar3b + 0.5·(this vector)**.

## The problem: the SAME recipe on pillar3f degrades it

We iterated: recorded pillar3f's own death games, ran the identical pipeline
(MCTS@4800/FV, train_crisis_ft on base=pillar3f, merge). Corpus: 10,990 corrections from
902 games (12.2/game, 27% near-zero-margin), held-match 0.26 (vs 0.35 on pillar3b).

**Dose-response (500 seeds; base pillar3f mean 35,810 / median 27,434 / P10 4,546 / <1000 1.2%):**

| α | mean | median | P10 | <1000 |
|---|------|--------|-----|-------|
| 0.2 | 34,129 (−4.7%) | 23,255 (**−15%**) | 4,750 | 1.4% |
| 0.4 | 32,542 | 23,026 | 4,233 | 1.6% |
| 0.5 | 30,615 | 22,290 | 4,512 | 1.0% |
| 0.7 | 23,398 | 15,231 | 2,580 | 2.4% |

Every α is mean- and median-negative, monotonic. Even α=0.2 drops the median 15%.
(NB: every game passes through crisis at least once — the policy hasn't mastered infinite
play — so a median drop is consistent with the corrections degrading the typical game's
crisis decisions, not just "spillover to non-crisis states".)

## Diagnostic: ARE the corrections genuine for pillar3f?

For 60 decisive corrections (margin ≥ 0.10) we played pillar3f GREEDILY from move A
(MCTS-top) vs move B (policy move), 64 common-RNG rollouts each to death:

- **A survives longer than B: 67% of corrections** (40/60); A lower-catastrophe: 58%.
- mean turns delta (A−B): **+25** (median **+6**); catastrophe delta **−3.6pp**.

**CRITICAL: the survival metric UNDERCOUNTS genuine corrections.** Filtering to
margin≥0.5 did NOT raise the win-rate (65%) but raised the *size* of the win (mean +49,
P75 +105) — high-confidence corrections are bigger when they land but the same ⅓ "don't
beat on survival". An expert (the project lead) GUI-validated 3 max-margin (0.9) forks:
**2 of 3 are confirmed GRANDMASTER moves** (one is a genuine 2-ply technique: "prepare a
clearance one move ahead"); the 3rd is in a **lost position** — the MCTS move is clearly
better but *no move survives longer there*, so survival scores it as "non-beating". So the
"non-beating ⅓" is a MIX of (a) genuinely-better-but-position-already-lost and (b) truly
false corrections — and the survival metric cannot separate them. **The genuine rate is
therefore materially HIGHER than 67%; the corrections are largely real knowledge.**

This RE-WEIGHTS the diagnosis AWAY from "teacher too weak" and TOWARD "distillation fails
to capture genuine knowledge." Corroboration that filtering is not the answer: an
independent **held-out R=500 CI-verified** catastrophe-fork corpus (rollout-confirmed real
forks, recoverable positions) ALSO washed through the same merge channel — so even
verified, recoverable-position corrections did not lift the policy.

## The questions for review

1. **THE CORE GAP (primary question):** the corrections are largely GENUINE (expert-
   confirmed grandmaster moves; survival-verified ≥67% with the rest undercounted by the
   metric), and *applying* them helps in rollout (+25–49 turns), yet *distilling* the
   corpus degrades gameplay (−5% mean, −15% median at the gentlest α). **Why does
   genuine, verified knowledge become net-negative once distilled + merged?** Candidates:
   (a) MULTI-PLY TACTICS distilled as 1-move targets — e.g. "prepare a clearance one move
   ahead" teaches the SETUP move, but a greedy policy that doesn't reliably play the
   FOLLOW-THROUGH gets a neutral-or-worse setup; the corpus has the setup state, not the
   completion state; (b) T=0.5 sharpening over-commits; (c) soft-visit target ≠ "play A"
   — it reshapes the whole legal distribution; (d) the merge perturbs a *converged* policy
   with no coherent global task direction (vs pillar3b's headroom); (e) the genuine gains
   are real but tiny per-state (+6–9 median turns), so any collateral swamps them. Which
   dominates, and how would you tell them apart?

2. **Salvage the 67%:** if a third of corrections are wrong, is the fix simply to FILTER
   (keep only rollout/CI-verified A≫B, or high-margin) and re-distill — or does the
   small per-correction gain (+6 median turns) make even a clean corpus too weak to lift
   a strong policy? (The verified catastrophe-fork wash suggests filtering alone is not
   enough.)

3. **Teacher strength:** the obvious lead is to replace the FV linear value with the
   neural value head (value_head_pillar3f exists) so MCTS@4800 genuinely exceeds the
   student (>67% reliable, larger margins). Is that the right primary lever, or a
   second-order fix to a method that is fundamentally headroom-limited at this strength?

4. **Edit method:** is full-network fine-tune + linear weight merge the wrong tool for
   sparse local policy edits into a converged net? Would localized editing (ROME-style),
   Fisher/EWC-protected merge, or a small LoRA on the policy head distill the genuine
   corrections without the collateral median drop?

5. **Is the lever simply near its ceiling?** pillar3b→pillar3f was the FIRST crisis
   correction and captured the large headroom. Is ~30–35k mean near the practical ceiling
   of greedy-policy crisis correction, such that the rational move is a different lever
   (deeper/value-stronger search, or attacking the floor a different way) rather than a
   second correction pass?

## Data we can produce on request
- Per-correction rollout deltas; margin-stratified win-rate (does high-margin → higher
  genuine fraction?); the full α-sweep at finer resolution; GUI-validated example forks;
  the scrambled-control on pillar3f; a neural-value-head re-mine.

---
## Review synthesis (Gemini + ChatGPT, 2026-06-22)

**Converged diagnosis:** the FULL-NETWORK task-arith merge causes BROAD drift on normal
play that swamps the small crisis gains. Tell: at α=0.2, P10 RISES (4546→4750) while median
COLLAPSES (27434→23255) — the vector helps some low-tail states while damaging normal
equilibrium play (collateral, not just false corrections). Revised conclusion (replaces
"genuine verified knowledge → net-negative"): **the local correction signal is REAL but
SMALL (median +6 turns), and the current full-network merge introduces broader policy drift
than the corrections can pay for.** Both: do NOT re-mine or chase the neural value head yet
(better labels still damage if the edit channel is broken).

**Split on root cause (the open question):**
- Gemini: OVERFIT NOISE — 902 games × 11.9M params ≈ memorization; held-match 0.35→0.26 is
  the smoking gun; the task vector is dominated by parameter-space noise. → more data (or
  LoRA / replay-buffer to cap capacity) fixes it.
- ChatGPT: EDIT CHANNEL TOO BROAD — full-network merge perturbs delicate converged
  representations regardless of data; restrict the edit (head-only / late-layer / low α /
  no sharpening / hard top-move) and PROVE local transfer before scaling data.

**Resolving experiment (no re-mine, existing vector): scripts/probe_merge_locality.py** —
merge RESTRICTED (head-only, last-block+head) vs FULL × small α; measure LOCAL-MATCH to
MCTS-top on correction states + BROAD-DRIFT on random pillar3f self-play states (v15).
- head-only high-local + low-broad → ChatGPT (edit too broad) → restricted merge / LoRA;
  more data alone would NOT fix it.
- all scopes high broad-drift → Gemini (overfit) → more data / LoRA.
- local-match never rises → distillation target/training failed.
Then gameplay only the best 2-3. Candidate fixes both endorse (cheapest first): lower α,
head-only / late-layer vector, NO T=0.5 sharpening, hard top-move targets on verified
corrections, LoRA, replay-buffer (1% corrections in 99% pillar3f self-play, soft-CE).
