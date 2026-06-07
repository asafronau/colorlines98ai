# Crisis-corrections corpus: did more data help, and how to make better labels? — review request

Please adjudicate the two competing claims below (the human's and Claude's), check the reasoning
against the data, and recommend the next experiment. Be skeptical of *both* of us.

## Setup (the floor-lifting pipeline)
Goal: lift the **floor** of a Color Lines 98 policy net — the worst-case early-death games (`<1000`
score), not the mean. Pipeline:
- A policy net (pillar3b) was distilled from **400-sim** MCTS self-play. Standalone policy mean ~17.5k.
- To improve it we **mine crisis corrections**: rewind each policy *death* game to its crisis band
  (states D-15..D-85 moves before death); at each state run **deep widened MCTS@4800** (top_k=300 =
  all legal + Dirichlet, 3 determinization seeds, leaf value = a **27-feature linear evaluator**,
  q_weight=2.0); the **soft visit distribution** is the training target.
- Distill: warm-start pillar3b, re-distill the base self-play tensor **+** the aux corrections
  (soft-CE, margin-weighted, λ=0.03, target-temperature 0.5, gentle lr 5e-5, 10 epochs).
- "margin" of a correction = `mcts_top_share − policy_share` (how much the deep MCTS disagrees with
  the policy). Corpus built with `--min-margin 0` (keep all).

## The data point that triggered this (5,000-game held-out eval, same seeds)
| | v2.1-ep2 (16.5k corpus) | v2.2-ep2 (26.8k corpus, +59%) |
|---|---|---|
| mean | **21,195** | 20,711 |
| P10 (floor) | **2,796** | 2,552 |
| P50 | **15,121** | 14,445 |
| <1000 | **2.6%** | 2.7% |
| >10k | **64%** | 62% |
| max | 201,365 | 174,191 |

Adding +59% more crisis corrections (same recipe, same mining config) **regressed on every metric**.
Consistent across all 8 → not noise. (A 2,000-game eval had *misleadingly* shown v2.2 edging the
mean; 5k reversed it.) The corpus is ~**26% near-zero-margin** corrections.

## Claim A (the human)
*"4800-sim MCTS is the best player we've ever had; self-play only used 400 sims. So every mined
position should be pure gold — strictly better supervision than the policy that generated the games.
How can more gold possibly hurt, and how could we even produce better data?"* Proposed fix: filter to
keep only the **strongest** corrections (high margin).

## Claim B (Claude)
The labels are **not** pure gold, for reasons specific to a *stochastic* game with a *weak leaf value*:
1. **MCTS@4800 is only as strong as its leaf value, and ours is a saturated survival proxy.** We
   measured score ≈ survival ≈ ~2.0/turn (near-constant), and afterstate value *identical* (2.549) for
   a dead-end move vs a constructive one. The **same** linear evaluator drove the 400-sim self-play,
   so 4800 sims isn't a categorically stronger player — it's a *more confident* version of the same
   value-biased judgment. (Deep search converges to optimal regardless of value only in
   *deterministic* games; here a biased leaf value isn't averaged away.)
2. **~40% of greedy-mined forks were phantoms** (measured corpus-wide with rollouts: the MCTS-preferred
   move didn't actually survive better). We pivoted to soft-visit relabel partly for this.
3. **Many crisis states are already lost.** Autopsy: at the cliff the policy plays the *best* move and
   *still dies* — the density spiral was set 100+ moves earlier. In a lost position every move is
   near-equal → the label is noise.
4. The policy is **already MCTS-distilled** (pillar3b's 17.5k came from MCTS targets), so corrections
   are a marginal top-up; the marginal/phantom/lost mass tips it into the regression seen above.
5. Distilling the **soft** distribution (near-uniform over top moves) can *flatten* the policy; and
   the corrections fight the base re-distillation (which pulls toward the old 400-sim policy).

How 4800 was chosen (from our notes, not recall): a sweep on **4 known-answer anchor states** — the
MCTS visit-share on the *correct* move was still climbing at 4800 (e.g., 42→58%, 32→42%, 37→50% from
3200→4800), i.e. it took ~4800 sims to overcome the policy's pathologically over-confident **wrong**
prior (~0.95 on a blunder) on *recoverable* positions. It was never certified as an optimal label in
already-lost positions or where the leaf value misleads.

## Where we landed + the experiment we built (v2.3)
Agreed on the human's filter as the cheap first test: rebuild the **same 26.8k mine** at
`--min-margin 0.05` → **11,582 decisive** corrections (drops the entire near-zero mass, 57%). Recipe
byte-identical → clean single-variable test of quality-over-quantity. Margin survival: 0.05→11.6k,
0.10→6.4k, 0.15→4.1k. Evaluating at 5k. Read:
- **beats v2.1** → it was dilution; decisive-only is the recipe (then floor-target the mine).
- **ties v2.1** → margin-filtering recovers but can't exceed → the limit is deeper than dilution,
  because **margin-filter keeps high-margin phantoms and high-margin already-lost labels.** Pivot to
  an **outcome filter**: keep a correction only if its move *demonstrably outlives* the policy's move
  in common-RNG paired rollouts (can't keep a phantom or a lost-position label by construction).

## Questions for you
1. Is Claim B's core right — that in a stochastic game **the label quality is bottlenecked by the
   leaf value, not the sim count**, so "more 4800-sim labels" reinforces the same value bias rather
   than teaching new skill? Or is the human right that 4800-sim soft-visit targets should be
   monotonically-good supervision and the regression must be a *training* artifact (λ, sharpening,
   base-vs-aux conflict) rather than label quality?
2. Is the **margin filter** a meaningful proxy for "strongest correction," or does it mostly select
   *confident* corrections (incl. confident-but-phantom / confident-but-lost)? Should we skip straight
   to the **rollout-outcome filter**, or is margin-first worth it because it's ~free?
3. Floor-specific: the autopsy says floor deaths are **set earlier** (density spirals), not at
   crisis-onset. Are we mining the wrong *states* for the floor entirely — should corrections target
   the move where the spiral *began*, or is crisis-onset correction still the right lever and we just
   need cleaner labels?
4. Is re-distilling the base self-play tensor **+** aux corrections the right structure, or is the
   base distillation diluting/fighting the corrections (would pure correction fine-tuning with a KL
   anchor be better)?
5. Anything we're both missing — a cleaner definition of a "gold" crisis label for a stochastic,
   sparse-reward, survival game than "deep-MCTS soft visits"?
