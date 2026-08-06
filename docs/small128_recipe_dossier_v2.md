# Recipe dossier v2 (FINAL results included): the imitation ceiling moves with data — theory + next-lever review

Self-contained brief for independent review (assume no prior context). Hobby research project, limited but real compute (A100 Colab sessions + a local M5 Max with a custom C++ eval/search engine, 18-35k NN evals/s; data generation is effectively unlimited in wall-clock, just slow). Owner's stance: *"We need a simple recipe and I'm certain we can find it. Amazing 400k-point games, crisis recovery — there should be a good and simple recipe. We need to be theoretically convinced. We can extract insights from data (we have a TON). Experiments over pure thought."*

## Setup (compressed)

Color Lines 98 (9×9, 7 colors, 3 spawns/turn, clear lines of 5+; score ≈ 2.03 × turns survived; the load-bearing skill is escaping rare RNG-driven crises). **Student:** small128_vh2 — 10-block × 128-ch ResNet, 3.0M params; greedy: **20k-seed mean 13,334 / median 9,364**. **Teacher:** MCTS (400-600 sims) over the student's own policy + a survival value head on its frozen backbone; **teacher's uncapped self-play: score median 99,144 / max 486,992 (~10× the greedy student)**. A frozen 256-ch "master" (greedy 43,390) may be used as a demonstrator/label source but never retrained; search-on-master would produce games of order 1M turns (untested — proposal below). Instruments (95% resolution on mean): 1k seeds ±1,100; 5k ±480; 20k paired ±240. Training targets in all corpora: MCTS visit counts + the played move per state; hard-CE (blend 0) trains on the played move only.

## Complete results (all warm-start from vh2 unless noted)

**Soft-visit-target era (5 arms, all corpora/weightings/geometries):** ALL regress, −9% to −73%, damage ∝ crisis fraction and gradient concentration. BN running stats measured as a first-class damage channel (component-swap: each of {weights, BN buffers} alone worse than both). Not repeated here in full.

**Hard-CE (played-move) era:**

| run | corpus | geometry/epochs | outcome |
|---|---|---|---|
| hall | 5.3M all-data | bs8192 lr1e-4, 8+10ep | climbs 10,690→12,454 (1k); ep7 at 20k: **−1,493 mean** vs vh2 |
| hall3 | same | bs32768 lr3e-4, 20ep | FLAT 11.1-11.9 from ep1 (instant convergence, same optimum) |
| hall4-γ0 | **10.7M — everything, uncapped** | bs32768 lr3e-4, 12ep | 1k: 11.3→12.0→12.9(ep5)→~12.1(ep7-9)→13.4(ep12); **20k: ep12 = −424 [−674,−181], ep5 = −1,070** |
| hall4-γ2 | same, crisis rows ×3 | same | 11.0-12.0 flat — amplification neutral-to-negative; closed |
| **x3k (cross-experiment)** | **v14_rev3 = the 256ch line's WINNING corpus** (2.26M), its native recipe (dw3/T0.7 soft) | bs32768 lr3e-4, 12ep | **collapses to ~8.5k at ep1, flat-bad for 12 epochs** |

Supporting: student-vs-master argmax agreement unchanged by hall training (72.6% vs vh2's 73.0 — fabric intact). Student agrees with its own search corpus argmax on ~90% of rows. 1k screens over-read the best checkpoint by ~500-600 in both measured cases (mirages #6, #7 of the campaign — hence 20k paired for all verdicts).

## The two decisive facts

1. **The cross-experiment (owner-designed) came back student/recipe-side**: the corpus that provably lifted the 256-ch line +18% destroys THIS student under the same soft-target recipe, identically to our own corpora. Failure follows the target family + student, not corpus origin. (This kills "our data is bad" accounts and confirms soft visit distributions as the poison for this warm-started 3M student.)
2. **The hard-CE ceiling MOVES with data**: 5.3M → 10.7M (+ longer horizon) improved the 20k gap from −1,493 to **−424**. One dose-response point, but the first live axis after optimizer geometry, horizon, entry, and weighting were all falsified. Naive extrapolation says another ~2× of demonstrations crosses vh2; demonstrations are mintable locally without limit (~4.7M rows per 80 uncapped games, days not dollars).

## Questions

1. **Theory, updated**: what account fits BOTH facts — soft targets destroy while played-move hard-CE converges cleanly to a data-dependent optimum just below the edited student? Our leading story: soft visit tails are prior-shaped noise that a greedy 3M student can't afford (x3k confirms it's not corpus-specific), while hard-CE learns real decisions but needs enormous coverage because ~90% of rows are already-agreed (information-thin); vh2 sits above the current optimum only because of its verified-edit stack. Critique; what measurement would falsify your preferred account? (We can compute anything on 11M rows + fresh rollouts locally.)
2. **The simple recipe may just be MORE DEMONSTRATIONS**: given fact 2, is brute demonstration-scaling (mint 2-4× more uncapped search self-play; retrain hard-CE) the highest-expected-value next step? What's your predicted scaling exponent, and what early measurement would tell us the axis is saturating before we spend a week of mining?
3. **The owner's stronger-teacher proposal**: generate demonstrations from search-on-the-MASTER (256ch + its own survival head; est. games of order 1M turns). Agreement with the student would drop well below 90% → denser signal per row (directly attacks the info-thin account) but the demonstrator is farther off-manifold (prior per-state corrections from the master famously refused to install into this student — though that was per-state supervision, not trajectory imitation). Rank against Q2's brute scaling; predict outcomes under your theory.
4. **Single-knob tweaks** still worth running alongside (label smoothing? top-2 targets? EMA? tiny-LR long tail?) — rank by information per A100-hour, or say "none beats data."
5. **Q-recording** (visits-only → re-mine a slice with root Q/priors, ~1 day): under your theory, does Q-aware imitation ("imitate only where it mattered") change the scaling curve, or is it another weighting scheme headed to the same fixed point (as γ-amplification was)?
6. **Closure criteria**: if demonstration-scaling to ~20M rows and/or the master-teacher corpus still lands within ±240 of vh2 at 20k, is the honest conclusion that the student+search SYSTEM (median 99k) is the deliverable and the imitation question is answered? State the numeric stopping rule you'd pre-register.
