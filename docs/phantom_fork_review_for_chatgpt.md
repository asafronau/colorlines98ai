# Crisis-Fork Mining: the "phantom fork" problem — review request

## 0. What we want from you
We have a method for finding training signal to improve a game-playing policy. We
discovered it systematically **mis-attributes blame** (flags *correct* moves as
blunders). We've diagnosed the mechanism precisely and tried several fixes, each
with a fundamental limitation. **We want your read on whether our framing is right,
whether our best current fix is sound, and whether there's a cleaner approach we're
missing.** Full context below — it's self-contained.

---

## 1. The game and the agent
- **Color Lines 98**: 9×9 board, 7 colors. Each turn you slide one ball along a
  BFS-reachable path to an empty cell; then **3 new balls spawn** at random empty
  cells — **unless** your move completed a line of ≥5 same-color (horizontal,
  vertical, or diagonal), in which case the line clears, **no balls spawn, and you
  move again**. Game ends when the board fills. No turn cap.
- **Key fact we proved empirically: score = survival.** The policy scores ~2.0–2.1
  points per turn at *all* skill levels, so final score ≈ 2.1 × turns_survived.
  Every scalar "value" target we tried (density, score-rate, TD returns) saturates
  because they're all proxies for "turns survived."
- **The agent**: a **policy-only** neural net (10-block, 256-channel ResNet; board →
  move distribution; greedy argmax at play time). No value head at inference. It's
  destined for browser deployment (ONNX/WASM), and it also powers "AI hint" in a GUI.
- **Current performance** (deployed policy `pillar3b_epoch_20`, 2000 held-out seeds,
  greedy/policy-only): **mean 17,581, median 12,255, P10 2,377, max 145,084,
  fraction scoring <500 = 0.3%**.
- **The goal is the FLOOR, not the mean.** We want to lift the worst-case games
  (early density-spiral deaths). ~1/3 of floor deaths are RNG-sealed (unavoidable);
  ~2/3 appear to have avoidable mistakes in the last ~15–45 plies before death.

## 2. The improvement idea: "crisis-fork mining"
We want to find specific states where the policy made an avoidable mistake — a move
that quietly raised its death probability — and use those as distillation targets to
**make the one policy better** (we are NOT deploying a second model; the mined signal
shapes the policy's own training).

**Procedure (per game):**
1. Play to natural death at turn D.
2. Rewind and treat each turn in the band **[D−45, D−15]** as an "anchor" state.
3. At each anchor, take the policy's move plus ~10–20 alternative candidate moves
   (policy top-k + a heuristic's top-k).
4. For each candidate move: from the anchor, **force that move, then continue with
   the greedy policy** for up to **H = 300 plies**, over many random seeds.
   **"catastrophe" = the game died within H plies** (anchor-relative, so it's valid
   at any game length).
5. **Flag a "fork"** if some alternative's catastrophe is ≥ ~10pp lower than the
   policy's move's catastrophe.

**Three rollout passes with increasing rigor:**
- **curve** (R=100, policy move only) → pick the band.
- **screen** (R=50–100, all candidates) → flag forks; stores `best_cat` = the
  catastrophe of the best candidate at each anchor.
- **confirm** (R=500, only the flagged (policy-move, best-move) pair, on *fresh*
  seeds, **paired bootstrap**) → keep only forks whose 95% CI on the gap excludes 0
  (a winner's-curse guard).

The continuation in every rollout is the **greedy policy itself**. Current corpus:
**~925 confirmed forks across ~340 games.**

## 3. Why this is theoretically reasonable
The catastrophe metric is estimating **Q^π(state, move)** — the survival of each move
*when continued by the current policy π*. Finding a state where some alternative B has
higher Q^π than the policy's move A, and training the policy toward B, is the **policy
improvement step of policy iteration**. The improvement theorem says acting greedily
w.r.t. Q^π yields a policy ≥ π. So in principle this is rollout-based approximate
policy iteration, and it should monotonically improve the policy (modulo
function-approximation effects). Some confirmed forks are genuinely great (human
inspection in a GUI confirms "the policy move was clearly bad; the alternative clearly
better").

## 4. The problem we discovered: "phantom forks"
Deep-dive on one confirmed fork (seed 50029, anchor turn 6771, confirmed gap 43pp):

- The board is nearly full (17 empties) — **dying**. The policy's move completes a
  **vertical line of 5 and clears it** (`[1,4]→[2,4]`, +5 score, opens a column).
  A human judges this **correct and necessary**.
- Yet the metric flags it: clear → **81%** catastrophe; an alternative "waiting" move
  (`[5,7]→[4,8]`, no clear) → **38%**. So the metric says the clear is a 43pp blunder
  and the wait is the fix.

We brute-forced **every legal continuation** (R=500 each) from the board *after* the
clear, and from the board *after* the wait:
- After the clear, **best achievable** catastrophe = **36.7%**; but the **greedy
  policy plays its 95th-best move out of 255** (catastrophe 81.7%). It cannot
  capitalize on its own correct clear.
- After the wait, best achievable = **43.3%**, reached by… **playing the clear one
  turn later** (which the greedy policy correctly does there).

**So with good follow-up, the clear (36.7%) is BETTER than the wait (43.3%). The clear
is the right move.** The 43pp "fork" was entirely an artifact of *what the greedy
policy does next*: it plays the correct clear at 6771, then **blunders at turn 6772**
(its argmax `[0,1]→[0,2]`, ranked 95/255 for survival; the right move was to relocate
a different ball). The greedy continuation carries that 6772 blunder, inflating the
clear's catastrophe. The "wait" move only "wins" because it reshuffles the board so the
6772-blunder position never arises.

We verified **there is no execution bug**: the post-clear board is deterministic
(clearing spawns nothing), and the rollout's greedy move from it (`[0,1]→[0,2]`,
prob 0.95) is **bit-identical to what the real recorded game played at 6772**. The
policy genuinely makes a correct clear and then confidently blunders.

**We call this a "phantom fork": the catastrophe metric, judged by the greedy policy
itself, blames an UPSTREAM correct move for a DOWNSTREAM blunder.** Because the judge
is the policy, the metric cannot distinguish "this move is bad" from "this move is fine
but the policy mismanages what follows."

## 5. The localization principle we agree on
A move at turn T can only be called a blunder if **T is where the mistake actually is**.
If the policy plays well at T but errs at T+k, the catastrophe belongs to T+k.

Crucially, the **real** blunder at 6772 **was independently flagged** by the same
mining run (it mines every turn): at 6772 the policy's `[0,1]→[0,2]` (80.6%) vs the
correct relocation (40.2%) = a confirmed 40pp fork — **with the right fix**. So the
corpus contains BOTH the phantom (6771, "don't clear") and the real fork (6772, "don't
play [0,1]→[0,2]"), with **near-identical confirmed gaps (43 vs 40pp)** — indistinguishable
by the metric. Distilling the phantom would teach the policy to **avoid a correct
clear** — actively harmful. We believe this is a major reason an earlier distillation
attempt ("pillar3c", listwise-margin on these forks) **regressed the policy 34%**.

## 6. How common are phantoms?
Phantoms appear whenever the policy plays a good move at T but blunders downstream, so
they cluster on the turns *before* a real blunder. Measured across the corpus:
- 72% of fork "runs" are isolated singletons.
- **But 47% of all forks live in a consecutive-turn cluster** (runs of length 2–6).
- These clusters are where phantoms concentrate.

## 7. Fixes we tried, and why each is limited
**(a) "Drop forks whose policy move clears a line."** Cheap, catches the clear-phantom.
But the user (correctly) refuses to blanket-ignore clearances — *sometimes (rarely)
waiting really is better*, and we'd throw away genuine clear-related forks.

**(b) "Keep the latest fork in a consecutive run, drop the earlier ones."** Directionally
right (a phantom is a shadow of a *later* blunder, so it sits earlier). But: 47% of
forks are in runs, so this drops ~27% of the corpus on a heuristic; we can't assume all
earlier-consecutive forks are shadows (the policy can make two genuine blunders in a
row); and "keep latest" can overshoot past the real blunder into a near-dead position.

**(c) Re-score: judge a move by where it LEADS under good play, not greedy survival.**
Define `corrected_gap(T) = best_cat(T+1) − best_cat(T)` — the rescuability of the board
the policy's move *leads to* (which, since the trajectory follows the policy, is exactly
the next mined anchor T+1) minus the alternative's catastrophe at T. Intuition: a real
blunder leads somewhere genuinely harder to rescue; a phantom leads somewhere just as
survivable. **Uses only existing data (no new rollouts).**
- **It passes all 4 cases where we have ground truth**: the two known phantoms (the
  6771 clear, and another confirmed-by-eye correct clear) score ≈0 (−1, −9 at R=500);
  the two human-confirmed real blunders score +47 and +48. The raw greedy-gap rated
  those phantoms 48pp and 14pp — indistinguishable from real.
- **But it has a structural flaw inside runs.** `corrected_gap(T)` and
  `corrected_gap(T+1)` share the term `best_cat(T+1)` with opposite signs, so
  consecutive values are **anti-correlated by construction** — first-differencing a
  turn-to-turn-bouncing `best_cat` series produces alternation (real/phantom/real/…)
  that is an artifact, not signal. Raising R from 100→500 did **not** remove it
  (confirming it's structural, not noise). Also, `best_cat(T+1)` is conditioned on the
  *single* realized spawn at T, so bad spawn RNG at T can masquerade as a blunder.
  → It's a useful **coarse** filter (cleanly separates isolated forks and obvious
  phantom-clears) but **not a reliable per-turn localizer** inside dense runs.

## 8. The fundamental difficulty
Per-move blame on a *dying* board appears **underdetermined**. Everything is bad, it's
a tangle of multi-move interactions, and an expert human cannot reliably assign blame
by eye (we tried; the user gave up on specific dense-run turns: "the board is dying and
it's complicated"). The deepest constraint: **we have no judge stronger than the
policy.** MCTS (policy + a feature-value evaluator + search) is only ~21k mean vs the
policy's 17k, and it's ~50× slower than a greedy rollout, so it's not a cheap oracle.
The only truly trustworthy judge is the **held-out floor eval** (run N fresh games,
measure the worst-case tail) — but that's a slow, *aggregate* signal that can't label
individual forks.

## 9. Where we've landed (and what we're unsure about)
Current plan (pragmatic): use `corrected_gap` as a **coarse filter** (drop clear
negatives = phantoms, keep clear large-positives = real), **confidence-weight** the
rest by confirmed gap, **distill carefully** (listwise-margin loss, heavy
augmentation, conservative loss weight to avoid mis-generalizing "avoid clearing"),
and let the **held-out floor eval** be the arbiter. Accept that some phantoms survive;
bet that the many real forks outweigh the few bad ones, and **iterate** (mine → distill
→ re-eval floor → re-mine on the improved policy, since the fork set shifts as π
improves). Escalate to an MCTS-as-per-state-oracle (evaluates each crisis state
independently — no differencing, no shared-term artifact) only if the floor doesn't
move.

## 10. Specific questions for you
1. **Is our diagnosis correct** — that greedy-continuation Q^π mining conflates move
   quality with the policy's *own downstream competence*, producing phantom forks that
   are systematic (not noise), and that distilling them can teach anti-correct moves?
2. **Is the "judge a move by where it leads under good play" reframing the right one?**
   Is there a clean, cheap estimator of it that avoids the first-difference
   anti-correlation artifact of `best_cat(T+1) − best_cat(T)`? (e.g., should we
   evaluate each anchor's *best achievable* survival absolutely, via a short
   rollout-improvement / 1-ply-lookahead controller, rather than differencing
   greedy-best_cat?)
3. **Is rollout-based policy iteration even the right tool here**, given "no judge
   stronger than the policy"? Or should we abandon per-move blunder labeling entirely
   and instead generate improved targets at crisis states with bounded MCTS
   (AlphaZero-style), accepting the cost — i.e., is the cheap-mining-as-blunder-
   detector a dead end, with search-as-teacher the only sound path?
4. **How would you separate "genuine consecutive blunders" from "a chain of shadows of
   one later blunder"** at low cost? Our same-source-ball heuristic (a shadow's "fix"
   reuses the same ball as the next turn's fix) was suggestive but not conclusive.
5. Any concern that this whole approach is **chasing a moving target** (the forks are
   defined relative to the current π; once π changes, the labels are stale), and if so,
   how should iteration be structured to stay stable?

## Appendix: concrete numbers for the worked example (seed 50029)
- Anchor turn 6771 (the phantom): policy `[1,4]→[2,4]` clears 5 (+5 score). Confirmed
  (R=500): policy 80.6% catastrophe, best alt `[5,7]→[4,8]` 37.6%, gap 43pp, CI [37.6, 48.4].
- Brute force from post-clear board (255 legal moves, R=500 each): best achievable
  36.7% via `[5,7]→[4,7]`; greedy picks `[0,1]→[0,2]` = 81.7%, **rank 95/255**.
- Brute force from post-wait board (104 legal moves): best achievable 43.3% via the
  clear `[1,4]→[2,4]` (rank 1); greedy correctly picks it.
- Anchor turn 6772 (the real blunder): policy `[0,1]→[0,2]` (its argmax, prob 0.95) =
  80.6% catastrophe; best alt `[5,7]→[1,4]` = 40.2%; confirmed gap 40.4pp, CI [34.8, 45.8].
- `corrected_gap` at R=500: 6771 → +2.6 (phantom ✓), 6772 → unscorable (deepest in run,
  no T+1 anchor) → fallback "keep". Two other human-confirmed real forks: 50249 d31 →
  +47, 90020 d28 → +48. The (4,2)-stuck-ball chain in seed 50034 shows the structural
  alternation: +3.8, −9, +22, −17, +24, −1 across consecutive turns.
