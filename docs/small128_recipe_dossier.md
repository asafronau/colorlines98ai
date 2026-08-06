# Recipe dossier: distilling search-strength into a small Color Lines model — full mechanism evidence + a proposal

Self-contained brief for independent review (no prior conversation assumed). Hobby research project; no deadlines; the owner's thesis: "we have a good teacher and a wrong recipe — there should be a natural recipe that absorbs the bulk of the data."

## Setup

Game: Color Lines 98 (9×9, 7 colors, 3 spawns/turn, clear 5+ lines; score ≈ 2.03 × turns survived; effectively infinite — the metric that matters is score distribution over held-out spawn seeds). Student: **small128_vh2**, a 10-block × 128-channel ResNet policy, 3.0M params. Its greedy play: **20k-seed eval mean 13,334 / median 9,364 / <1000-rate 3.9%** (seeds 775000-794999, never used in any training data). Instruments: 500-seed screens (±~8%, catastrophe filter only — 0-for-5 on close calls historically), 5k evals (±480 mean / ±550 median, 95%), 20k paired-seed bootstrap (±240 mean) for promotions. Frozen reference model: a 256-channel sibling ("the master," 11.9M params, greedy mean 43,390) — off-limits for further training by owner directive; the small model must be improved in place. Training: A100 Colab + local M5 Max; all data generation local (custom C++ engine, ~18-35k NN evals/s).

## The teacher and THE KEY FACT

The teacher is MCTS (~400-600 sims, PUCT, top-k 30) over the student's own policy plus a survival value head (4-horizon sigmoid head on the frozen student backbone, trained on 20k of the student's games; q_weight 2.0 at the root). **Its play strength: 80 uncapped self-play games @400 sims score median 99,144 / P75 158,790 / max 486,992 — ~10× the greedy student, and 2-10× the frozen 256ch master.** These games contribute 4,721,555 recorded states (board, next-spawn preview, MCTS visit counts over top-15 candidates, and the move actually played). Additionally ~5,000 "crisis replays": the search re-plays the student's dying games from 15-30 turns before death (@600 sims), escaping most deaths; ~2.9M states in the same format. **All tensors currently carry visit counts but ZERO root Q-values or priors** (a slim-format decision that now matters — see finding 4).

## What we tried: the complete recipe × corpus matrix (all warm-starts from vh2)

| arm | corpus (crisis%) | recipe | result vs vh2 |
|---|---|---|---|
| iter5a | 6.2M (23.5%) | dw3\*, T0.7 sharpen, soft visit-CE, lr3e-4 | 5k: **−32%** @~17k steps → −73% @8 epochs |
| a3 | 2.1M (70%) | same + literal big-model geometry (bs32768, 28ep) | screen: **−37% at epoch 1** (inside low-LR warmup!) → −52% @ep10 |
| a2 | 6.2M (23.5%) | dw0, T1, 0.5 soft + 0.5 hard argmax-CE, lr1e-4 | 5k ladder: **−9..−12%** at ep1/2/3 (11,663/11,620/12,157 mean) |
| a5 | 2.1M (70%) | identical to a2 | screen: −23% @ep1 → −35% @ep3, monotone |
| a4 | 2.1M (70%) | a2-loss + γ=3 on disagreement rows + frozen BN | screen: flat **−28%** |

\*dw3 = per-state CE weight (visit top-share)³ — the weighting that repeatedly WON on the 256ch line's own corpora (+18%, then +5-8%).

Prior related closures (summarized): a months-long "micro-channel" of judged-correction fine-tuning produced vh2 itself (+4.1% over its predecessor at 20k paired) then closed — verified-positive correction rows (+2-6pp per-move advantage over 64-256 paired rollouts) do not integrate a second time at any dose/loss tried, while ~7% quiet-state argmax churn accompanies every concentrated update.

## Mechanism findings (each independently measured)

1. **Composition toxicity**: the mildest recipe (a2) is near-least-bad at 23.5% crisis and monotonically worse at 70% (a5). Crisis-replay VISIT TARGETS harm this student under every weighting tried; self-play fraction was diluting harm, not signal.
2. **BN running statistics are a first-class damage channel**: swapping components between vh2 and a3's epoch-1 checkpoint — a3 weights + vh2 BN buffers screens 4,160; vh2 weights + a3 BN buffers screens 6,475; intact a3-ep1 8,592; vh2 13,700. Each component alone is worse than both together (entangled, partially compensating drifts). One warmup-LR epoch displaced BN buffers 8.4% (weights only 0.6%).
3. **Teacher-student proximity**: the student's full-legal argmax already matches the corpus argmax on ~90% of rows; the ~10% disagreements carry small per-row advantage (mean ~+0.4pp died-within-300 when each was individually judged by 64 paired rollouts in an earlier round). The winning 256ch-line corpora had ~78% agreement against THEIR base.
4. **Target representation suspicion**: a strong searcher can emit prior-dominated visit distributions (visits ∝ exploration prior when the prior is decent), i.e., deliberation traces rather than decisions. With zero recorded Q/priors we cannot separate "visits reflect value" from "visits reflect prior." The played-move column, however, IS the decision.
5. Miscellany: γ-disagreement masks are computed on unaugmented states but applied under color/dihedral augmentation (stale predicate); 500-seed screens invert vs 5k routinely; a frozen-BN "preservation" arm changes gradient direction radically (cos 0.23 vs normal at 4× norm — mechanically a different loss).

## The owner's proposal (lead candidate — please attack it)

**Imitate the decisions, not the deliberation**: train on the PLAYED MOVE (hard CE, one-hot; mechanically blend_alpha=0) over the **self-play-only** corpus (4.7M states of median-99k play; optionally + 1.2M more from 1,000-turn-capped games; crisis replays excluded as the measured poison — though their played escape-moves could be included as one-hots too). Precedent: this model line was BUILT by exactly this absorption mode — 0.5-hard-CE distillation of a stronger policy's choices over 3.85M states was the only method that ever absorbed bulk data here (and pure imitation of a 3.3×-stronger teacher; the demonstrations are now 10× and on-manifold). Arms under consideration: (i) gentle warm-start from vh2 (lr 1e-4); (ii) the literal proven from-scratch recipe (lr 1e-3, batch 4096, cosine, ~40-100 epochs — slow but historically monotone); masking each game's first ~30 temperature-exploration moves.

## Questions

1. Attack the proposal: what breaks when hard-CE-imitating a search player's trajectory moves? (Known worry: state distribution is the TEACHER's — positions a greedy student reaches differ; the games' 100k-level states may be unreachable/atypical for the student. DAgger-style counters exist but resemble the closed micro-channel.)
2. Warm-start vs from-scratch for bulk absorption at 10× teacher strength — which first, and what early metrics decide (our step-checkpoint + screen + 5k + 20k ladder is cheap)?
3. Corpus construction: selfplay-only vs selfplay + crisis-as-one-hots; include capped (1k-turn) games; temperature-move masking; per-game caps for the 400k-turn outliers?
4. If imitation also fails: the Q-bearing re-mine (--full-record exists; ~1 day) to test advantage/Q-weighted targets — worth it before or only after the imitation arms?
5. BN handling for whichever arm: normal BN (a2 survived it), or is finding 2 reason to re-estimate/protect stats even for mild imitation?
6. What result pattern would justify closing this line for good (ship vh2, bank the data), vs. continuing? Please state concrete numeric criteria on our instruments.
