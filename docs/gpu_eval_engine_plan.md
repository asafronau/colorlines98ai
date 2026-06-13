# GPU-resident policy eval engine — design + staged plan

## Why
Policy-only 5k evals are ~75M forwards at pillar3f strength (~2h on M5, growing with every
improvement). Profile: ~half GPU sync/compute, ~half CPU (obs build, legal filter, move apply,
Python glue). Colab GPUs are useless for the current eval (CPU-bound: L4 7k/s < M5 11k/s).
A GPU-resident loop (boards live on device; obs/legal/argmax/move/spawn all on GPU; only
per-game results come back) removes the CPU side entirely: est. ~25k+/s on M5 (B=256→1024),
~100k+/s on A100 at large batch ⇒ **5k eval in ~15 min on Colab**, and the engine carries over
to Phase-1 probes and future batched workloads. Reuses the validated parked GPU-miner
primitives (alphatrain/batched_engine_gpu.py).

## Protocol decision (v2)
SimpleRng = numpy **PCG64** (game/rng.py) — bit-replicating PCG64 + Generator.choice rejection
sampling on GPU is not practical. So:
- **Engine LOGIC is validated bit-exactly** against ColorLinesGame via **trace injection**
  (see Testing) — every rule identical.
- **Spawn randomness uses per-game SplitMix64 streams on GPU** (int64 tensor math, exact and
  vectorizable), seeded from the game seed ⇒ scores deterministic **per seed independent of
  batch composition** (STRONGER than today's fp16 protocol, where the same seed in a different
  list plays a different game).
- Consequence: scores differ from the CPU protocol ⇒ **protocol v2, re-baseline once**
  (pillar3f 5k + any active candidate). All historical comparisons stay within their protocol.

## Exact transition semantics to reproduce (from game/board.py, read 2026-06-12)
1. **move(src,tgt)**: valid iff board[src]≠0, board[tgt]==0, tgt's empty-component adjacent to
   src (`_is_reachable` on `_label_empty_components`). Execute: move ball, turns += 1.
2. **Move-clear**: `_clear_lines_at(tgt)` — 4 directions (E,S,SE,SW) of same-color runs THROUGH
   the cell; each direction's full run if length ≥ 5; union of cells **deduped across
   directions**; score = `n_total*(n_total-4)` computed ONCE on the deduped total (two crossing
   5-lines = 9 cells = 45 pts, not 2×5).
3. **If cleared == 0 → spawn**: place the 3 *intended* `next_balls` SEQUENTIALLY; a ball whose
   intended cell is occupied is **displaced** to a uniform-random empty cell (uniform over the
   empties AT THAT MOMENT — prior balls this turn affect it). Then clear-check each LANDED cell
   in landing order (skip if board[cell]==0 — an earlier spawn-clear may have removed it);
   each spawn-clear scores independently (`n*(n-4)` per landed-cell clear event).
4. **Next-ball generation**: `min(3, n_empty)` distinct empty cells uniform without
   replacement + colors uniform 1..7. (Production: per-game SplitMix64 keys; top-3-of-iid-keys
   over empty cells = exact uniform 3-subset.)
5. **Game over**: empty==0 after spawn/generation. **Death in eval** additionally: no legal
   move exists (eval_policy treats empty legal-set as death).
6. **Move selection (the player)**: argmax of policy logits over legal (src occupied, tgt
   empty, reachable) moves — CPU path is top-30-by-logit then max-prior, which equals global
   legal argmax (softmax monotone). GPU: masked argmax over all 6561. Tie-break may differ from
   the CPU insertion-sort order — irrelevant under protocol v2 (and fp16 exact ties are rare).

## Reusable validated primitives (alphatrain/batched_engine_gpu.py)
`label_components_sv` (empty-CC labels, validated bit-identical), `legal_priors_t` (legal mask
+ reachability), `build_observation_t` (validated vs build_observation),
`clear_lines_at_t` (CHECK semantics vs §2: dedup + single-score), `apply_move_t` /
`apply_move_nosync_t` (check spawn handling — miner-era variants used deterministic spawn for
CUDA-graph capture; production spawn here must be the SplitMix64 path), `_rand_empty_order` /
`_det_empty_order` (spawn ordering helpers).

## New files ONLY (live miner + eval_policy untouched)
- `alphatrain/gpu_eval_engine.py` — state container (boards [B,81] int8, next_pos [B,3],
  next_col [B,3], n_next [B], score/turns [B] int32, alive [B] bool, rng_state [B] int64) +
  `step(logits) -> done_info` + `spawn`/`clear`/`legal` kernels (imports primitives).
- `scripts/eval_policy_gpu.py` — production loop: refill slots from seed list, build obs,
  forward (fp16, large batch), step, collect (seed, score, turns); same stats printer +
  per-seed JSON as eval_policy.
- `scripts/test_gpu_engine_golden.py` — the gate (below).

## Testing (the ship gate, per feedback_perf_golden_tests)
1. **Trace injection**: record N=50 CPU games (eval_policy player): per turn (chosen move,
   intended spawn cells+colors, displacement landings, generated next_balls). Drive the GPU
   engine with the SAME moves and injected spawn decisions ⇒ boards, scores, turns, game-over
   must match **bit-exactly every turn**. Validates §1-§5 logic completely, independent of RNG.
2. **Legal-argmax equivalence**: random boards × random logits — GPU masked argmax ==
   CPU `_get_legal_priors_flat` + max (up to documented tie-break).
3. **Spawn statistics**: production SplitMix64 spawns — chi-square uniformity over empties +
   exact 3-subset coverage on small boards; per-seed determinism across runs AND across batch
   compositions (the protocol-v2 selling point).
4. **End-to-end determinism**: same seed list twice, different batch sizes ⇒ identical scores
   (per-seed RNG makes this hold on the ENGINE side; fp16 forward still ties scores to batch
   size ⇒ pin B in the protocol, same as today).
5. **Re-baseline**: pillar3f 5k under protocol v2 on M5 AND on a Colab GPU (cross-device fp16
   differs — each device class gets its own bar; production evals likely move to Colab once
   it's 15 min).

## Stages
1. State container + injected-spawn step() + golden test 1 (the bulk of the correctness work).
2. Legal-argmax + obs integration + test 2.
3. SplitMix64 spawn streams + tests 3-4.
4. Production script + benchmark (M5 + Colab) + re-baseline (test 5).
5. (later, separate) reuse engine for crisis Phase-1 probes and any batched rollout needs.

Status (2026-06-12):
- Stage 0 DONE — this doc.
- Stage 1 DONE — alphatrain/gpu_eval_engine.py (step with score/ordering fixes over
  apply_move_t) + scripts/test_gpu_engine_golden.py PASS with verified coverage (194
  move-clears incl. synthetic crossing-lines dedup, 251 displacements, 1869 turns).
  Lesson: the first "pass" was vacuous (random 7-color play never clears) — the test
  now asserts coverage; half the trace games use 3 colors.
- Stage 2 DONE — choose_moves = legal_priors_t(top_k=1); test_gpu_argmax_golden.py
  PASS (402 boards incl. full-board death case); miner-era obs/legal tests re-PASS.
- Stage 3 DONE — test_gpu_engine_rng.py PASS. CAUGHT A REAL BUG: stateful SplitMix64
  streams advanced on ANY batchmate's draw (batch-dependent games). Fixed with
  STATELESS counter keys: key = mix64(seed, turn, purpose, cell); purpose 0-2 =
  displacement ball i, 3 = regen cells, 4 = regen colors. Batch-independence now
  holds by construction (48 seeds × batch {48,16,7} identical).
- Stage 4 — scripts/eval_policy_gpu.py + scripts/bench_gpu_eval.py written, M5 smoke
  + benchmark DONE. RESULT (M5/mps, pillar3f, evals/s): b64 3089, b256 8333, b512
  11252, b1024 13339. ms/step grows 20.7→30.7→45.3→76.6 ⇒ ~LINEAR in batch (no
  tensor cores → batching buys nothing). **On M5 it only TIES eval_policy (~11k).**
  Decomposition @b512: net forward 26.3ms (~80%), engine ops total ~6.7ms (label 1.25
  / obs 3.22 / choose_moves 1.22). So the bottleneck is the FORWARD (identical to
  eval_policy), not my engine — the M5 wash is just MPS's linear forward scaling, and
  the engine only recoups the ~10ms CPU glue.
  VERDICT: M5 keeps eval_policy (GPU engine = wash there). The GPU engine's value is
  CUDA-ONLY (tensor cores make the forward sublinear in batch; eval_policy is CPU-bound
  on Colab at 7k/s). GO/NO-GO = run bench_gpu_eval.py --device cuda --batches 256 512
  1024 2048 on Colab: ≥~40k/s @1024 ⇒ ship as the Colab evaluator + 5k re-baseline;
  plateaus near 13k ⇒ shelve next to the parked miner (same compute-wall lesson).
  Orthogonal lever (forward dominates everywhere): distill 11.9M→~3M policy ⇒ ~4× on
  eval/selfplay/mining/browser at once. Distribution-sanity vs eval_policy (task 130)
  still pending — do it wherever the engine ends up running.
