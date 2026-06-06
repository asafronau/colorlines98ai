# Tier 2: Array-Based Batched-Tree MCTS for Mining

## Goal
Run K independent MCTS searches (band states × seeds) concurrently in one process,
advancing all K trees with **vectorized array ops** and issuing **one large GPU forward
per step**. Mining is CPU-bound on per-tree Python tree-work; batching K trees amortizes
that interpreter overhead ~K× and saturates the (currently idle, esp. on L4) GPU.

Target: 5–20× mining throughput per process vs the current per-worker scalar MCTS.

## Why this is the right lever (measured 2026-06-05)
Per scalar search @4800/top_k=300/bs=8 (~4.5s on M5): NN fwd+.cpu() 33%, **tree
bookkeeping (PUCT scans + backup + Node alloc) ~55%**, engine (move/spawn) ~10%, leaf
numba (obs/feature-value) ~7%. The bit-identical per-search ceiling is ~8% (scan loops);
the ~55% tree cost only collapses if K trees share **one vectorized pass**. So throughput
must come from batching trees, not speeding one tree.

## Correctness contract (non-negotiable)
- **Deterministic engine primitives** (move, line-clear, connected-components,
  reachability, obs, feature-value, legal-priors) MUST be **bit-identical** to the scalar
  numba functions for the same inputs → strict golden tests on random boards.
- **The search as a whole** need NOT be bit-identical to scalar MCTS: spawn RNG draws
  differ (independent determinization) and fp16 forward batching differs. Validate by
  **argmax agreement + low TV** of the root visit distribution vs scalar on real band
  states (same bar we used for bs/top_k checks). Both are valid open-loop MCTS over the
  same priors+values → must converge at 4800 sims.
- Backstop: `scripts/golden_search_test.py` (scalar) stays the reference; a new
  `scripts/validate_batched_mcts.py` does batched-vs-scalar agreement/TV.

## Open-loop structure (what makes this Color-Lines-specific)
The scalar MCTS is **open-loop with determinized spawns**: a node = an action path; its
board is RE-SIMULATED every visit with fresh random spawns (so the same node has different
boards across sims). Therefore we do NOT store a board per node — the descent re-applies
moves+spawns each sim, maintaining an ephemeral per-tree board. The tree stores only
visit/value/prior stats per (node, child-edge).

## Array representation (per batch of K trees)
Dense edge arrays (mctx-style), width W = top_k (300), N = num_sims+1 nodes:
```
node_visits     int32   [K, N]          # N visits
node_value_sum  float32 [K, N]
child_action    int32   [K, N, W]       # flat action (src*81+tgt); -1 = empty slot
child_prior     float32 [K, N, W]       # c_puct*prior precomputed
child_visits    float32 [K, N, W]
child_valuesum  float32 [K, N, W]
child_nodeid    int32   [K, N, W]       # -1 = unexpanded edge
n_children      int32   [K, N]
n_nodes         int32   [K]             # next free node id per tree
```
Memory @ K=64, N=4801, W=300: ~2 GB (5×[K,N,W] f32/i32). K is the throughput/memory knob
(K=64 → 64-tree batching, 512-leaf forwards at bs=8). Ephemeral per-sim: boards[K,9,9],
current_node[K], path buffers.

## Algorithm (one simulation across all K trees)
Per sim (vectorized over K; a depth loop with active-mask for variable depth):
1. **Select**: from each tree's root, repeatedly gather current node's child stats, compute
   PUCT `q_weight*q_norm + child_prior*sqrt(node_visits)/(1+child_visits)`, argmax (first-max)
   over valid slots; apply the chosen move to the per-tree board (engine), follow child_nodeid;
   stop a tree when it reaches an unexpanded edge (leaf). Virtual loss on the path.
   Reachability/occupancy: the chosen move must be legal on the current board (batched
   occupancy mask + batched reachability; ban+retry like scalar — vectorized).
2. **Evaluate leaves**: build obs for the K leaf boards → ONE forward (K×… leaves) →
   policy logits + feature-value.
3. **Expand**: create one new node per tree from top_k legal priors; write child_* slots.
4. **Backup**: add value along each tree's path (undo virtual loss). Update node + edge stats.

## Staged build (each stage validated before the next)
- **Stage 1 — batched engine primitives** (`alphatrain/batched_engine.py`), numpy [K,…],
  bit-identical golden tests vs scalar:
  - `label_components` (iterative min-label propagation) + `reachable` ← the de-risking piece.
  - `clear_lines_at` / line detection.
  - `apply_move` (move ball + clear at landed cells + spawn given RNG) + batched spawn sampler.
  - `build_obs` (18-ch) + `feature_value` (27-feat) + `legal_priors` (top_k).
- **Stage 2 — array-tree core** (`alphatrain/batched_mcts.py`): selection/expansion/backup
  vectorized; first version may LOOP the engine/leaf per-tree (reuse scalar numba) to get an
  end-to-end correct driver fast, then swap in Stage-1 batched primitives.
- **Stage 3 — validate** vs scalar (`scripts/validate_batched_mcts.py`): argmax agreement +
  TV on the 12 band states; tune until agreement ≥ ~11/12 and TV small.
- **Stage 4 — integrate** into mining as a NEW entry point (do not touch
  `gen_corrections_parallel.py` while the live run uses it): `scripts/gen_corrections_batched.py`.
- **Stage 5 — benchmark** M5 + L4; pick K; then optionally port hot primitives to torch-GPU
  (endgame: whole search on GPU, zero CPU bottleneck).

## Validation cadence
Every Stage-1 primitive: random-board golden vs scalar (exact). Stage 3: agreement/TV gate.
Never advance a stage on a failing gate.
```
PYTHONPATH=. python scripts/test_batched_engine.py        # strict, per primitive
PYTHONPATH=. python scripts/validate_batched_mcts.py      # agreement + TV vs scalar
```
