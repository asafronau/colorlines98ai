# GPU batched-MCTS throughput — request for ideas

We built a GPU batched-tree MCTS to speed up "teacher" search for a Color Lines 98 RL agent.
It is **correct** but **slower end-to-end than our CPU baseline**, even on an L4. We've localized
the bottleneck and want ideas before we either fix it or shelve it. Please poke holes and suggest
approaches we haven't considered.

## 1. The goal

We mine **teacher labels** to lift the policy's failure floor (worst-case early-death games).
A "label" = take one game state, run a **deep, widened MCTS** on it, and emit the **soft root
visit distribution** as a distillation target for the policy. Throughput unit = **trees/sec**
(states relabeled per second). More trees/sec = more/better training signal.

**Production baseline (must beat):** a scalar (per-tree) CPU MCTS, 16 parallel worker processes
on an M5 Max (18 CPU cores). Measured **3.56 trees/sec** aggregate.

## 2. The workload (why it's unusual)

- **Board:** 9×9, 7 colors. Tiny.
- **Action space:** move one ball (source cell → target cell) = 81×81 = **6561 actions**; legality
  by connected-empty-region reachability. We restrict to **top_k=300** prior-ranked legal moves per node.
- **Net:** a policy-only ResNet — input 18×9×9, **10 residual blocks, 256 channels**, policy head →
  6561 logits. ~7M params. The **value** at a leaf does NOT come from the net; it's a cheap
  **linear evaluator over 27 hand features** (`v = w·feat(board)`). (History: the learned value head
  underperformed the linear one, so MCTS uses the linear value + the net's policy prior.)
- **Search:** PUCT, `c_puct=2.5`, value mixed with `q_weight=2.0`. **sims=4800** per tree,
  Dirichlet root noise. **Open-loop / stochastic**: spawns are random, so we DON'T store a child
  board per node — each simulation **re-simulates the board from the root** applying the chosen
  moves (the env is cheap). Scalar averages **3 determinizations** (seeds) per state.
- Tree is deep and narrow-ish: typical descent depth is tens of plies before hitting a leaf.

## 3. What we built (GPU version)

One process drives **K independent trees** as torch tensors on the GPU.

- **Edge-based tree arrays** `[K, N, W]`, `W=top_k=300`, `N=sims+2=4802`: per-edge child action,
  prior, visit count, value sum, child-node-id (mctx-style: a child node id is allocated only when
  it's itself expanded). Per-node visit/value sums `[K, N]`.
- **Vectorized PUCT descent**: `cur_board[K,9,9]` re-simulated from root each sim; an `active[K]`
  mask; at each depth step we gather the current node's 300 edges, decode moves, compute legality
  (connected-components + reachability on `cur_board`), PUCT-argmax, and `apply_move` for the trees
  that picked a real move.
- **Engine primitives ported to torch** (all validated bit-identical / partition-identical vs a
  numpy reference): connected-components (Shiloach-Vishkin hooking + pointer-jumping, ~6 iters),
  reachability (gather + neighbor-compare), line-clear (padded board + cumprod), apply_move
  (move + clear + spawn with torch RNG).
- **One batched NN forward per sim** over the K leaf boards.
- **Leaf eval is on CPU, per-leaf** (v1 shortcut): each sim, we copy the K leaf boards GPU→CPU and
  Python-loop `build_observation` (18-ch) + `feature_value` (27-feat, numba) over the K leaves,
  then the NN forward runs on GPU. Expand + backup also Python-loop over K.

Execution model: **eager PyTorch**, driven by a Python `for sim in range(4800)` × `for depth in
range(≤96)` loop, with per-tree Python loops for leaf/expand/backup.

## 4. Measured numbers

**Isolated engine primitives** (the micro-bench that made us optimistic) — L4, per-call, GPU vs
same-machine numpy:

| K | net "descent-step" (1×CC + 1×reach + 4×clear) | note |
|---|---|---|
| 2048 | **8.7× faster on GPU** | GPU wall-time ~flat in K |
| 4096 | **17.1× faster on GPU** | GPU not saturated |

**Full search end-to-end** — L4 24GB, sims=800 (not yet the production 4800), K trees at once:

| K | wall (s) | trees/s | vs M5 16-worker (3.56/s) | peak GB |
|---|---|---|---|---|
| 256 | 140.4 | 1.82 | 0.51× | 1.0 |
| 512 | 196.0 | 2.61 | 0.73× | 1.9 |

(The production sims=4800 run is in progress; we expect ~these ÷6 in trees/s, i.e. **~0.4 trees/s**,
because wall scales ~linearly in sims — plus a bit worse since late sims descend deeper trees.)

## 5. Where the time goes (the important part)

The full-search wall is **NOT flat in K**. Fitting a line through the two points:

```
wall(K) ≈ 84.8 s  +  0.217 s · K          (at sims=800)
          \_______/    \___________/
           flat term    O(K) term
```

- **O(K) term ≈ 0.217·K** = the **per-leaf CPU loop** (`build_observation` + `feature_value` +
  GPU↔CPU transfer, one pass over K leaves per sim). Invisible at K=64 (3.5% in an early profile),
  but **57% of the wall at K=512**. This is the obvious first fix (vectorize leaf eval in torch).

- **Flat term ≈ 85 s** = the **sequential descent**: `sims × avg_depth` engine "steps", each a stack
  of small tensor ops on a 9×9×K board. We believe this is **almost entirely kernel-launch overhead**,
  not arithmetic:
  - The NN forward (the only real FLOPs) is negligible: 800 forwards × batch-512 through a
    10-block/256-ch ResNet on 9×9 ≈ **<1 s total**.
  - Launch-count estimate: per descent step ≈ CC(8 iters × ~10 ops) + reach(~20) + apply_move
    (4 clears × ~50 ops) ≈ **~350 kernel launches**. With ~20 avg depth that's ~7000 launches/sim ×
    800 sims ≈ **5.6M launches**. 85 s / 5.6M ≈ **~15 µs/launch** — which is exactly the eager-CUDA
    per-op launch overhead. **The flat term is the launch overhead, almost line-for-line.**

So: tiny boards (no arithmetic to hide latency) + a Python-driven eager loop issuing millions of
sub-microsecond-of-work kernels ⇒ **launch-bound**.

## 6. The ceiling, even if we fix the easy thing

If we vectorize the leaf eval (kill the O(K) term entirely), wall → the ~85 s flat floor (which
itself scales with sims: ~509 s at sims=4800). Then:

| K | trees/s @ sims=4800 (leaf-vectorized, optimistic) |
|---|---|
| 1024 | ~2.0 |
| 2048 | ~4.0 (but won't fit: see below) |

And **memory caps K**: tree arrays `[K, 4802, 300]` (×5 child arrays, int16/fp16 = 2× shrunk
already) are ~11 GB at K=512/sims=4800, so **K≈1024 is the max on 24 GB**. ⇒ even optimistically,
single-L4 lands around **~2 trees/s < 3.56**. One L4 does not beat the 16-worker M5 fleet, and you'd
need ~6 L4s to match one M5 on cost. That's why we paused to ask for ideas.

## 7. Constraints

- The **scalar CPU miner is the live production tool** and must keep running untouched (all GPU work
  is in separate files).
- **Correctness contract:** engine primitives must be bit-identical to scalar; the full search need
  not be (spawn RNG + fp16 batch differ) but must match on **argmax + low total-variation of root
  visits** vs scalar. (GPU search currently smoke-validated on 4 states; full TV validation pending.)
- Hardware on hand: **M5 Max** (18 CPU, 40-core GPU, 48 GB unified, Metal/MPS — no float64, weak
  scatter), **Colab L4 24GB / T4 16GB / A100 40GB**. Comfortable in **PyTorch**; willing to learn
  **JAX** or write a **custom kernel / Rust** if the payoff is clear. (User previously rejected a Rust
  rewrite of the engine, but it's back on the table if it's the right answer.)

## 8. What we're considering — please critique / extend

1. **Graph-capture the eager loop.** The flat term is launch overhead, so **CUDA Graphs** or
   `torch.compile`(mode="reduce-overhead") around the per-sim descent could cut it ~10×+. But our
   descent has **data-dependent control flow** (variable depth, per-tree `active` masks, dynamic node
   allocation). How well does CUDA-graph capture / `torch.compile` handle a fixed-`max_depth` padded
   descent with masks? Worth converting the variable loop to a fixed-length masked loop to enable capture?

2. **JAX + `lax.scan` / `jit`, or DeepMind's `mctx`.** `mctx` is literally batched MCTS jit-compiled
   on accelerators; a single `jit` over the whole search eliminates per-op Python/launch overhead.
   Our open-loop, linear-value, top_k-restricted, 6561-action, stochastic-spawn setting is non-standard
   — does mctx fit, or is a custom JAX scan the move? Is the dynamic tree (node allocation) expressible
   in `jit` (fixed `[K,N,W]` buffers + masking)?

3. **Fuse the engine.** ~350 launches/descent-step is the enemy. Could connected-components +
   reachability + clear be written as **one or a few custom fused kernels** (Triton / CUDA / Pallas)?
   Or restructure: do we even need full CC every step, or can reachability be maintained incrementally
   as the board changes by one move?

4. **Reduce sequential depth cost.** The descent is `sims × depth` sequential. Anything that cuts the
   *sequential* dimension: batched/parallel simulations within a tree? Larger leaf batches per
   descent? A shallower-but-wider search that trades depth for K-parallelism without hurting label quality?

5. **Vectorize leaf eval** (kills the O(K) term) — clearly necessary but, per §6, not sufficient alone.

6. **Is GPU just the wrong tool here?** Tiny 9×9 boards + deep sequential MCTS may be fundamentally
   launch/latency-bound. Is the honest answer a **native CPU (Rust) MCTS** (zero launch overhead,
   trivially parallel across cores/machines), with the GPU reserved only for a batched policy-prior
   server? Or does graph-capture/JAX genuinely rescue the GPU approach for this workload?

**Core question:** given the bottleneck is ~5.6M tiny eager kernel launches (≈85 s of pure launch
overhead at sims=800), what's the highest-leverage path to a real throughput win — CUDA Graphs,
`torch.compile`, JAX/mctx, custom fused kernels, an algorithmic restructure of the descent, or
conceding that tiny-board deep MCTS belongs on CPU?

---

# UPDATE — after your (ChatGPT's) review. Conceded points + new tooling + early data.

Your critique was right on the substance. Specifically:

- **The "~2 trees/s ceiling" in §6 is NOT established and we've stopped claiming it.** The leaf eval
  does `.cpu().numpy()` **every sim** (in `_evaluate` and again in the expand/backup block — ~7 syncs/
  sim). In eager CUDA those syncs stop the CPU from running ahead and queuing kernels, so they
  **expose** launch latency that would otherwise hide. Removing them may shrink the "flat" term too,
  not just the O(K) term. The ceiling is unproven until the sync is gone.
- **The launch-bound claim was inferred from 2 K points.** Now measured directly (below) and the K
  sweep is 128/256/512/1024.
- **The NN-forward "<1 s" was asserted, not measured.** Now an isolated microbench (`--nn-bench`):
  on mps it's 5.1 ms/fwd @K=64, 8.2 ms/fwd @K=128 → 4–7 s of pure forward over 800 sims. Not the
  bottleneck, but not negligible; we'll have the L4 number.

**Tooling we built to settle it** (all in `scripts/bench_full_search.py`, self-contained tarball):

- `--profile` — `torch.profiler` trace: kernel/op table + total launch count per sim + CPU syncs.
  **Early local (mps, K=24): 96 distinct ops, 1.84M calls / 100 sims = ~18,400 kernel launches per
  sim.** Direct confirmation that the search is launch-dominated (millions of sub-µs kernels), even
  before the L4 trace. This is the single most important artifact.
- `--ablate descent_only` — prefills the tree and runs `sims × descent` with **no leaf eval / expand /
  backup → zero CPU syncs**. Isolates the pure GPU descent cost. **Caveat we found:** it forces
  descent to `max_depth` every sim, whereas the real search hits unexpanded leaves shallowly in early
  sims — so it's an **upper bound** on the descent component, not a matched subtraction. The decisive
  read is the **ratio at sims=4800**: descent_only ≪ full ⇒ CPU-sync/leaf-bound (vectorize leaf eval);
  descent_only ≈ full ⇒ descent-launch-bound (graph capture / fused kernels).
- `--ablate no_net` — full search minus the NN forward (isolates the forward's contribution).

**Adopting your other suggestions into the plan:**

- **Progressive widening / `W_internal`.** We currently allocate `W=300` children at *every* node.
  You're right it's wasteful — most internal nodes need far fewer than 300 legal moves. `W_root=300,
  W_internal=64/128` attacks both the `[K,N,W]` memory wall (raising the K cap) and per-depth
  argmax/legality cost. We'll prototype and validate root-visit TV vs scalar.
- **Closed-loop / node-board caching.** The biggest *algorithmic* waste is open-loop re-simulating
  from the root and recomputing CC+reachability **every descent step**. If determinized/closed-loop
  semantics are acceptable for *teacher mining* (we think they likely are — the scalar teacher already
  averages only 3 determinizations), we can store the board + precomputed legal top-k **at expansion**,
  turning each descent step into a gather instead of a full CC/reachability/apply_move. Board storage
  (`[K,N,81]` int8) is cheap next to the `[K,N,W]` edge arrays. This removes most of the per-step
  engine launches and is plausibly higher leverage than CUDA Graphs.

**Revised plan (your ordering, which we agree with):**
1. Vectorize leaf eval fully on GPU + remove all per-sim CPU copies (kills the O(K) term *and* the
   sync). Needs a GPU `legal_priors` (CC/reachability over all top-k candidates → uniform/softmax
   prior) — golden-tested vs the numba `_legal_priors_jit`.
2. Run `--ablate`/`--profile` to isolate descent vs expand vs backup vs NN vs sync on L4.
3. `W_internal=64` (root 300) + root-visit TV check vs scalar.
4. If open-loop is negotiable: prototype closed-loop node-board caching under fixed determinizations.
5. Only if millions of tiny kernels remain after the above → fused custom kernels (Triton/Pallas) or
   a JAX `lax.scan` rewrite with fixed arrays. (We agree `mctx` is likely a poor fit: nonstandard
   transition legality, top-k action restriction, stochastic spawns, custom linear value.)

The CPU miner keeps running throughout; the sims=4800 GPU number is treated as **provisional** until
the per-sim CPU sync is removed. We have not given up on the GPU path — we're de-risking it properly.

**Anything you'd reorder, or failure modes in the closed-loop/`W_internal`/leaf-vectorization plan we're
missing?**

---

# RESULT — L4 `torch.profiler` trace. Your hypothesis was right; the "ceiling" was a mirage.

Ran `--profile` on the L4 (K=256, sims=100). The picture **reverses** our earlier "descent is
launch-bound / GPU is capped at ~2 trees/s" framing:

```
Self CUDA time total:  2.578 s     <- the GPU does only 2.58s of real work over 100 sims
Self CPU  time total: 18.686 s     <- wall ~18.2s; the GPU sits IDLE ~86% of the time
total ops: 211 distinct, 4,676,083 calls / 100 sims = 46,761 kernel/op calls PER SIM
```

So it is **CPU-orchestration-bound, not GPU-bound and not descent-launch-bound.** Top CPU costs:

| op | CPU total | # calls | what it is |
|---|---|---|---|
| `aten::copy_` | 5.61 s (30%) | **287,826** | per-sim `.cpu()` syncs + per-child `.to(dev)` writes in the expand loop |
| `aten::to` / `_to_copy` | 3.9 s (21%) | 112,806 | host↔device transfers |
| `aten::index` | 2.42 s (13%) | 34,479 | descent gathers + the per-tree (`for k in range(K)`) expand/backup loops |
| `aten::index_put_` | 1.25 s (7%) | 26,632 | clear scatter + backup |

GPU side (the small 2.58 s): the **NN forward is ~59% of it** (cudnn_convolution 0.74 s + cutlass
gemm 0.56 s + batchnorm) — so your point #5 was right to demand a measurement; it's the dominant
*GPU* cost but irrelevant while the GPU is 86% idle.

**Conclusions we're now acting on:**
1. The engine work is **cheap on GPU** (2.58 s/100 sims, most of it the unavoidable NN forward). The
   v1 throughput tax is entirely the **per-sim host transfers + O(K) Python expand/backup loops** —
   exactly the sync you flagged. The "~2 trees/s ceiling" does not exist; it was a v1 artifact.
2. **Stage 1 = vectorize leaf eval + expand + backup on GPU, delete all per-sim `.cpu()`** → collapses
   the CPU wall toward the GPU floor, and (since the GPU is 86% idle) lets us push K to 1024–2048 with
   `W_internal`. Estimated landing ≈ **M5 parity (one L4 ≈ the 16-worker fleet)**.
3. **Stage 2 = CUDA Graphs / `torch.compile` on the fixed-`max_depth` masked descent** to kill the
   remaining per-op CPU *dispatch* (~4 µs × hundreds of ops/sim). That's where the GPU's idle headroom
   should convert into the 5–20× we originally wanted.

The `descent_only`/`no_net` ablations became moot — the profiler answered it more directly.

**Questions for you on the build:** (a) For Stage 1's keystone — a GPU `legal_priors` (connected
components + reachability over the top-k candidate moves → softmax prior) — any pitfalls vectorizing
"which empty target cells are reachable from each occupied source" across `[K, 6561]` candidates that
you'd watch for? (b) For Stage 2, is a fixed-length padded descent with per-tree `active` masks the
right shape to make CUDA-graph / `torch.compile` capture actually stick, given dynamic node
allocation? (c) Does the closed-loop node-board cache (store board + legal top-k at expansion) change
your Stage ordering now that we know transfers/loops — not descent compute — are the cost?
