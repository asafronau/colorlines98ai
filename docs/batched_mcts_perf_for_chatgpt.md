# GPU batched-MCTS throughput — request for ideas

> **READING NOTE: UPDATE 3 (bottom) is the CURRENT state. Sections 5-6, 'RESULT', and UPDATE 1-2 are historical/superseded — kept for the trail.**


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

> **⚠️ SECTIONS 5–6 ARE SUPERSEDED by the L4 profiler result at the bottom ("RESULT — L4
> torch.profiler trace").** The "flat term = launch overhead" and "~2 trees/s ceiling" reasoning
> below was an extrapolation from 2 K-points and turned out WRONG: the profiler shows the GPU is
> only 14% utilized and the wall is dominated by per-sim CPU `.cpu()` syncs + O(K) Python
> expand/backup loops, not descent kernel launches. Kept for the record; read §RESULT for the
> actual diagnosis and plan.

## 5. Where the time goes (the important part) — [SUPERSEDED, see §RESULT]

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

---

# UPDATE 2 — Stage 1 BUILT + measured on L4. It did NOT improve throughput. We need your read.

We implemented the full Stage-1 vectorization you recommended and golden-tested it. Then we measured
on the L4. **The throughput did not move.** This contradicts the earlier diagnosis that the per-tree
Python loops + host syncs were the cost, so we're bringing it back to you before picking Stage 2.

## What we built (all golden-tested EXACT vs the numba reference)

- `legal_priors_t` (GPU): full legal set 1536/1536, prior err 1e-7, **top-300 selection 100% identical**.
  Uses CC-once + neighbour-component membership + full-6561 mask then top-k, exactly as you specified.
- `build_observation_t` (GPU 18-channel obs): **bit-exact**, 0.0 error over 1024 boards.
- Rewrote the search: obs + logits + legal_priors + **expand** + **backup** all on-device.
  - Expand: vectorized — every tree writes children into its next-free node slot for all K at once;
    only expanders bump the counter / link the parent edge. **No `for k in range(K)`, no per-child
    `.to(dev)`, no `nonzero`.**
  - Backup: single `scatter_add` over the whole path. **No Python loop, no `.item()`.**
  - Dirichlet on GPU; coarse early-exit (one `bool(active.any())` sync per 8 descent steps).
  - The ONLY remaining per-sim host sync is the feature-value round-trip (one batched `.cpu()` →
    numba `board_features_with_next` → `.to(dev)`). We left it as the measured-or-not question.
- Argmax still EXACT vs scalar on the smoke states (5405/4064/3274/3515).

## The L4 result (K=256, sims=100, torch.profiler)

```
                     old v1      Stage-1 (now)
ops / sim            46,761      46,747          <- essentially UNCHANGED
Self CUDA time        2.58 s      2.715 s
Self CPU time        ~18.7 s     20.8 s          <- the wall; CPU-dispatch-bound
GPU utilization       14 %        ~13 %
trees/s @ sims=4800  ~0.29       ~0.29           <- no change
```

The op *composition* changed exactly as designed — `aten::copy_` 287k→198k, the per-tree loop is
gone, replaced by vectorized `aten::add` (scatter backup, 81k) and `aten::bitwise_and` (legal mask,
74k) — but the **total stayed ~46.7k ops/sim**. NN forward (conv+cutlass+bn) is ~55% of the small
2.7 s GPU time; everything else is the descent's elementwise/index/scatter ops.

## Our revised diagnosis (please confirm or correct)

The per-tree Python expand loop was a **red herring** — it was a small fraction. The real cost is the
**descent itself**, which was always vectorized: ~depth × (connected-components + reachability +
`apply_move`'s 4 line-clears + PUCT scoring) = tens of thousands of tiny **eager** ops per sim. The
GPU executes them in 2.7 s, but the CPU can't *dispatch* ~4.7M ops/100-sims fast enough (~4 µs each →
~20 s). So it's **eager-dispatch-bound on the descent**, and vectorizing the leaf/expand/backup
trimmed the wrong 5 %. Stage 1 was necessary groundwork (correct, K-independent, nearly sync-free →
graph-capturable) but not sufficient.

Correctness of the rewrite: full argmax+TV vs scalar = 3/4 agree, meanTV 0.298 (this run used 1 scalar
determinization, vs an earlier 6/6 at 3 seeds — the one divergence at depth-60 is most likely
open-loop spawn-RNG variance, but we'll firm it up).

## The fork — which Stage 2, and in what order?

Both seem to ultimately require porting the feature-value (`board_features_with_next`, ~250 lines incl.
next-ball spawn simulation) to GPU, since that `.cpu()` is the last sync either way.

1. **CUDA Graphs / `torch.compile`** on the (open-loop) descent — capture the ~46k-op per-sim
   sequence, replay with near-zero CPU dispatch. Since the GPU only needs ~2.7 s, naively this could
   reach ~1.8 trees/s @K=256 and scale to **>M5 at K=1024**, keeping semantics. Blockers to remove:
   the feature-value `.cpu()`, the early-exit sync (→ truly fixed-length descent), and dynamic node
   allocation inside the captured region (n_nodes is a tensor; is tensor-indexed in-place scatter
   into fixed `[K,N,W]` buffers capturable, or does the changing `new_id` break the graph?).
2. **Closed-loop node-board caching** — store the board + precomputed legal top-k at expansion so each
   descent step is a *gather* instead of re-running CC/reachability/apply_move. This **deletes** the
   bulk of the 46k ops rather than dispatching them faster. Bigger structural win, but a **semantic
   change** (determinized per node) needing TV re-validation. (Teacher mining may tolerate it — the
   scalar teacher already averages only 3 determinizations.)

**Questions:**
- (a) Given the bottleneck is eager *dispatch* of ~46k ops/sim (GPU itself is 87% idle), is CUDA
  Graphs the right first lever, or will the dynamic node allocation + variable descent depth make
  capture impractical without first going closed-loop?
- (b) For CUDA-graph capture: can a per-sim region with in-place scatter into fixed `[K,N,W]` tree
  buffers (indices computed from tensors, never `.item()`'d) be captured and replayed across sims,
  even though the *logical* tree grows? Or is the standard pattern to capture only the
  fixed-structure inner kernels (CC / reachability / clear / PUCT) and leave the outer loop in Python?
- (c) Is closed-loop caching actually the higher-leverage move here (deletes ops vs. dispatches them
  faster), and would you do it *before* attempting graph capture given it also shrinks the op count
  that any graph would have to replay?
- (d) Anything that says "stop — tiny-board deep MCTS at 4800 sims is the wrong job for a GPU; the
  46k-eager-op descent is intrinsic, put this on a multicore-CPU / Rust path and use the GPU only for
  a batched prior server"? We want the honest version.

---

# UPDATE 3 — Full-sim CUDA-graph capture WORKS (6.4x) but loses to the production search. Why, + a fork.

We followed your spike recipe to completion and captured the FULL per-sim. It captures and replays
correctly. But sims-matched against the real workload it does NOT beat the CPU miner, for a reason
specific to MCTS, and we want your read before picking the next lever.

## What we did (all the capture blockers, in order)

Made the per-sim fully capture-safe and bisected the failures with a per-component probe (each
component captured in its own subprocess, since a failed capture corrupts the CUDA context):
1. `torch.rand` in `apply_move` -> "Offset increment outside graph capture". Fixed: deterministic
   RNG-free spawn for the spike (randomness doesn't affect timing).
2. `build_observation` had a hidden `bool(.any())` + boolean-index in the next-ball channels (correct
   eagerly, breaks capture). Fixed: masked full-K writes. Probe then showed all 7 components capture OK
   (label_cc, build_obs, net_forward, legal_priors, apply_move, scatter_add, topk).
3. Node-buffer overrun: `n_nodes` increments in place every `per_sim`, and the graph path runs
   per_sim more than `sims` times (warmup + capture + replays), so it ran past the `[K,N,W]` tree
   buffer -> device-side index assert. Fixed: clamp the slot + size N with headroom.

## Result (L4, full per-sim captured, FAKE gpu value)

```
            eager(fixed-depth)   graph        speedup
K=256          136.4 s           21.4 s        6.4x
K=512          136.3 s           24.2 s        5.6x
```

Capture works; 6.4x over the (fixed-depth) eager baseline. **But that eager baseline is a strawman**
(fixed depth, nobody runs it). Two honest comparisons:

- **vs the real workload (sims-matched).** The graph replays one sim per call, so wall is linear in
  sims. At the mining sims=4800 (48x the 100 measured): K=512 -> 24.2*48 ≈ 1162 s for 512 trees =
  **0.44 trees/s**; K=256 -> **0.25 trees/s**. **M5 = 3.56 trees/s. So the captured graph is ~8x
  BELOW the CPU miner at equal sims.** K=1024 (~0.8 trees/s) still loses.
- **vs the production eager search** (the one with coarse early-exit, profiled earlier at K=256
  sims=100 = 18.4 s wall, 2.7 s GPU, dispatch-bound): the captured graph is **21.4 s — slightly
  SLOWER**.

## Diagnosis (please sanity-check)

Graph capture did NOT lose to dispatch — it eliminated dispatch (the 21.4 s is now almost all GPU
compute). It lost to the **fixed-depth requirement clashing with MCTS's growing tree**: capture needs
a static loop, so every sim runs all 64 descent steps. But in MCTS the *early* sims descend a young,
shallow tree (1-2 plies); only late sims are deep. The production early-exit search naturally does
shallow early sims (its 2.7 s total GPU compute reflects mostly-shallow descents). The fixed-depth
graph does ~8x more descent step-executions (64 x 100 vs ~avg-low x 100), so the ~6x capture win is
canceled by ~8x more compute. **Dispatch is solved; we're now compute-bound on the descent engine
ops (CC + reachability + apply_move's 4 clears), executed depth x sims times.**

## The fork

1. **Block capture (open-loop, keeps semantics).** Capture a SMALL fixed descent block (e.g. 8
   steps) as a graph; Python-loop it with an early-exit `bool(active.any())` check *between* blocks
   (one sync per 8 steps). Dispatch-free within a block AND depth-matched (early sims do 1 block then
   stop). If wall approaches the ~2.7 s real GPU compute, K=512-1024 could hit ~3-6 trees/s -> beat M5.
2. **Closed-loop node-board caching (semantic change).** Store board + precomputed legal top-k at
   expansion so each descent step is a gather, DELETING the per-step CC/reachability/apply_move (the
   compute floor itself, not just its dispatch). Bigger win; changes the teacher (determinized);
   needs TV re-validation (current open-loop TV already only ~0.30).

## Questions

- (a) Is the diagnosis right — fixed-depth capture is fundamentally mismatched to MCTS's
  shallow-early / deep-late descent, so full-sim capture can't win even though the mechanism works?
- (b) Block capture (capture an 8-step descent block, loop with inter-block early-exit) — is this the
  right way to get BOTH dispatch-free execution AND depth-matched work, or does the inter-block sync
  reintroduce enough overhead to negate it? Any better granularity (capture leaf-eval/expand/backup
  as one graph, descent steps as another)?
- (c) Given we're now COMPUTE-bound (not dispatch-bound), is closed-loop caching actually the only
  thing that moves the needle (it deletes the ops; block-capture only schedules them better)? Would
  you go straight to closed-loop now?
- (d) Or is this the point where the honest answer is "even optimally-scheduled, K-parallel deep MCTS
  on 9x9 boards is ~the same total work as the CPU fleet does, and a single L4 just doesn't have the
  throughput; keep mining on CPU and use the GPU only for a batched prior/eval server"?

---

# UPDATE 4 — Closed-loop caching: throughput lever works, but the determinized teacher looks UNFAITHFUL.

We built closed-loop node-board caching (your option 2 / the post-block-capture lever). It does what
you predicted on *speed*, but the *fidelity* gate is the deciding factor and the early signal is bad.

## What we built
Each node CACHES its board + its legal top-k children (computed once, at expansion). Descent is then
a pure gather + PUCT walk down cached nodes — **no per-step CC / reachability / apply_move**. The
engine ops (1 apply_move + 1 CC + 1 NN forward + 1 legal_priors) run **once per sim**, at the single
expansion, instead of depth-times. Board cache is cheap (`node_board [K,N,9,9]` int8, ~0.2 GB).
**Semantic change:** spawns are determinized per node (a node's board is sampled once at creation and
reused on every revisit).

## Throughput (L4, eager, real value) — the engine cut is real
| | @sims4800 vs M5 |
|---|---|
| open-loop eager (production) | ~0.08x |
| open-loop block-capture (best) | 0.81x |
| **closed-loop EAGER** K=512 / K=1024 | 0.44x / 0.52x |

Closed-loop eager is **3.5x faster than open-loop eager** (K=256: 5.3 s vs 18.4 s) — deleting the
per-step engine ops worked. It's *below* block-capture's 0.81x only because it's still **eager**
(dispatch-bound: 64 cheap gather steps + the expansion, all eager-dispatched, + a per-sim feature-value
`.cpu()`). **Key point:** closed-loop's descent is gather-only, so the fixed-depth full-sim capture
that *backfired* on open-loop (64 expensive engine steps) is now 64 *cheap* steps. closed-loop (cuts
compute) + capture (cuts dispatch) together should clear M5 by several×. So speed is not the blocker.

## The fidelity gate — the actual blocker (preliminary, UNFAVORABLE)
Closed-loop vs scalar (re-sampling) MCTS, mps, sims=200-400, 1 scalar seed, on real crisis band states:
**0/N argmax agreement, meanTV ~0.63** — vs open-loop's ~0.27 (open-loop re-samples like scalar). So
closed-loop diverges ~2.3x more from the faithful (scalar) teacher.

We think this is a **real bias, not low-sims noise**: the true game is stochastic, and open-loop /
scalar MCTS average value over the spawn *distribution* (E[value | move]). Closed-loop freezes ONE
spawn per node, so each subtree explores a single realized future — the root's 300 children each get
one frozen spawn and revisits can't average them out. The visit distribution then reflects "value
under this one sampled future," which is a different (biased) objective. For a *teacher we distill
into the policy*, that bias could mislead.

Proper gate (sims=4800, 3 scalar seeds, n=8) is queued; more sims/seeds may pull TV down some, but we
expect the structural bias to persist.

## Questions
- (a) Is the determinization-bias diagnosis right — closed-loop optimizes a single spawn realization
  rather than the expectation, so for a *stochastic* game it's a systematically biased teacher (not
  just noisier), and the ~0.63 TV reflects that?
- (b) Middle grounds worth it, or do they kill the speed win? (i) **M determinizations per node**
  (sample M spawns at expansion, average their value / store M boards) — restores expectation but
  re-introduces ~M× the expansion compute + storage; (ii) **open-loop near the root, closed-loop in
  the deep tail** (where spawn-averaging may matter less); (iii) accept the bias because *crisis* move
  choice is more forced (escape-or-die) and less spawn-dependent than mid-game.
- (c) If the full gate confirms TV ~0.5+, is the honest conclusion "for this stochastic game a single
  L4 can't be both fast AND a faithful deep-MCTS teacher — keep mining on the CPU fleet, use the GPU
  as a batched prior/eval server"? Open-loop tops out at 0.8x M5 (faithful but slow); closed-loop is
  fast but biased. Is there a fourth option we're missing?
- (d) Or: is determinized (closed-loop) MCTS actually a *defensible* teacher for distillation on its
  own terms (afterstate-style), such that we should judge it by downstream policy improvement (does
  distilling its labels lift the floor?) rather than by TV-vs-open-loop — i.e., is TV-vs-scalar even
  the right gate here?
