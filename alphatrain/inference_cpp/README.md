# Color Lines 98 — C++ engine

Native C++ for the deployed policy, built on **LibTorch** (PyTorch's official C++
API). Training stays in Python. Executables:

- **`infer`** — single-state inference demo + CPU/MPS benchmark (`src/main.cc`).
- **`eval`** — `native_eval_policy`: a full C++ game loop + batched greedy play,
  the C++ port of `scripts/eval_policy.py` (`src/eval.cc` + the game engine).
- **`mcts_eval`** — feature-value MCTS eval over a seed range (port of the
  `eval_parallel` MCTS path): game threads + one shared inference-server
  thread that coalesces requests into big MPS batches.
- **`mcts_selfplay`** — MCTS selfplay data gen: records visit distributions in
  the moves-schema JSON that `alphatrain/scripts/build_expert_v2_tensor.py`
  consumes directly (integration-tested end-to-end).
- **`mcts_crisis`** — crisis mining: bulk greedy probes to death → rewind
  anchors (recovery/prevention) → deep-MCTS replays. Two-phase, resume-safe.
- **`rollout_judge`** — adjudicates move pairs by died-within-H rates over R
  greedy rollouts (correction screening, ranking validation, target audits).
- **`game_test`**, **`feature_test`** — golden tests (no LibTorch needed): game
  engine + 27-feature leaf-value evaluator, bit-exact vs Python.

**NN value head (`--value-module`)**: `export_policy_value.py` fuses the policy
with a survival ValueHead into one TorchScript module returning
`(logits[B,6561], V[B])`; pass it via `--value-module` to mcts_eval /
mcts_selfplay / mcts_crisis and the MCTS leaf value comes from the head
(q_weight 2.0 = validated operating point) instead of the linear features.
Gate-validated 2026-07-09 (HISTORY 177): head-guided corpora improve the
policy where FV-guided ones regressed. Retrain the head per base model
(HISTORY 158) and re-export; ALWAYS verify exports against the checkpoint
(logit diff) before mining — a stale default export once poisoned a corpus.

LibTorch lives inside your venv's `torch`, so there's nothing extra to download;
`CMakeLists.txt` auto-locates it via the venv python.

## Build
```bash
source .venv/bin/activate          # from repo root

# export the TorchScript net + golden test vectors
python -m alphatrain.inference_cpp.export_ts --model alphatrain/data/<MODEL>.pt
python -m alphatrain.inference_cpp.export_game_golden      # game-engine goldens
python -m alphatrain.inference_cpp.export_feature_weights  # 27-feat FV weights + golden

cd alphatrain/inference_cpp
cmake -B build && cmake --build build -j               # builds all targets
```

## Run (from this dir, so it finds `data/`)
```bash
./build/game_test          # engine golden test: obs/legal/clear/LegalPriors (must PASS)
./build/feature_test       # 27-feature leaf-value evaluator vs Python (must PASS)
./build/infer              # single-state demo + CPU-vs-MPS benchmark

# native_eval_policy: greedy policy play over a seed range, score distribution
./build/eval --device mps --seed-start 50000 --seed-end 50256 --batch 256
#   flags: --model --device cpu|mps --seed-start/--seed-end (end EXCLUSIVE)
#          --batch --max-turns --fp32   (default: fp16 on MPS)

# MCTS eval (feature-value leaf; the eval_parallel MCTS path)
./build/mcts_eval --seed-start 775000 --seed-end 775048 --sims 100 \
    --q-weight 1.0 --early-stop --max-turns 12000 --threads 14

# MCTS selfplay data gen (visit distributions -> moves-schema JSON)
./build/mcts_selfplay --seed-start 900000 --seed-end 900040 --sims 1600 \
    --threads 14 --out-dir ../../data/selfplay_cpp_v1
# then: python -m alphatrain.scripts.build_expert_v2_tensor \
#           --games-dir data/selfplay_cpp_v1 --policy-only-data --output <tensor.pt>
```

MCTS notes: the per-search sim RNG is ours (not Python's MD5+PCG64), so visit
counts don't bit-match Python — validation is by score DISTRIBUTION over a seed
set (never per-seed; one different move forks the whole game). fp16 server
batching makes runs non-bit-reproducible run-to-run (same as Python's server).

## Correctness & the RNG caveat
The **deterministic** kernels (obs, legal-move mask, line-clear) are golden-tested
bit-for-bit against the authoritative Python (`game_test` → `max|diff| = 0`).

The **game RNG is NOT** matched to Python: `game/rng.py` uses numpy PCG64, whose
`choice()/integers()` internals are fragile to replicate, so this engine uses its
own RNG (SplitMix64). Consequence: for a given seed the C++ engine plays a
*different specific game* than Python, but the **same score distribution** (same
policy + rules). That's all the consumers (eval/selfplay/mining) need. Validated:
on the same model+seeds the distributions match (tail fractions ~identical,
medians close); per-seed games differ by design.

## Performance (vs `scripts/eval_policy.py`, fair capped comparison)
Both forward-bound (the long-tail games run solo at tiny batch and gate wall-clock
— same limitation as Python). Matched 256 seeds / batch 256 / cap 3000:

| precision | C++ | Python | C++ speedup |
|-----------|-----|--------|-------------|
| fp16 (default) | 24.9s | 27.0s | ~1.1× |
| fp32           | 36.5s | 44.8s | ~1.2× |

fp16 is the bigger lever (~1.5× over fp32). C++'s edge grows at smaller batch
(more host overhead per forward — the AI-hint regime). Compare speed with
`--max-turns` capped: uncapped wall-clock is dominated by whichever run's RNG
happens to deal the longest game.

## Files
```
export_ts.py              PolicyNet -> data/policy_ts.pt + example_obs/logits/legal.f32
export_game_golden.py     game-engine golden vectors (obs/legal/clear)
export_feature_weights.py 27-feature FV weights (from feature_value_weights_2y_nb.npz) + golden
src/main.cc               inference demo + benchmark
src/eval.cc               native_eval_policy (batched greedy eval)
src/mcts.h/.cc            MCTS core (port of alphatrain/mcts.py, feature-value leaf)
src/infer_server.h        shared batching inference-server thread
src/mcts_eval.cc          MCTS eval driver
src/mcts_selfplay.cc      MCTS selfplay recorder (moves-schema JSON)
src/feature_value.h/.cc   27-feature linear leaf-value evaluator
src/game.h/.cc            game engine (port of game/board.py)
src/obs.cc                18-channel observation (port of observation.py)
src/rng.h                 SplitMix64 (+ normal/gamma/Dirichlet)
src/game_test.cc          engine golden test    src/feature_test.cc  FV golden test
data/                     generated artifacts (git-ignored)
from_scratch/             OPTIONAL deep-dive: the net hand-written op-by-op with abseil
```

## Next
- Crisis mining in C++ (death recorder = frames-schema JSON from greedy eval;
  fix-mining = this MCTS at high sims from rewind anchors).
- **WASM/browser** build (likely ONNX Runtime) for deployment.
