# Color Lines 98 — C++ engine

Native C++ for the deployed policy, built on **LibTorch** (PyTorch's official C++
API). Training stays in Python. Two executables:

- **`infer`** — single-state inference demo + CPU/MPS benchmark (`src/main.cc`).
- **`eval`** — `native_eval_policy`: a full C++ game loop + batched policy play,
  the C++ port of `scripts/eval_policy.py` (`src/eval.cc` + the game engine).
- **`game_test`** — golden test for the game engine (no LibTorch needed).

LibTorch lives inside your venv's `torch`, so there's nothing extra to download;
`CMakeLists.txt` auto-locates it via the venv python.

## Build
```bash
source .venv/bin/activate          # from repo root

# export the TorchScript net + golden test vectors
python -m alphatrain.inference_cpp.export_ts --model alphatrain/data/<MODEL>.pt
python -m alphatrain.inference_cpp.export_game_golden   # game-engine goldens

cd alphatrain/inference_cpp
cmake -B build && cmake --build build -j               # builds all 3 targets
```

## Run (from this dir, so it finds `data/`)
```bash
./build/game_test          # engine golden test: obs/legal/clear vs Python (must PASS)
./build/infer              # single-state demo + CPU-vs-MPS benchmark

# native_eval_policy: greedy policy play over a seed range, score distribution
./build/eval --device mps --seed-start 50000 --seed-end 50256 --batch 256
#   flags: --model --device cpu|mps --seed-start/--seed-end (end EXCLUSIVE)
#          --batch --max-turns --fp32   (default: fp16 on MPS)
```

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
export_ts.py          PolicyNet -> data/policy_ts.pt + example_obs/logits/legal.f32
export_game_golden.py game-engine golden vectors (obs/legal/clear)
src/main.cc           inference demo + benchmark
src/eval.cc           native_eval_policy (batched greedy eval)
src/game.h/.cc        game engine (port of game/board.py)
src/obs.cc            18-channel observation (port of observation.py)
src/rng.h             SplitMix64
src/game_test.cc      engine golden test
data/                 generated artifacts (git-ignored)
from_scratch/         OPTIONAL deep-dive: the net hand-written op-by-op with abseil
```

## Next
- **fp16 everywhere** is in; **WASM/browser** build (likely ONNX Runtime) is the
  remaining deploy step.
- Wire `BestMoves` / the game engine into native selfplay + crisis mining (same
  batched-MPS pattern: stack ~256 states, one forward, one masked argmax).
