# Color Lines 98 — C++ policy inference

A from-scratch C++ forward pass for the `PolicyNet` (the deployed Color Lines
policy), built against [abseil](https://abseil.io). Goal: a small, dependency-light
inference engine that runs the policy fast at batch=1 on CPU (the AI-hint regime)
and compiles to WASM for the browser. Training stays in Python; this is inference only.

## Layout
```
export_weights.py   # Python: dump PolicyNet -> data/weights.bin + a golden test vector
CMakeLists.txt      # build (fetches abseil)
src/net.h / net.cc  # Tensor, blob loader, ops (Conv2d done; BN/ReLU/Forward TODO)
src/main.cc         # milestone harness: run an op, diff against the golden
```

## Build & run
```bash
# 1) export weights + golden from a checkpoint (run from the repo root, venv active)
python -m alphatrain.inference_cpp.export_weights \
    --model alphatrain/data/pillar3k_small128_epoch_15.pt

# 2) build (first build compiles abseil; ~minutes)
cd alphatrain/inference_cpp
cmake -B build && cmake --build build -j

# 3) run (from this dir so it finds data/)
./build/infer
```
Expected at Milestone 1: `stem conv: max|diff| = …e-07 -> PASS ✅`.

## Implementation milestones
The forward pass is built op-by-op; the **golden vector** (PyTorch output on a real
board, in `data/golden.bin`) is the correctness oracle at each step.

1. **Stem conv** — `Conv2d(obs, stem.0.weight, pad=1)` vs `golden["stem_conv_out"]`. *(done — the harness checks it)*
2. **ReLU** — `net.cc::ReluInPlace`. (one-liner)
3. **BatchNorm** — `net.cc::BatchNorm` (per-channel affine, inference form). Verify a stem→BN→ReLU chain against a golden you add for it.
4. **ResBlock** — pre-activation residual; compose into the body loop.
5. **Head + reshape** — 1×1 convs to 81 channels → 6561 logits.
6. **`Forward`** — wire it all up; flip `kRunFullForward` in `main.cc` and match `golden["logits"]` to ~1e-2.
7. **Obs builder** — port `_build_obs_core` (board+next_balls → 18×9×9) so C++ runs end-to-end from a board, not a precomputed obs.

Later (after correctness): legal-move masking + argmax (the actual move), then
optimization (contiguous loops, SIMD, fixed-size buffers), then an Emscripten/WASM target.

## Binary format (`CLNW`)
`magic 'CLNW'`, `uint32 num_tensors`, then per tensor: `uint32 name_len`, name,
`uint32 ndim`, `int32[ndim] dims`, `float32[prod] data` (row-major). Both
`weights.bin` and `golden.bin` use it; `net.cc::LoadBlob` reads it.
