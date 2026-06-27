# Color Lines 98 — C++ policy inference

Run the deployed policy net from C++ using **LibTorch** (PyTorch's official C++
API). Training stays in Python; this is inference only. The whole engine is
`src/main.cc` (~40 lines) — LibTorch provides conv/batchnorm/relu, so there is
no math to hand-write.

## Why LibTorch
The model is already a PyTorch net. We export it once to a self-contained
TorchScript file (`policy_ts.pt`), and C++ just loads and runs it. LibTorch
already lives inside your venv's `torch` package, so there's nothing extra to
download.

## Build & run
```bash
# 0) from the repo root, venv active
source .venv/bin/activate

# 1) export the net + a test vector (TorchScript module + example obs/logits)
python -m alphatrain.inference_cpp.export_ts \
    --model alphatrain/data/pillar3k_small128_epoch_15.pt

# 2) configure + build (point CMake at the LibTorch inside your venv)
cd alphatrain/inference_cpp
cmake -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')"
cmake --build build -j

# 3) run (from this dir, so it finds data/)
./build/infer
```
Expected output:
```
predicted move index <n>  (source cell .., target cell ..)
max|diff| vs PyTorch = 0.000…   PASS ✅
```

## Files
```
export_ts.py     Python: PolicyNet -> data/policy_ts.pt + example_obs/logits.f32
CMakeLists.txt   build (finds LibTorch in the venv)
src/main.cc      load policy_ts.pt, run forward, print the move, check vs PyTorch
data/            generated artifacts (git-ignored)
from_scratch/    OPTIONAL deep-dive: the same net hand-written op-by-op with
                 abseil (conv/BN/relu from scratch). Great for understanding the
                 internals later; not needed for the working engine.
```

## What's next (small, in order)
1. **Run it** — confirm the PASS above.
2. **Argmax over *legal* moves** — the real game masks illegal source/target
   cells before argmax. Port that mask so the move is always playable.
3. **Obs builder in C++** — port `_build_obs_core` (board + next balls → 18×9×9)
   so C++ runs end-to-end from a raw board, not a precomputed obs.
4. **Speed / deploy** — batch=1 latency, then a WASM build for the browser
   (likely via ONNX Runtime at that point).
