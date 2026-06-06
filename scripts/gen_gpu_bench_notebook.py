"""Generate alphatrain/gpu_bench_colab.ipynb — batched-engine GPU benchmark for Colab/CUDA.

Validates the torch engine primitives (label_components_pj / reachable_many_t /
clear_lines_at_t) against numpy and benchmarks them GPU-vs-numpy across a K sweep, on
whatever CUDA GPU the Colab session has. Upload colorlines_gpu_bench.tar.gz (6.9KB,
self-contained). The decision number is the DESCENT-STEP net ratio (it grows with K).
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'gpu_bench_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = []

cells.append(md(r"""
# Batched-engine GPU benchmark (CUDA)

Measures whether the **array-based batched-tree MCTS** descent is faster on GPU than the
numpy/CPU version — the go/no-go for porting the full search to the GPU. It validates the
three hot per-descent-step primitives against numpy (must all pass), then times them
GPU-vs-numpy across a **K sweep** (more trees per batch → more GPU amortization).

**Read the `DESCENT STEP 1L+1R+4C` row** — that's the net per-step cost (1 label-components
+ 1 reachable_many + ~4 clear_lines_at per descent step). Ratio > 1 = GPU faster. It should
climb with K. (Per-machine numpy baselines vary with the Colab vCPU, so compare the *trend*
and the descent-step ratio, not absolute numpy ms.)

## Upload
`colorlines_gpu_bench.tar.gz` (6.9 KB, self-contained: alphatrain/{__init__,batched_engine,
batched_engine_gpu}.py + game/{__init__,config}.py). No numba/scipy needed — just torch+numpy.
""".strip()))

cells.append(code(r"""
import torch
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0),
          '| capability', torch.cuda.get_device_capability(0))
""".strip()))

cells.append(md("## Upload the tarball"))
cells.append(code(r"""
from google.colab import files
up = files.upload()          # pick colorlines_gpu_bench.tar.gz
!tar xzf colorlines_gpu_bench.tar.gz
!ls alphatrain game
""".strip()))

cells.append(md(r"""
## Validate + benchmark (K sweep)

All three correctness lines must read full (512/512, 100%, boards+counts match). Then watch
the `DESCENT STEP` ratio grow with K.
""".strip()))

cells.append(code(r"""
import sys; sys.path.insert(0, '.')
from alphatrain.batched_engine_gpu import _bench
for K in [512, 1024, 2048, 4096]:
    _bench(device='cuda', K=K)
    print()
""".strip()))

cells.append(md(r"""
## Read it
- **Correctness** must pass at every K (it's K-independent, but confirms the port is sound).
- **DESCENT STEP ratio** is the decision number. On L4 it was ~3x at K=512; expect it to rise
  with K as the GPU amortizes launch overhead. If it stays comfortably >1 and climbing →
  finishing the full torch descent port (apply_move + PUCT selection + tree arrays on GPU) is
  justified; the NN forward is already GPU-batched on top of this.
- **clear_lines_at** is the laggard (its scatter). If it caps the descent step, the next
  optimization is skipping no-op spawn-clears / a scatter-free marking.
""".strip()))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
