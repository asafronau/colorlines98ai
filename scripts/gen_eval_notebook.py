"""Generate a STANDALONE eval-only Colab notebook: hardcode a model filename (on Drive), it copies the
code tarball + the model from Drive and runs scripts.eval_policy on the canonical 5k held-out seeds
(775000..779999, fp16, batch 256). No training. Prints eval_policy's own progress + percentile stats.
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'eval_policy_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = [
    md(r"""
# Eval-only — `scripts.eval_policy` on the canonical 5k held-out seeds

Hardcode a model filename below; this notebook copies the code tarball + that model from Drive and runs
the single-process batched policy player (fp16, batch 256) on seeds **775000..779999**. Prints its own
progress + full percentile stats (min/max/mean, P1..P95, <500/<1000/>10000).

**Upload to `MyDrive/alphatrain/`:** `colorlines_pillar3d_v2.tar.gz` (must contain the top-level
`scripts/` package — eval_policy.py, batched_rollout.py, __init__.py — plus `alphatrain/` and `game/`),
and your **model `.pt`**.
"""),
    code(r"""
# ==== EDIT: model file (in MyDrive/alphatrain/) and seed range ====
MODEL = 'pillar3d_mC_dec_T05_epoch_2.pt'   # <-- the .pt on Drive to evaluate
SEED_START, SEED_END = 775000, 779999      # canonical 5k held-out (inclusive)
BATCH = 256
"""),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    code(r"""
import os, shutil
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
assert os.path.exists('/content/scripts/eval_policy.py'), \
    "tarball is missing top-level scripts/eval_policy.py — rebuild it to include scripts/"
os.makedirs('/content/models', exist_ok=True)
shutil.copy(f'{DRIVE}/{MODEL}', f'/content/models/{MODEL}')
print('model:', MODEL, os.path.getsize(f'/content/models/{MODEL}'), 'bytes')
%cd /content
!pip install -q numpy numba scipy
"""),
    code("import torch\nprint('CUDA:', torch.cuda.get_device_name(0) if "
         "torch.cuda.is_available() else 'NONE')"),
    md("## Eval (5,000 games)"),
    code(r"""
%cd /content
print(f'===== {MODEL}  (5k {SEED_START}..{SEED_END}) =====', flush=True)
!python -m scripts.eval_policy --model /content/models/{MODEL} \
    --seed-start {SEED_START} --seed-end {SEED_END} --device cuda --batch {BATCH}
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(OUT, 'w'), indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
