"""Generate alphatrain/train_pillar3d_v2_1_colab.ipynb — v2.1 (gentle-LR recipe fix).

v2 result: ep1 (warmup, val 2.23) won big on HELD-OUT 777k (mean +17.6%, P10 +15.6%,
<1000 2.9%->2.0% — floor moved AND generalized). But ep2+ REGRESSED below control because
the base re-distillation's full LR (3e-4) destroyed the warm-started policy (val jump to
2.45+ was LR-driven, λ-independent). v2.1 = keep every epoch as gentle as ep1: low LR
throughout (no 3e-4 ramp). Built from gen_pillar3d_v2_notebook.py.
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'train_pillar3d_v2_1_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = []

cells.append(md(r"""
# Pillar3d-v2.1: gentle-LR MCTS-corrections distillation

## Why v2.1
v2 worked — ep1 lifted the **held-out** floor (777k: mean 17,581→20,678 **+17.6%**, P10
+15.6%, <1000 2.9%→2.0%, generalizing). But **only ep1**: ep2+ regressed *below* control
because the base re-distillation ran at lr **3e-4**, which perturbs the converged
warm-start policy (val jumped 2.23→2.45+, and that jump was **LR-driven, λ-independent** —
λ=0.10 and λ=0.03 gave the same ep2 val).

**Fix: gentle LR throughout** (`--lr 5e-5`, no 3e-4 phase), so every epoch is as soft as
v2's ep1 — the corrections accumulate without the high-LR phase undoing them. Same corpus
(now **16,547 corrections / 956 games**), same soft-CE + margin weight + sharpening,
**λ stays 0.03** (the damage was the LR, not λ).

## Required Drive uploads (`MyDrive/alphatrain/`)
1. **`colorlines_pillar3d_v2.tar.gz`** — code (unchanged from v2; re-upload if unsure)
2. **`corrections_corpus.pt`** — the **rebuilt 16,547-correction** corpus (~6 MB) — RE-UPLOAD
3. `v13_pillar3a.pt.gz` — already on Drive
4. `pillar3b_epoch_20.pt` — already on Drive

(The tarball is identical to v2's — the only change is CLI flags below.)
""".strip()))

cells.append(code("from google.colab import drive\ndrive.mount('/content/drive')"))

cells.append(code(r"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True)
os.makedirs('/content/crisis', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt',
            '/content/alphatrain/data/pillar3b_epoch_20.pt')
shutil.copy(f'{DRIVE}/corrections_corpus.pt', '/content/crisis/corrections_corpus.pt')
print('corrections corpus:',
      f"{os.path.getsize('/content/crisis/corrections_corpus.pt')/1e6:.1f} MB")
t0 = time.time()
!gzip -dc {DRIVE}/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
print(f"V13 tensor: {os.path.getsize('/content/alphatrain/data/v13_pillar3a.pt')/1e9:.1f} GB "
      f"({time.time()-t0:.0f}s)")
%cd /content
!pip install -q numpy numba scipy
""".strip()))

cells.append(code(r"""
import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
""".strip()))

cells.append(md(r"""
## Train pillar3d-v2.1 (gentle LR — ~12h)

**Key change: `--lr 5e-5` (was 3e-4), `--epochs 10`.** Every epoch stays as soft as v2's
winning ep1, so the corrections accumulate instead of being undone by a high-LR phase.

**Watch:**
- **`V12 val` should NOT jump** (no 3e-4 phase) — it should hover near ~2.2–2.3, not spike to
  2.45+. If it climbs hard, lower `--lr` to 3e-5.
- `soft-heldout match=` should keep rising (corrections generalizing).
- The best checkpoint is now **unknown** (no destruction past ep1) — likely mid-run. **Eval
  several epochs**, don't assume the last is best.
""".strip()))

cells.append(code(r"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \
    --tensor-file alphatrain/data/v13_pillar3a.pt \
    --amp --compile \
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \
    --epochs 10 --batch-size 32768 --lr 5e-5 --warmup-epochs 1 \
    --target-temperature 0.5 \
    --aux-corrections-corpus crisis/corrections_corpus.pt --aux-weighted \
    --aux-lambda 0.03 --aux-target-temperature 0.5 \
    --aux-holdout-frac 0.15 --aux-split-seed 0 \
    --aux-batch-size 256 --aux-warmup-epochs 2.0 \
    --aux-preflight-every 200 \
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_v2_1_best.pt \
    --save-dir /content/checkpoints/pillar3d_v2_1 2>&1 | tee /content/pillar3d_v2_1_train.log
""".strip()))

cells.append(code(
    "!grep -E 'soft-preflight|soft-heldout|Train:|val:' /content/pillar3d_v2_1_train.log"))

cells.append(code(r"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_v2_1/epoch_*.pt')):
    dst = f'{DRIVE}/pillar3d_v2_1_{os.path.basename(f)}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_v2_1_train.log', f'{DRIVE}/pillar3d_v2_1_train.log')
""".strip()))

cells.append(md(r"""
## Floor eval — sweep epochs (no destruction now, so the winner may be mid-run)

Control (pillar3b) on **777k held-out**: mean 17,581, P10 2,377, P5 1,452, <1000 2.9%.
v2 ep1 hit mean 20,678 / P10 2,747 / <1000 2.0%. Beat that. Sweep several epochs on the
held-out set (the real metric); run the winner on 50000–50300 too for the in-sample upper
bound. `--device cuda`, policy-only.
""".strip()))

cells.append(code(r"""
%cd /content
for EP in [3, 5, 7, 10]:        # edit; sweep since the best epoch is unknown
    m = f'/content/checkpoints/pillar3d_v2_1/epoch_{EP}.pt'
    if not os.path.exists(m): continue
    print(f'===== v2.1 epoch {EP}  (held-out 777000..778999) =====')
    !python -m alphatrain.scripts.eval_parallel \
        --model {m} --policy-only \
        --seeds $(seq 777000 778999) --device cuda --workers 8 \
        2>&1 | grep -E 'P5|P10|mean|<1000|<500'
""".strip()))

cells.append(md(r"""
## Read it
- Pick the epoch with the best **777k floor** (P10 / <1000) at mean ≥ ~v2's 20,678.
- If v2.1's best beats v2's ep1 → the gentle-LR fix worked; that's the new deployable policy.
- Then **iterate**: re-mine crisis corrections on this improved policy (its deaths shifted),
  rebuild the corpus, distill again — walking the floor (incl. the <500 tail) further down.
""".strip()))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
