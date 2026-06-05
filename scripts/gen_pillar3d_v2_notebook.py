"""Generate alphatrain/train_pillar3d_v2_colab.ipynb — MCTS-corrections distillation.

pillar3d-v2: distill pillar3b toward the dense, MCTS-validated CORRECTIONS corpus
(soft visit-distribution targets, margin-weighted + sharpened) instead of the
greedy-mined listwise forks. Built from gen_pillar3d_notebook.py.
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'train_pillar3d_v2_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = []

cells.append(md(r"""
# Pillar3d-v2: MCTS-corrections distillation (the floor pipeline)

## Goal
Lift pillar3b's policy-only **floor** (fixed-engine baseline: mean 17,581, P5 1,452,
P10 2,377, <1000 2.9%, <500 0.3%) without regressing the mean >~5%.

## Method
Distill pillar3b toward **~12k dense, MCTS-validated corrections** — states where deep
widened MCTS@4800 picks a different move than the policy. The target is the **soft MCTS
visit distribution** (not a hard flip), the per-state weight is the **MCTS margin**
(top − policy visit share, mean 1) so decisive corrections dominate and marginal ones
collapse to ~0, and the target is **sharpened** (`--aux-target-temperature 0.5`) so the
model commits to the high-visit move.

Why this should beat Track-1 (479 greedy forks → +9% mean but flat floor, and held-out
forks didn't generalize): the corpus is ~25× larger, phantom-free, soft, and dense. In
the local smoke the **held-out match-rate rose *with* train** (generalizing, not
memorizing) — exactly Track-1's missing piece.

| Track-1 (forks) | pillar3d-v2 (corrections) |
|---|---|
| 479 forks, ~40% phantom, greedy-mined | ~12k corrections, MCTS-validated, phantom-free |
| listwise hard flip | soft visit-distribution target |
| weight = greedy catastrophe gap | weight = MCTS margin (decisive dominate) |

## Required Drive uploads (`MyDrive/alphatrain/`)
1. **`colorlines_pillar3d_v2.tar.gz`** — code (build cmd below)
2. **`corrections_corpus.pt`** — the ~12k-correction corpus (tiny, ~4 MB)
3. `v13_pillar3a.pt.gz` — V13 base tensor (already on Drive from pillar3b)
4. `pillar3b_epoch_20.pt` — warm-start checkpoint (already on Drive)

Build the tarball locally (repo root):
```bash
tar czf colorlines_pillar3d_v2.tar.gz --exclude='**/__pycache__' \
    alphatrain/*.py alphatrain/scripts/*.py scripts/*.py game/
```
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
## Train pillar3d-v2 (~12h on L4/A100)

**Watch the soft-preflight prints** (every 200 main steps):
- `soft-preflight match=` (TRAIN, policy argmax == MCTS top move) and
  `soft-heldout match=` (UNSEEN games) should BOTH rise. **Held-out rising = the
  corrections generalize** (Track-1's held-out stayed flat — the failure mode).
- `softCE` should fall on both.
- **`V12 val: loss`** must stay within ~5% of pillar3b's (~2.18). If it climbs, the aux
  is over-pushing → lower `--aux-lambda`.

`--aux-lambda 0.10` is the starting point (cleaner corpus than Track-1's 0.05 forks);
the held-out monitor + val are the guardrails.
""".strip()))

cells.append(code(r"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \
    --tensor-file alphatrain/data/v13_pillar3a.pt \
    --amp --compile \
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \
    --epochs 17 --batch-size 32768 --lr 3e-4 --warmup-epochs 1 \
    --target-temperature 0.5 \
    --aux-corrections-corpus crisis/corrections_corpus.pt --aux-weighted \
    --aux-lambda 0.10 --aux-target-temperature 0.5 \
    --aux-holdout-frac 0.15 --aux-split-seed 0 \
    --aux-batch-size 256 --aux-warmup-epochs 2.0 \
    --aux-preflight-every 200 \
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_v2_best.pt \
    --save-dir /content/checkpoints/pillar3d_v2 2>&1 | tee /content/pillar3d_v2_train.log
""".strip()))

cells.append(code(
    "# match-rate + val trajectory at a glance\n"
    "!grep -E 'soft-preflight|soft-heldout|Train:|val:' /content/pillar3d_v2_train.log"))

cells.append(code(r"""
# Copy all epoch checkpoints + the log to Drive
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_v2/epoch_*.pt')):
    dst = f'{DRIVE}/pillar3d_v2_{os.path.basename(f)}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_v2_train.log', f'{DRIVE}/pillar3d_v2_train.log')
""".strip()))

cells.append(md(r"""
## Floor eval — in-sample vs held-out (the verdict)

Your design: **50000–50300** (crisis bands ARE in the corpus → "did it learn + watch the
fixes", upper bound) vs **777000–778999** (held strictly OUT → generalization, the real
number). Run pillar3d-v2 on both; pillar3b on 50xxx gives the before. Policy-only, fixed
engine, `--device cuda` (NOT cpu).
""".strip()))

cells.append(code(r"""
%cd /content
EP = 17  # pick an epoch (e.g. where held-out match plateaued)
for model, tag in [('/content/checkpoints/pillar3d_v2/epoch_%d.pt' % EP, 'v2_ep%d' % EP),
                   ('alphatrain/data/pillar3b_epoch_20.pt', 'pillar3b')]:
    for lo, n in [(50000, 301), (777000, 2000)]:
        print(f'===== {tag}  seeds {lo}..{lo+n-1} =====')
        !python -m alphatrain.scripts.eval_parallel \
            --model {model} --policy-only \
            --seeds $(seq {lo} {lo+n-1}) --device cuda --workers 8 \
            2>&1 | grep -E 'P5|P10|mean|<1000|<500|>5000'
""".strip()))

cells.append(md(r"""
## Read it
- **777k held-out** is the metric: P10 / <1000 / mean vs pillar3b (17,581 / 2,377 / 2.9%).
- **50xxx − 777k gap** = generalization quality. Big in-sample lift + flat held-out =
  memorizing (need more/denser corrections — re-build the corpus as more `corr_*.json`
  land and retrain). Both move together = learning the pattern → scale up.
- Then replay a couple of 50xxx games before/after to literally watch the policy play the
  MCTS-corrected move at the crisis.
""".strip()))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
