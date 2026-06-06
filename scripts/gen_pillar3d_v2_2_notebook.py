"""Generate alphatrain/train_pillar3d_v2_2_colab.ipynb — v2.2 (data-scaling iteration).

Same gentle-LR recipe as v2.1 (lr 5e-5, λ 0.03, target-temp 0.5, warm-start pillar3b_epoch_20),
the ONLY change is a BIGGER corpus: 26,310 corrections / 1520 games (v2.1 used 16,547 / 956).
Purpose is twofold: (1) push pillar3b's floor further; (2) measure how pillar3d reacts to MORE
crisis data — does the +59% corpus break the saturation v2.1 hit, or plateau? Compare v2.2 vs v2.1
(deployed ep2) vs control on the SAME 777k held-out. Built from gen_pillar3d_v2_1_notebook.py.
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'train_pillar3d_v2_2_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = []

cells.append(md(r"""
# Pillar3d-v2.2: data-scaling iteration (bigger crisis corpus, same gentle-LR recipe)

## Why v2.2
v2.1 (gentle LR throughout) is the deployed floor-lifter: held-out 777k **mean 20,609 (+17.2%)**,
P10 2,736, <1000 2.5% — but its corpus (16,547 corr / 956 games) had **saturated**, and the
**<500 tail was untouched**. We've since mined **more crises on fresh seeds**, so the corpus grew
to **26,310 corrections / 1,520 games (+59%)**.

**This run changes ONE thing: the corpus.** Recipe is byte-identical to v2.1 (warm-start
pillar3b_epoch_20, `--lr 5e-5`, `--aux-lambda 0.03`, `--target-temperature 0.5`, 10 epochs). Two
questions:
1. Does the bigger corpus push the floor **past v2.1** (mean > 20,609, P10 > 2,736, <1000 < 2.5%,
   and ideally the **<500 tail** finally moves)?
2. **The data-scaling curve** — how does pillar3d react to more crisis data: keep improving, or
   plateau (saturation is a property of the *method*, not just this corpus)? This is the pattern
   we'll repeat as more crises are mined.

## Required Drive uploads (`MyDrive/alphatrain/`)
1. `colorlines_pillar3d_v2.tar.gz` — code (unchanged; re-upload if unsure)
2. **`corrections_corpus.pt`** — the **rebuilt 26,310-correction** corpus (~9 MB) — **RE-UPLOAD** (this is the only real change)
3. `v13_pillar3a.pt.gz` — already on Drive
4. `pillar3b_epoch_20.pt` — already on Drive
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
import torch
_c = torch.load('/content/crisis/corrections_corpus.pt')
print('corrections corpus:',
      f"{os.path.getsize('/content/crisis/corrections_corpus.pt')/1e6:.1f} MB",
      _c['_stats'], 'anchors', _c['boards'].shape[0])
assert _c['boards'].shape[0] > 20000, "expected the rebuilt 26k corpus — re-upload corrections_corpus.pt"
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
## Train pillar3d-v2.2 (gentle LR — ~12h)

Identical recipe to v2.1; only `corrections_corpus.pt` is bigger (26,310 vs 16,547).

**Watch (same as v2.1):**
- **`V12 val` should NOT jump** — hover ~2.2–2.3 (it's the same gentle 5e-5, no high-LR phase).
- `soft-heldout match=` should keep rising; with more data the aux holdout is a different 15%
  split, so don't compare its absolute value to v2.1 — the **777k game floor below is the metric**.
- Best epoch unknown → **sweep several**.
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
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_v2_2_best.pt \
    --save-dir /content/checkpoints/pillar3d_v2_2 2>&1 | tee /content/pillar3d_v2_2_train.log
""".strip()))

cells.append(code(
    "!grep -E 'soft-preflight|soft-heldout|Train:|val:' /content/pillar3d_v2_2_train.log"))

cells.append(code(r"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_v2_2/epoch_*.pt')):
    dst = f'{DRIVE}/pillar3d_v2_2_{os.path.basename(f)}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_v2_2_train.log', f'{DRIVE}/pillar3d_v2_2_train.log')
""".strip()))

cells.append(md(r"""
## Floor eval — sweep epochs, compare against v2.1 and control (the data-scaling read)

Baselines on the **same 777k held-out (777000..778999)**:
- **control (pillar3b)**: mean 17,581, P10 2,377, P5 1,452, <1000 2.9%
- **v2.1 (deployed)**: mean 20,609 (+17.2%), P10 2,736, <1000 2.5% — **the bar to beat**

The question is whether +59% more corpus pushes past v2.1 (and whether the **<500 tail** moves).
""".strip()))

cells.append(code(r"""
%cd /content
for EP in [3, 5, 7, 10]:        # edit; sweep since the best epoch is unknown
    m = f'/content/checkpoints/pillar3d_v2_2/epoch_{EP}.pt'
    if not os.path.exists(m): continue
    print(f'===== v2.2 epoch {EP}  (held-out 777000..778999) =====')
    !python -m alphatrain.scripts.eval_parallel \
        --model {m} --policy-only \
        --seeds $(seq 777000 778999) --device cuda --workers 8 \
        2>&1 | grep -E 'P5|P10|mean|<1000|<500'
""".strip()))

cells.append(md(r"""
## Read it — the data-scaling verdict
- **Beats v2.1** (mean > 20,609, P10 > 2,736, <1000 < 2.5%) → more crisis data still helps; keep
  mining + rebuilding. Note especially whether **<500** finally drops (v2.1 left it untouched).
- **Ties / plateaus v2.1** → saturation is a *method* limit, not a data limit — the next lever is
  the training objective / teacher (e.g. the action-risk teacher or the MCTS-relabel teacher),
  not more of the same corpus.
- **Regresses** → too much marginal data diluting; raise `build_corrections_corpus.py --min-margin`
  or lower `--aux-lambda`.
- Whatever wins, record the full curve in HISTORY.md (this is iteration #2 of the data-scaling
  pattern; v2.1 was #1).
""".strip()))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
