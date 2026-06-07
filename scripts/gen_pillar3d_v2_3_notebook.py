"""Generate alphatrain/train_pillar3d_v2_3_colab.ipynb — v2.3 (quality filter: decisive corrections).

Why v2.3: v2.2 (26.8k corrections, ALL kept) REGRESSED vs v2.1 on the 5k eval (mean 20,711 vs
21,195; floor + every percentile worse) — the dilution signature (26% of the corpus was near-zero
margin). v2.3 tests quality-over-quantity: SAME 26.8k mine, but margin-filtered to the DECISIVE
corrections only (--min-margin 0.05 -> 11,582 kept, the entire near-zero mass dropped). Recipe is
otherwise byte-identical to v2.1/v2.2. Clean single-variable test: does decisive-only beat v2.1?
Built from gen_pillar3d_v2_2_notebook.py.
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'train_pillar3d_v2_3_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = []

cells.append(md(r"""
# Pillar3d-v2.3: quality filter (decisive corrections only)

## Why v2.3
v2.2 added +59% more crises (26.8k corrections, all kept) and **REGRESSED** vs v2.1 on the
**5,000-game** eval — mean 20,711 vs **21,195**, and *every* percentile + the floor slightly worse.
That's the **dilution signature**: 26% of the corpus was near-zero margin (MCTS barely prefers a
different move than the policy = near-noise). More data of mixed quality hurt.

**v2.3 = quality over quantity.** Same 26.8k mine, but margin-filtered to the **decisive**
corrections (`build_corrections_corpus.py --min-margin 0.05` → **11,582 kept**, the entire near-zero
mass dropped, 0% marginal remaining). Recipe otherwise **byte-identical** to v2.1/v2.2 (warm-start
pillar3b_epoch_20, lr 5e-5, target-temp 0.5, aux-lambda 0.03, 10 epochs). Clean single-variable
test: **does decisive-only beat v2.1?**

## Required Drive uploads (`MyDrive/alphatrain/`)
1. `colorlines_pillar3d_v2.tar.gz` — code (unchanged; re-upload if unsure)
2. **`corrections_corpus_mm05.pt`** — the **11,582-decisive** corpus (~3.8 MB) — **UPLOAD** (this is the change)
3. `v13_pillar3a.pt.gz`, `pillar3b_epoch_20.pt` — already on Drive
""".strip()))

cells.append(code("from google.colab import drive\ndrive.mount('/content/drive')"))

cells.append(code(r"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True)
os.makedirs('/content/crisis', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt', '/content/alphatrain/data/pillar3b_epoch_20.pt')
# the FILTERED (decisive-only) corpus, copied to the path train_path_b reads:
shutil.copy(f'{DRIVE}/corrections_corpus_mm05.pt', '/content/crisis/corrections_corpus.pt')
import torch
_c = torch.load('/content/crisis/corrections_corpus.pt')
print('corpus:', f"{os.path.getsize('/content/crisis/corrections_corpus.pt')/1e6:.1f} MB",
      _c['_stats'], 'anchors', _c['boards'].shape[0])
assert 9000 < _c['boards'].shape[0] < 14000, "expected the ~11.6k decisive corpus — upload corrections_corpus_mm05.pt"
assert _c['_stats']['min_margin'] >= 0.05, "expected min_margin>=0.05 (the filtered corpus)"
t0 = time.time()
!gzip -dc {DRIVE}/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
print(f"V13 tensor ({time.time()-t0:.0f}s)")
%cd /content
!pip install -q numpy numba scipy
""".strip()))

cells.append(code(r"""
import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
""".strip()))

cells.append(md(r"""
## Train pillar3d-v2.3 (~12h) — identical recipe, only the corpus is filtered
"""))

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
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_v2_3_best.pt \
    --save-dir /content/checkpoints/pillar3d_v2_3 2>&1 | tee /content/pillar3d_v2_3_train.log
""".strip()))

cells.append(code(
    "!grep -E 'soft-preflight|soft-heldout|Train:|val:' /content/pillar3d_v2_3_train.log"))

cells.append(code(r"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_v2_3/epoch_*.pt')):
    dst = f'{DRIVE}/pillar3d_v2_3_{os.path.basename(f)}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_v2_3_train.log', f'{DRIVE}/pillar3d_v2_3_train.log')
""".strip()))

cells.append(md(r"""
## Floor eval — sweep epochs, **5,000 games**, vs v2.1 (the bar) and v2.2

Baselines on 5,000 held-out (777000..781999):
- **v2.1-ep2 (bar):** mean **21,195**, P10 2,796, P50 15,121, <1000 2.6%, >10k 64%
- v2.2-ep2 (all-26.8k, regressed): mean 20,711, P10 2,552, <1000 2.7%

Use 5k (not 2k) — the 2k eval misled us on v2.2 (looked like a win, reversed at 5k).
""".strip()))

cells.append(code(r"""
%cd /content
for EP in [2, 3, 5, 7]:          # decisive corpus is smaller -> best epoch may be earlier; sweep
    m = f'/content/checkpoints/pillar3d_v2_3/epoch_{EP}.pt'
    if not os.path.exists(m): continue
    print(f'===== v2.3 epoch {EP}  (5k held-out 777000..781999) =====')
    !python -m alphatrain.scripts.eval_parallel \
        --model {m} --policy-only \
        --seeds $(seq 777000 781999) --device cuda --workers 8 \
        2>&1 | grep -E 'P5|P10|mean|<1000|<500|>10000'
""".strip()))

cells.append(md(r"""
## Read it — the quality-vs-quantity verdict
- **Beats v2.1** (mean > 21,195, P10 > 2,796, <1000 < 2.6%) → it WAS dilution; decisive-only is the
  recipe. Next: floor-target the mine (`--max-final-score`) + keep filtering.
- **Ties v2.1** → filtering recovers but doesn't exceed → the limit is deeper than dilution
  (already-lost positions / leaf-value), → rollout-grounded teacher.
- **Below v2.1** → the marginal corrections were actually helping (unlikely given v2.2's regression);
  reconsider the filter.
""".strip()))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
