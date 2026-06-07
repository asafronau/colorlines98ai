"""Generate the v2.3 lambda-sweep notebooks: train_pillar3d_v2_3_lam0.01_colab.ipynb and
train_pillar3d_v2_3_lam0.003_colab.ipynb.

Motivation: the grad-audit (train_path_b --grad-audit) showed the aux stream at λ=0.03 is NOT a
gentle nudge — effective share λ|g_aux|/|g_main| = **2.09** (aux drives training 2:1 over the base;
cos≈0 orthogonal). pillar3b is converged so |g_main| is tiny (~0.23) while the OOD crisis
corrections have |g_aux|~15.8. So v2.2's regression is largely a training-mechanics over-index, and
the result is hypersensitive to corpus composition. These runs combine BOTH fixes the audit points
at: the DECISIVE corpus (v2.3 filter, 11.6k) + a GENTLER λ. Predicted effective shares (linear in λ):
  λ=0.01  -> ~0.70 (aux-heavy)      λ=0.003 -> ~0.21 (real gentle nudge)
Everything else byte-identical to v2.3/v2.1. Corpus fixed -> isolates λ.
"""
import os, json

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alphatrain')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


def build(lam, suf, share, corpus_file='corrections_corpus_mm05.pt',
          corpus_label='decisive (v2.3 filter, 11,702)', n_lo=9000, n_hi=20000, mm_assert=0.05):
    cells = []
    cells.append(md(f"""
# Pillar3d-v2.3-{suf}: aux λ sweep on the decisive corpus (λ={lam})

## Why
The gradient audit (`train_path_b --grad-audit`) showed λ=0.03 is **aux-dominated**: effective share
`λ·|g_aux|/|g_main| = 2.09` (the corrections drive training 2:1 over the base; gradients orthogonal,
cos≈0). pillar3b is converged → tiny `|g_main|≈0.23`; OOD crisis corrections → huge `|g_aux|≈15.8`.
So "λ=0.03 gentle" was an illusion, and v2.2's regression is largely a training-mechanics over-index.

**This run: corpus = {corpus_label} + gentler `--aux-lambda {lam}`**
→ predicted effective aux share ≈ **{share}**. Everything else byte-identical to v2.1/v2.3 (warm-start
pillar3b_epoch_20, lr 5e-5, target-temp 0.5, 10 epochs). Corpus fixed across the λ sweep so this
isolates λ. Question: does a gentler aux **generalize / lift the floor better**, or do the gains need
the strong regime?

## Required Drive uploads (`MyDrive/alphatrain/`)
1. `colorlines_pillar3d_v2.tar.gz` — code (recipe flags unchanged)
2. **`{corpus_file}`** — the {corpus_label} corpus
3. `v13_pillar3a.pt.gz`, `pillar3b_epoch_20.pt` — already on Drive
""".strip()))

    cells.append(code("from google.colab import drive\ndrive.mount('/content/drive')"))

    cells.append(code(r"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True); os.makedirs('/content/crisis', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt', '/content/alphatrain/data/pillar3b_epoch_20.pt')
shutil.copy(f'{DRIVE}/CORPUS_FILE', '/content/crisis/corrections_corpus.pt')
import torch
_c = torch.load('/content/crisis/corrections_corpus.pt')
print('corpus anchors', _c['boards'].shape[0], _c['_stats'])
assert N_LO < _c['boards'].shape[0] < N_HI and _c['_stats']['min_margin'] >= MM_ASSERT, \
    "unexpected corpus size/min_margin — upload CORPUS_FILE"
t0 = time.time()
!gzip -dc {DRIVE}/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
print(f"V13 tensor ({time.time()-t0:.0f}s)")
%cd /content
!pip install -q numpy numba scipy
""".strip().replace('CORPUS_FILE', corpus_file).replace('N_LO', str(n_lo))
       .replace('N_HI', str(n_hi)).replace('MM_ASSERT', str(mm_assert))))

    cells.append(code(
        "import torch\nprint('CUDA:', torch.cuda.is_available(),\n"
        "      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"))

    cells.append(md(f"## Train pillar3d-v2.3-{suf} (~12h) — only `--aux-lambda {lam}` differs"))

    cells.append(code(f"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \\
    --tensor-file alphatrain/data/v13_pillar3a.pt \\
    --amp --compile \\
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \\
    --epochs 10 --batch-size 32768 --lr 5e-5 --warmup-epochs 1 \\
    --target-temperature 0.5 \\
    --aux-corrections-corpus crisis/corrections_corpus.pt --aux-weighted \\
    --aux-lambda {lam} --aux-target-temperature 0.5 \\
    --aux-holdout-frac 0.15 --aux-split-seed 0 \\
    --aux-batch-size 256 --aux-warmup-epochs 2.0 \\
    --aux-preflight-every 200 \\
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_v2_3_{suf}_best.pt \\
    --save-dir /content/checkpoints/pillar3d_v2_3_{suf} 2>&1 | tee /content/pillar3d_v2_3_{suf}_train.log
""".strip()))

    cells.append(code(
        f"!grep -E 'soft-preflight|soft-heldout|Train:|val:' /content/pillar3d_v2_3_{suf}_train.log"))

    cells.append(code(f"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_v2_3_{suf}/epoch_*.pt')):
    dst = f'{{DRIVE}}/pillar3d_v2_3_{suf}_{{os.path.basename(f)}}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_v2_3_{suf}_train.log', f'{{DRIVE}}/pillar3d_v2_3_{suf}_train.log')
""".strip()))

    cells.append(md(r"""
## Floor eval — 5,000 games, vs v2.1 (the bar) and the other λ
Baselines on 5,000 held-out (777000..781999):
- **v2.1-ep2 (bar):** mean **21,195**, P10 2,796, P50 15,121, <1000 2.6%, >10k 64%
- v2.2-ep2 (26.8k all, λ=0.03): mean 20,711, P10 2,552 (regressed)

Sweep epochs; with a gentler aux the policy moves less per epoch → best epoch may be LATER.
""".strip()))

    cells.append(code(f"""
%cd /content
for EP in [3, 5, 7, 10]:
    m = f'/content/checkpoints/pillar3d_v2_3_{suf}/epoch_{{EP}}.pt'
    if not os.path.exists(m): continue
    print(f'===== v2.3-{suf} (λ={lam}) epoch {{EP}}  (5k held-out) =====')
    !python -m alphatrain.scripts.eval_parallel \\
        --model {{m}} --policy-only \\
        --seeds $(seq 777000 781999) --device cuda --workers 8 \\
        2>&1 | grep -E 'P5|P10|mean|<1000|<500|>10000'
""".strip()))

    cells.append(md(f"""
## Read it — λ={lam} (effective aux share ≈ {share})
- **Beats v2.1** → gentler aux generalizes better; the λ=0.03 regime was over-indexing. Pick the
  winning (λ, epoch); this becomes the recipe.
- **Ties v2.1** → λ not the limiter at this corpus; compare across the sweep (0.03 / 0.01 / 0.003).
- **Below v2.1** → too gentle, the gains need a stronger aux → label-quality / rollout-grounded teacher.
Compare the three λ side by side: if there's a monotone floor trend, that's the lever.
""".strip()))

    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = os.path.join(DIR, f'train_pillar3d_v2_3_{suf}_colab.ipynb')
    with open(out, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == '__main__':
    build(0.01, 'lam0.01', '~0.70')
    build(0.003, 'lam0.003', '~0.21')
    # "use them ALL with the right λ": full 27k corpus at the sweet-spot λ=0.01.
    # Tests whether the decisive filter was just a workaround for the over-strong λ=0.03.
    build(0.01, 'lam0.01_FULL', '~0.70', corpus_file='corrections_corpus.pt',
          corpus_label='FULL (all 27,103 corrections, min_margin 0)',
          n_lo=24000, n_hi=44000, mm_assert=0.0)
