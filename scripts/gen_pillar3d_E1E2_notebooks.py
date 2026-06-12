"""Generate the E1/E2 notebooks — dilution-vs-interference deciders.

Background (docs/distill_dilution_vs_interference_for_chatgpt.md): mC27k (27.6k corrections,
λ=0.01) regressed to 14,943 median vs the 13.8k bar's 16,738 on a clean config, at IDENTICAL
held-out match (~0.215) and with composition ruled out (all mining slices statistically
identical). Two hypotheses: H1 dilution (per-correction weight = (steps/N)·λ halved) vs
H2 interference (fixed 11.9M net can't encode 2× preferences at the same drift budget).

  E1  random 13.8k SUBSAMPLE of the 27.6k (corrections_corpus_sub13k.pt), λ=0.01
      ≈ bar  => size is the whole story.   Regresses => deeper problem.
  E2  full 27.6k, λ=0.02 (exact per-correction-weight match to the bar's recipe)
      Recovers bar => H1 (scale λ ∝ N).    Flat => H2 (capacity next).

3 epochs each (~2h A100); eval locally (1k, 775000..775999, batch 256) vs bar 16,738.
"""
import os, json

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alphatrain')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


def build(suf, corpus_file, n_lo, n_hi, lam, hypothesis):
    cells = [
        md(f"""
# Pillar3d **{suf}** — {hypothesis}

mC recipe (warm-start pillar3b_epoch_20, lr 5e-5, T 0.5, aux_T 0.5, weighted, warmup 0.5),
**corpus `{corpus_file}`, λ={lam}**, 3 epochs (~2h A100). Bar (local 1k eval_policy,
775000..775999, batch 256): **16,738** = 13.8k mC-ep2; mC27k-ep2 scored 14,943.
Checkpoints → Drive as `pillar3d_{suf}_epoch_*.pt` (fresh names).

**Upload to `MyDrive/alphatrain/`:** `colorlines_pillar3d_v2.tar.gz`, **`{corpus_file}`**,
`v13_pillar3a.pt.gz`, `pillar3b_epoch_20.pt`.
"""),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        code(rf"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {{DRIVE}}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True); os.makedirs('/content/crisis', exist_ok=True)
shutil.copy(f'{{DRIVE}}/pillar3b_epoch_20.pt', '/content/alphatrain/data/pillar3b_epoch_20.pt')
shutil.copy(f'{{DRIVE}}/{corpus_file}', '/content/crisis/{corpus_file}')
import torch
_c = torch.load('/content/crisis/{corpus_file}')
print('corpus anchors', _c['boards'].shape[0], _c['_stats'])
assert {n_lo} < _c['boards'].shape[0] < {n_hi} and _c['_stats']['min_margin'] >= 0.05, \
    "unexpected corpus — upload {corpus_file}"

t0 = time.time()
!cp {{DRIVE}}/v13_pillar3a.pt.gz /content/v13_pillar3a.pt.gz
gz_size = os.path.getsize('/content/v13_pillar3a.pt.gz')
EXPECTED_GZ = 642_409_267
assert gz_size == EXPECTED_GZ, (
    f'.gz on Drive is truncated! got {{gz_size}} expected {{EXPECTED_GZ}}. Re-upload.')
!gunzip -t /content/v13_pillar3a.pt.gz && echo '.gz integrity OK'
!gzip -dc /content/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
pt_size = os.path.getsize('/content/alphatrain/data/v13_pillar3a.pt')
EXPECTED_PT = 5_433_958_495
assert pt_size == EXPECTED_PT, f'.pt size wrong! got {{pt_size}} expected {{EXPECTED_PT}}.'
print(f'V13 tensor: {{pt_size/1e9:.2f}} GB ({{time.time()-t0:.0f}}s)')
!rm /content/v13_pillar3a.pt.gz
%cd /content
!pip install -q numpy numba scipy
"""),
        code("import torch\nprint(f'PyTorch: {torch.__version__}')\n"
             "print('CUDA:', torch.cuda.get_device_name(0) if "
             "torch.cuda.is_available() else 'NONE')"),
        md(f"## Train {suf}  (~40 min/epoch A100)"),
        code(f"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \\
    --tensor-file alphatrain/data/v13_pillar3a.pt --amp --compile \\
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \\
    --epochs 3 --batch-size 32768 --lr 5e-5 --warmup-epochs 1 \\
    --target-temperature 0.5 \\
    --aux-corrections-corpus crisis/{corpus_file} --aux-weighted \\
    --aux-lambda {lam} --aux-target-temperature 0.5 \\
    --aux-holdout-frac 0.15 --aux-split-seed 0 \\
    --aux-batch-size 256 --aux-warmup-epochs 0.5 --aux-preflight-every 200 \\
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_{suf}_best.pt \\
    --save-dir /content/checkpoints/pillar3d_{suf} 2>&1 | tee /content/pillar3d_{suf}_train.log
"""),
        code(f"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_{suf}/epoch_*.pt')):
    dst = f'{{DRIVE}}/pillar3d_{suf}_{{os.path.basename(f)}}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_{suf}_train.log', f'{{DRIVE}}/pillar3d_{suf}_train.log')
"""),
        md(f"""
## After training (local, M5)
```
PYTHONPATH=. python -m scripts.eval_policy --model pillar3d_{suf}_epoch_2.pt \\
    --seed-start 775000 --seed-end 775999 --device mps --batch 256
```
vs bar **16,738** (13.8k mC-ep2) and mC27k-ep2's **14,943**.
"""),
    ]
    out = os.path.join(DIR, f'train_pillar3d_{suf}_colab.ipynb')
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(out, 'w'), indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == '__main__':
    build('mE1sub13k', 'corrections_corpus_sub13k.pt', 13000, 14500, 0.01,
          'E1 size-control: 13.8k random subsample of the 27.6k (full diversity), λ=0.01')
    build('mE2lam02', 'corrections_corpus_mm05.pt', 26000, 30000, 0.02,
          'E2 dilution-match: full 27.6k at λ=0.02 (per-correction weight = the bar recipe)')
    # σ_train replicates (Gemini step 1): the EXACT bar config (original 13.8k corpus rebuilt via
    # --first-n-by-mtime 1837), fresh runs — the trainer is unseeded so each run draws its own
    # aux-sampling/augmentation. Tight around 16.6k => the bar config is reliably strong (λ/corpus
    # effects real); spread ~15-16.5k => σ_train ≈ 1k and the bar was a lucky draw (channel flat).
    build('mCrep1', 'corrections_corpus_orig13k.pt', 13000, 14500, 0.01,
          'σ_train replicate 1 of the bar config (original 13.8k corpus, λ=0.01)')
    build('mCrep2', 'corrections_corpus_orig13k.pt', 13000, 14500, 0.01,
          'σ_train replicate 2 of the bar config (original 13.8k corpus, λ=0.01)')
