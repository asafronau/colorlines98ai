"""Generate the aux-distillation run-matrix notebooks (ChatGPT's design) to test the 'use them all
+ sharpen' thesis vs the decisive-corpus control.

Matrix (all: warm-start pillar3b, λ=0.01, lr 5e-5, warmup-ep(lr)=1, aux-warmup 0.5, 6 epochs):
  A  full 31.8k corpus, aux_T=0.5, weighted on
  B  full 31.8k corpus, aux_T=0.3, weighted on   <- KEY: use-them-all + sharpen
  C  decisive 13.8k,    aux_T=0.5, weighted on   <- CONTROL (current best recipe)
  D  full 31.8k corpus, aux_T=0.3, weighted OFF  (optional)
Eval epochs 2/3/4/5 on 5k seeds, judged on MEDIAN (+ mean); floor = no-regress guardrail.
Read: B>C => use-all+sharpen wins. C>B => filter removes bad labels, not just flatness.
      D>B => margin weights over-focus the decisive subset.
Why sharpen: the teacher is widened (top_k=300 + Dirichlet) to find buried fixes, so low-margin
soft-visit targets carry exploration mass, not move quality; aux_T=0.3 extracts move quality.
(Note: aux-warmup shortened 2.0->0.5 so full λ acts from ep1 — 2.0 made early-epoch reads misleading
in a 6-epoch run.)
"""
import os, json

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alphatrain')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


def build(suf, corpus_file, corpus_label, n_lo, n_hi, mm_assert, aux_t, weighted,
          lam=0.01, warmup=0.5, epochs=6):
    aux_w = '--aux-weighted ' if weighted else ''
    cells = []
    cells.append(md(f"""
# Pillar3d matrix run **{suf}**  (λ={lam}, aux_T={aux_t}, weighted={'on' if weighted else 'OFF'})

Corpus: **{corpus_label}**.  Tests the 'use them all + sharpen' thesis (see the run matrix). The
teacher is widened (top_k=300 + Dirichlet) so low-margin soft-visit targets carry exploration mass;
`--aux-target-temperature {aux_t}` extracts move-quality. Recipe otherwise = warm-start
pillar3b_epoch_20, lr 5e-5, **aux-warmup 0.5** (shortened from 2.0 so full λ acts from ep1), {epochs} epochs.

**Upload to `MyDrive/alphatrain/`:** `colorlines_pillar3d_v2.tar.gz`, **`{corpus_file}`**,
`v13_pillar3a.pt.gz`, `pillar3b_epoch_20.pt`.
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
    "unexpected corpus — upload CORPUS_FILE"
t0 = time.time()
!gzip -dc {DRIVE}/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
print(f"V13 ({time.time()-t0:.0f}s)")
%cd /content
!pip install -q numpy numba scipy
""".strip().replace('CORPUS_FILE', corpus_file).replace('N_LO', str(n_lo))
       .replace('N_HI', str(n_hi)).replace('MM_ASSERT', str(mm_assert))))
    cells.append(code("import torch\nprint('CUDA:', torch.cuda.get_device_name(0) if "
                      "torch.cuda.is_available() else 'NONE')"))
    cells.append(md(f"## Train {suf} (~12h)"))
    cells.append(code(f"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \\
    --tensor-file alphatrain/data/v13_pillar3a.pt --amp --compile \\
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \\
    --epochs {epochs} --batch-size 32768 --lr 5e-5 --warmup-epochs 1 \\
    --target-temperature 0.5 \\
    --aux-corrections-corpus crisis/corrections_corpus.pt {aux_w}\\
    --aux-lambda {lam} --aux-target-temperature {aux_t} \\
    --aux-holdout-frac 0.15 --aux-split-seed 0 \\
    --aux-batch-size 256 --aux-warmup-epochs {warmup} --aux-preflight-every 200 \\
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_{suf}_best.pt \\
    --save-dir /content/checkpoints/pillar3d_{suf} 2>&1 | tee /content/pillar3d_{suf}_train.log
""".strip()))
    cells.append(code(f"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_{suf}/epoch_*.pt')):
    dst = f'{{DRIVE}}/pillar3d_{suf}_{{os.path.basename(f)}}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_{suf}_train.log', f'{{DRIVE}}/pillar3d_{suf}_train.log')
""".strip()))
    cells.append(md(r"""
## Eval — 5,000 games, judged on **MEDIAN (+ mean)**; floor = no-regress guardrail
Bar (5k 777000..781999): v2.1-ep2 median 15,121 / mean 21,195 / <1000 2.6%.
Decisive λ=0.01 ep2 (5k 775000..779999): median 15,447 / mean 22,203 (current best).
Sweep ep2/3/4/5 (aux-warmup 0.5 → full λ early, but median may still peak mid-run).
""".strip()))
    cells.append(code(f"""
%cd /content
for EP in [2, 3, 4, 5]:
    m = f'/content/checkpoints/pillar3d_{suf}/epoch_{{EP}}.pt'
    if not os.path.exists(m): continue
    print(f'===== {suf} epoch {{EP}}  (5k) =====')
    !python -m alphatrain.scripts.eval_parallel --model {{m}} --policy-only \\
        --seeds $(seq 777000 781999) --device cuda --workers 8 \\
        2>&1 | grep -E 'P5|P10|mean|P50|<1000|<500|>10000'
""".strip()))
    out = os.path.join(DIR, f'train_pillar3d_{suf}_colab.ipynb')
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(out, 'w'), indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


# size bounds widened for the 3.5k/5k-game data-scaling study (decisive grows ~7.5/game →
# ~26k @3.5k games, ~37k @5k games; full ~2.4x that). Just sanity asserts, kept generous.
FULL = ('corrections_corpus.pt', 'FULL (min_margin 0)', 20000, 130000, 0.0)
DEC = ('corrections_corpus_mm05.pt', 'DECISIVE (min_margin 0.05)', 9000, 60000, 0.05)
if __name__ == '__main__':
    build('mA_full_T05',     *FULL, aux_t=0.5, weighted=True)
    build('mB_full_T03',     *FULL, aux_t=0.3, weighted=True)    # KEY: use-all + sharpen
    build('mC_dec_T05',      *DEC,  aux_t=0.5, weighted=True)    # CONTROL
    build('mD_full_T03_unw', *FULL, aux_t=0.3, weighted=False)   # optional
