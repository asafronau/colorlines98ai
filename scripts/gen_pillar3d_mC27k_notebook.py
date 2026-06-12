"""Generate the pillar3d mC27k notebook — the mC keeper recipe on the 27.6k corpus.

THE data-scaling decider: the 16,738 bar (local 1k eval_policy config) is the original
13.8k/1,837-game mC (proven by scripts/fingerprint_corpus_membership.py). Same recipe on
27,605 decisive corrections from 3,676 games. If ep2 >= bar => more data helps, the old
"19.6k regression" was a seed-list artifact; if it loses => the channel genuinely
saturates with data. The trust-region channel already lost to this recipe at every β
(project_trust_region_negative), so this IS the channel.

Fresh run name mC27k everywhere — never overwrite checkpoints across runs again.
4 epochs only (peak is ep2; ep5-6 never helped). No eval cells — evals run locally on M5
(eval_policy, 1k triage 775000..775999 then 5k, batch 256).
"""
import os, json

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alphatrain')
SUF = 'mC27k'


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = [
    md(f"""
# Pillar3d **{SUF}** — mC keeper recipe, 27.6k corpus (3,676 games). The data-scaling decider.

Recipe unchanged from mC (λ=0.01, aux_T=0.5, weighted, aux-warmup 0.5, lr 5e-5, warm-start
pillar3b_epoch_20), corpus grown 13.8k → **27,605** (2.0× the bar's) decisive corrections. **4 epochs** (peak ep2).
Bar to beat (local 1k eval_policy, seeds 775000..775999, batch 256): **median 16,738 / mean 24,249 /
P10 3,196 / <1000 1.6%** = the original 13.8k mC-ep2.

Checkpoints copy to Drive as `pillar3d_{SUF}_epoch_*.pt` (FRESH name — no overwrites). Evals run
locally; this notebook only trains. ~3h to ep4 on A100.

**Upload to `MyDrive/alphatrain/`:** `colorlines_pillar3d_v2.tar.gz`, **`corrections_corpus_mm05.pt`
(the 27,605 build)**, `v13_pillar3a.pt.gz`, `pillar3b_epoch_20.pt`.
"""),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    code(r"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True); os.makedirs('/content/crisis', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt', '/content/alphatrain/data/pillar3b_epoch_20.pt')
shutil.copy(f'{DRIVE}/corrections_corpus_mm05.pt', '/content/crisis/corrections_corpus.pt')
import torch
_c = torch.load('/content/crisis/corrections_corpus.pt')
print('corpus anchors', _c['boards'].shape[0], _c['_stats'])
assert 26000 < _c['boards'].shape[0] < 30000 and _c['_stats']['min_margin'] >= 0.05, \
    "unexpected corpus — upload the 27,605-correction corrections_corpus_mm05.pt (3,676 games)"

# V13: copy .gz local FIRST + verify sizes (Drive FUSE can silently truncate streams).
t0 = time.time()
!cp {DRIVE}/v13_pillar3a.pt.gz /content/v13_pillar3a.pt.gz
gz_size = os.path.getsize('/content/v13_pillar3a.pt.gz')
EXPECTED_GZ = 642_409_267
assert gz_size == EXPECTED_GZ, (
    f'.gz on Drive is truncated! got {gz_size} expected {EXPECTED_GZ}. Re-upload.')
!gunzip -t /content/v13_pillar3a.pt.gz && echo '.gz integrity OK'
!gzip -dc /content/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
pt_size = os.path.getsize('/content/alphatrain/data/v13_pillar3a.pt')
EXPECTED_PT = 5_433_958_495
assert pt_size == EXPECTED_PT, f'.pt size wrong! got {pt_size} expected {EXPECTED_PT}.'
print(f'V13 tensor: {pt_size/1e9:.2f} GB ({time.time()-t0:.0f}s)')
!rm /content/v13_pillar3a.pt.gz
%cd /content
!pip install -q numpy numba scipy
"""),
    code("import torch\nprint(f'PyTorch: {torch.__version__}')\n"
         "print('CUDA:', torch.cuda.get_device_name(0) if "
         "torch.cuda.is_available() else 'NONE')"),
    md(f"## Train {SUF}  (~45 min/epoch A100; watch heldout match + V12 val)"),
    code(f"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \\
    --tensor-file alphatrain/data/v13_pillar3a.pt --amp --compile \\
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \\
    --epochs 4 --batch-size 32768 --lr 5e-5 --warmup-epochs 1 \\
    --target-temperature 0.5 \\
    --aux-corrections-corpus crisis/corrections_corpus.pt --aux-weighted \\
    --aux-lambda 0.01 --aux-target-temperature 0.5 \\
    --aux-holdout-frac 0.15 --aux-split-seed 0 \\
    --aux-batch-size 256 --aux-warmup-epochs 0.5 --aux-preflight-every 200 \\
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_{SUF}_best.pt \\
    --save-dir /content/checkpoints/pillar3d_{SUF} 2>&1 | tee /content/pillar3d_{SUF}_train.log
"""),
    code(f"""
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d_{SUF}/epoch_*.pt')):
    dst = f'{{DRIVE}}/pillar3d_{SUF}_{{os.path.basename(f)}}'
    if not os.path.exists(dst):
        shutil.copy(f, dst); print('copied', os.path.basename(f))
shutil.copy('/content/pillar3d_{SUF}_train.log', f'{{DRIVE}}/pillar3d_{SUF}_train.log')
"""),
    md(f"""
## After training (local, M5)
Download `pillar3d_{SUF}_epoch_2.pt` (+3, 4) and run the 1k triage:
```
PYTHONPATH=. python -m scripts.eval_policy --model pillar3d_{SUF}_epoch_2.pt \\
    --seed-start 775000 --seed-end 775999 --device mps --batch 256
```
vs bar 16,738 (13.8k mC-ep2, same config). Within ±600 → run the 5k. Also worth a
fingerprint sanity check: `scripts/fingerprint_corpus_membership.py --model ...` should show
A ≈ B ≈ C all elevated (trained on all three chronological slices).
"""),
]

out = os.path.join(DIR, f'train_pillar3d_{SUF}_colab.ipynb')
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(out, 'w'), indent=1)
print(f"wrote {out} ({len(cells)} cells)")
