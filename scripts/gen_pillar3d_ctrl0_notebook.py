"""Generate the pillar3d ctrl0 notebook — the λ=0 CONTROL (gentle re-distill, NO corrections).

The missing measurement: every run in the crisis program (v2, v2.1, mC, mA-mG) trained with
corrections in the loss; the base recipe alone — warm-start pillar3b_epoch_20 + gentle V13
re-distill (lr 5e-5, T=0.5) — was never evaluated. This isolates:
  * the corrections' true contribution: mC bar (16,738 @1k local) minus ctrl0-ep2;
  * the ep2-peak mechanism: if ctrl0 ALSO peaks at ep2, the peak is a property of the
    main-loss drift dynamics (identical in every run), not of any aux knob.
3 epochs only (~2h A100). No corpus needed. Evals local (ep1/2/3, 1k each, ~12 min).
"""
import os, json

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alphatrain')
SUF = 'ctrl0'


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = [
    md(f"""
# Pillar3d **{SUF}** — λ=0 control: gentle V13 re-distill, NO corrections

Same base recipe as every crisis run (warm-start pillar3b_epoch_20, lr 5e-5, warmup 1,
target_T 0.5) with the aux term REMOVED. Measures how much of the mC gain is self-distillation
polish vs corrections. 3 epochs (~2h A100). Checkpoints → Drive as `pillar3d_{SUF}_epoch_*.pt`.

Read (local 1k eval_policy, 775000..775999, batch 256; bar = 13.8k mC-ep2 16,738):
ctrl0-ep2 ≈ bar ⇒ the channel transmits ~nothing from corrections — rethink everything.
ctrl0-ep2 well below ⇒ corrections' isolated lift = bar − ctrl0; channel validated.
ctrl0 peaking at ep2 ⇒ the universal ep2 peak is main-loss drift dynamics, not an aux artifact.

**Upload to `MyDrive/alphatrain/`:** `colorlines_pillar3d_v2.tar.gz`, `v13_pillar3a.pt.gz`,
`pillar3b_epoch_20.pt` (all already there). No corpus needed.
"""),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
    code(r"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt', '/content/alphatrain/data/pillar3b_epoch_20.pt')

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
    md(f"## Train {SUF}  (~40 min/epoch A100)"),
    code(f"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \\
    --tensor-file alphatrain/data/v13_pillar3a.pt --amp --compile \\
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \\
    --epochs 3 --batch-size 32768 --lr 5e-5 --warmup-epochs 1 \\
    --target-temperature 0.5 \\
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
```
for EP in 1 2 3:
PYTHONPATH=. python -m scripts.eval_policy --model pillar3d_{SUF}_epoch_$EP.pt \\
    --seed-start 775000 --seed-end 775999 --device mps --batch 256
```
Compare each to the bar 16,738 (13.8k mC-ep2) AND to pillar3b base (worth one 1k eval too,
if not yet measured on this config) — that brackets the self-distillation polish from both sides.
"""),
]

out = os.path.join(DIR, f'train_pillar3d_{SUF}_colab.ipynb')
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open(out, 'w'), indent=1)
print(f"wrote {out} ({len(cells)} cells)")
