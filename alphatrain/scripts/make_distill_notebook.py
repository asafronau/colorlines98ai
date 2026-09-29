"""Write the Colab notebook for a from-scratch PAIR2 student distilled from a teacher corpus (HISTORY 258).

    python -m alphatrain.scripts.make_distill_notebook --tag A4 --blocks 9 --channels 64 \
        --corpus alphatrain/data/distill_A4.pt --tarball colorlines_pillar3d_v8.tar.gz \
        --out alphatrain/train_scratch9b64_pair2_distill_colab.ipynb
The notebook asserts the exact sizes of the uploaded .pt.gz and .pt, so a truncated upload fails fast.
"""
import argparse
import json
import os

import torch


def cell(kind, src):
    c = {'cell_type': kind, 'metadata': {}, 'source': src}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--blocks', type=int, default=9)
    ap.add_argument('--channels', type=int, default=64)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--tarball', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pt_size = os.path.getsize(a.corpus)
    gz_size = os.path.getsize(a.corpus + '.gz')
    rows = torch.load(a.corpus, map_location='cpu', weights_only=False)['boards'].shape[0]
    corpus = os.path.basename(a.corpus)
    shape = f'{a.blocks}b{a.channels}'
    cells = [
        cell('markdown', f'# scratch{shape}_pair2_distill_{a.tag}\n\n'
             f'From-scratch **{a.blocks} blocks x {a.channels} channels** PAIR2 student (+ legal-mask loss, A1 recipe), '
             f'distilled from teacher **{a.tag}**: {rows:,} states (teacher games, epoch-4 games, crisis states) all '
             f'labeled with the teacher\'s 8-symmetry-averaged policy (top-5). HISTORY 258.\n\n'
             f'Two arms, pick one per runtime in the CONFIG cell: `TARGET = "soft"` (full top-5 distribution) or '
             f'`TARGET = "hard"` (the teacher\'s move only).\n\n'
             f'Drive files (MyDrive/alphatrain/): `{a.tarball}` + `{corpus}.gz`.'),
        cell('code', "from google.colab import drive\ndrive.mount('/content/drive')"),
        cell('code', f"""import os, time
DRIVE='/content/drive/MyDrive/alphatrain'
!cp {{DRIVE}}/{a.tarball} /content/
!cd /content && tar xzf {a.tarball}
os.makedirs('/content/alphatrain/data', exist_ok=True)
t0=time.time()
!cp {{DRIVE}}/{corpus}.gz /content/{corpus}.gz
gz=os.path.getsize('/content/{corpus}.gz'); print(f'.gz: {{gz:,}} bytes')
assert gz == {gz_size}, f'.gz truncated! got {{gz}}; re-upload {corpus}.gz'
!gunzip -t /content/{corpus}.gz && echo '.gz integrity OK'
!gzip -dc /content/{corpus}.gz > /content/alphatrain/data/{corpus}
pt=os.path.getsize('/content/alphatrain/data/{corpus}')
assert pt == {pt_size}, f'.pt size wrong! got {{pt}}'
print(f'corpus: {{pt/1e9:.2f}} GB, {rows:,} rows ({{time.time()-t0:.0f}}s)')
!rm /content/{corpus}.gz
!pip install -q numpy numba scipy
_src = open('/content/alphatrain/dataset.py').read()
assert 'def _build_line_direction_luts' in _src and 'def _transform_observation' in _src, 'STALE CODE TARBALL (old D4 bug)'
_tp = open('/content/alphatrain/train_path_b.py').read()
assert 'class MixedLoader' in _tp and 'def legal_mask_from_obs' in _tp, 'STALE CODE TARBALL (trainer)'
print('code tarball OK')"""),
        cell('code', """import torch
print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()}')
if torch.cuda.is_available():
    g=torch.cuda.get_device_properties(0); print(f'GPU {torch.cuda.get_device_name(0)} | {g.total_memory/1e9:.0f} GB')"""),
        cell('code', f"""# ===== CONFIG =====
TARGET     = "soft"   # "soft": the teacher's top-5 distribution; "hard": the teacher's move only
NUM_BLOCKS = {a.blocks}
CHANNELS   = {a.channels}
EPOCHS     = 40
BATCH      = 32768
LR         = 3e-3
SEED       = 42
SAVE_STEPS = 1000
BLEND      = 1.0 if TARGET == "soft" else 0.0   # train_path_b --blend-alpha: 1 = soft CE, 0 = hard CE
RUN        = f"scratch{shape}_pair2_distill_{a.tag}_{{TARGET}}"
print(f'RUN={{RUN}} ep={{EPOCHS}} bs={{BATCH}} lr={{LR}} blend={{BLEND}}')"""),
        cell('code', f"""# Resume from the last COMPLETED epoch if Colab stopped. Checkpoints go straight to Drive every epoch.
LATEST = f'/content/drive/MyDrive/alphatrain/{{RUN}}_ckpts/latest.pt'
RESUME_ARGS = f'--resume {{LATEST}}' if os.path.exists(LATEST) else ''
print('Resume:', LATEST if RESUME_ARGS else 'none (new run)')
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \\
    {{RESUME_ARGS}} --tensor-file alphatrain/data/{corpus} \\
    --num-blocks {{NUM_BLOCKS}} --channels {{CHANNELS}} --seed {{SEED}} --amp --compile \\
    --epochs {{EPOCHS}} --batch-size {{BATCH}} --lr {{LR}} --warmup-epochs 1 \\
    --target-temperature 1.0 --blend-alpha {{BLEND}} --policy-head pair2 --pair-dim 64 --legal-mask-loss \\
    --save-every-steps {{SAVE_STEPS}} \\
    --copy-to /content/drive/MyDrive/alphatrain/{{RUN}}_best.pt \\
    --save-dir /content/drive/MyDrive/alphatrain/{{RUN}}_ckpts 2>&1 | tee /content/{{RUN}}_train.log
# val loss is informational only - the gate is the local C++ eval (death rate over 2k games)."""),
        cell('code', """import glob
for f in sorted(glob.glob(f'/content/drive/MyDrive/alphatrain/{RUN}_ckpts/*.pt')):
    print(f)"""),
        cell('markdown', f"""## Evaluation (local M5, C++ only)

Download epochs as they land (e.g. 7 / 14 / 20 / 27 / 33 / 40) into `alphatrain/data/`, then gate each one
(folded export, 2,000 games, cap 100k, logged to EVAL.md with the death rate):

```bash
python -m alphatrain.scripts.eval_log run --model alphatrain/data/<run>_ckpts_epoch_N.pt --desc "{shape} student of {a.tag}, epoch N"
```

Compare the death rate (deaths per 100k turns, 95% CI) with the teacher's; the teacher's 8-view-averaged
policy is the label source, so matching or beating the single-pass teacher is the goal."""),
    ]
    nb = {'cells': cells, 'metadata': {'accelerator': 'GPU', 'colab': {'provenance': []},
                                       'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}},
          'nbformat': 4, 'nbformat_minor': 0}
    json.dump(nb, open(a.out, 'w'), indent=1)
    print(f'wrote {a.out}: corpus {corpus} ({rows:,} rows, .pt {pt_size:,} B, .gz {gz_size:,} B)')


if __name__ == '__main__':
    main()
