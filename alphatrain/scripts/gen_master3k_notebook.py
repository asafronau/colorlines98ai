"""Generate the Colab capacity arm for the large 3f/3k master corpus."""

from __future__ import annotations

import json
import os


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TEMPLATE = os.path.join(ROOT, 'alphatrain',
                        'train_scratch128_lr3e3_colab.ipynb')
TENSOR = os.path.join(ROOT, 'alphatrain', 'data',
                      'master_3f3k_pillar3k.pt')
TENSOR_GZ = TENSOR + '.gz'
TARBALL = os.path.join(ROOT, 'colorlines_pillar3d_v5.tar.gz')
OUTPUT = os.path.join(ROOT, 'alphatrain',
                      'train_small128_strong_colab.ipynb')
ROWS = 13_412_868


def source(cell):
    return ''.join(cell.get('source', []))


def set_source(cell, value):
    cell['source'] = value.splitlines(keepends=True)


def main():
    for path in (TEMPLATE, TENSOR, TENSOR_GZ, TARBALL):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    pt_size = os.path.getsize(TENSOR)
    gz_size = os.path.getsize(TENSOR_GZ)
    tar_size = os.path.getsize(TARBALL)

    with open(TEMPLATE) as fh:
        nb = json.load(fh)

    title_done = config_done = data_done = train_done = False
    for cell in nb['cells']:
        text = source(cell)
        if cell['cell_type'] == 'markdown' and text.startswith('# scratch128'):
            set_source(cell, f"""# small128_strong — strong-corpus capacity stress test

This is a **source-pure historical-master arm**, deliberately separate from the
current vh/flywheel lineage.  It asks a one-sided capacity question: can a
10b×128ch model trained from scratch with the newly promising high-LR recipe
absorb trajectories produced by much stronger players?

Corpus: **{ROWS:,} states** (3.5× the old `distill_pillar3k` corpus): all V14
pillar3f self-play + crisis, only the genuinely new V15 self-play suffix, all
later 3f crises, and all available pillar3k `crisis_v17` games.  Every state was
relabelled uniformly by `pillar3k_r3_dw3_T0.7_epoch_22` using its exact legal
top-5 FP16 policy.  Original heterogeneous visit targets are not mixed.

Recipe is the current `scratch128_lr3e3` recipe with **only the corpus changed**:
`bs=32768`, `lr=3e-3`, one-epoch warmup, plain cosine, `T=1`, pure soft CE,
unweighted, corrected D4 + color augmentation.  A win establishes capacity; a
failure does not prove incapacity because the historical successful student used
hard/soft blending and different optimizer geometry.

The promoted `small128_vh3` **20K reference bar** is mean **13,765**, P50
**9,658**, P5 **1,140**, P10 **1,908**, `<1000` **3.9%**.  Use short 1K
screens only to choose checkpoints, then compare `small128_strong` and vh3 on
the same 5K seed distribution.  Individual seed trajectories are not paired
evidence in this stochastic game.

Upload to `MyDrive/alphatrain/`:

1. `colorlines_pillar3d_v5.tar.gz` — {tar_size:,} bytes
2. `master_3f3k_pillar3k.pt.gz` — {gz_size:,} bytes
""")
            title_done = True
        elif '===== CONFIG' in text:
            set_source(cell, """# ===== CONFIG (small128_strong) =====
CHANNELS = 128
EPOCHS   = 40
BATCH    = 32768
LR       = 3e-3
T        = 1.0
DW       = 0
BLEND    = 0.0
GAMMA    = 0  # primary capacity arm is unweighted
SEED     = 42
SAVE_STEPS = 1000
RUN      = "small128_strong"
print(f'RUN={RUN} ep={EPOCHS} bs={BATCH} lr={LR} gamma={GAMMA}')
""")
            config_done = True
        elif (cell['cell_type'] == 'code' and 'r2_bulk.pt.gz' in text
              and 'gunzip' in text):
            set_source(cell, f"""import os, time, torch
DRIVE='/content/drive/MyDrive/alphatrain'
!cp {{DRIVE}}/colorlines_pillar3d_v5.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v5.tar.gz

# Refuse the historical corrupt D4 implementation.
_dataset_source = open('/content/alphatrain/dataset.py').read()
assert ('def _build_line_direction_luts' in _dataset_source and
        'def _transform_observation' in _dataset_source), (
    'STALE CODE TARBALL: rebuild/upload colorlines_pillar3d_v5.tar.gz')
print('Exact directional-channel D4 transform confirmed.')

os.makedirs('/content/alphatrain/data', exist_ok=True)
t0=time.time()
!cp {{DRIVE}}/master_3f3k_pillar3k.pt.gz /content/master_3f3k_pillar3k.pt.gz
gz=os.path.getsize('/content/master_3f3k_pillar3k.pt.gz')
print(f'.gz: {{gz:,}} bytes')
assert gz == {gz_size:_}, f'.gz truncated! got {{gz}}; re-upload'
!gunzip -t /content/master_3f3k_pillar3k.pt.gz && echo '.gz integrity OK'
!gzip -dc /content/master_3f3k_pillar3k.pt.gz > /content/alphatrain/data/master_3f3k_pillar3k.pt
pt=os.path.getsize('/content/alphatrain/data/master_3f3k_pillar3k.pt')
assert pt == {pt_size:_}, f'.pt size wrong! got {{pt}}'
d=torch.load('/content/alphatrain/data/master_3f3k_pillar3k.pt',
             map_location='cpu', weights_only=False, mmap=True)
assert d['boards'].shape[0] == {ROWS:_}
assert d.get('relabeled_by','').endswith('pillar3k_r3_dw3_T0.7_epoch_22.pt')
assert d.get('value_mode') == 'policy_slim'
assert d.get('metadata', {{}}).get('policy_status') == 'teacher_topk'
assert d.get('metadata', {{}}).get('policy_precision') == 'fp16'
assert d['pol_indices'].shape == ({ROWS:_}, 5)
print(f'corpus: {{pt/1e9:.2f}} GB, {{len(d["boards"]):,}} uniformly relabelled states '
      f'({{time.time()-t0:.0f}}s)')
del d
!rm /content/master_3f3k_pillar3k.pt.gz
!pip install -q numpy numba scipy
""")
            data_done = True
        elif cell['cell_type'] == 'code' and 'train_path_b' in text:
            text = text.replace('alphatrain/data/r2_bulk.pt',
                                'alphatrain/data/master_3f3k_pillar3k.pt')
            set_source(cell, text)
            train_done = True

    assert title_done and config_done and data_done and train_done
    with open(OUTPUT, 'w') as fh:
        json.dump(nb, fh, indent=1)
    print(f'{OUTPUT}: {ROWS:,} rows, pt={pt_size:,}, gz={gz_size:,}, '
          '128ch bs32768 lr3e-3')


if __name__ == '__main__':
    main()
