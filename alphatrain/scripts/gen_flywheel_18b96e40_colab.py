"""Generate the primary 18b96 iteration-1 aggregate Colab notebook.

The notebook consumes the non-overlapping, current-lineage mixed tensor built
from exploit, explore, pilot-crisis, and crisis-expansion streams.  It is the
whole-trajectory flywheel arm; sparse edit-only diagnostics are deliberately
not part of this recipe.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ALPHATRAIN = ROOT / "alphatrain"
TARBALL = ROOT / "colorlines_pillar3d_v5.tar.gz"
TARGET = ALPHATRAIN / "data/flywheel_18b96e40_i1_mixed_targets.pt"
TARGET_GZ = TARGET.with_suffix(TARGET.suffix + ".gz")
ANCHOR = ALPHATRAIN / "data/flywheel_18b96e40_i1_anchors.pt"
ANCHOR_GZ = ANCHOR.with_suffix(ANCHOR.suffix + ".gz")
BASE = ALPHATRAIN / "data/scratch18b96_lr3e3_ckpts_epoch_40.pt"
OUTPUT = ALPHATRAIN / "train_flywheel_18b96e40_i1_colab.ipynb"

TARGET_ROWS = 4_809_533
ANCHOR_ROWS = 1_798_692
SOURCE_NAMES = [
    "18b96e40_i1_exploit_400",
    "18b96e40_i1_explore_200x400",
    "18b96e40_i1_crisis_pilot_600_1600",
    "18b96e40_i1_crisis_expand_600_1600",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    required = (TARBALL, TARGET, TARGET_GZ, ANCHOR, ANCHOR_GZ, BASE)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    artifacts = {
        "colorlines_pillar3d_v5.tar.gz": TARBALL,
        TARGET_GZ.name: TARGET_GZ,
        ANCHOR_GZ.name: ANCHOR_GZ,
        BASE.name: BASE,
    }
    sizes = {name: path.stat().st_size for name, path in artifacts.items()}
    hashes = {name: sha256(path) for name, path in artifacts.items()}
    pt_sizes = {TARGET.name: TARGET.stat().st_size,
                ANCHOR.name: ANCHOR.stat().st_size}
    pt_hashes = {TARGET.name: sha256(TARGET), ANCHOR.name: sha256(ANCHOR)}

    artifact_lines = "\n".join(
        f"- `{name}` — {sizes[name]:,} bytes, SHA256 `{hashes[name]}`"
        for name in artifacts)

    cells = [
        markdown(f"""# 18b96e40 flywheel iteration 1 — aggregate selfplay + crisis

This is the primary **play → mine → train** arm.  It warm-starts the current
18-block × 96-channel actor at epoch 40 and trains on every unique, same-actor
search trajectory generated in iteration 1:

- 2,945,600 clean exploit-search states;
- 723,704 explore states with behavior noise separated from clean labels;
- 1,140,229 recovery/prevention crisis states from two disjoint, independently
  causally validated tranches;
- 1,798,692 actor-native greedy states for frozen-policy KL preservation.

The target tensor has **{TARGET_ROWS:,} unique states**.  Pilot crisis is included
once; the expansion adds only its new seeds.  No vh3, 192/256-channel, 3f, or 3k
games enter this run.

The whole-trajectory objective is intentionally different from the sparse-edit
diagnostic: hard clean-search CE on all search states plus frozen-base KL on all
anchors.  Crisis sources receive 2× loss weight, so broad board diversity is
kept while crisis is not diluted.  BatchNorm is frozen, policy forwards are
FP16, losses are FP32, batches are pooled, legality is exact, color augmentation
is enabled, and D4 is held out of this first arm.  Twelve epochs at batch 4096
give about 18k optimizer updates—comparable to the successful 40-epoch
from-scratch update count—while epoch checkpoints expose the entire trajectory.

Upload these files to `MyDrive/alphatrain/`:

{artifact_lines}
"""),
        code("""from google.colab import drive
drive.mount('/content/drive')
"""),
        code(f"""import gc, hashlib, os, shutil, time

DRIVE = '/content/drive/MyDrive/alphatrain'
ARTIFACTS = {json.dumps({name: {'size': sizes[name], 'sha256': hashes[name]}
                              for name in artifacts}, indent=2)}

def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)

for name, expected in ARTIFACTS.items():
    src = os.path.join(DRIVE, name)
    dst = os.path.join('/content', name)
    assert os.path.exists(src), f'missing Drive upload: {{src}}'
    shutil.copy2(src, dst)
    size = os.path.getsize(dst)
    digest = sha256(dst)
    assert size == expected['size'], (name, size, expected['size'])
    assert digest == expected['sha256'], (name, digest)
    print(f'verified {{name}}: {{size:,}} bytes')

os.makedirs('/content/alphatrain/data', exist_ok=True)
shutil.unpack_archive('/content/colorlines_pillar3d_v5.tar.gz', '/content')

for gz_name, pt_name, pt_size, pt_sha in [
    ('{TARGET_GZ.name}', '{TARGET.name}', {pt_sizes[TARGET.name]},
     '{pt_hashes[TARGET.name]}'),
    ('{ANCHOR_GZ.name}', '{ANCHOR.name}', {pt_sizes[ANCHOR.name]},
     '{pt_hashes[ANCHOR.name]}'),
]:
    src = os.path.join('/content', gz_name)
    dst = os.path.join('/content/alphatrain/data', pt_name)
    t0 = time.time()
    import gzip
    with gzip.open(src, 'rb') as fin, open(dst, 'wb') as fout:
        shutil.copyfileobj(fin, fout, length=8 << 20)
    assert os.path.getsize(dst) == pt_size, (pt_name, os.path.getsize(dst))
    assert sha256(dst) == pt_sha, f'decompressed SHA256 mismatch: {{pt_name}}'
    os.remove(src)
    print(f'unpacked {{pt_name}}: {{pt_size:,}} bytes in {{time.time()-t0:.0f}}s')

shutil.copy2('/content/{BASE.name}', '/content/alphatrain/data/{BASE.name}')

source = open('/content/alphatrain/train_flywheel.py').read()
for marker in ('class PooledBatchLoader', '--resume-every-steps',
               '_quantize_frozen_bn_for_deployment_', '--source-weights'):
    assert marker in source, f'STALE CODE TARBALL: missing {{marker}}'
print('Flywheel pooled/resumable/frozen-BN trainer confirmed.')
"""),
        code("""import torch
assert torch.cuda.is_available(), 'Select a CUDA Colab runtime'
gpu = torch.cuda.get_device_properties(0)
print(f'PyTorch {torch.__version__} | {torch.cuda.get_device_name(0)} | '
      f'{gpu.total_memory/1e9:.1f} GB')
"""),
        code("""# ===== PRIMARY AGGREGATE FLYWHEEL CONFIG =====
RUN = '18b96e40_i1_mixed_warm'
EPOCHS = 12
BATCH = 4096
LR = 1e-4
SOURCE_WEIGHTS = [1.0, 1.0, 2.0, 2.0]  # exploit, explore, old crisis, new crisis
ANCHOR_WEIGHT = 1.0
SEED = 20260818
SAVE_STEPS = 250     # model-only dose checkpoints within every full epoch
RESUME_STEPS = 1000  # overwrite crash-safe latest.pt within long epochs
print(RUN, 'epochs=', EPOCHS, 'batch=', BATCH, 'lr=', LR,
      'source_weights=', SOURCE_WEIGHTS)
"""),
        code(f"""# Cheap metadata/provenance preflight; tensors stay memory-mapped.
import torch

target_path = '/content/alphatrain/data/{TARGET.name}'
anchor_path = '/content/alphatrain/data/{ANCHOR.name}'
target = torch.load(target_path, map_location='cpu', weights_only=False, mmap=True)
anchor = torch.load(anchor_path, map_location='cpu', weights_only=False, mmap=True)

assert len(target['boards']) == {TARGET_ROWS}
assert len(anchor['boards']) == {ANCHOR_ROWS}
assert target['metadata']['source_names'] == {SOURCE_NAMES!r}
assert target['metadata']['lineage'] == 'scratch18b96_lr3e3_e40'
assert anchor['metadata']['lineage'] == 'scratch18b96_lr3e3_e40'
assert target['metadata']['base_checkpoint_sha256'] == \
       '233809b150e94341b250d31a112111fc36381f5f9604c6d67afcaa738740da93'
assert target['metadata']['split_semantics'] == 'group_seed_mod_20; 1=validation'
print('target rows:', len(target['boards']), target['metadata']['source_names'])
print('anchor rows:', len(anchor['boards']))
print('validation:', int((target['split'] == 1).sum()), 'target rows;',
      int((anchor['split'] == 1).sum()), 'anchor rows')
del target, anchor
gc.collect()
"""),
        code(f"""# Resume exactly from Drive after a Colab interruption.
SAVE_DIR = f'{{DRIVE}}/{{RUN}}_ckpts'
LATEST = f'{{SAVE_DIR}}/latest.pt'
RESUME_ARGS = f'--resume {{LATEST}}' if os.path.exists(LATEST) else ''
SOURCE_WEIGHT_ARGS = ' '.join(str(x) for x in SOURCE_WEIGHTS)
print('Resume:', LATEST if RESUME_ARGS else 'none (new warm-start run)')

%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_flywheel \
    --targets alphatrain/data/{TARGET.name} \
    --anchors alphatrain/data/{ANCHOR.name} \
    --base alphatrain/data/{BASE.name} \
    {{RESUME_ARGS}} \
    --objective aggregate --soft-alpha 0 \
    --source-weights {{SOURCE_WEIGHT_ARGS}} --anchor-weight {{ANCHOR_WEIGHT}} \
    --epochs {{EPOCHS}} --batch-size {{BATCH}} --lr {{LR}} --weight-decay 0 \
    --schedule cosine --warmup-fraction 0.05 --min-lr-ratio 0.05 \
    --device cuda --precision fp16 --legal-support-loss \
    --no-dihedral-augment --color-augment --augment-factor 1 \
    --save-every-steps {{SAVE_STEPS}} --resume-every-steps {{RESUME_STEPS}} \
    --archive-every-epochs 1 --audit-batches 10 --log-every 25 \
    --gate-min-retain 0.90 --gate-max-legal-kl 0.05 \
    --gate-max-new-bn-nonfinite 0 \
    --seed {{SEED}} --save-dir {{SAVE_DIR}} 2>&1 | tee /content/{{RUN}}_train.log
"""),
        code("""# Inspect the resumable state and per-epoch functional diagnostics.
import glob, json, os

for path in sorted(glob.glob(f'{SAVE_DIR}/diagnostics_epoch_*.json')):
    report = json.load(open(path))
    target = report['audit']['target']
    anchor = report['audit']['anchor']
    print(os.path.basename(path),
          f"step={report['global_step']}",
          f"target retain={target['retain']:.4f} KL={target['mean_legal_kl']:.6f}",
          f"anchor retain={anchor['retain']:.4f} KL={anchor['mean_legal_kl']:.6f}",
          'gate=', report['functional_gate']['failures'])

print('Model archives:')
for path in sorted(glob.glob(f'{SAVE_DIR}/epoch_*.pt')):
    print(path, f'{os.path.getsize(path)/1e6:.1f} MB')
"""),
        markdown("""## Local M5 promotion gate

Download epochs 1, 2, 3, 5, 8, and 12 first.  Export each checkpoint directly
to its own TorchScript path, then use aggregate-only MPS FP16 evaluation.  A
999-game run is only a catastrophe/trajectory screen; do not interpret
individual seeds or rank close candidates from it.

```bash
cd /path/to/colorlines98
PYTHONPATH=. NUMBA_CACHE_DIR=/tmp/colorlines98-numba-cache \
  caffeinate -i -s .venv/bin/python -m alphatrain.inference_cpp.export_ts \
  --model alphatrain/data/18b96e40_i1_mixed_warm_epoch_3.pt \
  --output alphatrain/inference_cpp/data/18b96e40_i1_mixed_warm_epoch_3_ts.pt

cd alphatrain/inference_cpp
caffeinate -i -s ./build/eval \
  --model data/18b96e40_i1_mixed_warm_epoch_3_ts.pt --device mps --batch 500 \
  --seed-start 1000000 --seed-end 1000999 --max-turns 1000 \
  --scores-out data/18b96e40_i1_mixed_warm_epoch_3_cap1k_1000k_999.csv
```

Take at most two trajectory survivors to the full 5,000-game development
distribution (`--seed-end 1005000`) and evaluate the untouched epoch-40 base on
that distribution once.  Compare them with independent resampling, never
per-seed deltas:

```bash
cd /path/to/colorlines98
.venv/bin/python -m alphatrain.scripts.compare_score_distributions \
  alphatrain/inference_cpp/data/18b96e40_base_cap1k_1000k_5k.csv \
  alphatrain/inference_cpp/data/18b96e40_i1_mixed_warm_epoch_3_cap1k_1000k_5k.csv
```

Promotion requires a distributional improvement without a material floor or
cap-rate regression.  Matching numeric seed IDs are environmental controls,
not paired stochastic-game evidence.  The reserved final bank
1,100,000–1,119,999 remains untouched until a development winner exists.
"""),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with OUTPUT.open("w") as handle:
        json.dump(notebook, handle, indent=1)
        handle.write("\n")
    print(f"wrote {OUTPUT}")
    for name in artifacts:
        print(f"  {name}: {sizes[name]:,} bytes {hashes[name]}")


if __name__ == "__main__":
    main()
