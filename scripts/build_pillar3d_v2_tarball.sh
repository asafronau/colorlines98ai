#!/bin/bash
# Build the code tarball used by ALL pillar3d/3e/3i Colab notebooks.
#
# Output: colorlines_pillar3d_v2.tar.gz  (repo root, ~0.5-0.7 MB)
# Upload to: MyDrive/alphatrain/  (replaces the stale copy)
#
# Pure CODE only (no checkpoints, no tensors): the whole alphatrain/ + game/ +
# scripts/ Python trees plus game test fixtures. Everything large or generated is
# excluded — most importantly alphatrain/data/ (213 GB of .pt corpora) and notebooks.
#
# This was historically built ad-hoc, so the Drive copy drifted stale and lacked
# train_gumbel.py (-> "No module named alphatrain.train_gumbel" in Colab). Re-run this
# whenever you add/modify a module a notebook imports (train_gumbel, gumbel, dataset, ...).

set -e
cd "$(dirname "$0")/.."  # repo root

OUT=colorlines_pillar3d_v2.tar.gz

echo "Building $OUT..."
tar czf "$OUT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='alphatrain/data' \
    --exclude='*.ipynb' \
    --exclude='*.tar.gz' \
    --exclude='*.nbi' \
    --exclude='*.nbc' \
    --exclude='*.pt' \
    --exclude='*.npz' \
    alphatrain \
    game \
    scripts \
    requirements.txt

ls -lh "$OUT"
echo
echo "Sanity (these MUST be present):"
for f in alphatrain/train_gumbel.py alphatrain/gumbel.py alphatrain/dataset.py \
         alphatrain/train_path_b.py alphatrain/train.py alphatrain/model.py \
         game/board.py scripts/eval_policy.py; do
    if tar tzf "$OUT" | grep -qx "$f"; then echo "  OK  $f"; else echo "  MISSING $f"; exit 1; fi
done
echo
echo "Upload to Google Drive: MyDrive/alphatrain/$OUT  (overwrite the old one)"
