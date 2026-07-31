#!/bin/bash
# R2 arm 1 (HISTORY 183): corrections-only margin fine-tune on vh1 (frozen BN)
# -> task-vector merges -> pre-registered gate (fp16).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== stage 1: margin fine-tune (25,116 gap>=1.0 rows) ==="
PYTHONPATH=. python scripts/train_crisis_ft.py \
    --corpus alphatrain/data/dagger_r2_gap1.pt \
    --base alphatrain/data/small128_vh1.pt \
    --loss margin --margin 0.15 \
    --epochs 20 --lr 1e-4 --batch 1024 \
    --save-dir checkpoints/dagger_r2_ft

echo "=== stage 2: task-vector merges ==="
mkdir -p checkpoints/dagger_r2_merges
for A in 0.05 0.1 0.2 0.4; do
  PYTHONPATH=. python scripts/merge_checkpoints.py \
      --base alphatrain/data/small128_vh1.pt \
      --crisis checkpoints/dagger_r2_ft/ft_epoch_20.pt \
      --alpha $A --out checkpoints/dagger_r2_merges/a${A//./}.pt
done

echo "=== stage 3: pre-registered gate ==="
python -m alphatrain.scripts.gate_dagger_r2 \
    --models checkpoints/dagger_r2_merges/a005.pt \
             checkpoints/dagger_r2_merges/a01.pt \
             checkpoints/dagger_r2_merges/a02.pt \
             checkpoints/dagger_r2_merges/a04.pt \
             checkpoints/dagger_r2_ft/ft_epoch_20.pt
echo "R2 CHAIN DONE"
