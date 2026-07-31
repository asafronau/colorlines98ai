#!/bin/bash
# R2d: map the legalmax adoption-vs-drift frontier — stronger KL anchors
# (lambda 6, 10) + earlier epochs of the existing lambda 1/3 runs.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

for L in 6.0 10.0; do
  TAG="s0_l${L//./}"
  echo "=== train legalmax $TAG ==="
  PYTHONPATH=. python scripts/train_crisis_ft.py \
      --corpus alphatrain/data/dagger_r2_gap1.pt \
      --base alphatrain/data/small128_vh1.pt \
      --loss legalmax --margin 0.15 --kl-anchor-weight $L \
      --shuffle-seed 0 --epochs 20 --lr 1e-4 --batch 1024 \
      --save-dir checkpoints/dagger_r2c_$TAG
done

echo "=== gate the frontier ==="
python -m alphatrain.scripts.gate_dagger_r2 --models \
    checkpoints/dagger_r2c_s0_l10/ft_epoch_5.pt \
    checkpoints/dagger_r2c_s0_l10/ft_epoch_10.pt \
    checkpoints/dagger_r2c_s0_l30/ft_epoch_5.pt \
    checkpoints/dagger_r2c_s0_l30/ft_epoch_10.pt \
    checkpoints/dagger_r2c_s0_l60/ft_epoch_5.pt \
    checkpoints/dagger_r2c_s0_l60/ft_epoch_10.pt \
    checkpoints/dagger_r2c_s0_l60/ft_epoch_20.pt \
    checkpoints/dagger_r2c_s0_l100/ft_epoch_5.pt \
    checkpoints/dagger_r2c_s0_l100/ft_epoch_10.pt \
    checkpoints/dagger_r2c_s0_l100/ft_epoch_20.pt
echo "R2D CHAIN DONE"
