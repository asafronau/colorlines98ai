#!/bin/bash
# R2c: the review's final supervised arm — legalmax hard-negative margin +
# KL anchor to vh1 on independent quiet states, frozen BN.
# 3 shuffle seeds x lambda {1.0, 3.0}, then gate every ep20.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

MODELS=""
for S in 0 1 2; do
  for L in 1.0 3.0; do
    TAG="s${S}_l${L//./}"
    echo "=== train legalmax $TAG ==="
    PYTHONPATH=. python scripts/train_crisis_ft.py \
        --corpus alphatrain/data/dagger_r2_gap1.pt \
        --base alphatrain/data/small128_vh1.pt \
        --loss legalmax --margin 0.15 --kl-anchor-weight $L \
        --shuffle-seed $S --epochs 20 --lr 1e-4 --batch 1024 \
        --save-dir checkpoints/dagger_r2c_$TAG
    MODELS="$MODELS checkpoints/dagger_r2c_$TAG/ft_epoch_20.pt"
  done
done

echo "=== gate all 6 ==="
python -m alphatrain.scripts.gate_dagger_r2 --models $MODELS
echo "R2C CHAIN DONE"
