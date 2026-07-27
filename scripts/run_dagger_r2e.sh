#!/bin/bash
# R2e: gameplay verdict on the two best legalmax frontier points
# (lambda=10/ep5, lambda=6/ep5): export TS -> 500 catastrophe screen -> 5k.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

for M in l100_ep5:checkpoints/dagger_r2c_s0_l100/ft_epoch_5.pt \
         l60_ep5:checkpoints/dagger_r2c_s0_l60/ft_epoch_5.pt; do
  TAG="${M%%:*}"; CKPT="${M#*:}"
  echo "=== export $TAG ==="
  python -m alphatrain.inference_cpp.export_ts --model $CKPT 2>&1 | grep -E "arch|diff"
  mv alphatrain/inference_cpp/data/policy_ts.pt \
     alphatrain/inference_cpp/data/r2c_${TAG}_ts.pt
done

cd alphatrain/inference_cpp
for TAG in l100_ep5 l60_ep5; do
  echo "=== 500 screen: $TAG ==="
  ./build/eval --model data/r2c_${TAG}_ts.pt --device mps --batch 500 \
      --seed-start 775000 --seed-end 775500 2>&1 | tail -4
done
for TAG in l100_ep5 l60_ep5; do
  echo "=== 5k: $TAG ==="
  ./build/eval --model data/r2c_${TAG}_ts.pt --device mps --batch 1024 \
      --seed-start 775000 --seed-end 780000 2>&1 | tail -4
done
echo "R2E CHAIN DONE"
