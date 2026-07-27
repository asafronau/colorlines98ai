#!/bin/bash
# R2b (ChatGPT follow-up review): (1) zero-cost gates of the epoch 5/10/15
# task vectors (only Delta_ep20 was tested); (2) generate 20k fresh on-policy
# games (disjoint seed range) for the value-head scaling replication.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=== stage 1: alpha merges from ft_epoch_{5,10,15} ==="
for EP in 5 10 15; do
  for A in 0.1 0.2 0.4; do
    PYTHONPATH=. python scripts/merge_checkpoints.py \
        --base alphatrain/data/small128_vh1.pt \
        --crisis checkpoints/dagger_r2_ft/ft_epoch_$EP.pt \
        --alpha $A --out checkpoints/dagger_r2_merges/ep${EP}_a${A//./}.pt
  done
done

echo "=== stage 2: gate all 9 ==="
python -m alphatrain.scripts.gate_dagger_r2 --models \
    checkpoints/dagger_r2_merges/ep5_a01.pt \
    checkpoints/dagger_r2_merges/ep5_a02.pt \
    checkpoints/dagger_r2_merges/ep5_a04.pt \
    checkpoints/dagger_r2_merges/ep10_a01.pt \
    checkpoints/dagger_r2_merges/ep10_a02.pt \
    checkpoints/dagger_r2_merges/ep10_a04.pt \
    checkpoints/dagger_r2_merges/ep15_a01.pt \
    checkpoints/dagger_r2_merges/ep15_a02.pt \
    checkpoints/dagger_r2_merges/ep15_a04.pt

echo "=== stage 3: 20k games for value-head replication (seeds 870000-890000) ==="
mkdir -p ../data/dagger_games_v2 2>/dev/null || mkdir -p data/dagger_games_v2
cd alphatrain/inference_cpp
./build/eval --model data/vh1_policy_ts.pt --device mps --batch 512 \
    --seed-start 870000 --seed-end 890000 --max-turns 40000 \
    --record-dir ../../data/dagger_games_v2
echo "R2B CHAIN DONE"
