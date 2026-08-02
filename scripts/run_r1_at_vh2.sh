#!/bin/bash
# Final-candidate (review #6): round-1 vector re-derived AT vh2.
# Fresh R=256 rejudge weights, fixed-batch x30 steps, 4 averaged replicas,
# pre-registered alpha=0.2 (the winning dose), ONE 20k paired eval.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

python -m alphatrain.scripts.build_r1_at_vh2 --stage export 2>&1 | tail -1
(cd $CPP && ./build/rollout_judge --states data/r1vh2_judge.bin \
    --model data/vh2_policy_ts.pt --reps 256 --seed-offset 900000000 \
    --out data/r1vh2_results.csv) 2>&1 | tail -3
python -m alphatrain.scripts.build_r1_at_vh2 --stage corpus 2>&1 | tail -1

for S in 0 1 2 3; do
  PYTHONPATH=. python scripts/train_crisis_ft.py \
      --corpus alphatrain/data/advfilt_r1vh2.pt \
      --base alphatrain/data/small128_vh2.pt \
      --loss soft --epochs 30 --lr 1e-4 --batch 1024 \
      --kl-anchor-weight 3.0 --anchor-games-dir data/vh2_games_v1 \
      --holdout-frac 0 --shuffle-seed $S --save-every 30 \
      --save-dir checkpoints/r1vh2/s$S 2>&1 | grep -E "ep30"
done
python -m alphatrain.scripts.avg_vectors \
    --base alphatrain/data/small128_vh2.pt \
    --checkpoints checkpoints/r1vh2/s0/ft_epoch_30.pt \
                  checkpoints/r1vh2/s1/ft_epoch_30.pt \
                  checkpoints/r1vh2/s2/ft_epoch_30.pt \
                  checkpoints/r1vh2/s3/ft_epoch_30.pt \
    --out checkpoints/r1vh2/ft_avg.pt
PYTHONPATH=. python scripts/merge_checkpoints.py \
    --base alphatrain/data/small128_vh2.pt \
    --crisis checkpoints/r1vh2/ft_avg.pt --alpha 0.2 \
    --out checkpoints/r1vh2/final_a02.pt 2>&1 | tail -1
python -m alphatrain.inference_cpp.export_ts \
    --model checkpoints/r1vh2/final_a02.pt 2>&1 | grep diff
mv $CPP/data/policy_ts.pt $CPP/data/r1vh2_a02_ts.pt

echo "=== 20k paired ==="
(cd $CPP && ./build/eval --model data/r1vh2_a02_ts.pt --device mps \
    --batch 1024 --seed-start 775000 --seed-end 795000 \
    --scores-out data/pair20k_r1vh2.csv) 2>&1 | tail -4
python -m alphatrain.scripts.paired_bootstrap \
    --base $CPP/data/pair20k_m02.csv \
    --candidates $CPP/data/pair20k_r1vh2.csv
echo "R1VH2 DONE"
