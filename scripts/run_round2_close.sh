#!/bin/bash
# Round-2 close: advantage-filter -> fine-tune (base vh2) -> merges -> screens
# -> 20k paired vs vh2. The exact HISTORY 192 winning recipe.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

python -m alphatrain.scripts.build_advfiltered_corpus \
    --tensor alphatrain/data/vh2r2_crisis.pt \
    --rows $CPP/data/adv2_judge_states_rows.npz \
    --results $CPP/data/adv2_judge_results.csv \
    --output alphatrain/data/advfilt2.pt

PYTHONPATH=. python scripts/train_crisis_ft.py \
    --corpus alphatrain/data/advfilt2.pt \
    --base alphatrain/data/small128_vh2.pt \
    --loss soft --epochs 15 --lr 1e-4 --batch 512 \
    --kl-anchor-weight 3.0 --anchor-games-dir data/vh2_games_v1 \
    --holdout-frac 0.15 --save-every 5 \
    --save-dir checkpoints/advfilt2_ft 2>&1 | grep -E "corpus|anchor|ep0|ep15"

for AL in 0.1 0.2 0.4; do
  PYTHONPATH=. python scripts/merge_checkpoints.py \
      --base alphatrain/data/small128_vh2.pt \
      --crisis checkpoints/advfilt2_ft/ft_epoch_15.pt \
      --alpha $AL --out checkpoints/advfilt2_ft/m${AL//./}.pt 2>&1 | tail -1
done

for M in m01 m02 m04; do
  echo "===== screen $M ====="
  python -m alphatrain.inference_cpp.export_ts \
      --model checkpoints/advfilt2_ft/$M.pt 2>&1 | grep diff
  mv $CPP/data/policy_ts.pt $CPP/data/advfilt2_${M}_ts.pt
  (cd $CPP && ./build/eval --model data/advfilt2_${M}_ts.pt --device mps \
      --batch 500 --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
done

for M in m01 m02 m04; do
  echo "===== 20k $M ====="
  (cd $CPP && ./build/eval --model data/advfilt2_${M}_ts.pt --device mps \
      --batch 1024 --seed-start 775000 --seed-end 795000 \
      --scores-out data/pair20k_r2_${M}.csv) 2>&1 | tail -4
done

python -m alphatrain.scripts.paired_bootstrap \
    --base $CPP/data/pair20k_m02.csv \
    --candidates $CPP/data/pair20k_r2_m01.csv \
                 $CPP/data/pair20k_r2_m02.csv \
                 $CPP/data/pair20k_r2_m04.csv
echo "ROUND2 CLOSE DONE"
