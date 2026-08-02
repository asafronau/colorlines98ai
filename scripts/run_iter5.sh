#!/bin/bash
# Iteration-5: the LITERAL 256ch-track round at literal scale (user directive
# 2026-08-02). Base vh2. 70/30 crisis/selfplay, ~2.3M states, dw3/T0.7
# full-corpus distillation, step-matched to the +18% round (~15.5k steps).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

echo "=== phase 0: wait for rejudges to free the GPU ==="
until grep -q "REJUDGE_DONE" logs/rejudge.log 2>/dev/null; do sleep 60; done

echo "=== phase 1: crisis mine @600 (2,600 seeds) ==="
mkdir -p data/crisis_iter5
(cd $CPP && ./build/mcts_crisis --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 960000 --seed-end 962600 \
    --recovery-turns 15 --recovery-sims 600 \
    --prevention-turns 30 --prevention-sims 600 --q-weight 2.0 \
    --out-dir ../../data/crisis_iter5) 2>&1 | grep -vE "^  \[replay" | tail -2

echo "=== phase 2: selfplay @400 (80 games) ==="
mkdir -p data/selfplay_iter5
(cd $CPP && ./build/mcts_selfplay --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 970000 --seed-end 970080 --sims 400 --q-weight 2.0 \
    --out-dir ../../data/selfplay_iter5) 2>&1 | tail -2

echo "=== phase 3: tensor (70/30 by construction) ==="
python -m alphatrain.scripts.build_expert_v2_tensor \
    --games-dir data/crisis_iter5 data/selfplay_iter5 --policy-only-data \
    --output alphatrain/data/iter5.pt 2>&1 | tail -2

echo "=== phase 4: train (dw3, T0.7, pure soft, step-matched) ==="
PYTHONPATH=. python -m alphatrain.train_path_b \
    --tensor-file alphatrain/data/iter5.pt \
    --resume alphatrain/data/small128_vh2.pt --warm-start \
    --channels 128 --seed 42 --epochs 7 --batch-size 8192 --lr 3e-4 \
    --warmup-epochs 1 --target-temperature 0.7 --decisiveness-power 3.0 \
    --save-every-steps 500 \
    --save-dir checkpoints/iter5 2>&1 | grep -E "Epoch|val|ckpt|Done"

echo "=== phase 5: screens (catastrophe filter) ==="
for T in e1_s500 e1_s1000 e1_s2000 epoch_1 epoch_2 epoch_3 epoch_5 epoch_7; do
  [ -f checkpoints/iter5/$T.pt ] || continue
  echo "===== $T ====="
  python -m alphatrain.inference_cpp.export_ts \
      --model checkpoints/iter5/$T.pt 2>&1 | grep diff
  mv $CPP/data/policy_ts.pt $CPP/data/iter5_${T}_ts.pt
  (cd $CPP && ./build/eval --model data/iter5_${T}_ts.pt --device mps \
      --batch 500 --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
done
echo "ITER5 PIPELINE DONE"
