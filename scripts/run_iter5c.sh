#!/bin/bash
# Iteration-5 SCALE C ("just for science"): ~4x Scale A. Total 10k crisis
# seeds + ~4,000 short selfplay games -> ~8.6M states @ ~70/30. Identical
# recipe, step-matched (2 epochs). The third point on the dose-response curve.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

echo "=== phase 0: wait for Scale B ==="
until grep -q "ITER5B PIPELINE DONE" logs/iter5b.log 2>/dev/null; do sleep 300; done

echo "=== phase 1: crisis mine +5,000 seeds (total 10k) ==="
(cd $CPP && ./build/mcts_crisis --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 965000 --seed-end 970000 \
    --recovery-turns 15 --recovery-sims 600 \
    --prevention-turns 30 --prevention-sims 600 --q-weight 2.0 \
    --out-dir ../../data/crisis_iter5) 2>&1 | grep -vE "^  \[replay" | tail -2

echo "=== phase 2: selfplay +2,700 games @400, 1k-turn cap ==="
(cd $CPP && ./build/mcts_selfplay --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 972220 --seed-end 974920 --sims 400 --q-weight 2.0 \
    --max-turns 1000 \
    --out-dir ../../data/selfplay_iter5b) 2>&1 | tail -2

echo "=== phase 3: tensor (~8.6M @ ~70/30) ==="
python -m alphatrain.scripts.build_expert_v2_tensor \
    --games-dir data/crisis_iter5 data/selfplay_iter5 data/selfplay_iter5b \
    --policy-only-data --output alphatrain/data/iter5c.pt 2>&1 | tail -2

echo "=== phase 4: train (same recipe, step-matched: 2 epochs) ==="
PYTHONPATH=. python -m alphatrain.train_path_b \
    --tensor-file alphatrain/data/iter5c.pt \
    --resume alphatrain/data/small128_vh2.pt --warm-start \
    --channels 128 --seed 42 --epochs 2 --batch-size 8192 --lr 3e-4 \
    --warmup-epochs 1 --target-temperature 0.7 --decisiveness-power 3.0 \
    --save-every-steps 1000 \
    --save-dir checkpoints/iter5c 2>&1 | grep -E "Epoch|val|Done"

echo "=== phase 5: screens ==="
for T in e1_s2000 e1_s4000 e1_s6000 epoch_1 epoch_2; do
  [ -f checkpoints/iter5c/$T.pt ] || continue
  echo "===== $T ====="
  python -m alphatrain.inference_cpp.export_ts \
      --model checkpoints/iter5c/$T.pt 2>&1 | grep diff
  mv $CPP/data/policy_ts.pt $CPP/data/iter5c_${T}_ts.pt
  (cd $CPP && ./build/eval --model data/iter5c_${T}_ts.pt --device mps \
      --batch 500 --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
done
echo "ITER5C PIPELINE DONE"
