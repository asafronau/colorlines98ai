#!/bin/bash
# Iteration-5 SCALE B (user directive: measure the scale-response, don't
# assert it): total 5k crisis seeds + ~1,300 short selfplay games (1k-turn
# cap, many-games diversity), ~4.3M states at ~70/30, identical recipe,
# step-matched. Compares against Scale A (2.3M) and vh2 at the 20k bar.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

echo "=== phase 0: wait for Scale A pipeline ==="
until grep -q "ITER5 PIPELINE DONE" logs/iter5.log 2>/dev/null; do sleep 120; done

echo "=== phase 1: crisis mine +2,400 seeds (total 5k) ==="
(cd $CPP && ./build/mcts_crisis --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 962600 --seed-end 965000 \
    --recovery-turns 15 --recovery-sims 600 \
    --prevention-turns 30 --prevention-sims 600 --q-weight 2.0 \
    --out-dir ../../data/crisis_iter5) 2>&1 | grep -vE "^  \[replay" | tail -2

echo "=== phase 2: selfplay +1,220 games @400, 1k-turn cap ==="
(cd $CPP && ./build/mcts_selfplay --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 971000 --seed-end 972220 --sims 400 --q-weight 2.0 \
    --max-turns 1000 \
    --out-dir ../../data/selfplay_iter5b) 2>&1 | tail -2

echo "=== phase 3: tensor (~4.3M @ ~70/30) ==="
python -m alphatrain.scripts.build_expert_v2_tensor \
    --games-dir data/crisis_iter5 data/selfplay_iter5 data/selfplay_iter5b \
    --policy-only-data --output alphatrain/data/iter5b.pt 2>&1 | tail -2

echo "=== phase 4: gzip for Colab (training moved off-box per user) ==="
gzip -9 -c alphatrain/data/iter5b.pt > alphatrain/data/iter5b.pt.gz
ls -la alphatrain/data/iter5b.pt alphatrain/data/iter5b.pt.gz
echo "ITER5B PIPELINE DONE"
exit 0

echo "=== (disabled) local train ==="
PYTHONPATH=. python -m alphatrain.train_path_b \
    --tensor-file alphatrain/data/iter5b.pt \
    --resume alphatrain/data/small128_vh2.pt --warm-start \
    --channels 128 --seed 42 --epochs 4 --batch-size 8192 --lr 3e-4 \
    --warmup-epochs 1 --target-temperature 0.7 --decisiveness-power 3.0 \
    --save-every-steps 500 \
    --save-dir checkpoints/iter5b 2>&1 | grep -E "Epoch|val|Done"

echo "=== phase 5: screens ==="
for T in e1_s1000 e1_s2000 epoch_1 epoch_2 epoch_3 epoch_4; do
  [ -f checkpoints/iter5b/$T.pt ] || continue
  echo "===== $T ====="
  python -m alphatrain.inference_cpp.export_ts \
      --model checkpoints/iter5b/$T.pt 2>&1 | grep diff
  mv $CPP/data/policy_ts.pt $CPP/data/iter5b_${T}_ts.pt
  (cd $CPP && ./build/eval --model data/iter5b_${T}_ts.pt --device mps \
      --batch 500 --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
done
echo "ITER5B PIPELINE DONE"
