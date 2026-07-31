#!/bin/bash
# Iteration-4 pilot (review #5): source-aware weighting on the EXISTING deep
# corpus, preservation via KL anchor (no rehearsal mix).
#   arm A: w = top_share^1.5 * (1 + 2*disagree), lambda=1
#   arm B: w = top_share^3   * (1 + 5*disagree), lambda=1
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

run () {  # name P gamma
  PYTHONPATH=. python -m alphatrain.train_path_b \
      --tensor-file alphatrain/data/vh2c_crisis.pt \
      --resume alphatrain/data/small128_vh1.pt --warm-start \
      --channels 128 --seed 42 --epochs 6 --batch-size 4096 --lr 1e-4 \
      --warmup-epochs 1 --target-temperature 1.0 --blend-alpha 0.5 \
      --decisiveness-power $2 --disagree-gamma $3 \
      --kl-anchor-weight 1.0 --save-every-steps 100 \
      --save-dir checkpoints/$1
}

echo "=== arm A: P=1.5 gamma=2 ===" && run iter4_a 1.5 2
echo "=== arm B: P=3 gamma=5 ==="   && run iter4_b 3 5

for ARM in iter4_a iter4_b; do
  for T in e1_s100 epoch_1 e2_s100 epoch_2 epoch_3 epoch_4 epoch_6; do
    [ -f checkpoints/$ARM/$T.pt ] || continue
    echo "===== $ARM/$T ====="
    python -m alphatrain.inference_cpp.export_ts \
        --model checkpoints/$ARM/$T.pt 2>&1 | grep diff
    mv alphatrain/inference_cpp/data/policy_ts.pt \
       alphatrain/inference_cpp/data/${ARM}_${T}_ts.pt
    (cd alphatrain/inference_cpp && ./build/eval \
        --model data/${ARM}_${T}_ts.pt --device mps --batch 500 \
        --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
  done
done
echo "PILOT DONE"
