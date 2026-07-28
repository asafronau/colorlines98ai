#!/bin/bash
# vh2b: tax-reduction arms on the EXISTING iteration-3 corpus.
#   r6  = 6:1 rehearsal (14% signal), lr 1e-4, 2 epochs
#   r10 = 10:1 rehearsal (9% signal), lr 1e-4, 2 epochs
#   lr5 = 3:1 (25% signal), lr 5e-5, 3 epochs
# Then: screen a fixed checkpoint set per arm (catastrophe filter),
# and 5k the floor-best of each arm.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate

train () {  # name tensor lr epochs
  PYTHONPATH=. python -m alphatrain.train_path_b \
      --tensor-file $2 \
      --resume alphatrain/data/small128_vh1.pt --warm-start \
      --channels 128 --seed 42 --epochs $4 --batch-size 4096 --lr $3 \
      --warmup-epochs 1 --target-temperature 1.0 --decisiveness-power 0 \
      --blend-alpha 0.5 --save-every-steps 100 \
      --save-dir checkpoints/$1
}

echo "=== train r6 ==="  && train vh2b_r6  alphatrain/data/vh2b_r6_mix.pt  1e-4 2
echo "=== train r10 ===" && train vh2b_r10 alphatrain/data/vh2b_r10_mix.pt 1e-4 2
echo "=== train lr5 ===" && train vh2b_lr5 alphatrain/data/vh2try_mix.pt   5e-5 3

for ARM in vh2b_r6 vh2b_r10 vh2b_lr5; do
  for T in e1_s200 e1_s400 e1_s700 epoch_1 epoch_2; do
    [ -f checkpoints/$ARM/$T.pt ] || continue
    echo "===== $ARM/$T ====="
    python -m alphatrain.inference_cpp.export_ts \
        --model checkpoints/$ARM/$T.pt 2>&1 | grep -E "diff"
    mv alphatrain/inference_cpp/data/policy_ts.pt \
       alphatrain/inference_cpp/data/${ARM}_${T}_ts.pt
    (cd alphatrain/inference_cpp && ./build/eval \
        --model data/${ARM}_${T}_ts.pt --device mps --batch 500 \
        --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
  done
done
echo "ARMS + SCREENS DONE"
