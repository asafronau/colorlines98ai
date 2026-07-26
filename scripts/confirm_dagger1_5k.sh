#!/bin/bash
# 5k-seed confirmation evals (775000-779999) for the dagger1 shortlist.
# Decides vs the vh1 bar: mean 13,080 / P50 9,323 / P5 1,222 / <1000 3.5%.
set -e
cd "$(dirname "$0")/../alphatrain/inference_cpp"
for t in e2_s200 epoch_2 e3_s400; do
  echo "===== 5k: $t ====="
  ./build/eval --model data/dagger1_${t}_ts.pt --device mps --batch 1024 \
      --seed-start 775000 --seed-end 780000 2>&1 | tail -4
done
echo "ALL 5K DONE"
