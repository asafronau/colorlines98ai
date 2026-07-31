#!/bin/bash
# Screen all small128_dagger1 checkpoints: export TS + 500-seed eval each.
# 500 seeds = CATASTROPHE FILTER ONLY (HISTORY 181 gate protocol).
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
TAGS="e1_s100 e1_s200 e1_s300 e1_s400 epoch_1 e2_s100 e2_s200 e2_s300 e2_s400 epoch_2 e3_s100 e3_s200 e3_s300 e3_s400 epoch_3"
for t in $TAGS; do
  echo "===== $t ====="
  python -m alphatrain.inference_cpp.export_ts \
      --model alphatrain/data/small128_dagger1_$t.pt 2>&1 | grep -E "arch|diff"
  mv alphatrain/inference_cpp/data/policy_ts.pt \
     alphatrain/inference_cpp/data/dagger1_${t}_ts.pt
  (cd alphatrain/inference_cpp && ./build/eval \
      --model data/dagger1_${t}_ts.pt --device mps --batch 500 \
      --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
done
echo "ALL SCREENS DONE"
