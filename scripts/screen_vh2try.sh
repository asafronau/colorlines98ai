#!/bin/bash
# Screen vh2try checkpoints: export TS + 500-seed catastrophe filter each.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
TAGS="e1_s100 e1_s200 e1_s300 e1_s400 e1_s500 e1_s600 e1_s700 epoch_1 e2_s100 e2_s200 e2_s300 epoch_2 e3_s100 epoch_3"
for t in $TAGS; do
  echo "===== $t ====="
  python -m alphatrain.inference_cpp.export_ts \
      --model checkpoints/small128_vh2try/$t.pt 2>&1 | grep -E "arch|diff"
  mv alphatrain/inference_cpp/data/policy_ts.pt \
     alphatrain/inference_cpp/data/vh2try_${t}_ts.pt
  (cd alphatrain/inference_cpp && ./build/eval \
      --model data/vh2try_${t}_ts.pt --device mps --batch 500 \
      --seed-start 775000 --seed-end 775500) 2>&1 | tail -4
done
echo "ALL SCREENS DONE"
