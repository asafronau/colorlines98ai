#!/bin/bash
# vh5x decision judge (gate-2 protocol, reserved seeds >= 888000):
# arms = q0 control | q2 + falsified fresh head | q2 + new 5x head.
# Same rng -> identical sampled states across arms.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
D=alphatrain/inference_cpp/data

echo "=== relabel: vh5x head, q={0,2} ==="
PYTHONPATH=. python -m alphatrain.scripts.gate2_relabel \
    --model alphatrain/data/small128_vh1.pt \
    --head alphatrain/data/value_head_vh5x.pt \
    --games-dir data/dagger_games_v2 --min-seed 888000 \
    --n 400 --sims 600 --q-values 0.0 2.0 --out-prefix $D/g2vh5x

echo "=== relabel: OLD fresh head (falsified ref), q=2 ==="
PYTHONPATH=. python -m alphatrain.scripts.gate2_relabel \
    --model alphatrain/data/small128_vh1.pt \
    --head alphatrain/data/value_head_small128_vh1.pt \
    --games-dir data/dagger_games_v2 --min-seed 888000 \
    --n 400 --sims 600 --q-values 2.0 --out-prefix $D/g2old

cd alphatrain/inference_cpp
for TAG in g2vh5x_q0 g2vh5x_q2 g2old_q2; do
  echo "=== judge: $TAG ==="
  ./build/rollout_judge --states data/$TAG.bin --model data/vh1_policy_ts.pt \
      --out data/${TAG}_results.csv
done
cd ../..

for TAG in g2vh5x_q0 g2vh5x_q2 g2old_q2; do
  echo "=== analysis: $TAG ==="
  python -m alphatrain.scripts.rowjudge_analysis \
      --meta $D/${TAG}_meta.csv --results $D/${TAG}_results.csv
done
echo "VH5X JUDGE DONE"
