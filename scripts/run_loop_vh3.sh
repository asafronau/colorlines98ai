#!/bin/bash
# Round 2 of the loop from vh3 — the SIMPLE option (the 256ch cycle, hard-label
# form): fresh games -> fresh head on vh3 backbone -> fresh mining (bulk @600 +
# deep frontier tranche @1600/2400 + capped selfplay). All-new data; no stale
# vh2-era games in the primaries.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

echo "=== phase 1: 20k fresh vh3 games ==="
mkdir -p data/vh3_games
(cd $CPP && ./build/eval --model data/vh3_policy_ts.pt --device mps \
    --batch 512 --seed-start 1000000 --seed-end 1020000 --max-turns 40000 \
    --record-dir ../../data/vh3_games) 2>&1 | tail -3

echo "=== phase 2: value head on vh3 backbone ==="
python -m alphatrain.scripts.build_value_targets_slim \
    --games-dir data/vh3_games \
    --output alphatrain/data/value_targets_vh3.pt \
    --val-min-seed 1016000 --val-max-seed 1018000 --broad-keep 0.15 2>&1 | tail -2
python -m alphatrain.scripts.train_value_head \
    --backbone alphatrain/data/small128_vh3.pt \
    --train-data alphatrain/data/value_targets_vh3.pt \
    --out alphatrain/data/value_head_vh3.pt --epochs 5 2>&1 | tail -2
python -m alphatrain.inference_cpp.export_policy_value \
    --model alphatrain/data/small128_vh3.pt \
    --head alphatrain/data/value_head_vh3.pt 2>&1 | grep -E "traced|values"
mv $CPP/data/policy_value_ts.pt $CPP/data/pv_vh3_ts.pt

echo "=== phase 3: bulk crisis mine @600 (2,600 seeds) ==="
mkdir -p data/crisis_vh3
(cd $CPP && ./build/mcts_crisis --model data/vh3_policy_ts.pt \
    --value-module data/pv_vh3_ts.pt --device mps \
    --seed-start 1030000 --seed-end 1032600 \
    --recovery-turns 15 --recovery-sims 600 \
    --prevention-turns 30 --prevention-sims 600 --q-weight 2.0 \
    --out-dir ../../data/crisis_vh3) 2>&1 | grep -vE "^  \[replay" | tail -2

echo "=== phase 4: DEEP frontier tranche @1600/2400 (300 seeds) ==="
mkdir -p data/crisis_vh3_deep
(cd $CPP && ./build/mcts_crisis --model data/vh3_policy_ts.pt \
    --value-module data/pv_vh3_ts.pt --device mps \
    --seed-start 1040000 --seed-end 1040300 \
    --recovery-turns 15 --recovery-sims 2400 \
    --prevention-turns 30 --prevention-sims 1600 --q-weight 2.0 \
    --out-dir ../../data/crisis_vh3_deep) 2>&1 | grep -vE "^  \[replay" | tail -2

echo "=== phase 5: fresh capped selfplay (1,500 games @1k turns) ==="
mkdir -p data/selfplay_vh3
(cd $CPP && ./build/mcts_selfplay --model data/vh3_policy_ts.pt \
    --value-module data/pv_vh3_ts.pt --device mps \
    --seed-start 1050000 --seed-end 1051500 --sims 400 --q-weight 2.0 \
    --max-turns 1000 \
    --out-dir ../../data/selfplay_vh3) 2>&1 | tail -2
echo "LOOP_VH3 GENERATION DONE"
