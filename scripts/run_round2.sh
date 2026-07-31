#!/bin/bash
# Round 2 of the winning advantage-filtered channel (HISTORY 192), base = vh2.
# Phases: 20k games -> value head on vh2 backbone -> fused export -> mine 300
# seeds -> tensor + full-legal mask -> row-judge every disagreement.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

echo "=== phase 1: 20k vh2 on-policy games (seeds 910000-930000) ==="
mkdir -p data/vh2_games_v1
(cd $CPP && ./build/eval --model data/vh2_policy_ts.pt --device mps \
    --batch 512 --seed-start 910000 --seed-end 930000 --max-turns 40000 \
    --record-dir ../../data/vh2_games_v1) 2>&1 | tail -4

echo "=== phase 2: value head on vh2 backbone ==="
python -m alphatrain.scripts.build_value_targets_slim \
    --games-dir data/vh2_games_v1 \
    --output alphatrain/data/value_targets_vh2_5x.pt \
    --val-min-seed 926000 --val-max-seed 928000 --broad-keep 0.15 2>&1 | tail -3
python -m alphatrain.scripts.train_value_head \
    --backbone alphatrain/data/small128_vh2.pt \
    --train-data alphatrain/data/value_targets_vh2_5x.pt \
    --out alphatrain/data/value_head_vh2_5x.pt --epochs 5 2>&1 | tail -3

echo "=== phase 3: fused export + mine 300 seeds @1200/800 ==="
python -m alphatrain.inference_cpp.export_policy_value \
    --model alphatrain/data/small128_vh2.pt \
    --head alphatrain/data/value_head_vh2_5x.pt 2>&1 | grep -E "traced|values"
mv $CPP/data/policy_value_ts.pt $CPP/data/pv_vh2_ts.pt
mkdir -p data/crisis_vh2_r2
(cd $CPP && ./build/mcts_crisis --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 950000 --seed-end 950300 \
    --recovery-turns 15 --recovery-sims 1200 \
    --prevention-turns 30 --prevention-sims 800 --q-weight 2.0 \
    --out-dir ../../data/crisis_vh2_r2) 2>&1 | tail -2

echo "=== phase 4: tensor + full-legal mask + judge export ==="
python -m alphatrain.scripts.build_expert_v2_tensor \
    --games-dir data/crisis_vh2_r2 --policy-only-data \
    --output alphatrain/data/vh2r2_crisis.pt 2>&1 | tail -1
python -m alphatrain.scripts.add_fulllegal_mask \
    --tensor alphatrain/data/vh2r2_crisis.pt \
    --base alphatrain/data/small128_vh2.pt 2>&1 | tail -1
python -m alphatrain.scripts.export_advantage_judge \
    --tensor alphatrain/data/vh2r2_crisis.pt \
    --base alphatrain/data/small128_vh2.pt \
    --out $CPP/data/adv2_judge_states.bin 2>&1 | tail -1

echo "=== phase 5: row-judge every disagreement (vh2 continuation) ==="
(cd $CPP && ./build/rollout_judge --states data/adv2_judge_states.bin \
    --model data/vh2_policy_ts.pt --out data/adv2_judge_results.csv) 2>&1 | tail -3
echo "ROUND2 PIPELINE DONE"
