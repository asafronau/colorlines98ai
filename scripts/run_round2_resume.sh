#!/bin/bash
# Round-2 resume: mining (scan-based skip) -> tensor/mask -> row-judge.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
CPP=alphatrain/inference_cpp

echo "=== phase 3 (resume): mining ==="
(cd $CPP && ./build/mcts_crisis --model data/vh2_policy_ts.pt \
    --value-module data/pv_vh2_ts.pt --device mps \
    --seed-start 950000 --seed-end 950300 \
    --recovery-turns 15 --recovery-sims 1200 \
    --prevention-turns 30 --prevention-sims 800 --q-weight 2.0 \
    --out-dir ../../data/crisis_vh2_r2) 2>&1 | grep -vE "^  \[replay" | tail -3

echo "=== phase 4: tensor + mask + judge export ==="
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

echo "=== phase 5: row-judge (vh2 continuation) ==="
(cd $CPP && ./build/rollout_judge --states data/adv2_judge_states.bin \
    --model data/vh2_policy_ts.pt --out data/adv2_judge_results.csv) 2>&1 | tail -3
echo "ROUND2 RESUME DONE"
