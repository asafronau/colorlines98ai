#!/bin/bash
# One flywheel turn (HISTORY 250): actor -> own games -> survival head on its frozen backbone
# (HISTORY-138 rule) -> fixed mcts_crisis windows on its deaths -> fine-tune (frozen BN) ->
# task-arithmetic sweep -> gate. Every step logs to logs/<tag>_*.log.
#
#   caffeinate -is scripts/flywheel_turn.sh <actor.pt> <tag> <record_seed> <mine_seed> "<alphas>" [prev_windows_dir ...]
#   e.g. caffeinate -is scripts/flywheel_turn.sh alphatrain/data/ta_gen1b_e4_a1.3.pt A2 3400000 3500000 "1.0 1.3 1.6" data/gen1b_crisis_A1_w45
#
# record_seed: 1,000 fresh seeds for the actor's own games (cap 100k turns) (also a fresh-bank eval of it).
# mine_seed:   3,000 fresh probe seeds for crisis mining. prev_windows_dir: earlier generations'
# window corpora replayed in the fine-tune.
set -euo pipefail
cd "$(dirname "$0")/.."
ACTOR=$1; TAG=$2; RSEED=$3; MSEED=$4; ALPHAS=$5; shift 5; PREV=("$@")
CPP=alphatrain/inference_cpp
D=alphatrain/data
PY=.venv/bin/python
mkdir -p logs checkpoints/${TAG}_ft $D/greedy_${TAG}_cap100k

echo "=== [1/7] export $TAG ($(date)) ==="
[ -f $CPP/data/${TAG}_ts.pt ] || $PY -m alphatrain.inference_cpp.export_ts --model "$ACTOR" \
    --output $CPP/data/${TAG}_ts.pt > logs/${TAG}_export.log 2>&1

echo "=== [2/7] record 1,000 own games (cap 100k turns), seeds $RSEED+ ($(date)) ==="
(cd $CPP && ./build/eval --model data/${TAG}_ts.pt --device mps --batch 500 --seed-start $RSEED \
    --seed-end $((RSEED + 1000)) --max-turns 100000 --record-dir ../../$D/greedy_${TAG}_cap100k --record-every 8 \
    --record-tail 300 --scores-out data/greedy_${TAG}_record.csv) > logs/${TAG}_record.log 2>&1
$PY -m alphatrain.scripts.eval_log csv --csv $CPP/data/greedy_${TAG}_record.csv --model $TAG \
    --cap 100000 --desc "$TAG own-games recording run (fresh seeds, greedy, cap 100k turns)"
tail -n 4 logs/${TAG}_record.log

echo "=== [3/7] survival head on the frozen $TAG backbone ($(date)) ==="
$PY -m alphatrain.scripts.build_value_targets_from_records --games-dir $D/greedy_${TAG}_cap100k \
    --output $D/value_targets_${TAG}.pt --every 8 --tail 300 > logs/${TAG}_value_targets.log 2>&1
$PY -m alphatrain.scripts.train_value_head --backbone "$ACTOR" --train-data $D/value_targets_${TAG}.pt \
    --out $D/value_head_${TAG}.pt --epochs 5 --batch-size 4096 --lr 1e-3 --device mps \
    > logs/${TAG}_value_head.log 2>&1
tail -n 2 logs/${TAG}_value_head.log
$PY -m alphatrain.inference_cpp.export_policy_value --model "$ACTOR" --head $D/value_head_${TAG}.pt \
    > logs/${TAG}_export_pv.log 2>&1
mv $CPP/data/policy_value_ts.pt $CPP/data/pv_${TAG}_ts.pt

echo "=== [4/7] fixed-teacher crisis windows, probe seeds $MSEED+ ($(date)) ==="
(cd $CPP && ./build/mcts_crisis --run-id ${TAG}_crisis_w45 --model data/${TAG}_ts.pt \
    --value-module data/pv_${TAG}_ts.pt --device mps --seed-start $MSEED --seed-end $((MSEED + 3000)) \
    --recovery-turns 15 --recovery-sims 600 --prevention-turns 30 --prevention-sims 600 \
    --c-puct 1.5 --q-weight 2.0 --virtual-mean --dirichlet-weight 0 --continue-turns 45 \
    --policy-max-turns 100000 --threads 14 --out-dir ../../data/${TAG}_crisis_w45) \
    > logs/${TAG}_crisis_w45.log 2>&1
tail -n 1 logs/${TAG}_crisis_w45.log
files=(data/${TAG}_crisis_w45/game_seed*_prevention_*.json)
grep -q '"value_kind": "neural"' "${files[0]}" || { echo "FATAL: not a neural-leaf search"; exit 1; }

echo "=== [5/7] corpus ($(date)) ==="
$PY -m alphatrain.scripts.build_expert_v2_tensor --games-dir data/${TAG}_crisis_w45 ${PREV[@]+"${PREV[@]}"} \
    --policy-only-data --output $D/${TAG}_crisis_w45.pt | grep -E 'boards:'
$PY -m alphatrain.scripts.anomaly_checks --tensor $D/${TAG}_crisis_w45.pt --model "$ACTOR" --n 20000 \
    | grep -E '^ *(B|C)\.'

echo "=== [6/7] fine-tune $TAG on the windows ($(date)) ==="
$PY -m alphatrain.train_path_b --tensor-file $D/${TAG}_crisis_w45.pt --resume "$ACTOR" --warm-start \
    --freeze-bn --policy-head pair2 --pair-dim 64 --legal-mask-loss --num-blocks 18 --channels 96 \
    --epochs 4 --batch-size 1024 --lr 1e-4 --warmup-epochs 0 --target-temperature 0.5 \
    --augment-factor 1 --seed 42 --save-dir checkpoints/${TAG}_ft > logs/${TAG}_ft.log 2>&1
grep -E 'val:' logs/${TAG}_ft.log | tail -1

echo "=== [7/7] task-arithmetic sweep + gate, alphas $ALPHAS ($(date)) ==="
scripts/ta_sweep.sh "$ACTOR" checkpoints/${TAG}_ft/epoch_4.pt ta_${TAG}_e4 \
    "flywheel: $TAG + alpha*($TAG fine-tuned 4ep on its fixed-teacher crisis windows${PREV:+ + ${PREV[*]}}); frozen BN, lr1e-4, T0.5" \
    $ALPHAS
echo "=== done ($(date)) ==="
