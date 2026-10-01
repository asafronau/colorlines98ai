#!/bin/bash
# One flywheel turn (HISTORY 250): actor -> own games -> survival head on its frozen backbone
# (HISTORY-138 rule) -> fixed mcts_crisis windows on its deaths -> fine-tune (frozen BN) ->
# task-arithmetic sweep -> gate. Every step logs to logs/<tag>_*.log.
#
#   caffeinate -is scripts/flywheel_turn.sh <actor.pt> <tag> <record_seed> <mine_seed> "<alphas>" [prev_windows_dir ...]
#   e.g. caffeinate -is scripts/flywheel_turn.sh alphatrain/data/ta_gen1b_e4_a1.3.pt A2 3400000 3500000 "1.0 1.3 1.6" data/gen1b_crisis_A1_w45
#
# record_seed: 1,000 fresh seeds for the actor's own games (cap 100k turns) (also a fresh-bank eval of it).
# mine_seed:   $PROBES (default 3,000) fresh probe seeds for crisis mining. prev_windows_dir: earlier generations'
# window corpora replayed in the fine-tune. $SIMS (default 600): search sims per crisis move. Any actor size works
# (the fine-tune reads the trunk shape from the checkpoint).
# Window placement (HISTORY 265): REC_TURNS / PREV_TURNS (default 15 / 30) = how many moves before the probe's
# death the two search windows start, CONT_TURNS (default 45) = searched moves per window. HW = leaf-value weights
# over the survival horizons 25,50,100,200 (default 1.0,0.8,0.5,0.25). RUN (default = tag) names the windows,
# fine-tune and merges, so an arm with other windows reuses the actor's recorded games and survival head.
set -euo pipefail
cd "$(dirname "$0")/.."
ACTOR=$1; TAG=$2; RSEED=$3; MSEED=$4; ALPHAS=$5; shift 5; PREV=("$@")
CPP=alphatrain/inference_cpp
D=alphatrain/data
PY=.venv/bin/python
PROBES=${PROBES:-3000}   # crisis probe seeds; raise as deaths get rare (e.g. PROBES=5000 scripts/flywheel_turn.sh ...)
# Default mining source (owner, 2026-09-28): a WEAKER player's deaths, replayed by the actor's search. The
# actor's own deaths get rare as it nears infinite play; epoch-4 deaths teach the same (HISTORY 255).
PROBE_MODEL=${PROBE_MODEL-data/scratch18b96_lr3e3_ckpts_epoch_4_ts.pt}   # empty = the actor probes itself
PROBE_SLIP=${PROBE_SLIP:-0}   # per-move mouse-slip probability for the probe player
SIMS=${SIMS:-600}             # search sims per move in the crisis windows (recovery and prevention)
THREADS=${THREADS:-256}       # crisis games in flight: sets the shared GPU batch, not the search (HISTORY 260)
BATCH_WAIT=${BATCH_WAIT:-3000}  # us the GPU server waits to batch every thread's request (mcts_crisis --batch-wait-us)
SHAPE=$($PY -m alphatrain.scripts.model_shape "$ACTOR")   # fine-tune any actor size (18x96, 9x64 student, ...)
REC_TURNS=${REC_TURNS:-15}; PREV_TURNS=${PREV_TURNS:-30}; CONT_TURNS=${CONT_TURNS:-45}
HW=${HW:-}
RUN=${RUN:-$TAG}
WIN=${RUN}_crisis_w${CONT_TURNS}
PV=pv_${TAG}${HW:+_hw$(echo "$HW" | tr ",." "_p")}_ts.pt
mkdir -p logs checkpoints/${RUN}_ft $D/greedy_${TAG}_cap100k

echo "=== [1/7] export $TAG ($(date)) ==="
[ -f $CPP/data/${TAG}_ts.pt ] || $PY -m alphatrain.inference_cpp.export_ts --model "$ACTOR" \
    --output $CPP/data/${TAG}_ts.pt > logs/${TAG}_export.log 2>&1

# Steps 2-3 are skipped when their final output exists, so a stopped turn resumes by rerunning the same command
# (mcts_crisis resumes per seed from its out-dir; steps 5-7 are cheap to redo).
echo "=== [2/7] record 1,000 own games (cap 100k turns), seeds $RSEED+ ($(date)) ==="
if [ -f $CPP/data/greedy_${TAG}_record.csv ]; then echo "(done earlier: $CPP/data/greedy_${TAG}_record.csv)"; else
(cd $CPP && ./build/eval --model data/${TAG}_ts.pt --device mps --batch 1000 --seed-start $RSEED \
    --seed-end $((RSEED + 1000)) --max-turns 100000 --record-dir ../../$D/greedy_${TAG}_cap100k --record-every 8 \
    --record-tail 300 --scores-out data/greedy_${TAG}_record.csv) > logs/${TAG}_record.log 2>&1
$PY -m alphatrain.scripts.eval_log csv --csv $CPP/data/greedy_${TAG}_record.csv --model $TAG \
    --cap 100000 --desc "$TAG own-games recording run (fresh seeds, greedy, cap 100k turns)"
tail -n 4 logs/${TAG}_record.log
fi

echo "=== [3/7] survival head on the frozen $TAG backbone ($(date)) ==="
if [ -f $D/value_head_${TAG}.pt ]; then echo "(done earlier: $D/value_head_${TAG}.pt)"; else
$PY -m alphatrain.scripts.build_value_targets_from_records --games-dir $D/greedy_${TAG}_cap100k \
    --output $D/value_targets_${TAG}.pt --every 8 --tail 300 > logs/${TAG}_value_targets.log 2>&1
$PY -m alphatrain.scripts.train_value_head --backbone "$ACTOR" --train-data $D/value_targets_${TAG}.pt \
    --out $D/value_head_${TAG}.pt --epochs 5 --batch-size 4096 --lr 1e-3 --device mps \
    > logs/${TAG}_value_head.log 2>&1
tail -n 2 logs/${TAG}_value_head.log
fi
if [ -f $CPP/data/$PV ]; then echo "(done earlier: $CPP/data/$PV)"; else
$PY -m alphatrain.inference_cpp.export_policy_value --model "$ACTOR" --head $D/value_head_${TAG}.pt \
    ${HW:+--horizon-weights $HW} > logs/${TAG}_export_pv${HW:+_hw}.log 2>&1
mv $CPP/data/policy_value_ts.pt $CPP/data/$PV
fi

echo "=== [4/7] fixed-teacher crisis windows $WIN (start $REC_TURNS / $PREV_TURNS moves before death, $CONT_TURNS searched), $PROBES probe seeds from $MSEED ($(date)) ==="
(cd $CPP && ./build/mcts_crisis --run-id $WIN --model data/${TAG}_ts.pt \
    --value-module data/$PV --device mps --seed-start $MSEED --seed-end $((MSEED + PROBES)) \
    --recovery-turns $REC_TURNS --recovery-sims $SIMS --prevention-turns $PREV_TURNS --prevention-sims $SIMS \
    --c-puct 1.5 --q-weight 2.0 --virtual-mean --dirichlet-weight 0 --continue-turns $CONT_TURNS \
    ${PROBE_MODEL:+--probe-model $PROBE_MODEL} --probe-slip $PROBE_SLIP \
    --policy-max-turns 100000 --threads $THREADS --batch-wait-us $BATCH_WAIT --out-dir ../../data/$WIN) \
    > logs/$WIN.log 2>&1
tail -n 1 logs/$WIN.log
files=(data/$WIN/game_seed*_prevention_*.json)
grep -q '"value_kind": "neural"' "${files[0]}" || { echo "FATAL: not a neural-leaf search"; exit 1; }

echo "=== [5/7] corpus ($(date)) ==="
$PY -m alphatrain.scripts.build_expert_v2_tensor --games-dir data/$WIN ${PREV[@]+"${PREV[@]}"} \
    --policy-only-data --output $D/$WIN.pt | grep -E 'boards:'
$PY -m alphatrain.scripts.anomaly_checks --tensor $D/$WIN.pt --model "$ACTOR" --n 20000 \
    | grep -E '^ *(B|C)\.'

echo "=== [6/7] fine-tune $TAG on $WIN ($(date)) ==="
$PY -m alphatrain.train_path_b --tensor-file $D/$WIN.pt --resume "$ACTOR" --warm-start \
    --freeze-bn --policy-head pair2 --pair-dim 64 --legal-mask-loss $SHAPE \
    --epochs 4 --batch-size 1024 --lr 1e-4 --warmup-epochs 0 --target-temperature 0.5 \
    --augment-factor 1 --seed 42 --save-dir checkpoints/${RUN}_ft > logs/${RUN}_ft.log 2>&1
grep -E "val:" logs/${RUN}_ft.log | tail -1

echo "=== [7/7] task-arithmetic sweep + gate, alphas $ALPHAS ($(date)) ==="
scripts/ta_sweep.sh "$ACTOR" checkpoints/${RUN}_ft/epoch_4.pt ta_${RUN}_e4 \
    "flywheel: $TAG + alpha*($TAG fine-tuned 4ep on its fixed-teacher crisis windows ($SIMS sims, windows $REC_TURNS/$PREV_TURNS+$CONT_TURNS${HW:+, leaf weights $HW})${PREV:+ + ${PREV[*]}}); frozen BN, lr1e-4, T0.5" \
    $ALPHAS
echo "=== done ($(date)) ==="
