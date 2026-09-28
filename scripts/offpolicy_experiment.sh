#!/bin/bash
# Off-policy crisis experiment (HISTORY 254). All arms start from A3 and use A3's search (A3's survival
# head, 600 sims, c1.5 q2 virtual mean) and the turn-3 fine-tune recipe:
#   arm a = turn 3 (A3's own deaths, 45 searched moves)          -> already gated (ta_A3_e4_a*)
#   arm b = deaths of the human-level epoch-4 player, 45 moves    -> do spirals from weak play teach as much?
#   arm c = epoch-4 deaths, 200 searched moves (crisis + quiet)   -> do quiet-play labels add value?
# Plus escape benchmarks with a 200-move horizon: expert trouble (A1's deaths) and human trouble
# (epoch-4 deaths on held-out seeds). Every GPU job runs sequentially.
#   caffeinate -is scripts/offpolicy_experiment.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CPP=alphatrain/inference_cpp
D=alphatrain/data
PY=.venv/bin/python
EP4=data/scratch18b96_lr3e3_ckpts_epoch_4_ts.pt
A3=$D/ta_A2_e4_a2.0.pt
SEARCH="--c-puct 1.5 --q-weight 2.0 --virtual-mean --dirichlet-weight 0 --recovery-turns 15 --recovery-sims 600 --prevention-turns 30 --prevention-sims 600 --threads 14"
mkdir -p logs

echo "=== [1/5] confirm the A4 candidate on fresh seeds ($(date)) ==="
$PY -m alphatrain.scripts.eval_log run --model $D/ta_A3_e4_a1.0.pt --seed-start 3000000 --seed-end 3001000 \
    --desc "CONFIRMATION on fresh seeds (A3 there: P10 31,404 / 56.3% capped): A4 candidate = A3 + 1.0*(A3 fine-tuned 4ep on its crisis windows - A3)"

echo "=== [2/5] escape benchmarks, 200-move horizon ($(date)) ==="
awk '{$3 = 200; print}' $CPP/data/gen1_anchors.txt > $CPP/data/bench_expert_h200.txt
(cd $CPP && ./build/mcts_crisis --model $EP4 --device mps --seed-start 3900000 --seed-end 3901000 \
    --probe-batch 500 --continue-turns 200 --probe-only --anchors-out data/bench_human_h200.txt \
    --out-dir ../../data/bench_human_ep4) > logs/offp_bench_human.log 2>&1
grep -E 'phase 1 done|wrote' logs/offp_bench_human.log

escape() {  # escape <bench> <name:ts> ...  (greedy escape rates on one benchmark)
  local bench=$1; shift
  local runs=()
  for m in "$@"; do
    (cd $CPP && ./build/eval --model data/${m#*:} --device mps --batch 500 --anchors data/$bench.txt \
        --scores-out data/${bench}_${m%%:*}.csv > /dev/null)
    runs+=("${m%%:*}=$CPP/data/${bench}_${m%%:*}.csv")
  done
  $PY -m alphatrain.scripts.escape_benchmark --anchors $CPP/data/$bench.txt "${runs[@]}"
}
BASE_MODELS="A1:A1_pair2_orig_e40_ts.pt A2:ta_gen1b_e4_a2.5_ts.pt A3:ta_A2_e4_a2.0_ts.pt A4a1.0:ta_A3_e4_a1.0_ts.pt A4a2.0:ta_A3_e4_a2.0_ts.pt"
for B in bench_expert_h200 bench_human_h200; do echo "-- $B"; escape $B $BASE_MODELS; done

arm() {  # arm <tag> <probe seed> <probes> <searched moves>
  echo "=== arm $1: $3 epoch-4 probes from $2, $4 searched moves ($(date)) ==="
  (cd $CPP && ./build/mcts_crisis --run-id $1 --model data/A3_ts.pt --value-module data/pv_A3_ts.pt \
      --probe-model $EP4 --device mps --seed-start $2 --seed-end $(($2 + $3)) --continue-turns $4 \
      $SEARCH --out-dir ../../data/$1) > logs/$1.log 2>&1
  grep -E 'phase 1 done|^done:' logs/$1.log
  $PY -m alphatrain.scripts.build_expert_v2_tensor --games-dir data/$1 --policy-only-data \
      --output $D/$1.pt | grep -E 'boards:'
  $PY -m alphatrain.scripts.anomaly_checks --tensor $D/$1.pt --model $A3 --n 20000 | grep -E '^ *(B|C)\.'
  $PY -m alphatrain.train_path_b --tensor-file $D/$1.pt --resume $A3 --warm-start --freeze-bn \
      --policy-head pair2 --pair-dim 64 --legal-mask-loss --num-blocks 18 --channels 96 --epochs 4 \
      --batch-size 1024 --lr 1e-4 --warmup-epochs 0 --target-temperature 0.5 --augment-factor 1 \
      --seed 42 --save-dir checkpoints/$1_ft > logs/$1_ft.log 2>&1
  grep -E 'val:' logs/$1_ft.log | tail -1
  scripts/ta_sweep.sh $A3 checkpoints/$1_ft/epoch_4.pt ta_$1 \
      "off-policy arm $1: A3 + alpha*(A3 fine-tuned 4ep on epoch-4-death crises replayed by A3's search, $4 searched moves)" \
      1.0 2.0
}
echo "=== [3/5] arm b ==="; arm offp_b_ep4_w45 3950000 2160 45
echo "=== [4/5] arm c ==="; arm offp_c_ep4_w200 3960000 500 200

echo "=== [5/5] escape benchmarks for the arms ($(date)) ==="
ARM_MODELS="b1.0:ta_offp_b_ep4_w45_a1.0_ts.pt b2.0:ta_offp_b_ep4_w45_a2.0_ts.pt c1.0:ta_offp_c_ep4_w200_a1.0_ts.pt c2.0:ta_offp_c_ep4_w200_a2.0_ts.pt"
for B in bench_expert_h200 bench_human_h200; do echo "-- $B"; escape $B $ARM_MODELS; done
echo "=== gates ==="
grep -E '\| (ta_A3_e4_a|ta_offp_)' alphatrain/EVAL.md | awk -F'|' '{print $3, $5, "P1",$9,"P5",$10,"P10",$11,"P25",$12,"capped",$21}'
echo "=== done ($(date)) ==="
