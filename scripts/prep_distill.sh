#!/bin/bash
# Distillation corpus for a smaller student (HISTORY 258). Three state sources, ALL labeled with the teacher's
# 8-symmetry-averaged policy (top-5, softmax over the 5; build_tta_corpus / relabel_tta_tensor):
#   1. the teacher's own games: 2,000 at cap 100k, every 25th move + the last 300 (its death spirals)
#   2. a human-level player's games (epoch 4, mean ~2.4k): 1,000 games, every move
#   3. the states of the fixed-teacher crisis corpora (spirals from several actors and from epoch-4 deaths)
# GPU steps run sequentially.   caffeinate -is scripts/prep_distill.sh <teacher.pt> <tag>
set -euo pipefail
cd "$(dirname "$0")/.."
TEACHER=$1; TAG=$2
CPP=alphatrain/inference_cpp; D=alphatrain/data; PY=.venv/bin/python
EP4=data/scratch18b96_lr3e3_ckpts_epoch_4_ts.pt
NAME=$(basename "$TEACHER" .pt)
CRISIS="gen1b_crisis_A1_w45 A2_crisis_w45 A3_crisis_w45 offp_b_ep4_w45 offp_c_ep4_w200"
mkdir -p logs $D/distill_${TAG}_teacher_games $D/distill_${TAG}_ep4_games

echo "=== [1/5] teacher games ($NAME): 2,000 at cap 100k, every 25th move + last 300 ($(date)) ==="
[ -f $CPP/data/${NAME}_fold_ts.pt ] || $PY -m alphatrain.inference_cpp.export_ts --model "$TEACHER" \
    --output $CPP/data/${NAME}_fold_ts.pt --fold-bn > logs/distill_${TAG}_export.log 2>&1
(cd $CPP && ./build/eval --model data/${NAME}_fold_ts.pt --device mps --batch 2000 --seed-start 4300000 \
    --seed-end 4302000 --max-turns 100000 --record-dir ../../$D/distill_${TAG}_teacher_games --record-every 25 \
    --record-tail 300 --scores-out data/distill_${TAG}_teacher_4300000_2000.csv) > logs/distill_${TAG}_teacher_games.log 2>&1
tail -n 4 logs/distill_${TAG}_teacher_games.log

echo "=== [2/5] epoch-4 games: 1,000, every move ($(date)) ==="
(cd $CPP && ./build/eval --model $EP4 --device mps --batch 1000 --seed-start 4400000 --seed-end 4401000 \
    --record-dir ../../$D/distill_${TAG}_ep4_games --record-every 1 --record-tail 300 \
    --scores-out data/distill_${TAG}_ep4_4400000_1000.csv) > logs/distill_${TAG}_ep4_games.log 2>&1
tail -n 3 logs/distill_${TAG}_ep4_games.log

echo "=== [3/5] label recorded states with the teacher's 8-view average ($(date)) ==="
$PY -m alphatrain.scripts.build_tta_corpus --games-dir $D/distill_${TAG}_teacher_games --base "$TEACHER" \
    --out $D/distill_${TAG}_teacher.pt --holdout-games 40 --holdout-out $D/distill_${TAG}_teacher_holdout.clrj \
    > logs/distill_${TAG}_label_teacher.log 2>&1
tail -n 2 logs/distill_${TAG}_label_teacher.log
$PY -m alphatrain.scripts.build_tta_corpus --games-dir $D/distill_${TAG}_ep4_games --base "$TEACHER" \
    --out $D/distill_${TAG}_ep4.pt --holdout-games 20 --holdout-out $D/distill_${TAG}_ep4_holdout.clrj \
    > logs/distill_${TAG}_label_ep4.log 2>&1
tail -n 2 logs/distill_${TAG}_label_ep4.log

echo "=== [4/5] relabel the crisis corpora with the teacher ($(date)) ==="
PARTS=("$D/distill_${TAG}_teacher.pt" "$D/distill_${TAG}_ep4.pt")
for c in $CRISIS; do
  $PY -m alphatrain.scripts.relabel_tta_tensor --tensor $D/$c.pt --model "$TEACHER" \
      --out $D/distill_${TAG}_$c.npz > logs/distill_${TAG}_relabel_$c.log 2>&1
  $PY -m alphatrain.scripts.build_tta_r2_corpus --src $D/$c.pt --sidecar $D/distill_${TAG}_$c.npz --mode tta \
      --out $D/distill_${TAG}_$c.pt | tail -n 1
  PARTS+=("$D/distill_${TAG}_$c.pt")
done

echo "=== [5/5] merge ($(date)) ==="
$PY -m alphatrain.scripts.merge_policy_tensors --out $D/distill_${TAG}.pt "${PARTS[@]}"
gzip -k -1 -f $D/distill_${TAG}.pt
ls -la $D/distill_${TAG}.pt $D/distill_${TAG}.pt.gz
echo "=== done ($(date)) ==="
