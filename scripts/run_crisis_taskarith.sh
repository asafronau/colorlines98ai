#!/bin/bash
# Task-arithmetic crisis correction of pillar3f — the PROVEN local recipe that made
# pillar3f itself (HISTORY 167-168: pillar3f = pillar3b + 0.5*decisive_vector).
#
# Stage 1: fine-tune pillar3f ONLY on the catastrophe-corrections corpus (frozen BN) -> task vector
# Stage 2: merge  pillar3f + alpha*(ft - pillar3f), sweeping alpha
# Gate A : normal_play_drift.py  — confirm NO normal-play forgetting (healthy bin ~unchanged)
# Gate B : eval_policy floor sweep — mean+median+ceiling hold AND floor (P10/<1000) rises
#
# All local on M5 (train ~min/epoch, merge ~seconds, eval ~slow). Logs tee'd to logs/taskarith/.
#
#   bash scripts/run_crisis_taskarith.sh
#
# Tune via env: BASE, CORPUS, FT_EPOCHS, FT_LR, ALPHAS, EVAL_LO/EVAL_HI, DEVICE.
set -e
cd "$(dirname "$0")/.."

BASE=${BASE:-alphatrain/data/pillar3f.pt}
CORPUS=${CORPUS:-alphatrain/data/catastrophe_corrections_pillar3f_h2.pt}
FT_EPOCHS=${FT_EPOCHS:-15}    # ta15 = ft_epoch_15 is the proven deployable vector (HISTORY 167)
FT_LR=${FT_LR:-1e-4}          # proven recipe (HISTORY 167): soft-CE T0.5, weighted, lr 1e-4, frozen BN
FT_TT=${FT_TT:-0.5}
ALPHAS=${ALPHAS:-"0.2 0.4 0.5 0.7"}   # proven plateau α∈[0.4,0.7]; 0.2 as a low-dose check
EVAL_LO=${EVAL_LO:-775000}    # proven eval band; 1k for the cheap sweep, 775000-779999 (5k) to confirm
EVAL_HI=${EVAL_HI:-775999}
DEVICE=${DEVICE:-mps}
OUT=${OUT:-logs/taskarith}
FT_DIR=$OUT/crisis_ft
mkdir -p "$OUT"

echo "=== Stage 1: fine-tune $BASE on $CORPUS (frozen BN) ==="
PYTHONPATH=. python3 scripts/train_crisis_ft.py \
    --corpus "$CORPUS" --base "$BASE" \
    --epochs "$FT_EPOCHS" --lr "$FT_LR" --batch 1024 \
    --target-temperature "$FT_TT" --weighted \
    --device "$DEVICE" --save-dir "$FT_DIR" 2>&1 | tee "$OUT/ft_train.log"

FT=$FT_DIR/ft_epoch_${FT_EPOCHS}.pt
[ -f "$FT" ] || FT=$(ls -t "$FT_DIR"/ft_epoch_*.pt | head -1)
echo "fine-tune checkpoint: $FT"

echo "=== Stage 2: merge alpha-sweep + gates ($BASE + alpha*(ft-base)) ==="
echo "  baseline (alpha=0 = $BASE) first, then each alpha:"
for A in 0 $ALPHAS; do
    if [ "$A" = "0" ]; then
        MODEL=$BASE; TAG=base
    else
        MODEL=$OUT/ta_a$A.pt; TAG=a$A
        PYTHONPATH=. python3 scripts/merge_checkpoints.py \
            --base "$BASE" --crisis "$FT" --alpha "$A" --out "$MODEL"
    fi
    echo "--- [$TAG] floor eval seeds $EVAL_LO-$EVAL_HI ---"
    PYTHONPATH=. python3 -m scripts.eval_policy --model "$MODEL" \
        --seed-start "$EVAL_LO" --seed-end "$EVAL_HI" \
        --device "$DEVICE" --batch 256 2>&1 | tee "$OUT/eval_$TAG.log"
    if [ "$A" != "0" ]; then
        echo "--- [$TAG] GATE A: normal-play drift vs base ---"
        PYTHONPATH=. python3 scripts/normal_play_drift.py \
            --base "$BASE" --corrected "$MODEL" --n 400 \
            --device "$DEVICE" --out "$OUT/drift_$TAG.json" 2>&1 | tee "$OUT/drift_$TAG.log"
    fi
done

echo
echo "=== DONE. Read $OUT/eval_*.log (floor) + $OUT/drift_*.log (forgetting) ==="
echo "Pick the alpha that raises the floor (P10/<1000) while: (B) mean+median+ceiling hold,"
echo "(A) healthy-bin top1-agree stays ~1.0 (no normal-play forgetting). That's the new policy."
