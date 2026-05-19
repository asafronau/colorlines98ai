#!/bin/bash
# Pillar 3a-v3 pilot: stable-label pairwise mining + unweighted BPR training.
# ETA ~10-15 min on M5 MAX.

set -uo pipefail
cd /Users/andreis/local/source/colorlines98
source .venv/bin/activate

STAMP=$(date +%Y%m%d_%H%M)
WORK_DIR=/tmp/pilot_clean_${STAMP}
mkdir -p "$WORK_DIR"
MAIN_LOG=$WORK_DIR/main.log
exec > >(tee "$MAIN_LOG") 2>&1

MODEL=alphatrain/data/pillar2z_epoch_19.pt
STAGE1=alphatrain/data/anchors_stage1_20000.pt
PAIRWISE_OUT=alphatrain/data/pairwise_pilot.pt
HEAD_OUT=alphatrain/data/value_head_pilot.pt

echo "=========================================================="
echo "Pillar 3a-v3 PILOT: stable-label pairwise"
echo "Started: $(date)"
echo "Work dir: $WORK_DIR"
echo "=========================================================="

echo
echo "[$(date +%H:%M:%S)] MINE clean pairwise (200 anchors, K=16 split into halves)..."
python -m alphatrain.scripts.pilot_clean_pairwise \
  --model "$MODEL" \
  --stage1 "$STAGE1" \
  --num-anchors 200 \
  --top-moves 4 --k-continuations 16 --horizon 200 \
  --workers 16 --batch-size 8 \
  --output "$PAIRWISE_OUT" \
  2>&1 | tee "$WORK_DIR/mine.log" \
  || { echo "MINE FAILED"; exit 1; }

# Read stable count from log
STABLE=$(grep -oE "STABLE \(.+\): [0-9]+" "$WORK_DIR/mine.log" | head -1 \
         | grep -oE "[0-9]+$" || echo 0)
echo
echo "[$(date +%H:%M:%S)] MINE done — $STABLE stable pairs"

if [[ "$STABLE" -lt 50 ]]; then
  echo "Too few stable pairs ($STABLE). Reporting and stopping."
  echo "Diagnosis below in mine.log"
  exit 0
fi

echo
echo "[$(date +%H:%M:%S)] TRAIN unweighted BPR..."
python -m alphatrain.scripts.train_ranking_head \
  --backbone "$MODEL" \
  --pairwise "$PAIRWISE_OUT" \
  --epochs 25 --batch-size 256 --lr 1e-3 \
  --unweighted \
  --device mps \
  --out "$HEAD_OUT" \
  2>&1 | tee "$WORK_DIR/train.log" \
  || { echo "TRAIN FAILED"; exit 1; }

echo
echo "=========================================================="
echo "PILOT COMPLETE: $(date)"
echo "=========================================================="
echo "Logs at: $WORK_DIR/"
echo
echo "Key numbers:"
grep -E "Pass FULL|Pass BOTH|winner agreement|STABLE \(|Stable-label" \
  "$WORK_DIR/mine.log" | head -10
echo
echo "Training trajectory (last 5 epochs):"
grep -E "^Epoch " "$WORK_DIR/train.log" | tail -5
