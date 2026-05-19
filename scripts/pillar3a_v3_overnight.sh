#!/bin/bash
# Pillar 3a-v3 PRODUCTION — overnight: stable-label pairwise (all 7,162 separable
# anchors from Stage 1) + train SpatialValueHead + q-sweep A/B.
#
# Builds on pilot result: 88.8% inter-half agreement → labels are stable when
# both halves of K=16 continuations pass threshold. Scaling from 200 → 7,162
# anchors should yield ~2,750 stable pairs.
#
# Pipeline:
#   Mine (K=16 split-half stable filter):  ~5h on M5 MAX
#   Train SpatialValueHead:                ~1-2 min
#   Q-sweep (5 q-values × 100 seeds):      ~60 min
# Total: ~6-7h.
#
# Reference baseline (pillar2z + v11-targets head + q=2.0, 100 sims, 100 seeds):
#   mean=9138 P10=1934 P25=4512 P50=9394 %>=10K=50%
#
# Usage: ./scripts/pillar3a_v3_overnight.sh

set -uo pipefail
cd /Users/andreis/local/source/colorlines98
source .venv/bin/activate

caffeinate -dimsu &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null || true" EXIT INT TERM
echo "Master caffeinate started (PID $CAFFEINATE_PID)"

# ── Knobs ───────────────────────────────────────────────────────────────
NUM_ANCHORS=7200              # all 7,162 separable from Stage 1 (cap+small slack)
WORKERS=16
BATCH=8
MODEL=alphatrain/data/pillar2z_epoch_19.pt
STAGE1=alphatrain/data/anchors_stage1_20000.pt
Q_VALUES=(0.5 1.0 1.5 2.0 3.0)
EVAL_SEEDS=$(seq 0 99 | tr '\n' ' ')
# ────────────────────────────────────────────────────────────────────────

STAMP=$(date +%Y%m%d_%H%M)
WORK_DIR=/tmp/pillar3a_v3_${STAMP}
mkdir -p "$WORK_DIR"
MAIN_LOG=$WORK_DIR/main.log
exec > >(tee "$MAIN_LOG") 2>&1

PAIRWISE_OUT=alphatrain/data/pairwise_v3_${STAMP}.pt
HEAD_OUT=alphatrain/data/value_head_v3_${STAMP}.pt

echo "=========================================================="
echo "Pillar 3a-v3 PRODUCTION"
echo "Started: $(date)"
echo "Anchors: $NUM_ANCHORS"
echo "Workers: $WORKERS"
echo "Q sweep: ${Q_VALUES[*]}"
echo "Log dir: $WORK_DIR"
echo "Outputs: $PAIRWISE_OUT, $HEAD_OUT"
echo "=========================================================="

echo
echo "[$(date +%H:%M:%S)] MINE stable pairs (K=16 split-half filter)..."
caffeinate -dimsu python -m alphatrain.scripts.pilot_clean_pairwise \
  --model "$MODEL" \
  --stage1 "$STAGE1" \
  --num-anchors $NUM_ANCHORS \
  --top-moves 4 --k-continuations 16 --horizon 200 \
  --workers $WORKERS --batch-size $BATCH \
  --output "$PAIRWISE_OUT" \
  2>&1 | tee "$WORK_DIR/mine.log" \
  || { echo "MINE FAILED"; exit 1; }

STABLE=$(grep -oE "STABLE \(.+\): [0-9]+" "$WORK_DIR/mine.log" | head -1 \
         | grep -oE "[0-9]+$" || echo 0)
echo
echo "[$(date +%H:%M:%S)] MINE done — $STABLE stable pairs"

if [[ "$STABLE" -lt 200 ]]; then
  echo "FAIL: only $STABLE stable pairs (<200). Stopping before training."
  exit 1
fi

echo
echo "[$(date +%H:%M:%S)] TRAIN unweighted BPR + weight decay..."
caffeinate -dimsu python -m alphatrain.scripts.train_ranking_head \
  --backbone "$MODEL" \
  --pairwise "$PAIRWISE_OUT" \
  --epochs 40 --batch-size 256 --lr 5e-4 \
  --weight-decay 1e-3 \
  --unweighted \
  --device mps \
  --out "$HEAD_OUT" \
  2>&1 | tee "$WORK_DIR/train.log" \
  || { echo "TRAIN FAILED"; exit 1; }

BEST_VAL=$(grep -oE "Best val_loss=[0-9.]+" "$WORK_DIR/train.log" | head -1)
echo "[$(date +%H:%M:%S)] TRAIN done — $BEST_VAL"

echo
echo "=========================================================="
echo "Q-SWEEP A/B vs v11-targets baseline"
echo "Baseline: mean=9138 P10=1934 P25=4512 P50=9394 %>=10K=50% (q=2.0, 100 sims)"
echo "=========================================================="

for QW in "${Q_VALUES[@]}"; do
  echo
  echo "[$(date +%H:%M:%S)] eval q=$QW ..."
  caffeinate -dimsu python -m alphatrain.scripts.eval_parallel \
    --model "$MODEL" \
    --value-head-path "$HEAD_OUT" \
    --simulations 100 --top-k 30 --batch-size $BATCH \
    --device mps --workers $WORKERS --max-turns 12000 \
    --q-weight "$QW" --mcts-only --early-stop \
    --seeds $EVAL_SEEDS \
    > "$WORK_DIR/eval_q${QW}.log" 2>&1 \
    || { echo "EVAL q=$QW FAILED (continuing sweep)"; continue; }
  echo "--- q=$QW ---"
  grep -E "MEAN|MCTS percentiles|<1000|>=10000" "$WORK_DIR/eval_q${QW}.log" | head -3
done

echo
echo "=========================================================="
echo "OVERNIGHT COMPLETE: $(date)"
echo "=========================================================="
echo
echo "BASELINE (pillar2z + v11-targets head + q=2.0):"
echo "  mean=9138 P10=1934 P25=4512 P50=9394 %>=10K=50%"
echo
echo "Q-SWEEP SUMMARY:"
for QW in "${Q_VALUES[@]}"; do
  log="$WORK_DIR/eval_q${QW}.log"
  [[ -f "$log" ]] || continue
  mean=$(grep "MEAN" "$log" | awk -F'|' '{print $2}' | tr -d ' ' | head -1)
  echo
  echo "--- q=$QW (mean=$mean) ---"
  grep -E "MCTS percentiles|<1000|>=10000" "$log" | head -2 | sed 's/^/    /'
done
echo
echo "Mining audit:"
grep -E "Pass FULL|Pass BOTH|winner agreement|STABLE \(|Stable-label" \
  "$WORK_DIR/mine.log" | head -10
echo
echo "Training trajectory (best 3 epochs by val_loss):"
grep -E "^Epoch " "$WORK_DIR/train.log" | sort -k7 | head -3
echo
echo "All logs at: $WORK_DIR/"
