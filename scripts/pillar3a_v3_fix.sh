#!/bin/bash
# Pillar 3a-v3 FIX experiment: retrain on cap_rate-only stable pairs with
# val_acc checkpoint selection and terminal-V calibration. Then q-sweep.
#
# Diagnoses confirmed in previous run:
#   - 84.4% inter-half label agreement at scale (labels real)
#   - 73.9% val acc on cap_rate pairs vs 54% on turns (turns are noise)
#   - Head outputs ~[-1, +1] mean ≈ 0 → terminal V=0 is mid-distribution
#     → MCTS pulled toward death (q-sweep monotonically worse)
#
# This script applies the three fixes and re-runs the q-sweep.

set -uo pipefail
cd /Users/andreis/local/source/colorlines98
source .venv/bin/activate

caffeinate -dimsu &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null || true" EXIT INT TERM

# Reuse the existing pairwise dataset from production run.
PAIRWISE=alphatrain/data/pairwise_v3_20260514_1754.pt
MODEL=alphatrain/data/pillar2z_epoch_19.pt

STAMP=$(date +%Y%m%d_%H%M)
WORK_DIR=/tmp/pillar3a_v3_fix_${STAMP}
mkdir -p "$WORK_DIR"
exec > >(tee "$WORK_DIR/main.log") 2>&1

HEAD_OUT=alphatrain/data/value_head_v3_fix_${STAMP}.pt
WORKERS=16
BATCH=8
Q_VALUES=(0.25 0.5 1.0 1.5 2.0)
EVAL_SEEDS=$(seq 0 99 | tr '\n' ' ')

echo "=========================================================="
echo "Pillar 3a-v3 FIX: cap_rate-only + val_acc + terminal calibration"
echo "Started: $(date)"
echo "Pairwise: $PAIRWISE"
echo "Output head: $HEAD_OUT"
echo "Q sweep: ${Q_VALUES[*]}"
echo "Log dir: $WORK_DIR"
echo "=========================================================="

echo
echo "[$(date +%H:%M:%S)] TRAIN cap_rate-only + val_acc selection + terminal calibration..."
caffeinate -dimsu python -m alphatrain.scripts.train_ranking_head \
  --backbone "$MODEL" \
  --pairwise "$PAIRWISE" \
  --epochs 40 --batch-size 256 --lr 5e-4 \
  --weight-decay 1e-3 \
  --unweighted \
  --metric-filter cap_rate \
  --select-by val_acc \
  --calibrate-terminal --terminal-margin 0.1 \
  --device mps \
  --out "$HEAD_OUT" \
  2>&1 | tee "$WORK_DIR/train.log" \
  || { echo "TRAIN FAILED"; exit 1; }

echo
echo "[$(date +%H:%M:%S)] TRAIN done."

# Run analyzer for diagnostics on calibrated head.
echo
echo "[$(date +%H:%M:%S)] ANALYZE calibrated head..."
caffeinate -dimsu python -m alphatrain.scripts.analyze_pairwise_head \
  --backbone "$MODEL" \
  --pairwise "$PAIRWISE" \
  --head "$HEAD_OUT" \
  --device mps \
  2>&1 | tee "$WORK_DIR/analyze.log" \
  || echo "ANALYZE FAILED (continuing)"

echo
echo "=========================================================="
echo "Q-SWEEP vs baseline (v11-targets@q=2.0 mean=9138)"
echo "  + pure-prior mean=7045 (from previous run)"
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
echo "FIX RUN COMPLETE: $(date)"
echo "=========================================================="
echo
echo "BASELINES:"
echo "  v11-targets @ q=2.0:  mean=9138  %>=10K=50%"
echo "  pure-prior  @ q=0.0:  mean=7045  %>=10K=20%"
echo "  new head (bug)        best q=0.5  mean=6469"
echo
echo "FIX Q-SWEEP:"
for QW in "${Q_VALUES[@]}"; do
  log="$WORK_DIR/eval_q${QW}.log"
  [[ -f "$log" ]] || continue
  mean=$(grep "MEAN" "$log" | awk -F'|' '{print $2}' | tr -d ' ' | head -1)
  echo
  echo "--- q=$QW (mean=$mean) ---"
  grep -E "MCTS percentiles|<1000|>=10000" "$log" | head -2 | sed 's/^/    /'
done
echo
echo "All logs at: $WORK_DIR/"
