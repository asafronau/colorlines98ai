#!/bin/bash
# Pillar 3a-v2 — overnight: separability mine → pairwise dataset → train spatial
# ranking head → q-sweep A/B vs v11-targets baseline.
#
# Total ETA: ~7-9 hours on M5 MAX (18 P-cores, 16-worker server-mode).
#
# Pipeline:
#   Stage 1 (separability miner):   ~2.5h at 20K anchors
#   Stage 2 (pairwise labels):      ~2.5h on ~7K separable anchors
#   Spatial ranking head training:  ~5-15min on ~7K pairs
#   Q-sweep A/B (6 q values):       ~75min (6 × ~12.5 min)
#
# Reference baseline (pillar2z + v11-targets head + q=2.0, 100 sims, 100 seeds):
#   mean=9138 P10=1934 P25=4512 P50=9394 %>=10K=50%
#
# Usage: ./scripts/pillar3a_v2_overnight.sh

set -uo pipefail
cd /Users/andreis/local/source/colorlines98
source .venv/bin/activate

# Master caffeinate: keep the entire pipeline awake (display, disk, system,
# user idle, idle sleep). Killed automatically when this script exits.
caffeinate -dimsu &
CAFFEINATE_PID=$!
trap "kill $CAFFEINATE_PID 2>/dev/null || true" EXIT INT TERM
echo "Master caffeinate started (PID $CAFFEINATE_PID)"

# ── Knobs ───────────────────────────────────────────────────────────────
NUM_ANCHORS=20000        # Stage 1 candidate count
WORKERS=16                # M5 has 18 P-cores; 16 workers + 1 GPU + 1 master
BATCH=8
MODEL=alphatrain/data/pillar2z_epoch_19.pt
Q_VALUES=(0.25 0.5 1.0 1.5 2.0 3.0)
EVAL_SEEDS=$(seq 0 99 | tr '\n' ' ')
# ────────────────────────────────────────────────────────────────────────

STAMP=$(date +%Y%m%d_%H%M)
WORK_DIR=/tmp/pillar3a_v2_${STAMP}
mkdir -p "$WORK_DIR"
MAIN_LOG=$WORK_DIR/main.log
exec > >(tee "$MAIN_LOG") 2>&1

S1_OUT=alphatrain/data/anchors_stage1_${NUM_ANCHORS}.pt
S2_OUT=alphatrain/data/pairwise_v12_${NUM_ANCHORS}.pt
HEAD_OUT=alphatrain/data/value_head_spatial.pt

echo "=========================================================="
echo "Pillar 3a-v2 overnight run"
echo "Started: $(date)"
echo "Anchors: $NUM_ANCHORS"
echo "Workers: $WORKERS"
echo "Q sweep: ${Q_VALUES[*]}"
echo "Log dir: $WORK_DIR"
echo "=========================================================="

echo
echo "[$(date +%H:%M:%S)] STAGE 1: separability miner ($NUM_ANCHORS anchors)..."
caffeinate -dimsu python -m alphatrain.scripts.mine_separable_anchors \
  --model "$MODEL" \
  --crisis-dir data/crisis_v12 \
  --selfplay-dir data/selfplay_v12 \
  --num-anchors $NUM_ANCHORS \
  --crisis-frac 0.70 --selfplay-frac 0.10 \
  --top-moves 4 --k-rollouts 4 --horizon 150 \
  --workers $WORKERS --batch-size $BATCH \
  --output "$S1_OUT" \
  2>&1 | tee "$WORK_DIR/stage1.log" \
  || { echo "STAGE 1 FAILED"; exit 1; }

# Quick separability gate check (script also reports PASS/FAIL itself)
SEP_FRAC=$(grep -oE "PASS: [0-9.]+%" "$WORK_DIR/stage1.log" | head -1)
if [[ -z "$SEP_FRAC" ]]; then
  echo "STAGE 1 separability did not PASS gate. Aborting."
  exit 1
fi
echo "[$(date +%H:%M:%S)] STAGE 1 done — gate: $SEP_FRAC"

echo
echo "[$(date +%H:%M:%S)] STAGE 2: full pairwise labels (K=8, H=300)..."
caffeinate -dimsu python -m alphatrain.scripts.build_pairwise_dataset \
  --model "$MODEL" \
  --stage1 "$S1_OUT" \
  --top-moves 4 --k-rollouts 8 --horizon 300 \
  --workers $WORKERS --batch-size $BATCH \
  --cap-rate-threshold 0.375 --turns-threshold 50 --score-threshold 300 \
  --output "$S2_OUT" \
  2>&1 | tee "$WORK_DIR/stage2.log" \
  || { echo "STAGE 2 FAILED"; exit 1; }

# Sanity: count pairs
PAIRS=$(grep -oE "Accepted \(Stage 2 tight\): [0-9]+" "$WORK_DIR/stage2.log" \
        | grep -oE "[0-9]+$" | head -1)
echo "[$(date +%H:%M:%S)] STAGE 2 done — accepted pairs: $PAIRS"
if [[ -z "$PAIRS" || "$PAIRS" -lt 500 ]]; then
  echo "WARNING: fewer than 500 pairs ($PAIRS). Training is risky but proceeding."
fi

echo
echo "[$(date +%H:%M:%S)] TRAIN SpatialValueHead..."
caffeinate -dimsu python -m alphatrain.scripts.train_ranking_head \
  --backbone "$MODEL" \
  --pairwise "$S2_OUT" \
  --epochs 15 --batch-size 512 --lr 1e-3 \
  --device mps \
  --out "$HEAD_OUT" \
  2>&1 | tee "$WORK_DIR/train.log" \
  || { echo "TRAIN FAILED"; exit 1; }

echo "[$(date +%H:%M:%S)] TRAIN done."

echo
echo "=========================================================="
echo "Q-SWEEP A/B vs v11-targets baseline"
echo "Baseline: mean=9138 P10=1934 P25=4512 P50=9394 %>=10K=50%"
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

  echo "--- q=$QW result ---"
  grep -E "MEAN|MCTS percentiles|<1000|>=10000" "$WORK_DIR/eval_q${QW}.log" \
    | head -3
done

echo
echo "=========================================================="
echo "OVERNIGHT RUN COMPLETE: $(date)"
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
echo "All logs at: $WORK_DIR/"
