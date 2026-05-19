#!/bin/bash
# Overnight q-sweep for pillar2z + value_head_v12_v12targets.
#
# Question: does the V12-targets head want a different q-weight than the
# V11-targets head's q=2.0? The earlier 100-seed A/B at q=2.0 had the
# V12-targets head looking -23% worse on MCTS mean, but q may need
# re-tuning per head.
#
# Phase 1: 100 sims x 100 seeds, q in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
# Phase 2: 200 sims x 100 seeds at Phase-1 winner q
#
# Total ETA: ~6 hours on M5 Max + MPS, 16 workers.
#
# Usage: ./scripts/value_head_qsweep_overnight.sh
# (caffeinate wrapped inside each python call)

set -uo pipefail

CANDIDATES=(0.5 1.0 1.5 2.0 2.5 3.0)
SEEDS=$(seq 0 99 | tr '\n' ' ')
MODEL=alphatrain/data/pillar2z_epoch_19.pt
HEAD=alphatrain/data/value_head_v12_v12targets.pt

STAMP=$(date +%Y%m%d_%H%M)
LOG_DIR=/tmp/eval_2z_v12tgt_overnight_${STAMP}
mkdir -p "$LOG_DIR"

REFS="
  pillar2y2 + v11_head + q=2.0 @ 100 sims, 100 seeds:
    mean=9397  P10=2057  P25=3207  P50=8862  %>=10K=47%
  pillar2z + v11-targets head + q=2.0 @ 100 sims, 100 seeds:
    mean=9138  P10=1934  P25=4512  P50=9394  %>=10K=50%"

echo "=========================================================="
echo "Overnight q-sweep: pillar2z + value_head_v12_v12targets"
echo "Started: $(date)"
echo "Phase 1 candidates (100 sims, 100 seeds, 12K cap): ${CANDIDATES[*]}"
echo "Reference baselines:$REFS"
echo "Log dir: $LOG_DIR"
echo "=========================================================="

run_one() {
  local qw=$1
  local sims=$2
  local out=$3
  local tag=$4
  echo
  echo "[$(date +%H:%M:%S)] $tag: q=$qw sims=$sims (100 seeds)..."
  caffeinate -dimsu python -m alphatrain.scripts.eval_parallel \
    --model "$MODEL" \
    --value-head-path "$HEAD" \
    --simulations "$sims" --top-k 30 --batch-size 8 \
    --device mps --workers 16 --max-turns 12000 \
    --q-weight "$qw" --mcts-only --early-stop \
    --seeds $SEEDS \
    > "$out" 2>&1 || {
      echo "  FAILED q=$qw sims=$sims (see $out)"
      return 1
    }
  local mean
  mean=$(grep "MEAN" "$out" | awk -F'|' '{print $2}' | tr -d ' ' | head -1)
  echo "[$(date +%H:%M:%S)] $tag: q=$qw sims=$sims -> mean=$mean"
  return 0
}

extract_mean() {
  grep "MEAN" "$1" | awk -F'|' '{print $2}' | tr -d ' ' | head -1
}

# Phase 1: q-sweep at 100 sims
echo
echo "=========================================================="
echo "PHASE 1: 100 sims, 100 seeds, q-sweep"
echo "=========================================================="

for qw in "${CANDIDATES[@]}"; do
  run_one "$qw" 100 "$LOG_DIR/v12tgt_q${qw}_100sim.log" "Phase1" || true
done

# Pick winner from Phase 1 by mean
echo
echo "=========================================================="
echo "PHASE 1 RESULTS"
echo "=========================================================="
best_qw=""
best_mean=0
for qw in "${CANDIDATES[@]}"; do
  log="$LOG_DIR/v12tgt_q${qw}_100sim.log"
  if [[ ! -f "$log" ]]; then continue; fi
  mean=$(extract_mean "$log")
  if [[ -z "$mean" ]]; then
    echo "  q=$qw: NO MEAN (run may have failed)"
    continue
  fi
  echo "  q=$qw 100sim: mean=$mean"
  grep -E "MCTS percentiles|<1000|>=10000" "$log" | head -2 | sed 's/^/    /'
  if (( mean > best_mean )); then
    best_mean=$mean
    best_qw=$qw
  fi
done

if [[ -z "$best_qw" ]]; then
  echo "ERROR: no Phase 1 candidate produced a usable result. Aborting."
  exit 1
fi

echo
echo "=========================================================="
echo "PHASE 1 WINNER: q=$best_qw (mean=$best_mean at 100 sims)"
echo "=========================================================="

# Phase 2: 200 sims at winner
echo
echo "PHASE 2: 200 sims, 100 seeds, q=$best_qw"
run_one "$best_qw" 200 "$LOG_DIR/v12tgt_q${best_qw}_200sim.log" "Phase2" || {
    echo "Phase 2 failed"
}

# Final summary
echo
echo "=========================================================="
echo "FINAL SUMMARY"
echo "Reference baselines:$REFS"
echo "=========================================================="

echo
echo "--- Phase 1 (100 sims) ---"
for qw in "${CANDIDATES[@]}"; do
  log="$LOG_DIR/v12tgt_q${qw}_100sim.log"
  [[ -f "$log" ]] || continue
  echo
  echo "  v12tgt q=$qw 100sim:"
  grep -E "MEAN|MCTS percentiles|<1000|>=10000|MCTS done" "$log" | head -4
done

echo
echo "--- Phase 2 (200 sims, winner q=$best_qw) ---"
log="$LOG_DIR/v12tgt_q${best_qw}_200sim.log"
[[ -f "$log" ]] && grep -E "MEAN|MCTS percentiles|<1000|>=10000|MCTS done" "$log" | head -4

echo
echo "All logs at: $LOG_DIR"
echo "Finished: $(date)"
