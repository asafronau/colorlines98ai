#!/bin/bash
# Overnight q_weight refinement for pillar2z + value_head_v12.
#
# Phase 1: 200 sims x 50 seeds, --max-turns 10000  for q in {1.0, 1.5, 2.0}
# Phase 2: 400 sims x 50 seeds, --max-turns 10000  at the Phase-1 winner
# (max-turns 10000 to loosen the cap that pinned previous evals at ~16,400.)
#
# Total ETA: ~2-3 hours on M5 Max + MPS, 16 workers.
#
# Usage: ./scripts/value_head_qsweep_overnight.sh
# (caffeinate is wrapped inside, no need to wrap externally)

set -uo pipefail

CANDIDATES=(1.0 1.5 2.0)
SEEDS=$(seq 0 49 | tr '\n' ' ')
MODEL=alphatrain/data/pillar2z_epoch_19.pt
HEAD=alphatrain/data/value_head_v12.pt

STAMP=$(date +%Y%m%d_%H%M)
LOG_DIR=/tmp/eval_2z_overnight_${STAMP}
mkdir -p "$LOG_DIR"

PILLAR2Y2_REF="(pillar2y2 + v11_head + q=2.0 @ 400 sims, 8K cap: mean=13,964 P10=5,397 P50=16,440 %>=10K=78%)"

echo "=========================================================="
echo "Overnight q_weight sweep for pillar2z_epoch_19 + value_head_v12"
echo "Started: $(date)"
echo "Candidates Phase 1 (200 sims x 50 seeds, 10K cap): ${CANDIDATES[*]}"
echo "Reference: $PILLAR2Y2_REF"
echo "Log dir: $LOG_DIR"
echo "=========================================================="

run_one() {
  local qw=$1
  local sims=$2
  local max_turns=$3
  local out=$4
  echo
  echo "[$(date +%H:%M:%S)] Running q=$qw sims=$sims max_turns=$max_turns (50 seeds)..."
  caffeinate -dimsu python -m alphatrain.scripts.eval_parallel \
    --model "$MODEL" \
    --value-head-path "$HEAD" \
    --simulations "$sims" --top-k 30 --batch-size 8 \
    --device mps --workers 16 --max-turns "$max_turns" \
    --q-weight "$qw" --mcts-only --early-stop \
    --seeds $SEEDS \
    > "$out" 2>&1 || {
      echo "  FAILED q=$qw sims=$sims (see $out)"
      return 1
    }
  local mean
  mean=$(grep "MEAN" "$out" | awk -F'|' '{print $2}' | tr -d ' ' | head -1)
  echo "[$(date +%H:%M:%S)] q=$qw sims=$sims -> mean=$mean"
  return 0
}

extract_mean() {
  grep "MEAN" "$1" | awk -F'|' '{print $2}' | tr -d ' ' | head -1
}

# Phase 1: 200 sims x 50 seeds across candidates
echo
echo "=========================================================="
echo "PHASE 1: 200 sims x 50 seeds x ${#CANDIDATES[@]} q_weight values"
echo "=========================================================="

for qw in "${CANDIDATES[@]}"; do
  run_one "$qw" 200 10000 "$LOG_DIR/2z_q${qw}_200sim.log" || true
done

# Pick winner by mean
echo
echo "=========================================================="
echo "PHASE 1 RESULTS"
echo "=========================================================="
best_qw=""
best_mean=0
for qw in "${CANDIDATES[@]}"; do
  log="$LOG_DIR/2z_q${qw}_200sim.log"
  if [[ ! -f "$log" ]]; then continue; fi
  mean=$(extract_mean "$log")
  if [[ -z "$mean" ]]; then
    echo "  q=$qw: NO MEAN (run may have failed)"
    continue
  fi
  echo "  q=$qw 200sim: mean=$mean"
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
echo "WINNER: q=$best_qw (mean=$best_mean at 200 sims)"
echo "=========================================================="

# Phase 2: 400 sims x 50 seeds at winner
echo
echo "PHASE 2: 400 sims x 50 seeds at q=$best_qw"
run_one "$best_qw" 400 10000 "$LOG_DIR/2z_q${best_qw}_400sim.log" || {
  echo "Phase 2 failed"; exit 1
}

# Final summary
echo
echo "=========================================================="
echo "FINAL SUMMARY"
echo "Reference: $PILLAR2Y2_REF"
echo "=========================================================="
for qw in "${CANDIDATES[@]}"; do
  log="$LOG_DIR/2z_q${qw}_200sim.log"
  [[ -f "$log" ]] || continue
  echo
  echo "--- 2z q=$qw 200sim ---"
  grep -E "MEAN|MCTS percentiles|<1000|>=10000|MCTS done" "$log" | head -4
done

final_log="$LOG_DIR/2z_q${best_qw}_400sim.log"
echo
echo "--- 2z q=$best_qw 400sim (FINAL, comparable to pillar2y2 baseline) ---"
grep -E "MEAN|MCTS percentiles|<1000|>=10000|MCTS done" "$final_log" | head -4

echo
echo "All logs at: $LOG_DIR"
echo "Finished: $(date)"
