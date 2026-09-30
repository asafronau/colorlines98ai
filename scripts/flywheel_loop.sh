#!/bin/bash
# Autonomous flywheel (owner, 2026-09-29): repeated scripts/flywheel_turn.sh turns from the current actor.
# After each turn the alpha with the lowest 2k gate death rate (alphatrain/scripts/best_gate.py) is promoted
# if it beats the actor; when the largest swept alpha wins, one alpha 1.0 higher is gated first.
# Stops when the promoted rate reaches TARGET (default 0.52 deaths per 100k turns = the upper end of the
# 18x96 union teacher's 95% CI) or a turn cuts the rate by less than MIN_GAIN (default 0.15).
#
#   THREADS=64 caffeinate -is scripts/flywheel_loop.sh <actor.pt> <actor_rate> <prefix> <k> <record_seed> <mine_seed>
#   e.g. scripts/flywheel_loop.sh alphatrain/data/scratch9b64_pair2_distill_union_soft_ckpts_epoch_35.pt 1.52 S 1 4500000 4600000
# Turn k is tagged <prefix><k>; each later turn moves both seed banks up by STRIDE (default 500,000).
# A stopped loop resumes by rerunning it with the current actor, rate, k and seeds (flywheel_turn.sh skips
# finished steps and mcts_crisis resumes per seed).
set -euo pipefail
cd "$(dirname "$0")/.."
ACTOR=$1; RATE=$2; PREFIX=$3; K=$4; RSEED=$5; MSEED=$6
TARGET=${TARGET:-0.52}; MIN_GAIN=${MIN_GAIN:-0.15}; STRIDE=${STRIDE:-500000}
ALPHAS=${ALPHAS:-"1.0 2.0 3.0"}; SIMS=${SIMS:-600}; export SIMS
PY=.venv/bin/python
lt() { awk -v a="$1" -v b="$2" 'BEGIN { exit !(a < b) }'; }

while true; do
  TAG=${PREFIX}${K}
  echo "##### turn $TAG: actor $ACTOR ($RATE per 100k turns), seeds $RSEED / $MSEED ($(date))"
  scripts/flywheel_turn.sh "$ACTOR" "$TAG" "$RSEED" "$MSEED" "$ALPHAS"
  CANDS=(); for a in $ALPHAS; do CANDS+=("ta_${TAG}_e4_a$a"); done
  read -r BEST BRATE LO HI < <($PY -m alphatrain.scripts.best_gate "${CANDS[@]}")
  TOP=${ALPHAS##* }
  if [ "$BEST" = "ta_${TAG}_e4_a$TOP" ]; then
    MORE=$(awk -v a="$TOP" 'BEGIN { printf "%.1f", a + 1 }')
    echo "##### best alpha is the largest swept ($TOP): gating alpha $MORE too"
    scripts/ta_sweep.sh "$ACTOR" "checkpoints/${TAG}_ft/epoch_4.pt" "ta_${TAG}_e4" \
      "flywheel: $TAG + alpha*($TAG fine-tuned 4ep on its fixed-teacher crisis windows ($SIMS sims)); frozen BN, lr1e-4, T0.5" \
      "$MORE"
    CANDS+=("ta_${TAG}_e4_a$MORE")
    read -r BEST BRATE LO HI < <($PY -m alphatrain.scripts.best_gate "${CANDS[@]}")
  fi
  GAIN=$(awk -v p="$RATE" -v n="$BRATE" 'BEGIN { printf "%.3f", (p - n) / p }')
  echo "##### turn $TAG: best $BEST $BRATE [$LO, $HI] per 100k turns; previous $RATE; gain $GAIN ($(date))"
  if lt "$BRATE" "$RATE"; then ACTOR=alphatrain/data/$BEST.pt; RATE=$BRATE; fi
  if lt "$GAIN" "$MIN_GAIN"; then
    echo "##### STOP: gain $GAIN < $MIN_GAIN. Best actor: $ACTOR ($RATE per 100k turns)"; break
  fi
  if ! lt "$TARGET" "$RATE"; then
    echo "##### STOP: $RATE <= target $TARGET. Best actor: $ACTOR"; break
  fi
  K=$((K + 1)); RSEED=$((RSEED + STRIDE)); MSEED=$((MSEED + STRIDE))
done
