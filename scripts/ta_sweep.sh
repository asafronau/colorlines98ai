#!/bin/bash
# Task-arithmetic alpha sweep: theta = base + alpha * (finetuned - base), each merge gate-evaluated
# (1k uncapped greedy games, seeds 2,600,000-2,600,999) and logged to alphatrain/EVAL.md.
#   scripts/ta_sweep.sh <base.pt> <finetuned.pt> <tag> "<description>" <alpha> [<alpha> ...]
# Evals run concurrently (independent processes; results don't depend on concurrency).
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=$1; FT=$2; TAG=$3; DESC=$4; shift 4
mkdir -p logs
for A in "$@"; do
  .venv/bin/python scripts/merge_checkpoints.py --base "$BASE" --crisis "$FT" --alpha "$A" \
    --out "alphatrain/data/${TAG}_a${A}.pt"
done
for A in "$@"; do
  .venv/bin/python -m alphatrain.scripts.eval_log run --model "alphatrain/data/${TAG}_a${A}.pt" \
    --desc "$DESC, alpha=$A (base + alpha*(ft - base))" --seed-start 2600000 --seed-end 2601000 \
    > "logs/eval_${TAG}_a${A}.log" 2>&1 &
done
wait
for A in "$@"; do echo "== alpha=$A"; tail -n 3 "logs/eval_${TAG}_a${A}.log"; done
