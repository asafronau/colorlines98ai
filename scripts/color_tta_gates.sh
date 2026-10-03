#!/bin/bash
# Color-ensemble headroom of the best actor (HISTORY 279): same seeds/cap as the D4 TTA-8 measurement
# (HISTORY 270: D4 TTA-8 0.21, single pass over its first 25k turns 0.34 deaths per 100k turns).
set -euo pipefail
cd "$(dirname "$0")/.."
M=alphatrain/data/ta_A7_e4_a1.0.pt
.venv/bin/python -m alphatrain.scripts.eval_log run --model $M --color-tta 7 --max-turns 25000 \
    --desc "A8 = ta_A7_e4_a1.0, greedy on the average of logits over 7 color relabelings (cyclic shifts; eval --color-tta 7); no training"
.venv/bin/python -m alphatrain.scripts.eval_log run --model $M --tta 8 --color-tta 3 --max-turns 25000 \
    --desc "A8 = ta_A7_e4_a1.0, average over 8 board symmetries x 3 color relabelings (24 views; eval --tta 8 --color-tta 3); no training"
