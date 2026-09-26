#!/bin/bash
# Gen-1b (fixed-teacher crisis windows) follow-on, HISTORY 249: wait for the owner's mcts_crisis run,
# build + check the corpus, then train two arms from A1 and gate them.
#   arm 1: fine-tune on the new rows (frozen BN) + task-arithmetic blend at alpha 0.4 / 0.7
#   arm 2: continuous training, r2_bulk + 30% new rows per batch, EMA weights gated
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/gen1b_crisis_A1_w45
TENSOR=alphatrain/data/gen1b_crisis_A1_w45.pt
A1=alphatrain/data/A1_pair2_orig_e40.pt
mkdir -p logs checkpoints/gen1b_ft checkpoints/cont_gen1b_mix30

until [ -f "$OUT/generation_complete.json" ]; do sleep 60; done
files=("$OUT"/game_seed*_prevention_*.json)   # glob, not ls|head: pipefail turns SIGPIPE into exit 141
echo "mining complete: ${#files[@]} prevention files"
first=${files[0]}
grep -q '"value_kind": "neural"' "$first" || { echo "FATAL: $first is not a neural-leaf search"; exit 1; }

.venv/bin/python -m alphatrain.scripts.build_expert_v2_tensor --games-dir "$OUT" --policy-only-data \
    --output "$TENSOR" | grep -E 'boards:'
.venv/bin/python -m alphatrain.scripts.anomaly_checks --tensor "$TENSOR" --model "$A1" --n 20000 | grep -E '^ *(A|B|C)\.'

COMMON="--resume $A1 --warm-start --freeze-bn --policy-head pair2 --pair-dim 64 --legal-mask-loss --num-blocks 18 --channels 96 --warmup-epochs 0 --augment-factor 1 --seed 42"

(caffeinate -is .venv/bin/python -m alphatrain.train_path_b --tensor-file "$TENSOR" $COMMON \
    --epochs 4 --batch-size 1024 --lr 1e-4 --target-temperature 0.5 --save-dir checkpoints/gen1b_ft \
    > logs/gen1b_ft.log 2>&1 && \
 caffeinate -is scripts/ta_sweep.sh "$A1" checkpoints/gen1b_ft/epoch_4.pt ta_gen1b_e4 \
    "task arithmetic: A1 + alpha*(A1 fine-tuned 4ep on gen1b FIXED-teacher crisis windows (neural leaf, 600 sims, 45 moves/replay); frozen BN, lr1e-4 cosine, T0.5, bs1024, D4+color aug)" \
    0.4 0.7 > logs/ta_gen1b_e4_sweep.log 2>&1) &

(caffeinate -is .venv/bin/python -m alphatrain.train_path_b --tensor-file alphatrain/data/r2_bulk.pt $COMMON \
    --mix-tensor "$TENSOR" --mix-share 0.3 --ema-decay 0.999 --epochs 1 --flat-epochs 1 --batch-size 4096 \
    --lr 3e-5 --blend-alpha 0 --save-every-steps 1000 --save-dir checkpoints/cont_gen1b_mix30 \
    > logs/cont_gen1b_mix30.log 2>&1 && \
 cp checkpoints/cont_gen1b_mix30/ema_epoch_1.pt alphatrain/data/cont_gen1b_mix30_ema_e1.pt && \
 caffeinate -is .venv/bin/python -m alphatrain.scripts.eval_log run --model alphatrain/data/cont_gen1b_mix30_ema_e1.pt \
    --desc "continuous training: A1 warm, r2_bulk + 30% gen1b FIXED-teacher crisis windows per batch, 1 epoch constant lr3e-5 bs4096 hard CE, frozen BN, legal mask; EMA 0.999 at end of epoch" \
    --seed-start 2600000 --seed-end 2601000 > logs/eval_cont_gen1b_mix30_ema_e1.log 2>&1) &

wait
echo "== results"; grep -E 'ta_gen1b_e4|cont_gen1b_mix30' alphatrain/EVAL.md | cut -d'|' -f3,8,13,11,19
