#!/bin/bash
# Symmetric self-distillation turn (HISTORY 270): the actor's own recorded states labeled with its 8-view-averaged
# policy (top-5, softmax of the mean logits; build_tta_corpus), a gentle frozen-BN fine-tune on those soft targets,
# then the task-arithmetic sweep + 2k gates. The 8-view ensemble of A8 dies 38% less than its single pass.
#   caffeinate -is scripts/tta_distill_turn.sh <actor.pt> <records_dir> <tag> "<alphas>"
set -euo pipefail
cd "$(dirname "$0")/.."
ACTOR=$1; REC=$2; TAG=$3; ALPHAS=$4
D=alphatrain/data; PY=.venv/bin/python
LR=${LR:-3e-5}; EPOCHS=${EPOCHS:-1}; BS=${BS:-4096}
SHAPE=$($PY -m alphatrain.scripts.model_shape "$ACTOR")
mkdir -p logs checkpoints/${TAG}_ft
echo "=== [1/3] 8-view labels of $REC by $ACTOR ($(date)) ==="
[ -f $D/tta8_${TAG}.pt ] || $PY -m alphatrain.scripts.build_tta_corpus --games-dir $REC --base "$ACTOR" \
    --out $D/tta8_${TAG}.pt --holdout-games 20 --holdout-out $D/tta8_${TAG}_holdout.clrj > logs/${TAG}_tta_labels.log 2>&1
tail -n 1 logs/${TAG}_tta_labels.log
echo "=== [2/3] fine-tune on the 8-view labels (lr $LR, $EPOCHS ep, bs $BS, frozen BN, soft CE) ($(date)) ==="
$PY -m alphatrain.train_path_b --tensor-file $D/tta8_${TAG}.pt --resume "$ACTOR" --warm-start \
    --freeze-bn --policy-head pair2 --pair-dim 64 --legal-mask-loss $SHAPE \
    --epochs $EPOCHS --batch-size $BS --lr $LR --warmup-epochs 0 --target-temperature 1.0 --blend-alpha 1.0 \
    --augment-factor 1 --seed 42 --save-dir checkpoints/${TAG}_ft > logs/${TAG}_ft.log 2>&1
grep -E "val:" logs/${TAG}_ft.log | tail -1
echo "=== [3/3] task-arithmetic sweep + gate, alphas $ALPHAS ($(date)) ==="
scripts/ta_sweep.sh "$ACTOR" checkpoints/${TAG}_ft/epoch_${EPOCHS}.pt ta_${TAG}_e${EPOCHS} \
    "8-view self-distillation: actor + alpha*(actor fine-tuned on its own 8-view-averaged policy over its recorded games; lr $LR, $EPOCHS ep, frozen BN, soft CE)" \
    $ALPHAS
echo "=== done ($(date)) ==="
