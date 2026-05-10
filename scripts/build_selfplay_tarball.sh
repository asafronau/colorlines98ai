#!/bin/bash
# Build the code tarball for Colab self-play.
#
# Output: colorlines_selfplay_train.tar.gz  (repo root)
# Upload to: MyDrive/alphatrain/  (along with model + value head checkpoints)

set -e
cd "$(dirname "$0")/.."  # repo root

OUT=colorlines_selfplay_train.tar.gz

echo "Building $OUT..."
tar czf $OUT \
    alphatrain/__init__.py \
    alphatrain/model.py \
    alphatrain/dataset.py \
    alphatrain/train.py \
    alphatrain/evaluate.py \
    alphatrain/observation.py \
    alphatrain/mcts.py \
    alphatrain/inference_server.py \
    alphatrain/value_head.py \
    alphatrain/scripts/__init__.py \
    alphatrain/scripts/selfplay.py \
    alphatrain/scripts/crisis_mining.py \
    alphatrain/scripts/eval_parallel.py \
    alphatrain/scripts/build_selfplay_tensors.py \
    alphatrain/scripts/mine_death_features.py \
    game/__init__.py \
    game/board.py \
    game/config.py \
    game/fast_heuristic.py \
    game/rng.py

ls -lh $OUT
echo
echo "Upload to Google Drive: MyDrive/alphatrain/"
echo "Along with:"
echo "  - alphatrain/data/pillar2y2_epoch_40.pt   (~143 MB)"
echo "  - alphatrain/data/value_head_v11.pt       (~38 KB)"
