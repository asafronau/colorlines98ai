#!/bin/bash
# Build a single self-contained tarball for the Colab fleet miner.
#
# Output: /tmp/colorlines_fleet_mine.tar.gz  (~7-8 GB)
#
# Contains everything Colab needs:
#   - All Python code (alphatrain/ package + game/ engine)
#   - All mining input data (crisis_v12 + selfplay_v12 JSONs)
#   - The pillar2z model checkpoint
#   - requirements.txt
#
# Upload to MyDrive/alphatrain/ then run alphatrain/oracle_fleet_colab.ipynb.

set -e
cd "$(dirname "$0")/.."

OUT=/tmp/colorlines_fleet_mine.tar.gz

echo "Building $OUT (this can take 3-5 min on M5 MAX SSD)..."
tar czf "$OUT" \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='*.nbi' \
    --exclude='*.nbc' \
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
    alphatrain/scripts/phase1_oracle_fleet.py \
    alphatrain/scripts/phase1_oracle_fleet_gpu.py \
    alphatrain/scripts/phase1_oracle_label.py \
    alphatrain/scripts/fleet_jit.py \
    alphatrain/scripts/fleet_gpu.py \
    alphatrain/scripts/mine_death_features.py \
    alphatrain/data/pillar2z_epoch_19.pt \
    game/__init__.py \
    game/board.py \
    game/config.py \
    game/fast_heuristic.py \
    game/rng.py \
    requirements.txt \
    data/crisis_v12 \
    data/selfplay_v12

SIZE_MB=$(du -m "$OUT" | cut -f1)
echo
echo "Done: $OUT (${SIZE_MB} MB)"
echo
echo "Upload to Google Drive at: MyDrive/alphatrain/colorlines_fleet_mine.tar.gz"
echo "Then open alphatrain/oracle_fleet_colab.ipynb in Colab (L4 GPU)."
