#!/bin/bash
# Build the code+oracle tarball for Colab Path B training.
#
# Output: colorlines_path_b.tar.gz  (repo root, ~70 KB)
# Upload to: MyDrive/alphatrain/  (alongside V12 tensor + warm-start ckpt)
#
# Bundles:
#   - Path B trainer (alphatrain/train_path_b.py) + unit tests
#   - Supporting modules (model, dataset, observation)
#   - game/ engine
#   - The Path B oracle tensor (16,897 anchors, 3.3 MB)
#
# Already on Drive (do NOT re-upload):
#   - v12_pillar2z.pt.gz       — V12 training corpus (~2-3 GB compressed)
#   - pillar2y2_epoch_40.pt    — warm-start checkpoint (~143 MB)

set -e
cd "$(dirname "$0")/.."  # repo root

OUT=colorlines_path_b.tar.gz

echo "Building $OUT..."
tar czf $OUT \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    alphatrain/__init__.py \
    alphatrain/model.py \
    alphatrain/dataset.py \
    alphatrain/observation.py \
    alphatrain/train.py \
    alphatrain/train_path_b.py \
    alphatrain/mcts.py \
    alphatrain/inference_server.py \
    alphatrain/value_head.py \
    alphatrain/evaluate.py \
    alphatrain/scripts/__init__.py \
    alphatrain/scripts/mine_death_features.py \
    alphatrain/scripts/build_path_b_tensor.py \
    alphatrain/scripts/combine_oracle_datasets.py \
    alphatrain/scripts/analyze_target_alignment.py \
    alphatrain/scripts/analyze_legal_distribution.py \
    alphatrain/scripts/analyze_v12_target_entropy.py \
    alphatrain/scripts/analyze_path_b_checkpoint.py \
    alphatrain/scripts/overnight_analysis.py \
    alphatrain/scripts/eval_parallel.py \
    alphatrain/scripts/selfplay.py \
    alphatrain/scripts/crisis_mining.py \
    alphatrain/scripts/build_expert_v2_tensor.py \
    alphatrain/scripts/fit_feature_value.py \
    alphatrain/scripts/train_value_head.py \
    alphatrain/tests/__init__.py \
    alphatrain/tests/test_train_path_b.py \
    alphatrain/data/phase1_oracle_path_b.pt \
    game/__init__.py \
    game/board.py \
    game/config.py \
    game/fast_heuristic.py \
    game/rng.py \
    requirements.txt

ls -lh $OUT
echo
echo "Upload to Google Drive: MyDrive/alphatrain/$OUT"
echo
echo "Already on Drive (verify before running Colab):"
echo "  - v12_pillar2z.pt.gz       (V12 corpus, ~2-3 GB compressed)"
echo "  - pillar2y2_epoch_40.pt    (warm-start, ~143 MB)"
echo
echo "Then open alphatrain/train_path_b_colab.ipynb in Colab (H100)."
