#!/bin/bash
# Prepare Pillar 2g training data for Colab upload.
# Run this AFTER self-play iter2 (seeds 500-1500) completes.
#
# Steps:
#   1. Merge game files -> selfplay_iter2.pt
#   2. Sharpen self-play policy targets (T=0.3)
#   3. Compress for upload
#   4. Rebuild code tarball
#
# Output files to upload to Drive (MyDrive/alphatrain/):
#   - colorlines_selfplay_train.tar.gz  (~90 KB)
#   - selfplay_iter2.pt.gz              (~X GB, compressed)
#   - alphatrain_pairwise.pt            (already on Drive)
#   - pillar2f_best.pt                  (already on Drive)

set -e
cd "$(dirname "$0")/../.."  # repo root

echo "=== Step 1: Merge self-play game files ==="
python -m alphatrain.scripts.build_selfplay_tensors \
    --games-dir data/selfplay_iter2 \
    --output alphatrain/data/selfplay_iter2.pt \
    --max-score 500.0

echo ""
echo "=== Step 2: Sharpen self-play policy (T=0.3) ==="
python -c "
import torch, sys
d = torch.load('alphatrain/data/selfplay_iter2.pt', weights_only=False)
pol = d['policy_targets']
entropy_before = -(pol * (pol + 1e-10).log()).sum(dim=-1).mean().item()
# Sharpen: pol^(1/T), then renormalize
inv_t = 1.0 / 0.3
sharpened = pol ** inv_t
sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True).clamp(min=1e-8)
entropy_after = -(sharpened * (sharpened + 1e-10).log()).sum(dim=-1).mean().item()
d['policy_targets'] = sharpened
d['policy_temperature'] = 0.3
torch.save(d, 'alphatrain/data/selfplay_iter2.pt')
print(f'Policy sharpened: entropy {entropy_before:.2f} -> {entropy_after:.2f}')
print(f'Saved (T=0.3)')
" || exit 1

echo ""
echo "=== Step 3: Compress for upload ==="
echo "Compressing selfplay_iter2.pt..."
gzip -c alphatrain/data/selfplay_iter2.pt > selfplay_iter2.pt.gz
ls -lh selfplay_iter2.pt.gz

echo ""
echo "=== Step 4: Rebuild code tarball ==="
tar czf colorlines_selfplay_train.tar.gz \
    alphatrain/__init__.py \
    alphatrain/model.py \
    alphatrain/dataset.py \
    alphatrain/train.py \
    alphatrain/train_hybrid.py \
    alphatrain/evaluate.py \
    alphatrain/observation.py \
    alphatrain/mcts.py \
    alphatrain/inference_server.py \
    alphatrain/afterstate.py \
    alphatrain/tests/ \
    alphatrain/scripts/__init__.py \
    alphatrain/scripts/selfplay.py \
    alphatrain/scripts/eval_parallel.py \
    alphatrain/scripts/build_selfplay_tensors.py \
    alphatrain/scripts/build_mixed_tensors.py \
    alphatrain/train_pillar2g_colab.ipynb \
    game/__init__.py \
    game/board.py \
    game/config.py \
    game/fast_heuristic.py
ls -lh colorlines_selfplay_train.tar.gz

echo ""
echo "=== DONE ==="
echo ""
echo "Upload to Google Drive (MyDrive/alphatrain/):"
echo "  1. colorlines_selfplay_train.tar.gz  (code)"
echo "  2. selfplay_iter2.pt.gz              (self-play data)"
echo "  3. alphatrain_pairwise.pt            (already on Drive)"
echo "  4. pillar2f_best.pt                  (already on Drive)"
echo ""
echo "Then open train_pillar2g_colab.ipynb on Colab (A100/H100 runtime)"
