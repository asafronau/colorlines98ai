#!/bin/bash
# Self-play on RunPod (L4, 12 vCPU).
#
# 1. Create a RunPod instance with L4 GPU, PyTorch template
# 2. Upload files:
#    scp colorlines_selfplay_train.tar.gz root@<ip>:/workspace/
#    scp alphatrain/data/pillar2m_best.pt root@<ip>:/workspace/
#    scp scripts/runpod_selfplay.sh root@<ip>:/workspace/
#
# 3. SSH and run:
#    ssh root@<ip> bash /workspace/runpod_selfplay.sh
#
# 4. Monitor:
#    ssh root@<ip> tail -f /workspace/selfplay.log
#
# 5. Download results:
#    scp -r root@<ip>:/workspace/selfplay_v3/ data/selfplay_v3/

set -euo pipefail

cd /workspace

echo "=== Setup ==="
pip install -q numpy numba scipy

# Extract code
tar xzf colorlines_selfplay_train.tar.gz

# Model
mkdir -p alphatrain/data
cp pillar2m_best.pt alphatrain/data/

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
from alphatrain.evaluate import load_model
net, ms = load_model('alphatrain/data/pillar2m_best.pt', 'cpu')
del net
print(f'Model OK, max_score={ms}')
"

echo ""
echo "=== Starting self-play: seeds 31500-32250 (750 games) ==="
echo "=== 10 workers, 800 sims, batch 32, L4 CUDA ==="
echo ""

mkdir -p selfplay_v3

nohup python3 -m alphatrain.scripts.selfplay \
    --model alphatrain/data/pillar2m_best.pt \
    --seed-start 30200 --seed-end 30500 \
    --sims 800 --batch-size 32 \
    --save-dir selfplay_v3 \
    --workers 10 --device cuda \
    --temperature-moves 15 \
    > selfplay.log 2>&1 &

echo "Started (PID: $!)"
echo "Monitor: tail -f /workspace/selfplay.log"
echo "Progress: ls selfplay_v3/*.json 2>/dev/null | wc -l"
