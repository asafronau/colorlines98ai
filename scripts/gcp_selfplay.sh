#!/bin/bash
# Self-play on GCP A3 (H100 + 26 vCPUs).
#
# Upload:
#   gcloud compute scp colorlines_selfplay_train.tar.gz <instance>:~/
#   gcloud compute scp alphatrain/data/pillar2k_surv_best.pt <instance>:~/
#   gcloud compute scp scripts/gcp_selfplay.sh <instance>:~/
#
# Run:
#   gcloud compute ssh <instance> -- bash gcp_selfplay.sh
#
# Monitor:
#   gcloud compute ssh <instance> -- tail -f ~/selfplay.log
#
# Download:
#   gcloud compute scp --recurse <instance>:~/selfplay_v1/ data/selfplay_v1/

set -euo pipefail

cd ~

echo "=== Setup ==="

# Venv
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install CUDA torch + deps
pip install -q --upgrade pip
pip install -q torch numpy numba scipy

# Extract code
tar xzf colorlines_selfplay_train.tar.gz

# Model
mkdir -p alphatrain/data
cp pillar2k_surv_best.pt alphatrain/data/

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Starting self-play: seeds 10500-12000 (1500 games) ==="
echo "=== 20 workers, 400 sims, batch 32, H100 CUDA ==="
echo "=== Output: ~/selfplay.log ==="
echo ""

mkdir -p selfplay_v1

nohup python3 -m alphatrain.scripts.selfplay \
    --model alphatrain/data/pillar2k_surv_best.pt \
    --seed-start 10500 --seed-end 12000 \
    --sims 400 --batch-size 32 \
    --save-dir selfplay_v1 \
    --workers 20 --device cuda \
    --temperature-moves 15 \
    > selfplay.log 2>&1 &

echo "Started (PID: $!)"
echo "Monitor: tail -f ~/selfplay.log"
echo "Progress: ls ~/selfplay_v1/*.json 2>/dev/null | wc -l"
