#!/bin/bash
# GCP self-play deployment script.
#
# From local machine:
#
#   # 1. Start instance
#   gcloud compute instances start coloreval98 --zone=us-central1-b
#
#   # 2. Build tarball and upload code + model
#   tar czf /tmp/colorlines98.tar.gz \
#       --exclude='.venv' --exclude='data' --exclude='alphatrain/data' \
#       --exclude='*.tar.gz' --exclude='__pycache__' --exclude='.git' \
#       -C /Users/andreis/local/source colorlines98
#   gcloud compute scp /tmp/colorlines98.tar.gz coloreval98:~ --zone=us-central1-b
#   gcloud compute scp alphatrain/data/alphatrain_td_best.pt coloreval98:~/model.pt --zone=us-central1-b
#
#   # 3. SSH and run setup + selfplay
#   gcloud compute ssh coloreval98 --zone=us-central1-b -- bash -s < scripts/gcp_selfplay.sh
#
#   # 4. Monitor
#   gcloud compute ssh coloreval98 --zone=us-central1-b -- tail -f ~/selfplay.log
#
#   # 5. Download results
#   gcloud compute scp --recurse coloreval98:~/selfplay_data/ data/selfplay/ --zone=us-central1-b
#
#   # 6. Stop instance
#   gcloud compute instances stop coloreval98 --zone=us-central1-b

set -euo pipefail

GAMES=200
SEED_START=300
SIMS=800
BS=8
WORKERS=$(nproc)
SAVE_DIR="$HOME/selfplay_data"

echo "=== GCP Self-Play Setup ==="
echo "vCPUs: $WORKERS"
echo "Games: $GAMES (seeds $SEED_START-$((SEED_START + GAMES - 1)))"
echo "Sims: $SIMS, batch_size: $BS"

cd "$HOME"

# Install system dependencies
echo "Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip tar gzip > /dev/null

# Extract code tarball
if [ -f colorlines98.tar.gz ]; then
    echo "Extracting code..."
    tar xzf colorlines98.tar.gz
else
    echo "ERROR: ~/colorlines98.tar.gz not found. Upload it first."
    exit 1
fi

cd colorlines98

# Setup venv with CPU-only torch
if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing Python dependencies (CPU-only torch)..."
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy numba scipy pytest

# Copy model
if [ -f "$HOME/model.pt" ]; then
    mkdir -p alphatrain/data
    cp "$HOME/model.pt" alphatrain/data/alphatrain_td_best.pt
    echo "Model copied."
else
    echo "ERROR: ~/model.pt not found. Upload it first."
    exit 1
fi

# Sanity test
echo "Running tests..."
python -m pytest alphatrain/tests/test_mcts.py -v --tb=short 2>&1 | tail -3

# Start self-play
mkdir -p "$SAVE_DIR"
SEED_END=$((SEED_START + GAMES))

echo ""
echo "=== Starting Self-Play ==="
echo "Output: ~/selfplay.log"
echo ""

nohup python -m alphatrain.scripts.selfplay \
    --model alphatrain/data/alphatrain_td_best.pt \
    --seed-start "$SEED_START" \
    --seed-end "$SEED_END" \
    --sims "$SIMS" \
    --batch-size "$BS" \
    --device cpu \
    --workers "$WORKERS" \
    --save-dir "$SAVE_DIR" \
    > "$HOME/selfplay.log" 2>&1 &

echo "Self-play started (PID: $!)"
echo "Monitor: tail -f ~/selfplay.log"
echo "Check: ls $SAVE_DIR/game_*.pt 2>/dev/null | wc -l"
