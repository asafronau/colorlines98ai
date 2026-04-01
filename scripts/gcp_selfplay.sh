#!/bin/bash
# GCP self-play deployment script.
#
# Run from local machine:
#   1. gcloud compute instances start coloreval98 --zone=us-central1-b
#   2. gcloud compute scp alphatrain/data/alphatrain_td_best.pt coloreval98:~/model.pt --zone=us-central1-b
#   3. gcloud compute ssh coloreval98 --zone=us-central1-b -- bash -s < scripts/gcp_selfplay.sh
#   4. gcloud compute ssh coloreval98 --zone=us-central1-b -- tail -f ~/selfplay.log
#   5. gcloud compute scp --recurse coloreval98:~/selfplay_data/ data/selfplay/ --zone=us-central1-b
#   6. gcloud compute instances stop coloreval98 --zone=us-central1-b

set -euo pipefail

REPO="https://github.com/anthropics/colorlines98.git"
GAMES=300
SEED_START=0
SIMS=800
BS=8
WORKERS=$(nproc)
SAVE_DIR="$HOME/selfplay_data"

echo "=== GCP Self-Play Setup ==="
echo "vCPUs: $WORKERS"
echo "Games: $GAMES (seeds $SEED_START-$((SEED_START + GAMES - 1)))"
echo "Sims: $SIMS, batch_size: $BS"

# Clone or pull repo
cd "$HOME"
if [ -d colorlines98 ]; then
    echo "Updating repo..."
    cd colorlines98 && git pull
else
    echo "Cloning repo..."
    git clone "$REPO" colorlines98
    cd colorlines98
fi

# Setup venv with CPU-only torch (smaller, faster install)
if [ ! -d .venv ]; then
    echo "Creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies (CPU-only torch)..."
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q numpy numba scipy pytest

# Copy model
if [ -f "$HOME/model.pt" ]; then
    mkdir -p alphatrain/data
    cp "$HOME/model.pt" alphatrain/data/alphatrain_td_best.pt
    echo "Model copied to alphatrain/data/"
else
    echo "ERROR: ~/model.pt not found. Upload it first:"
    echo "  gcloud compute scp alphatrain/data/alphatrain_td_best.pt coloreval98:~/model.pt --zone=us-central1-b"
    exit 1
fi

# Quick sanity test
echo "Running tests..."
python -m pytest alphatrain/tests/test_mcts.py -v --tb=short 2>&1 | tail -3

# Start self-play
mkdir -p "$SAVE_DIR"
SEED_END=$((SEED_START + GAMES))

echo ""
echo "=== Starting Self-Play ==="
echo "Output: ~/selfplay.log"
echo "Monitor: tail -f ~/selfplay.log"
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
echo "Check progress: tail -f ~/selfplay.log"
echo "Check games: ls $SAVE_DIR/game_*.pt | wc -l"
