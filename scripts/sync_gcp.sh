#!/bin/bash
# Sync completed games from GCP spot instance every 15 minutes.
# Safe to run even if VM is down — just prints a warning and retries next cycle.
#
# Usage: bash scripts/sync_gcp.sh
# Stop: Ctrl+C

ZONE="us-central1-b"
VM="colorlines98"
REMOTE="~/rust_engine/data/expert_v2/"
LOCAL="data/expert_v2/"

mkdir -p "$LOCAL"

while true; do
    BEFORE=$(ls "$LOCAL" 2>/dev/null | wc -l | tr -d ' ')
    gcloud compute scp --zone="$ZONE" \
        "${VM}:${REMOTE}game_*.json" "$LOCAL" 2>/dev/null
    AFTER=$(ls "$LOCAL" 2>/dev/null | wc -l | tr -d ' ')
    NEW=$((AFTER - BEFORE))
    echo "[$(date '+%H:%M')] $AFTER games local (+$NEW new)"
    sleep 900
done
