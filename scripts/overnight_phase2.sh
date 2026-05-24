#!/bin/bash
# Overnight Phase 2 chain: rebuild dataset with new LEC labels, train spatial-head
# sanity, then mine stationary-boundary counterfactuals for ~8-12h.
#
# Each step is wrapped so failures in step N don't block step N+1. All logs in
# logs/overnight_phase2/.

set -u
cd /Users/andreis/local/source/colorlines98
source .venv/bin/activate

OUT=logs/overnight_phase2
mkdir -p "$OUT"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT/chain.log"
}

run_step() {
    local name="$1"
    shift
    log "=== START: $name ==="
    if "$@" 2>&1 | tee "$OUT/${name}.log"; then
        log "=== DONE: $name ==="
    else
        log "=== FAILED: $name (continuing chain) ==="
    fi
}

# Step 1: rebuild stationary-risk dataset with new LEC labels
run_step rebuild_dataset python scripts/build_stationary_risk_dataset.py \
    --selfplay-dirs data/selfplay_v13 data/selfplay_v14 \
    --crisis-dirs data/crisis_v13 \
    --H 100 --stride 50 \
    --output alphatrain/data/stationary_risk_v2.pt

# Step 2a: GAP-head sanity on v2 labels (validates pipeline)
run_step train_gap_v2 python scripts/train_stationary_risk_head.py \
    --backbone alphatrain/data/pillar3b_epoch_20.pt \
    --data alphatrain/data/stationary_risk_v2.pt \
    --out alphatrain/data/stationary_risk_head_v2_gap.pt \
    --head-type gap --epochs 5 --batch-size 4096 --hidden 128

# Step 2b: spatial head — does 1x1 conv preserve LEC info?
run_step train_spatial_v2 python scripts/train_stationary_risk_head.py \
    --backbone alphatrain/data/pillar3b_epoch_20.pt \
    --data alphatrain/data/stationary_risk_v2.pt \
    --out alphatrain/data/stationary_risk_head_v2_spatial.pt \
    --head-type spatial --epochs 8 --batch-size 4096 --hidden 256

# Step 3: mine Phase 2 counterfactuals — the slow step
# Budget: 1500 anchors x 10 candidates x 32 rollouts x ~50 turn average x ~10ms
# = ~24M turn-ops / 12 workers approx 5-12h depending on rollout length variance.
run_step mine_counterfactuals python scripts/mine_stationary_counterfactuals.py \
    --model alphatrain/data/pillar3b_epoch_20.pt \
    --selfplay-dir data/selfplay_v13 \
    --max-anchors 1000 \
    --top-k 10 --R 24 --H 100 \
    --workers 12 --device cpu \
    --out alphatrain/data/stationary_counterfactuals_v1.pt

log "=== CHAIN COMPLETE ==="
log "Outputs:"
log "  - alphatrain/data/stationary_risk_v2.pt                  (dataset with new LEC labels)"
log "  - alphatrain/data/stationary_risk_head_v2_gap.pt         (GAP head, 5 ep)"
log "  - alphatrain/data/stationary_risk_head_v2_spatial.pt     (spatial head, 8 ep)"
log "  - alphatrain/data/stationary_counterfactuals_v1.pt       (Phase 2 mining)"
log "  - logs/overnight_phase2/*.log"
