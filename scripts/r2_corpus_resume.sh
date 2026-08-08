#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
python - << 'PY'
import torch, os
t = torch.load('alphatrain/data/r2_bulk.pt', weights_only=True)
assert os.path.exists('alphatrain/data/r2_bulk_strata.npz')
print(f'bulk verified: {t["boards"].shape[0]:,} rows')
PY
python -m alphatrain.scripts.build_round2_corpora --which frontier
python -m alphatrain.scripts.danger_score_corpus --tensor alphatrain/data/r2_bulk.pt
gzip -9 -c alphatrain/data/r2_bulk.pt > alphatrain/data/r2_bulk.pt.gz
gzip -9 -c alphatrain/data/r2_frontier.pt > alphatrain/data/r2_frontier.pt.gz
ls -la alphatrain/data/r2_bulk.pt.gz alphatrain/data/r2_frontier.pt.gz
echo R2_CORPORA_DONE
