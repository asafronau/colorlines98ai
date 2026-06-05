"""Generate alphatrain/gen_crisis_colab.ipynb — the crisis-corrections pipeline on Colab.

Records policy death games (tail-90) for a seed range, then mines MCTS corrections
(deep widened multi-seed MCTS@4800). Everything synced to Drive, resume-friendly:
  - death games: synced to Drive AFTER recording completes (batch)
  - corrections: copied to Drive after EACH game (background loop, crash-safe)
Re-running pulls whatever's already on Drive and skips it.
"""
import os, json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'gen_crisis_colab.ipynb')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


cells = []

cells.append(md(r"""
# Crisis-corrections pipeline — Colab (seeds 51000–51500)

Two stages, both stored in Drive and **resume-friendly**:
1. **Record** policy death games to **tail-90** (`batch_record`) — needed for the
   D-15..D-85 crisis band. Synced to Drive **after recording completes** (batch).
2. **Mine** corrections — rewind each death game's crisis band, run **widened
   multi-seed MCTS@4800**, keep only states where MCTS's move != the policy's move,
   store the soft visit distribution. Each `corr_<seed>.json` is copied to Drive
   **as soon as it's written** (background loop, every 30s) — so a disconnect loses
   at most ~30s. Re-running pulls everything already on Drive and skips it.

## Uploads to `MyDrive/alphatrain/`
1. **`colorlines_crisis.tar.gz`** — code + feature-value weights (build cmd below)
2. **`pillar3b_epoch_20.pt`** — the policy (already on Drive from pillar3b)

Build the tarball locally (repo root):
```bash
tar czf colorlines_crisis.tar.gz --exclude='**/__pycache__' \
    alphatrain/*.py alphatrain/scripts/*.py scripts/*.py game/ \
    alphatrain/data/feature_value_weights_2y_nb.npz
```
Outputs land in `MyDrive/alphatrain/crisis_51000/{death_games,corrections}/`.
""".strip()))

cells.append(code("from google.colab import drive\ndrive.mount('/content/drive')"))

cells.append(code(r"""
import os, shutil
SEED_START, N_SEEDS = 51000, 501          # 51000..51500 inclusive
DRIVE = '/content/drive/MyDrive/alphatrain'
CR = f'{DRIVE}/crisis_{SEED_START}'        # this run's Drive folder

# Extract code + model + feature-value weights
!cp {DRIVE}/colorlines_crisis.tar.gz /content/
!cd /content && tar xzf colorlines_crisis.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt',
            '/content/alphatrain/data/pillar3b_epoch_20.pt')
%cd /content
os.makedirs('crisis/death_games', exist_ok=True)
os.makedirs('crisis/corrections', exist_ok=True)
os.makedirs(f'{CR}/death_games', exist_ok=True)
os.makedirs(f'{CR}/corrections', exist_ok=True)

# RESUME: pull anything already done from Drive so both stages skip it
!cp -n {CR}/death_games/*.json crisis/death_games/ 2>/dev/null
!cp -n {CR}/corrections/*.json crisis/corrections/ 2>/dev/null
print('resume: death_games =', len(os.listdir('crisis/death_games')),
      '| corrections =', len(os.listdir('crisis/corrections')))

# Crash-safe: copy each NEW correction to Drive every 30s (incremental sync)
get_ipython().system_raw(
    "nohup bash -c 'while true; do cp -u /content/crisis/corrections/*.json " + CR +
    "/corrections/ 2>/dev/null; sleep 30; done' >/dev/null 2>&1 &")
!pip install -q numpy numba scipy
print('correction sync (every 30s) ->', CR + '/corrections')
""".strip()))

cells.append(code(r"""
import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
import os
WORKERS = min(16, (os.cpu_count() or 8))
print('cpu_count:', os.cpu_count(), '-> workers:', WORKERS)
""".strip()))

cells.append(md(r"""
## Stage 1 — record death games (tail-90), then batch-sync to Drive

Batched replay-to-death on the GPU; resumable (skips death games already pulled
from Drive). All death games are copied to Drive **after** this cell finishes.
""".strip()))

cells.append(code(r"""
%cd /content
!PYTHONPATH=. python scripts/batch_record.py \
    --seed-start {SEED_START} --n-seeds {N_SEEDS} --tail 90 \
    --out-dir crisis/death_games --device cuda --batch 256

# Batch-sync ALL death games to Drive (after recording completes)
import glob, shutil
for f in glob.glob('/content/crisis/death_games/*.json'):
    shutil.copy(f, CR + '/death_games/')
print('synced', len(glob.glob(CR + '/death_games/*.json')), 'death games to Drive')
""".strip()))

cells.append(md(r"""
## Stage 2 — mine MCTS corrections (the long stage; ~hours)

Widened (top_k=all legal + Dirichlet) multi-seed MCTS@4800 over the D-15..D-85 band.
Per-game `corr_<seed>.json` written as each game finishes — the background loop
syncs each to Drive within 30s. Watch the `[k/N] seed … corrections` lines + the
running yield %. Re-runnable: skips games whose `corr_<seed>.json` already exists.
""".strip()))

cells.append(code(r"""
%cd /content
!PYTHONPATH=. python scripts/gen_corrections_parallel.py \
    --death-glob 'crisis/death_games/death_*.json' \
    --workers {WORKERS} --sims 4800 --mcts-seeds 3 --lo 15 --hi 85 \
    --out-dir crisis/corrections 2>&1 | tee crisis/mine.log
""".strip()))

cells.append(code(r"""
# Final flush — catch the last <30s of corrections + the log
import glob, shutil, os
n = 0
for f in glob.glob('/content/crisis/corrections/*.json'):
    shutil.copy(f, CR + '/corrections/'); n += 1
if os.path.exists('/content/crisis/mine.log'):
    shutil.copy('/content/crisis/mine.log', CR + '/')
print('final flush: corrections on Drive =',
      len(glob.glob(CR + '/corrections/*.json')))
""".strip()))

cells.append(md(r"""
## After it finishes (on the M5)

Download `MyDrive/alphatrain/crisis_51000/corrections/*.json` into the repo's
`crisis/corrections/` (next to the M5's own), then assemble the combined
corrections corpus for distillation. Death games are in
`crisis_51000/death_games/` if you want to re-mine on a future policy generation.
""".strip()))

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"wrote {OUT} ({len(cells)} cells)")
