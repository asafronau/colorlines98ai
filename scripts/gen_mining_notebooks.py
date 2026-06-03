"""Generate 3 Colab crisis-fork mining notebooks with DISJOINT seed ranges.

Each notebook plays games to natural death (no cap), rewinds, confirms forks
(R=500), and writes small logs/mine_<seed>.json files (policy move baked in, so
no death-game upload needed). A background loop copies those to a per-notebook
Drive folder every 5 min => a 24h Colab disconnect loses at most ~5 min.

Run all 3 on separate Colab runtimes simultaneously, in parallel with the M5.
Ranges are wide (10k each) so even a much-faster-than-M5 Colab can't exhaust
them in 24h, and they never overlap (M5: 50140+, Colab: 60000/70000/80000+).
"""
import os
import json

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'alphatrain')
COPIES = [(1, 60000), (2, 70000), (3, 80000)]

INTRO = """# Crisis-fork mining — Colab copy __IDX__ of 3 (seeds __SEED__..__SEEDEND__)

Parallel fork mining. Plays each game to **natural death (no cap)**, rewinds the
last 15-45 plies, and confirms avoidable "crisis fork" moves with R=500 fresh
rollouts. Each confirmed fork lands in a tiny `logs/mine_<seed>.json` (the policy
move is baked in, so death-game trajectories are NOT needed downstream and stay
local). A background job copies mine files to Drive every 5 min, so a disconnect
loses almost nothing.

**Run all 3 copies on separate Colab runtimes at the same time**, alongside the
M5. Seed ranges are disjoint (M5 50140+, copies 60000/70000/80000+) so results
merge cleanly. This is **copy __IDX__** -> `MyDrive/alphatrain/mine_colab___IDX__/`.

## Drive uploads (`MyDrive/alphatrain/`) — same for all 3 copies
1. `colorlines_mining.tar.gz` — code archive (build locally, repo root; explicit
   file list keeps it ~360 KB — do NOT tar `alphatrain/` wholesale, its `data/`
   holds multi-GB checkpoints):
   ```bash
   tar czf colorlines_mining.tar.gz --exclude='**/__pycache__' \\
       alphatrain/*.py alphatrain/scripts/*.py \\
       alphatrain/data/feature_value_weights_2y_nb.npz \\
       scripts/*.py game/
   ```
2. `pillar3b_epoch_20.pt` — already on Drive from pillar3b training.

Upload both, then Runtime > Run all."""

SETUP = """import os, shutil
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_mining.tar.gz /content/
!cd /content && tar xzf colorlines_mining.tar.gz
os.makedirs('/content/alphatrain/data', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt',
            '/content/alphatrain/data/pillar3b_epoch_20.pt')
%cd /content
os.makedirs('logs', exist_ok=True)
os.makedirs('alphatrain/data/death_games', exist_ok=True)
OUT = f'{DRIVE}/mine_colab___IDX__'
os.makedirs(OUT, exist_ok=True)
# RESUME: pull any already-mined files back so this run SKIPS them (idempotent
# — safe to re-run after a disconnect; it continues where it left off)
!cp -n {OUT}/mine_*.json logs/ 2>/dev/null
print('resuming with', len([x for x in os.listdir('logs') if x.startswith('mine_')]),
      'already-mined seeds')
# crash-safe: copy new mine files to Drive every 5 min (local logs stay fast)
get_ipython().system_raw(
    "nohup bash -c 'while true; do cp -u logs/mine_*.json " + OUT +
    "/ 2>/dev/null; sleep 300; done' >/dev/null 2>&1 &")
!pip install -q numpy numba scipy
print('crash-safe sync every 5 min ->', OUT)"""

GPU = """import torch
print('CUDA:', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"""

RUN_MD = """## Mine (≈23h, leaves 1h buffer under the 24h Colab limit)

Plays to natural death, mines, confirms. Watch the `death .. -> mine` lines and
`mine rc=0` counts. Mine files are streaming to Drive in the background."""

RUN = """%cd /content
# --r-screen 50: verified fork-preserving at 1.42x (vs 100); R=500 confirm gates
#   quality, the screen just flags candidates.
# --workers 8: cuda runs the per-process forwards CONCURRENTLY (unlike MPS, which
#   serializes), so workers parallelize the CPU game logic + overlap the GPU.
#   Single-process (workers 1) leaves the L4/A100 mostly idle. Tune up toward the
#   vCPU count if the GPU isn't saturated.
# Re-runnable: skips seeds already in logs/ (pulled from Drive), so restart resumes.
!python scripts/overnight_systematic.py \\
    --device cuda --batch 256 --r-screen 50 --workers 8 \\
    --seed-start __SEED__ --n-try 10000 --max-seconds 82800 \\
    2>&1 | tee -a logs/mining___IDX__.log \\
    | grep -E "death |mine rc|HARVEST|DONE|forks:|fork\\(s\\)|skip"
"""

FINAL = """# Final flush to Drive (catches the last <5 min not yet synced)
import glob, shutil, os
OUT = '/content/drive/MyDrive/alphatrain/mine_colab___IDX__'
n = 0
for f in glob.glob('logs/mine_*.json'):
    shutil.copy(f, OUT); n += 1
if os.path.exists('logs/mining___IDX__.log'):
    shutil.copy('logs/mining___IDX__.log', OUT)
print(f'flushed {n} mine files to {OUT}')"""

AFTER = """## After all runs finish (on the M5)

Download every `MyDrive/alphatrain/mine_colab_{1,2,3}/mine_*.json` into the repo's
`logs/` (next to the M5's own mine files), then rebuild the combined corpus:
```bash
PYTHONPATH=. python scripts/build_crisis_corpus_file.py \\
    --out alphatrain/data/crisis_corpus_v2.pt
```
`build_crisis_corpus` globs all `logs/mine_*.json`; new files carry the policy
move inline (no death games needed), old M5 files fall back to their death games."""


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t}


for idx, seed in COPIES:
    sub = lambda s: (s.replace('__IDX__', str(idx)).replace('__SEED__', str(seed))
                     .replace('__SEEDEND__', str(seed + 9999)))
    cells = [
        md(sub(INTRO)),
        code("from google.colab import drive\ndrive.mount('/content/drive')"),
        code(sub(SETUP)),
        code(GPU),
        md(RUN_MD),
        code(sub(RUN)),
        code(sub(FINAL)),
        md(AFTER),
    ]
    nb = {"cells": cells,
          "metadata": {"accelerator": "GPU",
                       "colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    path = os.path.join(OUT_DIR, f'mine_crisis_colab_{idx}.ipynb')
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path} (seeds {seed}..{seed + 9999})")
