"""Generate alphatrain/train_pillar3d_colab.ipynb (crisis-fork floor distillation).

Hand-authoring ipynb JSON with escaped newlines is error-prone; this builds the
cells from Python multi-line strings and json.dumps a valid nbformat-4 notebook.
Re-run to regenerate after editing the recipe. Mirrors train_pillar3c_colab.ipynb.
"""
import os
import sys
import json

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'alphatrain', 'train_pillar3d_colab.ipynb')


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n")}


cells = []

cells.append(md(r"""
# Pillar3d (Track 1): crisis-fork floor distillation on the CLEAN fork subset

## Goal

Lift pillar3b's policy-only **floor** (fixed-engine baseline: mean 17,581, P5 1,452,
P10 2,377, <1000 2.9%, <500 0.3%) WITHOUT regressing the mean more than ~5%.
Policy-only / browser-WASM target. This is the **conservative Track-1 calibration**:
the cleanest fork subset + a small aux nudge, to validate the augment->distill->floor-eval
pipeline before scaling to the harder clustered forks.

## Method

Listwise-margin auxiliary loss fed **CI-confirmed R=500 crisis forks**, but on the
**Track-1 CLEAN subset only**: forks that are (a) **isolated** (no adjacent confirmed
fork) and (b) **non-clearing** (the policy move does not complete a line). This removes
the "phantom forks" we diagnosed -- a *correct* move (often a forced clear) blamed for a
*downstream* blunder, because the greedy-rollout judge can't separate move quality from
the policy's own later mistake. Phantoms cluster on the turns before a real blunder, so
isolated + non-clearing forks are the highest-trust signal: **479 clean forks** (from
1,209 confirmed; dropped 572 clustered + 158 clearing-policy-move).

pillar3c's three root causes and the fixes carried here:

| pillar3c failure | fix |
|---|---|
| R=24 labels too noisy | R=500 paired-bootstrap, keep only CI-excludes-0 forks |
| argmax-flip too aggressive | confirmed winners + clean isolated subset + **lambda=0.05** (was 0.15) |
| "val loss missed it" | watch **held-out fork flip-rate** (by-seed split) + val CE within 5% |

Per confirmed fork: winner = confirmed safe move, top1(loser) = the move the policy
actually played (higher catastrophe), clean losers = candidates >=10pp worse than the
winner (noise guard). The listwise hinge pushes `logit[winner] > logit[policy_move] +
margin`. **Anti-dilution:** each fork is weighted by its normalized confirmed
catastrophe gap (`--aux-weighted`) so high-value forks dominate instead of being
averaged into noise.

**Baseline (pillar3b on the forks):** flip=0% (policy plays its own move), the safe
move sits at median rank ~5 (mean ~12, a buried tail), `margin(pol-win) ~ +3.3 logits`.
That +3.3 gap is what training must close. Local smoke already moved **held-out** flip
0 -> 8.6% in 4 steps -> the signal generalizes across games.

## Required Drive uploads (`MyDrive/alphatrain/`)

1. `colorlines_pillar3d.tar.gz` -- code archive (build command below)
2. `crisis_corpus_track1.pt` -- pre-built confirmed-fork corpus (tiny, ~0.1 MB)
3. `v13_pillar3a.pt.gz` -- V13 tensor (already on Drive from pillar3b)
4. `pillar3b_epoch_20.pt` -- warm-start checkpoint (already on Drive)

**Build the tarball + corpus locally first** (from repo root, AFTER the harvest finishes):
```bash
# 1. (re)build the Track-1 CLEAN corpus (isolated + non-clearing forks)
PYTHONPATH=. python scripts/build_crisis_corpus_file.py \
    --isolated --drop-clearing \
    --out alphatrain/data/crisis_corpus_track1.pt

# 2. code archive
tar czf colorlines_pillar3d.tar.gz \
    --exclude='**/__pycache__' \
    alphatrain/counterfactual.py \
    alphatrain/train_path_b.py \
    alphatrain/model.py \
    alphatrain/dataset.py \
    alphatrain/observation.py \
    alphatrain/evaluate.py \
    alphatrain/mcts.py \
    alphatrain/__init__.py \
    scripts/eval_fork_ranking.py \
    scripts/batched_rollout.py \
    scripts/__init__.py \
    game/
```

Then upload `colorlines_pillar3d.tar.gz` and `crisis_corpus_track1.pt` to `MyDrive/alphatrain/`.
""".strip()))

cells.append(code(
    "from google.colab import drive\n"
    "drive.mount('/content/drive')"))

cells.append(code(r"""
import os, shutil, time

DRIVE = '/content/drive/MyDrive/alphatrain'

# Extract code
!cp {DRIVE}/colorlines_pillar3d.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d.tar.gz

os.makedirs('/content/alphatrain/data', exist_ok=True)

# Warm-start checkpoint
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt',
            '/content/alphatrain/data/pillar3b_epoch_20.pt')
print(f'pillar3b: {os.path.getsize("/content/alphatrain/data/pillar3b_epoch_20.pt")/1e6:.0f} MB')

# Crisis-fork corpus (separate small upload; rebuilt after each harvest)
cc = '/content/alphatrain/data/crisis_corpus_track1.pt'
shutil.copy(f'{DRIVE}/crisis_corpus_track1.pt', cc)
print(f'crisis corpus: {os.path.getsize(cc)/1e6:.2f} MB')

# V13 training tensor (~5 GB after decompression)
t0 = time.time()
!gzip -dc {DRIVE}/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
print(f'V13 tensor: {os.path.getsize("/content/alphatrain/data/v13_pillar3a.pt")/1e9:.1f} GB ({time.time()-t0:.0f}s)')

!pip install -q numpy numba scipy
""".strip()))

cells.append(code(r"""
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    g = torch.cuda.get_device_properties(0)
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {g.total_memory / 1e9:.1f} GB')
""".strip()))

cells.append(md(r"""
## Baseline fork-ranking on pillar3b

Confirm the opportunity before burning ~12h of compute. Expected (matches M5 local):
- `flip = 0%` (policy ranks its own move first, by construction)
- `win_rank med ~5`, `top5 ~58%` (safe move under-ranked, with a buried tail)
- `margin ~ +3.3` logits (policy prefers its risky move by ~3.3 -- the gap to close)
- `conc ~71%` (policy already ranks the safe move above clearly-worse moves)
- held-out numbers ~= train (fork structure is consistent across games)

If these are way off, stop and investigate before training.
""".strip()))

cells.append(code(r"""
%cd /content
!PYTHONPATH=. python scripts/eval_fork_ranking.py \
    --model alphatrain/data/pillar3b_epoch_20.pt \
    --corpus alphatrain/data/crisis_corpus_track1.pt \
    --holdout-frac 0.2 --split-seed 0
""".strip()))

cells.append(md(r"""
## Train pillar3d Track 1 (lambda=0.05, weighted, held-out monitored)

~12h on L4/A100. Checkpoints copied to Drive every epoch via `--copy-to`. This is the
**conservative** run: small aux nudge (lambda=0.05) on the clean isolated fork subset.
No auto-abort guard here -- at lambda=0.05 the flip rises slowly during the 2-epoch
warmup, so a step-500 abort would false-trip; instead WATCH the prints and the val CE.

**Watch the preflight prints** (every 200 main steps; both `preflight` = train forks
and `heldout ` = unseen-game forks):
- `flip` should rise on BOTH train and heldout (heldout rising = the fix generalizes,
  not memorization). With lambda=0.05 expect a gentle rise, not a jump.
- `margin(win-pol)` should climb from negative toward 0+ (winner overtaking the policy move).
- `conc` should stay high (~0.71+); a drop means we're damaging the good ranking.
- **`val: loss`** must stay within ~5% of pillar3b's (~2.18). If it climbs, the aux is
  over-pushing and breaking the policy (pillar3c's failure mode) -> lower lambda further.
""".strip()))

cells.append(code(r"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_path_b \
    --tensor-file alphatrain/data/v13_pillar3a.pt \
    --amp --compile \
    --resume alphatrain/data/pillar3b_epoch_20.pt --warm-start \
    --epochs 17 --batch-size 32768 --lr 3e-4 --warmup-epochs 1 \
    --target-temperature 0.5 \
    --aux-crisis-corpus alphatrain/data/crisis_corpus_track1.pt --aux-weighted \
    --aux-holdout-frac 0.2 --aux-split-seed 0 \
    --aux-lambda 0.05 --aux-margin 0.25 \
    --aux-batch-size 128 --aux-warmup-epochs 2.0 \
    --aux-preflight-every 200 \
    --copy-to /content/drive/MyDrive/alphatrain/pillar3d_best.pt \
    --save-dir /content/checkpoints/pillar3d 2>&1 | tee /content/pillar3d_train.log
""".strip()))

cells.append(code(
    "# Preflight + heldout + val trajectory at a glance\n"
    "!grep -E 'preflight|heldout|Train: loss|val:' /content/pillar3d_train.log"))

cells.append(md(r"""
## Confirm the forks were learned (post-train fork-ranking)

Re-run the fork-ranking eval on a trained checkpoint. We want, vs the pillar3b baseline:
- `flip` up substantially on the **held-out** split (forks generalized, not memorized)
- `win_rank` median down toward 1-2, `top1`/`top5` up
- `margin` moved from +3.3 toward negative (safe move now preferred)
- `conc` not collapsed

Pick an epoch from the trajectory (e.g. the last, or where heldout flip plateaued).
""".strip()))

cells.append(code(r"""
%cd /content
EP = 17  # edit to taste
!PYTHONPATH=. python scripts/eval_fork_ranking.py \
    --model /content/checkpoints/pillar3d/epoch_{EP}.pt \
    --corpus alphatrain/data/crisis_corpus_track1.pt \
    --holdout-frac 0.2 --split-seed 0
""".strip()))

cells.append(code(r"""
# Copy all epoch checkpoints + the training log to Drive
import glob, shutil, os
DRIVE = '/content/drive/MyDrive/alphatrain'
for f in sorted(glob.glob('/content/checkpoints/pillar3d/epoch_*.pt')):
    dst = f'{DRIVE}/pillar3d_{os.path.basename(f)}'
    if not os.path.exists(dst):
        shutil.copy(f, dst)
        print(f'copied {os.path.basename(f)}')
shutil.copy('/content/pillar3d_train.log', f'{DRIVE}/pillar3d_train.log')
print('log copied')
""".strip()))

cells.append(md(r"""
## Post-run floor eval on M5 (the verdict)

The fork-ranking metric proves the signal was learned; the GAMEPLAY FLOOR proves it
helped. After downloading the checkpoints (eval_parallel prints mean/percentiles/<1000
to stdout -- tee it to a log; it has no --output flag):

```bash
# Policy-only floor eval (FIXED engine). MUST use the SAME 2000-seed range as the
# baseline so floor percentiles are comparable: seeds 777000-778999.
# NOTE: the policy player runs through the GPU inference server -- use --device
# mps (M5) or cuda, NOT cpu (eval_parallel crashes on cpu policy eval by design).
for ep in 5 8 10; do
    echo "===== pillar3d-track1 epoch ${ep} ====="
    python -m alphatrain.scripts.eval_parallel \
        --model alphatrain/data/pillar3d_epoch_${ep}.pt --policy-only \
        --seeds $(seq 777000 778999) \
        --device mps --workers 16 \
        2>&1 | tee alphatrain/data/pillar3d_ep${ep}_eval.log | grep -E 'P5|P10|mean|<1000|<500|>5000'
done
```

**Decision gate** (vs pillar3b FIXED-engine baseline: mean 17,581, P5 1,452, P10 2,377,
<1000 2.9%, <500 0.3%):

| outcome | criterion |
|---|---|
| Strong win | P10 >= 3,200 AND <1000 <= 1.8% AND mean >= 16,700 |
| Acceptable | P10 >= 2,700 AND <1000 <= 2.4% AND mean >= 16,700 |
| No-go | mean drops > 8% OR floor (P10 / <1000) doesn't improve |

Track 1 is a CALIBRATION: even "acceptable" validates the augment->distill->floor-eval
loop. If the floor improves but is undersized, go to Track 2 (the local-search judge
cleans the clustered forks -> ~2-3x more clean signal), then re-mine on this improved
policy (labels are pi-relative).
""".strip()))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

with open(OUT, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {OUT} ({os.path.getsize(OUT)} bytes, {len(cells)} cells)")
