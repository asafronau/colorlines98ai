"""Generate the pillar3e TRUST-REGION arm-A notebooks (ChatGPT's structural recipe).

Arm A: the 4800-sim corrections are the PRIMARY teacher (weighted, sharpened soft-CE on the
visit distribution), and general play is held by a KL(frozen pillar3b ‖ student) trust region
on broad V13 states — replacing the aux-exception-list channel (pillar3d), whose λ was
empirically flat and whose stale main teacher drifted every epoch past ep2.

β arms target teacher:anchor GRADIENT ratios ~3:1 / 1:1 / 1:3. The M5 smoke run measured
|g_teacher|≈5-12, |g_anchor|≈0.3-0.4 (anchor grows as the student drifts), so β ≈ 5 / 15 / 45.
--grad-audit-every prints the realized share during training — read it, don't trust the nominal β.

Eval: scripts.eval_policy (fp16, batch 256) on the CANONICAL 775000..779999. Judge on
MEDIAN + P10/P25 (+ mean); <1000 = no-regress guardrail. Re-establish the bar by evaluating
the old best (pillar3d_mC ep2) once under this exact config.
"""
import os, json

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'alphatrain')


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}


def build(suf, beta, epochs=120, save_every=10, eval_eps=(60, 120)):
    cells = []
    cells.append(md(f"""
# Pillar3e trust-region run **{suf}**  (β={beta}, arm A: visit-softmax teacher + KL anchor)

`loss = teacher_CE(4800-sim corrections) + β·KL(pillar3b ‖ student)` on broad V13 states.
Corpus: **DECISIVE 22.7k (3,042 games, min_margin 0.05)**. Teacher recipe otherwise = mC keeper
(T=0.5 sharpening, margin-weighted, by-seed holdout 0.15/seed 0).

**This run is SMALL.** An epoch = one pass over 22.7k corrections = ~23 steps; {epochs} epochs
≈ 2.8k small-batch steps total — about the compute of ONE pillar3b epoch. Wall estimate:
**~30-40 min on A100, ~1h on G4** for the whole training. The 5k evals are the expensive part
(~1h each) — eval ep {'/'.join(str(e) for e in eval_eps)} only, refine around the winner later.

Watch in the train log: **HELD match** must climb (generalization), **val V13ce / KLvsFrozen**
must stay pinned (the trust region working), and **[grad step]** lines show the realized
anchor share β|g_a|/|g_t| (target ≈ {'0.33' if beta == 5 else '1.0' if beta == 15 else '3.0'}).

**Upload to `MyDrive/alphatrain/`:** `colorlines_pillar3d_v2.tar.gz` (REBUILT — must contain
`alphatrain/train_trust_region.py` + top-level `scripts/`), **`corrections_corpus_mm05.pt`**
(the 22.7k / 3,042-game build), `v13_pillar3a.pt.gz`, `pillar3b_epoch_20.pt`.
""".strip()))
    cells.append(code("from google.colab import drive\ndrive.mount('/content/drive')"))
    cells.append(code(r"""
import os, shutil, time
DRIVE = '/content/drive/MyDrive/alphatrain'
!cp {DRIVE}/colorlines_pillar3d_v2.tar.gz /content/
!cd /content && tar xzf colorlines_pillar3d_v2.tar.gz
assert os.path.exists('/content/alphatrain/train_trust_region.py'), \
    "stale tarball — rebuild with alphatrain/train_trust_region.py and re-upload"
assert os.path.exists('/content/scripts/eval_policy.py'), \
    "stale tarball — must include top-level scripts/"
os.makedirs('/content/alphatrain/data', exist_ok=True); os.makedirs('/content/crisis', exist_ok=True)
shutil.copy(f'{DRIVE}/pillar3b_epoch_20.pt', '/content/alphatrain/data/pillar3b_epoch_20.pt')
shutil.copy(f'{DRIVE}/corrections_corpus_mm05.pt', '/content/crisis/corrections_corpus_mm05.pt')
import torch
_c = torch.load('/content/crisis/corrections_corpus_mm05.pt')
print('corpus anchors', _c['boards'].shape[0], _c['_stats'])
assert 20000 < _c['boards'].shape[0] < 60000 and _c['_stats']['min_margin'] >= 0.05, \
    "unexpected corpus — upload the 22.7k corrections_corpus_mm05.pt (3,042 games)"

# V13: copy .gz local FIRST + verify sizes (Drive FUSE can silently truncate
# streaming reads of files this size — pillar3b notebook lesson).
t0 = time.time()
!cp {DRIVE}/v13_pillar3a.pt.gz /content/v13_pillar3a.pt.gz
gz_size = os.path.getsize('/content/v13_pillar3a.pt.gz')
EXPECTED_GZ = 642_409_267
assert gz_size == EXPECTED_GZ, (
    f'.gz on Drive is truncated! got {gz_size} expected {EXPECTED_GZ}. Re-upload.')
!gunzip -t /content/v13_pillar3a.pt.gz && echo '.gz integrity OK'
!gzip -dc /content/v13_pillar3a.pt.gz > /content/alphatrain/data/v13_pillar3a.pt
pt_size = os.path.getsize('/content/alphatrain/data/v13_pillar3a.pt')
EXPECTED_PT = 5_433_958_495
assert pt_size == EXPECTED_PT, f'.pt size wrong! got {pt_size} expected {EXPECTED_PT}.'
print(f'V13 tensor: {pt_size/1e9:.2f} GB ({time.time()-t0:.0f}s)')
!rm /content/v13_pillar3a.pt.gz
%cd /content
!pip install -q numpy numba scipy
""".strip()))
    cells.append(code("import torch\nprint(f'PyTorch: {torch.__version__}')\n"
                      "print('CUDA:', torch.cuda.get_device_name(0) if "
                      "torch.cuda.is_available() else 'NONE')"))
    cells.append(md(f"## Train {suf}  (~30-40 min A100 / ~1h G4 total)"))
    cells.append(code(f"""
%cd /content
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m alphatrain.train_trust_region \\
    --corrections-corpus crisis/corrections_corpus_mm05.pt \\
    --anchor-tensor alphatrain/data/v13_pillar3a.pt \\
    --frozen alphatrain/data/pillar3b_epoch_20.pt \\
    --beta {beta} --epochs {epochs} --lr 5e-5 --amp \\
    --teacher-batch-size 1024 --anchor-batch-size 4096 \\
    --target-temperature 0.5 --weighted \\
    --holdout-frac 0.15 --split-seed 0 \\
    --grad-audit-every 100 --save-every {save_every} \\
    --copy-to /content/drive/MyDrive/alphatrain/pillar3e_{suf}_best.pt \\
    --save-dir /content/checkpoints/pillar3e_{suf} 2>&1 | tee /content/pillar3e_{suf}_train.log
""".strip()))
    cells.append(code(f"""
import shutil
shutil.copy('/content/pillar3e_{suf}_train.log',
            '/content/drive/MyDrive/alphatrain/pillar3e_{suf}_train.log')
""".strip()))
    cells.append(md(r"""
## Eval — `scripts.eval_policy` on the CANONICAL 5k (775000..779999, fp16, batch 256)
Judge on **MEDIAN (P50) + P10/P25** (+ mean); <1000 = no-regress guardrail. A crisis teacher
shows up in the lower quantiles first. Compare against the old best (pillar3d_mC ep2) evaluated
ONCE under this same config — different (tool, list, batch) numbers are NOT comparable.
""".strip()))
    cells.append(code(f"""
%cd /content
import os
for EP in {list(eval_eps)}:
    m = f'/content/checkpoints/pillar3e_{suf}/epoch_{{EP}}.pt'
    if not os.path.exists(m): continue
    print(f'===== {suf} epoch {{EP}}  (5k 775000..779999) =====', flush=True)
    !python -m scripts.eval_policy --model {{m}} \\
        --seed-start 775000 --seed-end 779999 --device cuda --batch 256
""".strip()))
    out = os.path.join(DIR, f'train_pillar3e_{suf}_colab.ipynb')
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
          "kernelspec": {"display_name": "Python 3", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
    json.dump(nb, open(out, 'w'), indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == '__main__':
    build('trA_b5',  beta=5)    # teacher:anchor ~3:1 — teacher-dominant
    build('trA_b15', beta=15)   # ~1:1 — balanced
    build('trA_b45', beta=45)   # ~1:3 — anchor-dominant (conservative)
