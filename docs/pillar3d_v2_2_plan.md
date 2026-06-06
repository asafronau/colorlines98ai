# pillar3d-v2.2 — data-scaling iteration (training note)

**Date:** 2026-06-06. **Run:** `alphatrain/train_pillar3d_v2_2_colab.ipynb` (gen by
`scripts/gen_pillar3d_v2_2_notebook.py`).

## Goal — two things at once
1. **Push pillar3b's floor further** (the deployable deliverable).
2. **Measure how pillar3d reacts to MORE crisis data** — the data-scaling curve. This is the
   *repeating pattern*: mine more crises → rebuild corpus → retrain → compare. v2.1 was iteration #1;
   v2.2 is #2.

## What changed: only the corpus
Rebuilt `crisis/corrections_corpus.pt` from all `crisis/corrections/corr_*.json`:

| | games | anchors | size |
|---|---|---|---|
| v2.1 corpus | 956 | 16,547 | 5.4 MB |
| **v2.2 corpus (new)** | **1,520** | **26,310** | **9.0 MB** |

`+59%` more games/anchors (new crises mined on fresh seeds, incl. corr_50xxx/51xxx). Weight
(margin, mean 1): P10 0.07, P50 0.49, P90 2.53, max 11.2, 26% near-zero. Old corpus backed up to
`crisis/corrections_corpus_956.pt.bak`. Rebuild command:

```
PYTHONPATH=. python scripts/build_corrections_corpus.py \
    --glob 'crisis/corrections/corr_*.json' --out crisis/corrections_corpus.pt
```

## Recipe — byte-identical to v2.1 (the only deliberate change is the corpus)
Warm-start `pillar3b_epoch_20.pt`, `train_path_b.py`, `--lr 5e-5` (gentle throughout, no 3e-4 phase),
`--target-temperature 0.5`, `--aux-lambda 0.03`, `--aux-target-temperature 0.5`, 10 epochs,
`--aux-holdout-frac 0.15 --aux-split-seed 0`. (Holding the recipe fixed is what makes this a clean
*data*-scaling measurement.)

## To run
1. **Re-upload `corrections_corpus.pt` (9 MB) to `MyDrive/alphatrain/`** — the only real change.
2. Code tarball `colorlines_pillar3d_v2.tar.gz` + `pillar3b_epoch_20.pt` + `v13_pillar3a.pt.gz`
   already on Drive (re-upload code only if unsure it matches current `train_path_b.py`).
3. Run the notebook. ~12h. Sweep epochs 3/5/7/10 on the 777k held-out.

## The read (baselines on the SAME 777k held-out, 777000..778999)
- **control (pillar3b):** mean 17,581, P10 2,377, P5 1,452, <1000 2.9%
- **v2.1 (deployed, the bar):** mean 20,609 (+17.2%), P10 2,736, <1000 2.5% — `<500` tail untouched

Verdict logic:
- **Beats v2.1** (mean/P10 up, <1000 down, esp. **<500 moves**) → more data still helps → keep
  the mine→rebuild→retrain loop.
- **Plateaus** → saturation is a *method* limit, not data → next lever is the teacher/objective
  (action-risk teacher, MCTS-relabel), not more corpus.
- **Regresses** → marginal-data dilution → raise `--min-margin` on the corpus or lower `--aux-lambda`.

Record the full epoch curve in HISTORY.md as data-scaling iteration #2.

## Related
- Pipeline: [[project_crisis_pipeline]]. Saturation note this tests: corpus was 16.5k-SATURATED.
- The GPU mining-throughput track (separate) is parked at the closed-loop-caching decision; the CPU
  miner remains production and is what generated these new crises.
