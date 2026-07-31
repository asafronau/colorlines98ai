# The small-model improvement loop — per-iteration checklist

Validated end-to-end 2026-07-09 (HISTORY 177: `small128_vh1` > ep87 on the 5k
gold standard). Each iteration takes a **base model** `M_k` and produces
`M_{k+1}`. Generation is all C++; training is Colab; the M5 is the generation
and eval box.

## 0. Promote + prepare (minutes, M5)
```bash
cp <winning ckpt>.pt alphatrain/data/small128_vh{k}.pt
# retrain the survival head ON THIS backbone (HISTORY 158: never reuse heads)
python -m alphatrain.scripts.train_value_head \
    --backbone alphatrain/data/small128_vh{k}.pt \
    --train-data alphatrain/data/value_targets_small128.pt \
    --val-data alphatrain/data/value_val_small128_K64_died.pt \
    --epochs 5 --batch-size 4096 --lr 1e-3 \
    --out alphatrain/data/value_head_small128_vh{k}.pt
# gate: death-balanced calibration r >= 0.75 at H25-H100
# (optional, better): rebuild labels + val set from the NEWEST games first
python -m alphatrain.inference_cpp.export_ts --model alphatrain/data/small128_vh{k}.pt
python -m alphatrain.inference_cpp.export_policy_value \
    --model alphatrain/data/small128_vh{k}.pt \
    --head alphatrain/data/value_head_small128_vh{k}.pt
# VERIFY: logit-diff exported-vs-checkpoint == 0 (the stale-export rule)
```

## 1. Mine the new base's own crises (hours-overnight, M5)
```bash
cd alphatrain/inference_cpp
caffeinate -dim ./build/mcts_crisis --model data/policy_ts.pt \
    --value-module data/policy_value_ts.pt \
    --seed-start <fresh range> --seed-end <+2000> \
    --recovery-turns 15 --recovery-sims 1200 \
    --prevention-turns 30 --prevention-sims 800 \
    --continue-turns 500 --q-weight 2.0 \
    --policy-max-turns 12000 --threads 14 \
    --out-dir ../../data/crisis_vh128_v{k}
```
Resume-safe; ~1M+ states per 2,000 probes. 600/400 sims also proven (cheaper);
1200/800 sits inside the historical gain zone.

## 2. Corpus (minutes, M5)
```bash
python -m alphatrain.scripts.build_expert_v2_tensor \
    --games-dir data/crisis_vh128_v{k} --policy-only-data \
    --output alphatrain/data/iter{k}_crisis.pt
PYTHONPATH=. python -m alphatrain.scripts.mix_tensors \
    --main alphatrain/data/iter{k}_crisis.pt \
    --rehearsal alphatrain/data/distill_pillar3k.pt \
    --rehearsal-ratio 3.0 --output alphatrain/data/iter{k}_mix.pt
# gate-3 proven mix = 25% signal (ratio 3.0); 39% passed iter-2's gates too
```

## 3. Train (Colab, ~1h A100)
Gate-3 recipe, unchanged until a gate says otherwise:
```
train_path_b --resume small128_vh{k}.pt --warm-start --channels 128 --seed 42
  --epochs 4 --batch-size 4096 --lr 1e-4 --warmup-epochs 1
  --target-temperature 1.0 --decisiveness-power 0 --blend-alpha 0.5
```
Per-epoch Drive saves. (Notebook template: train_small128_iter2_colab.ipynb.)

## 4. Gate (M5, C++ eval — Python is retired for eval)
1. **ep1 first**, 500 seeds 775000-775500, **floor-first** vs M_k's bar.
   ep1 regressing ⇒ recipe/corpus bug — STOP (healthy runs improve by ep1).
2. Pick by floor across epochs (winners are EARLY epochs).
3. **5k confirm** (775000-780000) — 500 seeds mislead close calls
   (gate-3: −6% median at 500, +4.6% at 5k).
4. Clear win ⇒ `small128_vh{k+1}`; HISTORY entry with exact commands
   (standing rule); loop to step 0.

## Failure playbook (measured, not guessed)
- ep1 regression → rollout-judge a correction sample (`export_judge_states` +
  `rollout_judge`): neutral/phantom corrections = generation problem;
  genuine corrections + regression = training problem.
- Value-head calibration drop → rebuild labels from newer games.
- Close/ambiguous evals → remember σ_train ≈ 1k unseeded (we seed now) and
  NEVER compare per-seed across models — distributions only.
