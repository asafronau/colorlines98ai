#!/bin/bash
# Round-2 merge sweep: build 8 merge candidates from the downloaded Colab
# epochs, export TS, screen each on dev slice 775000-775999 (1k, screen
# resolution +-1100 mean; vh3 same-slice baseline mean=14429 P50=10334).
set -e
cd /path/to/colorlines98
source .venv/bin/activate
mkdir -p checkpoints/r2m alphatrain/inference_cpp/data/r2m logs/r2m

DG1=alphatrain/data/r2bulk_dg_ckpts_epoch_1.pt
DG3=alphatrain/data/r2bulk_dg_ckpts_epoch_3.pt
F12=alphatrain/data/r2frontier_ckpts_epoch_12.pt
BASE=alphatrain/data/small128_vh3.pt

merge_one() {  # name, then --add args...
    local name=$1; shift
    python -m alphatrain.scripts.merge_vectors --base $BASE "$@" \
        --out checkpoints/r2m/$name.pt
    python -m alphatrain.inference_cpp.export_ts \
        --model checkpoints/r2m/$name.pt \
        --outdir alphatrain/inference_cpp/data/r2m/$name
    cp alphatrain/inference_cpp/data/r2m/$name/policy_ts.pt \
       alphatrain/inference_cpp/data/r2m/${name}_ts.pt
}

merge_one dg1_a03 --add 0.3 $DG1
merge_one dg1_a05 --add 0.5 $DG1
merge_one dg1_a07 --add 0.7 $DG1
merge_one dg3_a03 --add 0.3 $DG3
merge_one dg3_a05 --add 0.5 $DG3
merge_one dg3_a07 --add 0.7 $DG3
merge_one f12_b02 --add 0.2 $F12
merge_one dg1a05_f12b01 --add 0.5 $DG1 --add 0.1 $F12
echo "=== all 8 merges built + exported ==="

cd alphatrain/inference_cpp
for name in dg1_a03 dg1_a05 dg1_a07 dg3_a03 dg3_a05 dg3_a07 f12_b02 dg1a05_f12b01; do
    echo "=== screening $name ($(date +%H:%M:%S)) ==="
    ./build/eval --model data/r2m/${name}_ts.pt --device mps --batch 500 \
        --seed-start 775000 --seed-end 775999 \
        --scores-out ../../logs/r2m/${name}_slice1k.csv
done
echo "R2_MERGE_SCREEN_DONE"
