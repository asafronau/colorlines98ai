#!/bin/bash
# Born-again corpus (HISTORY 272): every row labeled with the TEACHER's 8-view-averaged policy (top-5 soft), for a
# from-scratch student of the same size. Sources: the teacher's own recorded games (already labeled by
# scripts/tta_distill_turn.sh step 1), the states of the crisis / slide corpora, and human-level (epoch-4) game
# states, relabeled (relabel_tta_tensor: mean of the 8 views' logits) -> merge -> gzip -> Colab notebook.
#   caffeinate -is scripts/prep_born_again.sh <teacher.pt> <own_tta_tensor.pt> <tag> <tensor> [<tensor> ...]
set -euo pipefail
cd "$(dirname "$0")/.."
TEACHER=$1; OWN=$2; TAG=$3; shift 3
D=alphatrain/data; PY=.venv/bin/python
PARTS=("$OWN")
for src in "$@"; do
  name=$(basename "$src" .pt)
  out=$D/ba_${TAG}_$name.pt
  if [ ! -f "$out" ]; then
    $PY -m alphatrain.scripts.relabel_tta_tensor --tensor "$src" --model "$TEACHER" --out $D/ba_${TAG}_$name.npz \
        > logs/ba_${TAG}_relabel_$name.log 2>&1
    $PY -m alphatrain.scripts.build_tta_r2_corpus --src "$src" --sidecar $D/ba_${TAG}_$name.npz --mode tta --out "$out" | tail -n 1
  fi
  PARTS+=("$out")
done
$PY -m alphatrain.scripts.merge_policy_tensors --out $D/born_again_${TAG}.pt "${PARTS[@]}"
gzip -k -1 -f $D/born_again_${TAG}.pt
ls -la $D/born_again_${TAG}.pt $D/born_again_${TAG}.pt.gz
