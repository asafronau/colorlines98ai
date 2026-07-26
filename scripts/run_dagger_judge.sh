#!/bin/bash
# Phase 0b (redesigned per ChatGPT review): judge pillar3k's move vs vh1's move
# on the 2,135 TRUE student-visited disagreement anchors, under 7 continuation
# conditions. Common (state, rep) seeds pair all arms and all conditions.
set -e
cd "$(dirname "$0")/../alphatrain/inference_cpp"
D=data
J=build/rollout_judge
S=$D/dagger_judge_states.bin
VH1=$D/vh1_policy_ts.pt
P3K=$D/pillar3k_ep22_policy_ts.pt

echo "=== condition S: student continuation (PRIMARY) ==="
$J --states $S --model $VH1 --out $D/dagger_S.csv

echo "=== condition T: teacher continuation ==="
$J --states $S --model $P3K --out $D/dagger_T.csv

for L in 1 2 4 8 16; do
  echo "=== condition L$L: teacher burst $L then student ==="
  $J --states $S --model $VH1 --burst-model $P3K --burst-len $L \
     --out $D/dagger_L$L.csv
done
echo "ALL CONDITIONS DONE"
