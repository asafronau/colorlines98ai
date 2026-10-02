"""Dense danger targets for the 4-horizon value head (HISTORY 269): label_H = 1 iff the game survives the next H
moves AND the board stays below --calm-balls balls over them ("calm"), from games recorded by `eval --record-dir`
(states every --record-every moves + the last record_tail moves; the future maximum is taken over the recorded
states, so a spike shorter than the record stride can be missed -- slides last 60-120 moves). Censoring: in a
capped game a horizon that runs past the cap is masked unless the board already reached the threshold.
Same schema as build_value_targets_from_records (target_type 'survival'), so train_value_head, the fused
policy+value export and the C++ search use it unchanged.

    python -m alphatrain.scripts.build_calm_targets_from_records --calm-balls 52 \
        --games-dir alphatrain/data/greedy_A8_cap100k --output alphatrain/data/calm52_targets_A8.pt
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '.')
from alphatrain.value_head import SURVIVAL_HORIZONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games-dir', required=True, nargs='+')
    ap.add_argument('--output', required=True)
    ap.add_argument('--calm-balls', type=int, required=True)
    ap.add_argument('--every', type=int, default=8, help='keep every Nth mid-game state (the tail is kept whole)')
    ap.add_argument('--tail', type=int, default=300)
    ap.add_argument('--val-frac', type=float, default=0.1)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    H = np.array(SURVIVAL_HORIZONS)
    files = sorted(f for d in a.games_dir for f in glob.glob(os.path.join(d, 'game_seed*.json')))
    B, NP, NC, NN, L, M, TR = [], [], [], [], [], [], []
    for fi, f in enumerate(files):
        g = json.load(open(f))
        T, died = g['final_turns'], g['died']
        st = sorted(g['states'], key=lambda s: s['turn'])
        turns = np.array([s['turn'] for s in st])
        balls = np.array([int(np.count_nonzero(np.array(s['board']))) for s in st])
        is_train = rng.random() >= a.val_frac
        for i, s in enumerate(st):
            t = s['turn']
            if t < T - a.tail and t % a.every:
                continue
            lab = np.zeros(len(H), np.int8); msk = np.ones(len(H), np.int8)
            for j, h in enumerate(H):
                end = np.searchsorted(turns, t + h, side='right')
                hot = balls[i:end].max() >= a.calm_balls
                if died and T - t <= h:
                    lab[j] = 0                      # dies within h
                elif hot:
                    lab[j] = 0                      # the board reaches the threshold within h
                elif not died and t + h > T:
                    msk[j] = 0                      # runs past the cap, threshold not reached yet: unknown
                else:
                    lab[j] = 1
            B.append(np.array(s['board'], np.int8)); nb = s['next_balls']; k = min(len(nb), 3)
            npos = np.zeros((3, 2), np.int8); ncol = np.zeros(3, np.int8)
            for q in range(k):
                npos[q] = (nb[q]['row'], nb[q]['col']); ncol[q] = nb[q]['color']
            NP.append(npos); NC.append(ncol); NN.append(k); L.append(lab); M.append(msk); TR.append(is_train)
        if (fi + 1) % 200 == 0:
            print(f'  {fi + 1}/{len(files)} games, {len(B):,} states', flush=True)
    L = np.stack(L); M = np.stack(M)
    out = {'boards': torch.from_numpy(np.stack(B)), 'next_pos': torch.from_numpy(np.stack(NP)),
           'next_col': torch.from_numpy(np.stack(NC)), 'n_next': torch.from_numpy(np.array(NN, np.int8)),
           'survive_labels': torch.from_numpy(L), 'survive_masks': torch.from_numpy(M),
           'is_train': torch.from_numpy(np.array(TR, bool)), 'horizons': list(SURVIVAL_HORIZONS),
           'target_type': 'survival', 'calm_balls': a.calm_balls}
    torch.save(out, a.output)
    rate = [(L[:, j][M[:, j] == 1] == 1).mean() for j in range(len(H))]
    print(f'wrote {a.output}: {len(B):,} states from {len(files)} games; calm < {a.calm_balls} balls: P(calm H) '
          f'{[round(float(r), 4) for r in rate]} for H {list(H)}; masked {100 * (1 - M.mean()):.1f}%')


if __name__ == '__main__':
    main()
