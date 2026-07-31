"""Build survive_H value targets from SLIM recorder games (eval --record-dir).

Same output tensor format + label/censoring semantics as build_value_targets
(labels from remaining = final_turns - state.turn; capped games censored past
the cap), but reads the slim states-schema and splits by SEED RANGE for the
disjoint-split protocol (R2 review): train < --val-min-seed; validation in
[--val-min-seed, --val-max-seed); seeds >= --val-max-seed are SKIPPED entirely
(reserved for the decision judge).

    python -m alphatrain.scripts.build_value_targets_slim \
        --games-dir data/dagger_games_v2 \
        --output alphatrain/data/value_targets_vh5x.pt \
        --val-min-seed 886000 --val-max-seed 888000
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from alphatrain.value_head import SURVIVAL_HORIZONS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--games-dir', default='data/dagger_games_v2')
    p.add_argument('--output', required=True)
    p.add_argument('--val-min-seed', type=int, default=886000)
    p.add_argument('--val-max-seed', type=int, default=888000)
    p.add_argument('--broad-keep', type=float, default=1.0,
                   help='subsample broad (non-death-band) states')
    a = p.parse_args()
    rng = np.random.default_rng(0)
    H = list(SURVIVAL_HORIZONS)

    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json')))
    print(f'{len(files)} games in {a.games_dir}; horizons {H}', flush=True)
    boards, npos, ncol, nn = [], [], [], []
    labels, masks, is_train, seeds = [], [], [], []
    t0 = time.time()
    kept = skipped = 0
    for gi, fp in enumerate(files):
        d = json.load(open(fp))
        seed = int(d['seed'])
        if seed >= a.val_max_seed:
            skipped += 1
            continue
        capped = not d['died']
        tail_start = d['final_turns'] - d['record_tail']
        for s in d['states']:
            if s['turn'] < tail_start and rng.random() > a.broad_keep:
                continue
            remaining = d['final_turns'] - s['turn']
            lab = np.zeros(len(H), dtype=np.int8)
            msk = np.ones(len(H), dtype=np.int8)
            for hi, h in enumerate(H):
                if remaining >= h:
                    lab[hi] = 1
                elif capped:
                    msk[hi] = 0
            boards.append(np.array(s['board'], dtype=np.int8))
            pp = np.zeros((3, 2), dtype=np.int8)
            cc = np.zeros(3, dtype=np.int8)
            nb = s['next_balls'][:s['num_next']]
            for t, x in enumerate(nb[:3]):
                pp[t] = (x['row'], x['col'])
                cc[t] = x['color']
            npos.append(pp)
            ncol.append(cc)
            nn.append(len(nb[:3]))
            labels.append(lab)
            masks.append(msk)
            is_train.append(seed < a.val_min_seed)
            seeds.append(seed)
            kept += 1
        if (gi + 1) % 2000 == 0:
            print(f'  {gi + 1}/{len(files)} games, {kept:,} rows '
                  f'({time.time() - t0:.0f}s)', flush=True)

    out = {
        'boards': torch.from_numpy(np.stack(boards)),
        'next_pos': torch.from_numpy(np.stack(npos)),
        'next_col': torch.from_numpy(np.stack(ncol)),
        'n_next': torch.tensor(nn, dtype=torch.int8),
        'survive_labels': torch.from_numpy(np.stack(labels)),
        'survive_masks': torch.from_numpy(np.stack(masks)),
        'is_train': torch.tensor(is_train, dtype=torch.bool),
        'seed': torch.tensor(seeds, dtype=torch.int64),
        'horizons': H,
    }
    torch.save(out, a.output)
    it = out['is_train']
    lab = out['survive_labels'].float()
    msk = out['survive_masks'].float()
    pos_rate = (lab * msk).sum(0) / msk.sum(0)
    print(f'\n{a.output}: {kept:,} rows ({int(it.sum()):,} train / '
          f'{int((~it).sum()):,} val), {skipped} judge-reserved games skipped')
    print('per-horizon survive rate (masked): '
          + ', '.join(f'H{h}={r:.3f}' for h, r in zip(H, pos_rate.tolist())))
    print('censored frac per horizon: '
          + ', '.join(f'H{h}={1 - r:.3f}'
                      for h, r in zip(H, (msk.mean(0)).tolist())))


if __name__ == '__main__':
    main()
