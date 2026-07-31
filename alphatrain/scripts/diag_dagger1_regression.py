"""Diagnose the dagger1 regression: absorption vs collateral drift.

Measures, for each trained checkpoint vs vh1:
  1. ABSORPTION on the 47,308 confident correction states (disagree_mask==1):
     % now playing pillar3k's move / still vh1's move / a third move.
  2. DRIFT on held-out quiet states (broad states from the recorded games that
     were NOT selected into the corpus): argmax match vs vh1 — collateral
     damage signature if low.
  3. DRIFT on the rehearsal distribution (sample of distill_pillar3k.pt).

    python -m alphatrain.scripts.diag_dagger1_regression \
        --models alphatrain/data/small128_dagger1_e3_s400.pt \
                 alphatrain/data/small128_dagger1_epoch_2.pt
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model


def legal_argmax(net, ds, idx, dev, batch=2048):
    out = np.full(len(idx), -1, dtype=np.int64)
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        obs = ds._build_obs_core(ds.boards[b], next_pos=ds.next_pos[b],
                                 next_col=ds.next_col[b], n_next=ds.n_next[b])
        with torch.no_grad():
            lg = net(obs).float().cpu().numpy()
        bd = ds.boards[b].cpu().numpy().astype(np.int8)
        for i in range(len(b)):
            k, fi, _ = _legal_priors_jit(bd[i], lg[i], 1)
            if k > 0:
                out[s + i] = int(fi[0])
    return out


def build_holdout(games_dir, corpus_path, n_games, per_game, out_path, rng,
                  skip=0):
    corpus = torch.load(corpus_path, map_location='cpu', weights_only=False)
    seen = set()
    cb = corpus['boards'].numpy().reshape(len(corpus['boards']), 81)
    for i in range(len(cb)):
        seen.add(cb[i].tobytes())
    boards, next_pos, next_col, n_next = [], [], [], []
    files = sorted(glob.glob(os.path.join(games_dir, 'game_*.json')))
    for fp in files[skip:skip + n_games]:
        d = json.load(open(fp))
        states = [s for s in d['states']][::4]
        rng.shuffle(states)
        took = 0
        for s in states:
            b = np.array(s['board'], dtype=np.int8).reshape(81)
            if b.tobytes() in seen:
                continue
            boards.append(b.reshape(9, 9))
            npos = np.zeros((3, 2), dtype=np.int8)
            ncol = np.zeros(3, dtype=np.int8)
            nb = s['next_balls'][:s['num_next']]
            for t, x in enumerate(nb[:3]):
                npos[t] = (x['row'], x['col'])
                ncol[t] = x['color']
            next_pos.append(npos)
            next_col.append(ncol)
            n_next.append(len(nb[:3]))
            took += 1
            if took >= per_game:
                break
    n = len(boards)
    torch.save({'boards': torch.from_numpy(np.stack(boards)),
                'next_pos': torch.from_numpy(np.stack(next_pos)),
                'next_col': torch.from_numpy(np.stack(next_col)),
                'n_next': torch.tensor(n_next, dtype=torch.int8),
                'pol_indices': torch.zeros((n, 5), dtype=torch.int64),
                'pol_values': torch.zeros((n, 5), dtype=torch.float32),
                'max_score': 0.0}, out_path)
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--corpus', default='alphatrain/data/dagger_v1.pt')
    p.add_argument('--meta', default='alphatrain/data/dagger_v1_states_meta.npz')
    p.add_argument('--games-dir', default='data/dagger_games_v1')
    p.add_argument('--rehearsal', default='alphatrain/data/distill_pillar3k.pt')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(0)

    meta = np.load(a.meta)
    mask = meta['disagree'] & (meta['gap'] >= 0.5)
    corr = np.where(mask)[0]
    t_mv, s_mv = meta['teacher_move'][corr], meta['student_move'][corr]
    print(f'{len(corr):,} confident correction states', flush=True)

    ds_corr = TensorDatasetGPU(a.corpus, augment=False, color_augment=False,
                               augment_factor=1, device=a.device)
    hold_path = a.corpus + '.holdout.tmp'
    n_hold = build_holdout(a.games_dir, a.corpus, 200, 100, hold_path, rng)
    ds_hold = TensorDatasetGPU(hold_path, augment=False, color_augment=False,
                               augment_factor=1, device=a.device)
    ds_reh = TensorDatasetGPU(a.rehearsal, augment=False, color_augment=False,
                              augment_factor=1, device=a.device)
    reh_idx = torch.from_numpy(
        np.sort(rng.choice(ds_reh.boards.shape[0], 20000, replace=False))).to(dev)
    hold_idx = torch.arange(n_hold).to(dev)
    corr_idx = torch.from_numpy(corr).to(dev)
    print(f'holdout {n_hold:,} quiet states, rehearsal sample 20,000', flush=True)

    results = {}
    for path in [a.base] + a.models:
        net, _ = load_model(path, dev, fp16=False)
        name = os.path.basename(path).replace('.pt', '')
        results[name] = {
            'corr': legal_argmax(net, ds_corr, corr_idx, dev),
            'hold': legal_argmax(net, ds_hold, hold_idx, dev),
            'reh': legal_argmax(net, ds_reh, reh_idx, dev),
        }
        del net
        print(f'  forwards done: {name}', flush=True)

    base = os.path.basename(a.base).replace('.pt', '')
    b = results[base]
    print(f'\n{"model":28s} {"absorb%":>8s} {"keep_vh1%":>10s} {"third%":>7s} '
          f'{"drift_hold%":>12s} {"drift_reh%":>11s}')
    for name, r in results.items():
        absorb = 100 * (r['corr'] == t_mv).mean()
        keep = 100 * (r['corr'] == s_mv).mean()
        third = 100 - absorb - keep
        dh = 100 * (r['hold'] != b['hold']).mean()
        dr = 100 * (r['reh'] != b['reh']).mean()
        print(f'{name:28s} {absorb:8.1f} {keep:10.1f} {third:7.1f} '
              f'{dh:12.1f} {dr:11.1f}')
    os.remove(hold_path)


if __name__ == '__main__':
    main()
