"""Build H70/H44 hard-CE corpora with the dual-review sampling hygiene.

Crisis rows (per replay): first 128 decisions post-anchor if the replay
SURVIVED to the cap ("capped": true = successful escape), first
min(96, len-20) if it died (keep difficult prefixes, drop the terminal tail).
Prevention/recovery kept stratified via a per-row strata array.
Broad rows: uncapped selfplay capped at 10,000 rows/game (anti-outlier);
capped 1k-turn games in full; first 30 temperature-exploration moves of
every selfplay game masked out.
Targets stay visit tensors — training uses --blend-alpha 0 (hard CE on the
target argmax = the played move at temperature 0).
Mixes: H70 = 70/30 crisis/broad, H44 = 44/56 (both capped by pool sizes).

    python -m alphatrain.scripts.build_hardce_corpora
"""
import glob
import json
import os

import numpy as np
import torch

CRISIS_DIRS = ['data/crisis_iter5']          # A+B era replays (same teacher)
BROAD_UNCAPPED = 'data/selfplay_iter5'
BROAD_CAPPED = 'data/selfplay_iter5b'
TEMP_MOVES = 30
ROW_B = {'board': None}  # placeholder


def rows_of_game(d, kind):
    mv = d['moves']
    if kind == 'crisis':
        if d.get('capped', False):
            keep = range(0, min(128, len(mv)))
        else:
            keep = range(0, max(0, min(96, len(mv) - 20)))
    elif kind == 'uncapped':
        idx = np.arange(TEMP_MOVES, len(mv))
        if len(idx) > 10000:
            idx = np.sort(np.random.default_rng(d['seed']).choice(
                idx, 10000, replace=False))
        keep = idx
    else:  # capped broad
        keep = range(TEMP_MOVES, len(mv))
    return [mv[i] for i in keep]


def pack(rows_meta):
    n = len(rows_meta)
    out = {
        'boards': torch.zeros((n, 9, 9), dtype=torch.int8),
        'next_pos': torch.zeros((n, 3, 2), dtype=torch.int8),
        'next_col': torch.zeros((n, 3), dtype=torch.int8),
        'n_next': torch.zeros(n, dtype=torch.int8),
        'pol_indices': torch.zeros((n, 5), dtype=torch.int64),
        'pol_values': torch.zeros((n, 5), dtype=torch.float32),
        'pol_nnz': torch.zeros(n, dtype=torch.int64),
    }
    strata = []
    for i, (m, st) in enumerate(rows_meta):
        out['boards'][i] = torch.tensor(m['board'], dtype=torch.int8)
        nb = m['next_balls'][:m['num_next']]
        for t, b in enumerate(nb[:3]):
            out['next_pos'][i, t, 0] = b['row']
            out['next_pos'][i, t, 1] = b['col']
            out['next_col'][i, t] = b['color']
        out['n_next'][i] = len(nb[:3])
        cm = m['chosen_move']
        flat = ((cm['sr'] * 9 + cm['sc']) * 81 + cm['tr'] * 9 + cm['tc'])
        k = min(5, len(m['cand_moves']))
        vis = np.array(m['cand_visits'][:k], dtype=np.float32)
        idxs = list(m['cand_moves'][:k])
        if flat in idxs:  # ensure argmax(pol_values) == played move
            j = idxs.index(flat)
            vis[j] = vis.max() + 1.0
        else:
            idxs = [flat] + idxs[:4]
            vis = np.concatenate([[vis.max() + 1.0 if len(vis) else 1.0],
                                  vis[:4]])
            k = len(idxs)
        s = vis.sum()
        out['pol_indices'][i, :k] = torch.tensor(idxs, dtype=torch.int64)
        out['pol_values'][i, :k] = torch.tensor(vis / max(s, 1e-8))
        out['pol_nnz'][i] = k
        strata.append(st)
    out['strata'] = np.array(strata)
    out.update({'num_channels': 18, 'max_score': 0.0,
                'value_mode': 'policy_slim', 'gamma': 0.99,
                'num_value_bins': 64})
    return out


def main():
    rng = np.random.default_rng(0)
    crisis, broad = [], []
    for d in CRISIS_DIRS:
        for fp in sorted(glob.glob(f'{d}/game_seed*.json')):
            g = json.load(open(fp))
            st = ('c_succ_' if g.get('capped') else 'c_fail_') + g.get('label', '?')
            crisis += [(m, st) for m in rows_of_game(g, 'crisis')]
    for fp in sorted(glob.glob(f'{BROAD_UNCAPPED}/game_seed*.json')):
        g = json.load(open(fp))
        broad += [(m, 'b_uncap') for m in rows_of_game(g, 'uncapped')]
    for fp in sorted(glob.glob(f'{BROAD_CAPPED}/game_seed*.json')):
        g = json.load(open(fp))
        broad += [(m, 'b_cap') for m in rows_of_game(g, 'capped')]
    print(f'pools: crisis={len(crisis):,}  broad={len(broad):,}')

    for name, cfrac in (('h70', 0.70), ('h44', 0.44)):
        n_c = len(crisis)
        n_b = min(len(broad), int(n_c * (1 - cfrac) / cfrac))
        if n_b == len(broad):  # broad-limited: shrink crisis instead
            n_c = min(n_c, int(len(broad) * cfrac / (1 - cfrac)))
        ci = rng.choice(len(crisis), n_c, replace=False)
        bi = rng.choice(len(broad), n_b, replace=False)
        rows = [crisis[i] for i in ci] + [broad[i] for i in bi]
        rng.shuffle(rows)
        out = pack(rows)
        path = f'alphatrain/data/{name}.pt'
        torch.save(out, path)
        from collections import Counter
        cnt = Counter(out['strata'].tolist())
        print(f'{path}: {len(rows):,} rows ({100*n_c/len(rows):.0f}% crisis) '
              f'| {dict(cnt)}')


if __name__ == '__main__':
    main()
