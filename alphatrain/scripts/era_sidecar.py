"""Reconstruct per-row ERA (fresh vh3 teacher vs old) for r2_bulk.pt WITHOUT
rebuilding: replay build_round2_corpora's deterministic construction (same dir
order, same rng(0) shuffle) over an index array, then verify against the
stored strata sidecar.

    python -m alphatrain.scripts.era_sidecar
"""
import glob
import json

import numpy as np

from alphatrain.scripts.build_hardce_corpora import rows_of_game
from alphatrain.scripts.build_round2_corpora import (BULK_CRISIS, BULK_UNCAP,
                                                     BULK_CAP)

FRESH_DIRS = {'data/crisis_vh3', 'data/crisis_vh3_deep', 'data/selfplay_vh3'}


def main():
    kinds, eras = [], []
    for d in BULK_CRISIS:
        for fp in sorted(glob.glob(f'{d}/game_seed*.json')):
            g = json.load(open(fp))
            st = (('c_succ_' if g.get('capped') else 'c_fail_')
                  + g.get('label', '?'))
            n = len(rows_of_game(g, 'crisis', 'all'))
            kinds += [st] * n
            eras += [d in FRESH_DIRS] * n
    for d in BULK_UNCAP:
        for fp in sorted(glob.glob(f'{d}/game_seed*.json')):
            g = json.load(open(fp))
            n = len(rows_of_game(g, 'uncapped'))
            kinds += ['b_uncapped'] * n
            eras += [d in FRESH_DIRS] * n
    for d in BULK_CAP:
        for fp in sorted(glob.glob(f'{d}/game_seed*.json')):
            g = json.load(open(fp))
            n = len(rows_of_game(g, 'capped'))
            kinds += ['b_capped'] * n
            eras += [d in FRESH_DIRS] * n

    kinds = np.array(kinds)
    eras = np.array(eras)
    stored = np.load('alphatrain/data/r2_bulk_strata.npz')['strata'].astype(str)
    assert len(kinds) == len(stored), (len(kinds), len(stored))

    # Replay the exact shuffle: builder did rng(0); rng.shuffle(rows-list).
    idx = np.arange(len(kinds), dtype=object)  # object dtype = list-like shuffle
    rng = np.random.default_rng(0)
    lst = list(idx)
    rng.shuffle(lst)
    perm = np.array(lst, dtype=np.int64)
    assert (kinds[perm] == stored).all(), 'permutation replay FAILED'
    era_shuffled = eras[perm]
    np.savez('alphatrain/data/r2_bulk_era.npz', fresh=era_shuffled)
    print(f'era sidecar verified against strata: {int(era_shuffled.sum()):,} '
          f'fresh rows / {len(era_shuffled):,} '
          f'({100 * era_shuffled.mean():.1f}%)')


if __name__ == '__main__':
    main()
