"""Round-2 corpora (loop from vh3): bulk (ALL data ever generated) and
frontier (concentrated escape windows from fresh vh3 replays + broad anchor).

  bulk    : every game dir, uncapped, hard-CE format (played-move targets);
            label-quality trims only (temp moves, terminal rows of failures).
  frontier: first 96 rows of every FRESH vh3 replay (bulk @600 + deep
            @1600/2400) = the escape windows, PLUS an equal-sized broad
            anchor sampled from fresh selfplay (review: 50% anchor).

    python -m alphatrain.scripts.build_round2_corpora --which bulk
    python -m alphatrain.scripts.build_round2_corpora --which frontier
"""
import argparse
import glob
import os
import json

import numpy as np
import torch

from alphatrain.scripts.build_hardce_corpora import rows_of_game, pack

BULK_CRISIS = ['data/crisis_iter5', 'data/crisis_vh3', 'data/crisis_vh3_deep']
BULK_UNCAP = ['data/selfplay_iter5']
BULK_CAP = ['data/selfplay_iter5b', 'data/selfplay_vh3']
FRESH_CRISIS = ['data/crisis_vh3', 'data/crisis_vh3_deep']
FRESH_CAP = ['data/selfplay_vh3']


def load(dirs, kind, mode='all'):
    rows = []
    for d in dirs:
        for fp in sorted(glob.glob(f'{d}/game_seed*.json')):
            g = json.load(open(fp))
            if kind == 'crisis':
                st = (('c_succ_' if g.get('capped') else 'c_fail_')
                      + g.get('label', '?'))
                rows += [(m, st) for m in rows_of_game(g, 'crisis', mode)]
            elif kind == 'frontier':
                if g.get('capped'):
                    rows += [(m, 'f_' + g.get('label', '?'))
                             for m in g['moves'][:96]]
                elif os.environ.get('FRONTIER_FAILED') == '1':
                    # OFF by default (Gemini+ChatGPT consensus 2026-08-08:
                    # hard-CE cannot distinguish unavoidable deaths from
                    # teacher errors — failed rows need independent
                    # validation before entering a concentrated arm).
                    end = max(0, min(96, len(g['moves']) - 20))
                    rows += [(m, 'ff_' + g.get('label', '?'))
                             for m in g['moves'][:end]]
            else:
                rows += [(m, 'b_' + kind) for m in rows_of_game(g, kind)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['bulk', 'frontier'], required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    if a.which == 'bulk':
        rows = (load(BULK_CRISIS, 'crisis') + load(BULK_UNCAP, 'uncapped')
                + load(BULK_CAP, 'capped'))
        out_path = 'alphatrain/data/r2_bulk.pt'
    else:
        fr = load(FRESH_CRISIS, 'frontier')
        anchor = load(FRESH_CAP, 'capped')
        take = rng.choice(len(anchor), min(len(fr), len(anchor)), replace=False)
        rows = fr + [anchor[i] for i in take]
        out_path = 'alphatrain/data/r2_frontier.pt'
    rng.shuffle(rows)
    out = pack(rows)
    strata = out.pop('strata')
    np.savez(out_path.replace('.pt', '_strata.npz'), strata=strata)
    torch.save(out, out_path)
    t = torch.load(out_path, weights_only=True)
    from collections import Counter
    print(f'{out_path}: {t["boards"].shape[0]:,} rows; weights_only OK; '
          f'{dict(Counter(strata.tolist()))}')


if __name__ == '__main__':
    main()
