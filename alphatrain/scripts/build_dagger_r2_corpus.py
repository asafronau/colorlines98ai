"""Build the round-2 corrections corpus: judged-domain gap>=1.0 rows only.

Per the R2 review + rowjudge validation (HISTORY 183): keep ONLY the 25,116
recovery/prevention disagreements with teacher logit gap >= 1.0 (+2.4..+2.8pp
row-validated); drop the 42% dead weight (gap<1.0: +0.1..+0.8pp, CIs cross 0).
Output = scripts/train_crisis_ft.py corpus format, plus `vh1_move` for the
pairwise-margin loss.

    python -m alphatrain.scripts.build_dagger_r2_corpus \
        --output alphatrain/data/dagger_r2_gap1.pt
"""
import argparse

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', default='alphatrain/data/dagger_v1.pt')
    p.add_argument('--meta', default='alphatrain/data/dagger_v1_states_meta.npz')
    p.add_argument('--output', default='alphatrain/data/dagger_r2_gap1.pt')
    a = p.parse_args()

    meta = np.load(a.meta)
    keep = (meta['disagree'] & (meta['gap'] >= 1.0)
            & (meta['band'] != 'broad'))
    idx = torch.from_numpy(np.where(keep)[0])
    c = torch.load(a.corpus, map_location='cpu', weights_only=False)
    seeds = torch.from_numpy(meta['seed'][keep.nonzero()[0]].astype(np.int64))
    out = {
        'boards': c['boards'][idx],
        'next_pos': c['next_pos'][idx],
        'next_col': c['next_col'][idx],
        'n_next': c['n_next'][idx],
        'tgt_idx': c['pol_indices'][idx],       # argmax-first (distill_relabel)
        'tgt_prob': c['pol_values'][idx],
        'vh1_move': torch.from_numpy(
            meta['student_move'][keep.nonzero()[0]].astype(np.int64)),
        'weight': torch.ones(len(idx)),
        'seed': seeds,
        'gap': torch.from_numpy(meta['gap'][keep.nonzero()[0]]),
        '_stats': {'n_seeds': len(set(seeds.tolist())),
                   'min_margin': 'teacher_gap>=1.0, recovery+prevention only'},
    }
    assert (out['tgt_idx'][:, 0] != out['vh1_move']).all(), \
        'row where teacher argmax == vh1 move slipped through'
    torch.save(out, a.output)
    print(f'{a.output}: {len(idx):,} rows from {out["_stats"]["n_seeds"]:,} '
          f'seeds (gap median {np.median(meta["gap"][keep]):.2f})')


if __name__ == '__main__':
    main()
