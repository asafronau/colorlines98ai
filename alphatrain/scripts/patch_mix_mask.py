"""Carry a corpus's disagree_mask into a mixed tensor (rehearsal rows = 0).

mix_tensors.py concatenates [main, rehearsal] but only copies its KEYS list, so
the main corpus's disagree_mask is dropped. This re-attaches it: mask =
[corpus.disagree_mask, zeros(n_rehearsal)]. Verifies row alignment by comparing
the first/last corpus boards.

    python -m alphatrain.scripts.patch_mix_mask \
        --corpus alphatrain/data/dagger_v1.pt \
        --mix alphatrain/data/dagger_v1_mix.pt
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corpus', required=True)
    p.add_argument('--mix', required=True)
    a = p.parse_args()

    corpus = torch.load(a.corpus, map_location='cpu', weights_only=False)
    mix = torch.load(a.mix, map_location='cpu', weights_only=False)
    mask = corpus['disagree_mask']
    n_c, n_m = mask.shape[0], mix['boards'].shape[0]
    assert corpus['boards'].shape[0] == n_c
    assert n_m > n_c, f'mix ({n_m}) not larger than corpus ({n_c})'
    assert torch.equal(mix['boards'][0], corpus['boards'][0]), 'row 0 mismatch'
    assert torch.equal(mix['boards'][n_c - 1], corpus['boards'][n_c - 1]), \
        f'row {n_c - 1} mismatch — mix is not [corpus, rehearsal]'
    mix['disagree_mask'] = torch.cat(
        [mask.to(torch.int8), torch.zeros(n_m - n_c, dtype=torch.int8)])
    torch.save(mix, a.mix)
    print(f'{a.mix}: disagree_mask attached '
          f'({int(mask.sum()):,} of {n_c:,} corpus rows set, '
          f'{n_m - n_c:,} rehearsal rows zero)')


if __name__ == '__main__':
    main()
