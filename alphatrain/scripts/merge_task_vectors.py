"""Task arithmetic with several fine-tunes of the SAME base: theta = base + sum_i alpha_i * (ft_i - base).

    python -m alphatrain.scripts.merge_task_vectors --base alphatrain/data/ta_A6_e4_a0.5.pt \
        --ft checkpoints/A7_ft/epoch_4.pt 1.0 --ft checkpoints/A7s_ft/epoch_4.pt 1.0 \
        --out alphatrain/data/ta_A7_A7s_a1.0_1.0.pt

BatchNorm running statistics come from the base and must be identical in every fine-tune (frozen BN), as in
scripts/merge_checkpoints.py (the single-vector version). Prints each vector's norm and their cosine.
"""
import argparse

import torch


def state_of(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    return {k.replace('_orig_mod.', ''): v for k, v in st.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--ft', nargs=2, action='append', metavar=('CHECKPOINT', 'ALPHA'), required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    base = state_of(a.base)
    fts = [(state_of(p), float(al), p) for p, al in a.ft]
    merged, vecs = {}, [[] for _ in fts]
    for k, v in base.items():
        for st, _, p in fts:
            if st.keys() != base.keys():
                raise SystemExit(f'{p}: key mismatch with the base')
        if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k:
            for st, _, p in fts:
                if not torch.equal(st[k].float(), v.float()):
                    raise SystemExit(f'{p}: BatchNorm stat {k} differs from the base (fine-tune without frozen BN?)')
            merged[k] = v
            continue
        if not torch.is_floating_point(v):
            merged[k] = v
            continue
        acc = v.float().clone()
        for i, (st, al, _) in enumerate(fts):
            d = st[k].float() - v.float()
            vecs[i].append(d.flatten())
            acc += al * d
        merged[k] = acc.to(v.dtype)
    flat = [torch.cat(x) for x in vecs]
    for (_, al, p), f in zip(fts, flat):
        print(f'  {p}: alpha {al}, task-vector norm {f.norm():.4f}')
    if len(flat) > 1:
        print(f'  cosine(vector 1, vector 2) = {torch.nn.functional.cosine_similarity(flat[0], flat[1], dim=0):.4f}')
    torch.save({'model': merged, 'policy_only': True,
                'args': {'base': a.base, 'fts': [(p, al) for _, al, p in fts]}}, a.out)
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
