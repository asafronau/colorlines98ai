"""Concatenate the policy fields of several train_path_b tensors into one shuffled training tensor.

Keeps only what the policy trainer reads (boards, next_pos, next_col, n_next, pol_indices, pol_values,
pol_nnz), so tensors from different builders (build_tta_corpus, build_tta_r2_corpus,
build_expert_v2_tensor) can be mixed. Records the source of every row in `source_id` + `sources`.

    python -m alphatrain.scripts.merge_policy_tensors --out alphatrain/data/distill.pt A.pt B.pt ...
"""
import argparse

import numpy as np
import torch

FIELDS = ['boards', 'next_pos', 'next_col', 'n_next', 'pol_indices', 'pol_values', 'pol_nnz']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('inputs', nargs='+')
    a = ap.parse_args()
    parts = {f: [] for f in FIELDS}
    source_id = []
    for i, path in enumerate(a.inputs):
        d = torch.load(path, map_location='cpu', weights_only=False)
        missing = [f for f in FIELDS if f not in d]
        if missing:
            raise SystemExit(f'{path}: missing {missing}')
        n = d['boards'].shape[0]
        for f in FIELDS:
            t = d[f]
            if f == 'pol_values':
                t = t.float()
            if f == 'pol_indices':
                t = t.long()
            parts[f].append(t)
        source_id.append(torch.full((n,), i, dtype=torch.int8))
        print(f'  {path}: {n:,} rows', flush=True)
    out = {f: torch.cat(parts[f]) for f in FIELDS}
    out['source_id'] = torch.cat(source_id)
    n = out['boards'].shape[0]
    perm = torch.from_numpy(np.random.default_rng(a.seed).permutation(n))
    out = {k: v[perm] for k, v in out.items()}
    out['sources'] = list(a.inputs)
    out['value_mode'] = 'policy_slim'
    out['num_channels'] = 18
    out['max_score'] = 0.0  # required by the dataset loader; unused for policy-only training
    torch.save(out, a.out)
    print(f'wrote {a.out}: {n:,} rows from {len(a.inputs)} tensors', flush=True)


if __name__ == '__main__':
    main()
