"""Assemble the MCTS-corrections corpus (.pt) for weighted/sharpened distillation.

Reads crisis/corrections/corr_*.json (gen_corrections_parallel output). Each
correction becomes a training anchor: board+next_balls -> the MCTS soft visit
distribution (the target), with a CONFIDENCE WEIGHT = margin (mcts_top_share -
pol_share) normalized to mean 1. Marginal corrections (MCTS barely prefers another
move) get ~0 weight; decisive ones dominate -- the anti-dilution lever, MCTS-grounded
(vs Track-1's greedy-gap weight). The soft target itself also encodes confidence
(peaked vs flat); the trainer sharpens it (--target-temperature).

    PYTHONPATH=. python scripts/build_corrections_corpus.py \\
        --glob 'crisis/corrections/corr_*.json' --out crisis/corrections_corpus.pt
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

K = 20   # top-K visit moves stored per anchor (matches gen --topk-visits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', default='crisis/corrections/corr_*.json')
    p.add_argument('--min-margin', type=float, default=0.0,
                   help='Drop corrections whose (top_share - pol_share) is below '
                        'this (0..1). 0 = keep all (weighting handles marginals).')
    p.add_argument('--first-n-by-mtime', type=int, default=0,
                   help='Use only the first N corr files by mining chronology '
                        '(mtime order). Reconstructs historical corpora exactly, '
                        'e.g. 1837 = the original 13.8k "bar" corpus. 0 = all.')
    p.add_argument('--out', default='crisis/corrections_corpus.pt')
    a = p.parse_args()

    files = sorted(glob.glob(a.glob))
    if a.first_n_by_mtime:
        files.sort(key=os.path.getmtime)
        files = files[:a.first_n_by_mtime]
    boards, npos, ncol, nnext = [], [], [], []
    tgt_idx, tgt_prob, weights, seeds = [], [], [], []
    n_files = n_corr = n_dropped = 0
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        n_files += 1
        for c in d['corrections']:
            margin = c['mcts_top_share'] - c['pol_share']
            if margin < a.min_margin:
                n_dropped += 1
                continue
            # next-ball features
            nposs, ncols = [[0, 0], [0, 0], [0, 0]], [0, 0, 0]
            for i, nb in enumerate(c['next_balls'][:3]):
                (nr, nc), color = nb
                nposs[i] = [int(nr), int(nc)]
                ncols[i] = int(color)
            # soft target: top-K (idx, prob), renormalized over the stored mass
            idxs = [int(v[0]) for v in c['visits'][:K]]
            prs = np.array([float(v[1]) for v in c['visits'][:K]], dtype=np.float64)
            prs = prs / prs.sum()
            pad = K - len(idxs)
            boards.append(c['board'])
            npos.append(nposs); ncol.append(ncols); nnext.append(min(len(c['next_balls']), 3))
            tgt_idx.append(idxs + [0] * pad)
            tgt_prob.append(prs.tolist() + [0.0] * pad)
            weights.append(max(margin, 0.0))
            seeds.append(int(c['seed']))
            n_corr += 1

    if not boards:
        raise SystemExit(f"No corrections in {len(files)} files (min_margin={a.min_margin}?).")
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean().clamp(min=1e-6)          # normalize to mean 1
    corpus = {
        'boards': torch.tensor(boards, dtype=torch.int8),
        'next_pos': torch.tensor(npos, dtype=torch.int8),
        'next_col': torch.tensor(ncol, dtype=torch.int8),
        'n_next': torch.tensor(nnext, dtype=torch.int8),
        'tgt_idx': torch.tensor(tgt_idx, dtype=torch.long),
        'tgt_prob': torch.tensor(tgt_prob, dtype=torch.float32),
        'weight': w,
        'seed': torch.tensor(seeds, dtype=torch.long),
        '_stats': {'n_files': n_files, 'n_corrections': n_corr,
                   'n_dropped': n_dropped, 'n_seeds': len(set(seeds)),
                   'min_margin': a.min_margin},
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(corpus, a.out)
    wn = w.numpy()
    print(f"Wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print(f"  {n_corr} corrections from {len(set(seeds))} seeds / {n_files} games "
          f"(dropped {n_dropped} below min_margin={a.min_margin})")
    print(f"  weight (margin, mean 1): P10={np.percentile(wn,10):.2f} "
          f"P50={np.percentile(wn,50):.2f} P90={np.percentile(wn,90):.2f} "
          f"max={wn.max():.1f}  | {100*np.mean(wn<0.2):.0f}% near-zero (marginal)")


if __name__ == '__main__':
    main()
