"""Map every row of a train_path_b tensor to its canonical form (D4 x color relabel, alphatrain/canonical.py):
board + preview are canonicalized, target move indices are mapped into the canonical view. Train the result
with --no-dihedral-augment --no-color-augment and evaluate with `eval --canon`: the net only ever sees one
representative per symmetry class."""
import argparse
import numpy as np, torch
from alphatrain.canonical import _canon_one, CELL, ACT


def main():
    p = argparse.ArgumentParser(); p.add_argument('--src', required=True); p.add_argument('--out', required=True); a = p.parse_args()
    d = torch.load(a.src, map_location='cpu', weights_only=False)
    B = d['boards'].numpy().reshape(-1, 81); NP = d['next_pos'].numpy(); NC = d['next_col'].numpy(); NN = d['n_next'].numpy()
    PI = d['pol_indices'].numpy().copy(); PV = d['pol_values'].numpy()
    n = len(B); outB = np.zeros_like(B); outNP = np.zeros_like(NP); outNC = np.zeros_like(NC); views = np.zeros(n, np.int64)
    for i in range(n):
        k = int(NN[i]); pcell = np.zeros(3, np.int64); pcol = np.zeros(3, np.int64)
        for t in range(k): pcell[t] = int(NP[i, t, 0]) * 9 + int(NP[i, t, 1]); pcol[t] = int(NC[i, t])
        v, cb, pc, pk = _canon_one(B[i].astype(np.int8), pcell, pcol, k, CELL)
        outB[i] = cb; views[i] = v
        for t in range(k): outNP[i, t] = (pc[t] // 9, pc[t] % 9); outNC[i, t] = pk[t]
        live = PV[i] > 0
        PI[i, live] = ACT[v][PI[i, live]]
        if (i + 1) % 200000 == 0: print(f'  {i+1:,}/{n:,}', flush=True)
    out = dict(d); out['boards'] = torch.from_numpy(outB.reshape(-1, 9, 9)); out['next_pos'] = torch.from_numpy(outNP)
    out['next_col'] = torch.from_numpy(outNC); out['pol_indices'] = torch.from_numpy(PI)
    torch.save(out, a.out); print(f'wrote {a.out}: {n:,} canonical rows; views used {np.bincount(views, minlength=8).tolist()}')


if __name__ == '__main__':
    main()
