"""Training-label audit: group r2_bulk rows by position up to the 8 board symmetries (canonical hash of
board + preview), map each row's target move into the canonical frame, and measure (1) exact duplicates,
(2) symmetric duplicates, (3) how often rows of the same position disagree on the target move."""
import sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.scripts.tta_check import CELL, ACT
d = torch.load('alphatrain/data/r2_bulk.pt', map_location='cpu', weights_only=True, mmap=True)
N = d['boards'].shape[0]
B = d['boards'].numpy().reshape(N, 81).astype(np.uint64); NP = d['next_pos'].numpy(); NC = d['next_col'].numpy(); NN = d['n_next'].numpy()
pi = d['pol_indices'].numpy(); pv = d['pol_values'].numpy(); tgt = pi[np.arange(N), pv.argmax(1)]
rng = np.random.default_rng(7); R = rng.integers(1, 2**63, size=(81, 8), dtype=np.uint64) | np.uint64(1)
PR = rng.integers(1, 2**63, size=(81, 8), dtype=np.uint64) | np.uint64(1)
pcell = (NP[:, :, 0].astype(np.int64) * 9 + NP[:, :, 1].astype(np.int64))          # (N,3)
valid = np.arange(3)[None, :] < NN[:, None]
H = np.zeros((N, 8), np.uint64)
with np.errstate(over='ignore'):
    for v in range(8):
        # board value at original cell i lands at CELL[v][i]; hash uses the landing position's random key
        Rv = R[CELL[v]]                                   # (81, 8): key for original cell i under view v
        h = np.zeros(N, np.uint64)
        for k in range(8): h ^= (B == k).astype(np.uint64) @ np.ascontiguousarray(Rv[:, k]) if k else np.zeros(N, np.uint64)
        for t in range(3):
            c = CELL[v][pcell[:, t]]; col = NC[:, t].astype(np.int64)
            h ^= np.where(valid[:, t], PR[c, col] * np.uint64(t + 1), np.uint64(0))
        H[:, v] = h
    print('hashes done', flush=True)
vmin = H.argmin(1); key = H[np.arange(N), vmin]; ctgt = ACT[vmin, tgt]
ident = H[:, 0]
_, inv_i, cnt_i = np.unique(ident, return_inverse=True, return_counts=True)
_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
print(f'rows {N:,}; distinct positions (exact) {len(cnt_i):,}; distinct up to symmetry {len(cnt):,}')
print(f'rows whose exact position occurs more than once: {100*(cnt_i[inv_i] > 1).mean():.1f}%; up to symmetry: {100*(cnt[inv] > 1).mean():.1f}%')
order = np.argsort(inv, kind='stable'); g = inv[order]; t = ctgt[order]
starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]]); ends = np.r_[starts[1:], len(g)]
multi = [(s, e) for s, e in zip(starts, ends) if e - s > 1]
dis = sum(len(set(t[s:e].tolist())) > 1 for s, e in multi)
rows_multi = sum(e - s for s, e in multi)
agree_frac = np.mean([np.bincount(np.unique(t[s:e], return_inverse=True)[1]).max() / (e - s) for s, e in multi]) if multi else float('nan')
print(f'positions seen >1 time (up to symmetry): {len(multi):,} covering {rows_multi:,} rows; target move DISAGREES within {100*dis/max(1,len(multi)):.1f}% of them; mean share of the majority target {100*agree_frac:.1f}%')
