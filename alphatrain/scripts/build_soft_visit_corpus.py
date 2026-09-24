"""Soft visit-target corpus from full-record self-search games (mcts_selfplay --full-record).
Every move of every game; targets = top-K (K=5) renormalized visit distribution of the CORRECTED
search (no argmax forcing, no cap per game); train_path_b-compatible. Optional: keep every k-th
move (--every) to thin long games."""
import argparse, glob, json, os
import numpy as np, torch


def main():
    p = argparse.ArgumentParser(); p.add_argument('--games-dir', nargs='+', required=True); p.add_argument('--out', required=True)
    p.add_argument('--every', type=int, default=1); p.add_argument('--skip', type=int, default=0, help='drop first N moves (temperature moves)'); p.add_argument('--k', type=int, default=5)
    a = p.parse_args(); K = a.k
    B, NP, NC, NN, PI, PV, NZ = [], [], [], [], [], [], []
    files = [f for d in a.games_dir for f in sorted(glob.glob(os.path.join(d, 'game_seed*.json')))]
    for fi, f in enumerate(files):
        g = json.load(open(f))
        for i, m in enumerate(g['moves']):
            if i < a.skip or (i - a.skip) % a.every: continue
            v = np.array(m['cand_visits'], np.float32); acts = np.array(m['cand_moves'], np.int64)
            o = np.argsort(-v)[:K]; v = v[o]; acts = acts[o]; k = int((v > 0).sum()) or 1
            pi = np.zeros(K, np.int64); pv = np.zeros(K, np.float32); pi[:k] = acts[:k]; pv[:k] = v[:k] / v[:k].sum()
            B.append(np.array(m['board'], np.int8)); nb = m['next_balls'][:m['num_next']]
            npos = np.zeros((3, 2), np.int8); ncol = np.zeros(3, np.int8)
            for t, b in enumerate(nb[:3]): npos[t] = (b['row'], b['col']); ncol[t] = b['color']
            NP.append(npos); NC.append(ncol); NN.append(len(nb[:3])); PI.append(pi); PV.append(pv); NZ.append(k)
        if (fi + 1) % 50 == 0: print(f'  {fi+1}/{len(files)} games, {len(B):,} rows', flush=True)
    n = len(B); perm = np.random.default_rng(0).permutation(n)
    out = {'boards': torch.from_numpy(np.stack(B)[perm]), 'next_pos': torch.from_numpy(np.stack(NP)[perm]), 'next_col': torch.from_numpy(np.stack(NC)[perm]),
           'n_next': torch.from_numpy(np.array(NN, np.int8)[perm]), 'pol_indices': torch.from_numpy(np.stack(PI)[perm]), 'pol_values': torch.from_numpy(np.stack(PV)[perm]),
           'pol_nnz': torch.from_numpy(np.array(NZ, np.int64)[perm]), 'num_channels': 18, 'max_score': 0.0, 'value_mode': 'policy_slim', 'gamma': 0.99, 'num_value_bins': 64}
    torch.save(out, a.out); ts = out['pol_values'].max(1).values
    print(f'wrote {a.out}: {n:,} rows from {len(files)} games; target top-share mean {ts.mean():.3f} P50 {ts.median():.3f}')


if __name__ == '__main__':
    main()
