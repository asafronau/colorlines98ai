"""r3_tail = r2_bulk + the student's own death-window states relabelled by the 10x teacher
(mcts_relabel CSV). New rows use pack() semantics (top-5 by visits, teacher argmax forced).
disagree_mask: 0 for r2_bulk rows, 1.0 for relabelled rows where teacher != student move,
--agree-mask for relabelled rows where they agree. train_path_b --disagree-gamma G weights
rows by 1 + G*mask."""
import argparse, csv, json, struct
import numpy as np, torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='alphatrain/data/r2_bulk.pt')
    p.add_argument('--states', default='alphatrain/inference_cpp/data/greedy_tail200_18b96e40_2600k.bin')
    p.add_argument('--csv', default='alphatrain/inference_cpp/data/greedy_tail200_18b96e40_2600k_vh2head.csv')
    p.add_argument('--agree-mask', type=float, default=0.25)
    p.add_argument('--tte-weights', default='60:1.0,120:0.5,201:0.25',
                   help='turns_to_end upper-bound:mask multiplier; judge excess is 14-18%% of flips at <60 turns, 6%% at 60-120, ~0 beyond')
    p.add_argument('--out', default='alphatrain/data/r3_tail.pt')
    a = p.parse_args()
    with open(a.states, 'rb') as f:
        assert f.read(4) == b'CLRJ'; n = struct.unpack('<i', f.read(4))[0]
        boards = np.zeros((n, 81), np.int8); npos = np.zeros((n, 3, 2), np.int8); ncol = np.zeros((n, 3), np.int8); nn = np.zeros(n, np.int8); stu = np.zeros(n, np.int64)
        for i in range(n):
            boards[i] = np.frombuffer(f.read(81), np.int8); k = struct.unpack('<i', f.read(4))[0]; nn[i] = k
            for t in range(3):
                r, c, col = struct.unpack('<iii', f.read(12))
                if t < k: npos[i, t] = (r, c); ncol[i, t] = col
            tm, bm, _ = struct.unpack('<iif', f.read(12)); stu[i] = tm
    rows = list(csv.DictReader(open(a.csv))); assert len(rows) == n
    meta = json.load(open(a.states + '.meta.json')); tte = np.array([m['turns_to_end'] for m in meta])
    bounds = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in a.tte_weights.split(',')]
    def tte_mult(t):
        for ub, w in bounds:
            if t < ub: return w
        return bounds[-1][1]
    pi = np.zeros((n, 5), np.int64); pv = np.zeros((n, 5), np.float32); nnz = np.zeros(n, np.int64); mask = np.zeros(n, np.float32)
    tea = np.zeros(n, np.int64)
    for i, r in enumerate(rows):
        cands = [c.split(':') for c in (r[f'cand{k}'] for k in range(15)) if not c.startswith('-1:')]
        acts = [int(c[0]) for c in cands][:5]; vis = np.array([float(c[1]) for c in cands][:5], np.float32)
        t = int(r['visit_argmax']); tea[i] = t
        if t in acts: vis[acts.index(t)] = vis.max() + 1.0
        else: acts = [t] + acts[:4]; vis = np.concatenate([[vis.max() + 1.0 if len(vis) else 1.0], vis[:4]])
        k = len(acts); pi[i, :k] = acts; pv[i, :k] = vis / max(vis.sum(), 1e-8); nnz[i] = k
        mask[i] = (1.0 if t != stu[i] else a.agree_mask) * tte_mult(int(tte[i]))
    base = torch.load(a.base, map_location='cpu', weights_only=False)
    N0 = base['boards'].shape[0]
    out = {
        'boards': torch.cat([base['boards'], torch.from_numpy(boards.reshape(n, 9, 9))]),
        'next_pos': torch.cat([base['next_pos'], torch.from_numpy(npos)]),
        'next_col': torch.cat([base['next_col'], torch.from_numpy(ncol)]),
        'n_next': torch.cat([base['n_next'], torch.from_numpy(nn)]),
        'pol_indices': torch.cat([base['pol_indices'], torch.from_numpy(pi)]),
        'pol_values': torch.cat([base['pol_values'], torch.from_numpy(pv)]),
        'pol_nnz': torch.cat([base['pol_nnz'], torch.from_numpy(nnz)]),
        'disagree_mask': torch.cat([torch.zeros(N0), torch.from_numpy(mask)]),
        'disagree_mask_protocol': 'r3_tail: 0 for r2_bulk rows; 1.0 relabelled death-window rows where 10x teacher != student move; '
                                  f'{a.agree_mask} where they agree, times turns_to_end multiplier {a.tte_weights}. Teacher = vh2_policy_ts + pv_vh2_ts, q2.0, c_puct1.5, virtual-mean, 400 sims.',
    }
    for k in ('num_channels', 'max_score', 'value_mode', 'gamma', 'num_value_bins'): out[k] = base[k]
    # shuffle so val split (by state) and batches see both sources
    perm = torch.randperm(N0 + n, generator=torch.Generator().manual_seed(0))
    for k in ('boards', 'next_pos', 'next_col', 'n_next', 'pol_indices', 'pol_values', 'pol_nnz', 'disagree_mask'):
        out[k] = out[k][perm].contiguous()
    torch.save(out, a.out)
    print(f'{a.out}: {N0 + n:,} rows = {N0:,} r2_bulk + {n:,} relabelled ({int((tea != stu).sum()):,} flips); '
          f'mask sum {mask.sum():.0f}; teacher top-share (top-5 renorm) mean {pv.max(1).mean():.3f}')


if __name__ == '__main__':
    main()
