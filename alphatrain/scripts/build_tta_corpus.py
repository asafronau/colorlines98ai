"""Symmetric self-distillation corpus. For states recorded by `eval --record-dir` (optionally played with
--tta 8), the target is the base net's own 8-symmetry-averaged policy: softmax of the mean of the 8 views'
logits (mapped back to the original frame), top-5 renormalized. The target is exactly D4-equivariant, so
dihedral augmentation is consistent (no view ever sees a contradictory label).
--holdout-games N keeps the last N games out (written as a CLRJ state file for consistency audits)."""
import argparse, glob, json, os, struct, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.evaluate import load_model
from alphatrain.mcts import _legal_priors_jit
from alphatrain.scripts.tta_check import views, ACT


def main():
    p = argparse.ArgumentParser(); p.add_argument('--games-dir', required=True); p.add_argument('--base', required=True)
    p.add_argument('--out', required=True); p.add_argument('--holdout-games', type=int, default=40); p.add_argument('--holdout-out', required=True)
    p.add_argument('--chunk', type=int, default=4096); a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json')))
    train_f, hold_f = files[:-a.holdout_games], files[-a.holdout_games:]
    def load(fs):
        S = []
        for f in fs:
            for s in json.load(open(f))['states']:
                S.append((np.array(s['board'], np.int8), [(b['row'], b['col'], b['color']) for b in s['next_balls']], int(s['move'])))
        return S
    tr, ho = load(train_f), load(hold_f)
    print(f'{len(train_f)} train games -> {len(tr):,} states; {len(hold_f)} held-out games -> {len(ho):,} states', flush=True)
    with open(a.holdout_out, 'wb') as f:                           # held-out states (teacher field = recorded TTA move)
        f.write(b'CLRJ'); f.write(struct.pack('<i', len(ho)))
        for b, nb, mv in ho:
            f.write(b.reshape(81).tobytes()); f.write(struct.pack('<i', len(nb)))
            for t in range(3): f.write(struct.pack('<iii', *nb[t]) if t < len(nb) else struct.pack('<iii', 0, 0, 0))
            f.write(struct.pack('<iif', mv, mv, 0.0))
    json.dump([{'stratum': 'holdout', 'chosen': mv, 'over': False} for _, _, mv in ho], open(a.holdout_out + '.meta.json', 'w'))
    dev = torch.device('mps'); net, _ = load_model(a.base, dev, fp16=False); n = len(tr)
    PI = np.zeros((n, 5), np.int64); PV = np.zeros((n, 5), np.float32); NZ = np.zeros(n, np.int64); agree = 0
    with torch.inference_mode():
        for s in range(0, n, a.chunk):
            batch = tr[s:s+a.chunk]
            obs = torch.from_numpy(np.stack([o for b, nb, _ in batch for o in views(b, nb)]).astype(np.float32))
            lg = net(obs.to(dev)).float().cpu().numpy().reshape(len(batch), 8, 6561)
            for j, (b, nb, mv) in enumerate(batch):
                avg = np.stack([lg[j, v, ACT[v]] for v in range(8)]).mean(0).astype(np.float32)
                c, idx, pri = _legal_priors_jit(b, avg, 5); k = int(c)
                PI[s + j, :k] = idx[:k]; PV[s + j, :k] = pri[:k] / pri[:k].sum(); NZ[s + j] = k; agree += int(idx[int(np.argmax(pri[:k]))] == mv)  # top-k is ASCENDING: idx[0] is the 5th best
            if (s // a.chunk) % 25 == 0: print(f'  {s:,}/{n:,}', flush=True)
    perm = np.random.default_rng(0).permutation(n)
    B = np.stack([b for b, _, _ in tr]); NP = np.zeros((n, 3, 2), np.int8); NC = np.zeros((n, 3), np.int8); NN = np.zeros(n, np.int8)
    for i, (_, nb, _) in enumerate(tr):
        NN[i] = len(nb)
        for t, (r, c, col) in enumerate(nb[:3]): NP[i, t] = (r, c); NC[i, t] = col
    out = {'boards': torch.from_numpy(B[perm]), 'next_pos': torch.from_numpy(NP[perm]), 'next_col': torch.from_numpy(NC[perm]), 'n_next': torch.from_numpy(NN[perm]),
           'pol_indices': torch.from_numpy(PI[perm]), 'pol_values': torch.from_numpy(PV[perm]), 'pol_nnz': torch.from_numpy(NZ[perm]),
           'num_channels': 18, 'max_score': 0.0, 'value_mode': 'policy_slim', 'gamma': 0.99, 'num_value_bins': 64}
    torch.save(out, a.out)
    print(f'wrote {a.out}: {n:,} rows; TTA target argmax == recorded (fp16 TTA) move on {100*agree/n:.1f}%; target top-1 mean {PV.max(1).mean():.3f}')


if __name__ == '__main__':
    main()
