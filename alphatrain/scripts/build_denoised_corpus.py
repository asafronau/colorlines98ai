"""Denoised hard-label corpus from full-record search games. Each override (search chosen != recorded prior
argmax) is kept only if >= --min-rep of K independent re-searches (mcts_relabel --seed-salt) reproduce it;
otherwise the row's target reverts to the prior move. --mode self sets EVERY row to the prior move
(pure self-distillation control: measures the fine-tuning tax with zero teacher signal)."""
import argparse, csv, glob, json, os
import numpy as np, torch


def flat(m): return (m['sr'] * 9 + m['sc']) * 81 + m['tr'] * 9 + m['tc']


def main():
    p = argparse.ArgumentParser(); p.add_argument('--games-dir', required=True); p.add_argument('--override-states', required=True)
    p.add_argument('--relabels', nargs='+', required=True); p.add_argument('--min-rep', type=int, default=2)
    p.add_argument('--mode', choices=['denoise', 'self'], default='denoise'); p.add_argument('--out', required=True); a = p.parse_args()
    meta = json.load(open(a.override_states + '.meta.json'))
    reps = np.zeros(len(meta), int)
    for path in a.relabels:
        rows = list(csv.DictReader(open(path))); assert len(rows) == len(meta)
        reps += np.array([int(r['visit_argmax']) == m['chosen'] for r, m in zip(rows, meta)])
    rep_of = {(m['seed'], m['turn']): int(c) for m, c in zip(meta, reps)}
    print(f'overrides: {len(meta):,}; reproduced by >= {a.min_rep}/{len(a.relabels)}: {(reps >= a.min_rep).sum():,} '
          f'({100*(reps >= a.min_rep).mean():.1f}%); distribution {np.bincount(reps, minlength=len(a.relabels)+1).tolist()}')
    B, NP, NC, NN, PI, PV, NZ, MK = [], [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(a.games_dir, 'game_seed*.json'))):
        g = json.load(open(f))
        for i, m in enumerate(g['moves']):
            ch = flat(m['chosen_move']); pa = m['cand_moves'][int(np.argmax(m['cand_prior']))]
            keep_override = a.mode == 'denoise' and ch != pa and rep_of.get((g['seed'], i), 0) >= a.min_rep
            tgt = ch if keep_override else pa
            B.append(np.array(m['board'], np.int8)); nb = m['next_balls'][:m['num_next']]
            npos = np.zeros((3, 2), np.int8); ncol = np.zeros(3, np.int8)
            for t, b in enumerate(nb[:3]): npos[t] = (b['row'], b['col']); ncol[t] = b['color']
            NP.append(npos); NC.append(ncol); NN.append(len(nb[:3]))
            pi = np.zeros(5, np.int64); pv = np.zeros(5, np.float32); pi[0] = tgt; pv[0] = 1.0
            PI.append(pi); PV.append(pv); NZ.append(1); MK.append(1.0 if keep_override else 0.0)
    n = len(B); perm = np.random.default_rng(0).permutation(n)
    out = {'boards': torch.from_numpy(np.stack(B)[perm]), 'next_pos': torch.from_numpy(np.stack(NP)[perm]), 'next_col': torch.from_numpy(np.stack(NC)[perm]),
           'n_next': torch.from_numpy(np.array(NN, np.int8)[perm]), 'pol_indices': torch.from_numpy(np.stack(PI)[perm]), 'pol_values': torch.from_numpy(np.stack(PV)[perm]),
           'pol_nnz': torch.from_numpy(np.array(NZ, np.int64)[perm]), 'disagree_mask': torch.from_numpy(np.array(MK, np.float32)[perm]),
           'disagree_mask_protocol': f'1 = override kept (reproduced by >= {a.min_rep}/{len(a.relabels)} independent re-searches); mode={a.mode}',
           'num_channels': 18, 'max_score': 0.0, 'value_mode': 'policy_slim', 'gamma': 0.99, 'num_value_bins': 64}
    torch.save(out, a.out); print(f'wrote {a.out}: {n:,} rows, {int(sum(MK)):,} robust-override targets ({100*sum(MK)/n:.2f}%), mode={a.mode}')


if __name__ == '__main__':
    main()
