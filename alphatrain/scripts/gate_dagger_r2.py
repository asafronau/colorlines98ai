"""Pre-registered early gate for dagger R2 candidates (fp16 deployment protocol).

For each candidate (merged or fine-tuned checkpoint), on the SAME by-seed
held-out correction split as train_crisis_ft (--holdout-frac/--split-seed must
match) plus a quiet-state holdout from the recorded games:

  - adoption%  : plays teacher move on held-out corrections (bar: >= +10pp)
  - pref%      : logit(teacher_mv) > logit(vh1_mv)
  - margin med : median logit(teacher_mv) - logit(vh1_mv) (bar: clearly up)
  - drift%     : quiet-state argmax changes vs vh1 (bar: <= 3%)
  - third%     : plays neither move on held-out corrections (bar: <= adoption)

    python -m alphatrain.scripts.gate_dagger_r2 \
        --models checkpoints/dagger_r2_merges/a010.pt ...
"""
import argparse
import os

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model
from alphatrain.scripts.diag_dagger1_fp16 import fp16_argmax_margin
from alphatrain.scripts.diag_dagger1_regression import build_holdout


def corr_holdout_rows(corpus, holdout_frac, split_seed):
    """Replicate train_crisis_ft.load_corpus's by-seed split exactly."""
    seeds_all = corpus['seed'].tolist()
    uniq = sorted(set(int(s) for s in seeds_all))
    n_hold = max(1, int(round(holdout_frac * len(uniq))))
    g = torch.Generator().manual_seed(split_seed)
    hperm = torch.randperm(len(uniq), generator=g).tolist()
    hold = set(uniq[i] for i in hperm[:n_hold])
    return np.array([int(s) in hold for s in seeds_all])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--corpus', default='alphatrain/data/dagger_r2_gap1.pt')
    p.add_argument('--holdout-frac', type=float, default=0.15)
    p.add_argument('--split-seed', type=int, default=0)
    p.add_argument('--games-dir', default='data/dagger_games_v1')
    p.add_argument('--full-corpus', default='alphatrain/data/dagger_v1.pt',
                   help='for excluding trained states from the quiet holdout')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(0)

    c = torch.load(a.corpus, map_location='cpu', weights_only=False)
    hold = corr_holdout_rows(c, a.holdout_frac, a.split_seed)
    hidx = np.where(hold)[0]
    print(f'held-out corrections: {len(hidx):,} rows '
          f'(must match train log heldout=N)', flush=True)
    t_mv = c['tgt_idx'][:, 0].numpy()[hidx]
    s_mv = c['vh1_move'].numpy()[hidx]

    tmp = a.corpus + '.gate.tmp'
    n = c['boards'].shape[0]
    torch.save({'boards': c['boards'], 'next_pos': c['next_pos'],
                'next_col': c['next_col'], 'n_next': c['n_next'],
                'pol_indices': torch.zeros((n, 5), dtype=torch.int64),
                'pol_values': torch.zeros((n, 5), dtype=torch.float32),
                'max_score': 0.0}, tmp)
    ds_corr = TensorDatasetGPU(tmp, augment=False, color_augment=False,
                               augment_factor=1, device=a.device)
    qtmp = a.corpus + '.gateq.tmp'
    n_q = build_holdout(a.games_dir, a.full_corpus, 200, 100, qtmp, rng)
    ds_q = TensorDatasetGPU(qtmp, augment=False, color_augment=False,
                            augment_factor=1, device=a.device)
    print(f'quiet holdout: {n_q:,} states', flush=True)

    def stats(path):
        net, _ = load_model(path, dev, fp16=True)
        arg_all, _ = fp16_argmax_margin(net, ds_corr, n, dev)
        arg = arg_all[hidx]
        # raw logit margin at the two moves on held-out rows
        margins = []
        bs = 2048
        for s in range(0, len(hidx), bs):
            b = torch.from_numpy(hidx[s:s + bs]).to(dev)
            obs = ds_corr._build_obs_core(
                ds_corr.boards[b], next_pos=ds_corr.next_pos[b],
                next_col=ds_corr.next_col[b], n_next=ds_corr.n_next[b])
            with torch.no_grad():
                lg = net(obs.to(torch.float16)).float().cpu().numpy()
            for i in range(lg.shape[0]):
                j = s + i
                margins.append(lg[i][t_mv[j]] - lg[i][s_mv[j]])
        qarg, _ = fp16_argmax_margin(net, ds_q, n_q, dev)
        del net
        return arg, np.array(margins), qarg

    base_name = os.path.basename(a.base).replace('.pt', '')
    b_arg, b_marg, b_qarg = stats(a.base)
    print(f'\n{"model":34s} {"adopt%":>7s} {"pref%":>6s} {"marg_med":>9s} '
          f'{"drift%":>7s} {"third%":>7s}')
    for path in [a.base] + a.models:
        name = os.path.basename(path).replace('.pt', '')
        if path == a.base:
            arg, marg, qarg = b_arg, b_marg, b_qarg
        else:
            arg, marg, qarg = stats(path)
        adopt = 100 * (arg == t_mv).mean()
        pref = 100 * (marg > 0).mean()
        third = 100 * ((arg != t_mv) & (arg != s_mv)).mean()
        drift = 100 * (qarg != b_qarg).mean()
        print(f'{name:34s} {adopt:7.1f} {pref:6.1f} {np.median(marg):9.3f} '
              f'{drift:7.1f} {third:7.1f}', flush=True)
    os.remove(tmp)
    os.remove(qtmp)


if __name__ == '__main__':
    main()
