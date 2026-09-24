"""Argmax-match of a policy checkpoint against a hard-CE corpus, by stratum and by
target top-share bin. Answers: is the residual mismatch label noise (near-tie states)
or unlearned structure (decisive states)?"""
import argparse, sys
import numpy as np, torch
sys.path.insert(0, '.')
from alphatrain.observation import build_observation
from alphatrain.evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default='alphatrain/data/r2_bulk.pt')
    p.add_argument('--strata', default='alphatrain/data/r2_bulk_strata.npz')
    p.add_argument('--model', default='alphatrain/data/scratch18b96_lr3e3_ckpts_epoch_40.pt')
    p.add_argument('--n', type=int, default=200000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    d = torch.load(a.tensor, map_location='cpu', weights_only=False, mmap=True)
    strata = np.load(a.strata)['strata']
    N = d['boards'].shape[0]
    idx = np.sort(np.random.default_rng(a.seed).choice(N, a.n, replace=False))
    boards = d['boards'][idx].numpy(); npos = d['next_pos'][idx].numpy(); ncol = d['next_col'][idx].numpy()
    nn_ = d['n_next'][idx].numpy(); pi = d['pol_indices'][idx].numpy(); pv = d['pol_values'][idx].numpy()
    st = strata[idx]
    tgt = pi[np.arange(a.n), pv.argmax(1)]; top = pv.max(1)
    dev = torch.device('mps')
    net, _ = load_model(a.model, dev, fp16=False)
    obs = np.zeros((a.n, 18, 9, 9), dtype=np.float32)
    for i in range(a.n):
        obs[i] = build_observation(boards[i], npos[i, :, 0].astype(np.int64), npos[i, :, 1].astype(np.int64),
                                   ncol[i].astype(np.int64), int(nn_[i]))
    arg = np.zeros(a.n, dtype=np.int64); margin = np.zeros(a.n, dtype=np.float32); in_top5 = np.zeros(a.n, bool)
    obs_t = torch.from_numpy(obs)
    with torch.inference_mode():
        for s in range(0, a.n, 2048):
            lg = net(obs_t[s:s+2048].to(dev)).float().cpu().numpy()
            for j in range(lg.shape[0]):
                i = s + j
                k = pi[i][pv[i] > 0]  # legal candidates stored (top-5); restrict argmax to them AND to all-legal? use stored set
                sub = lg[j][k]; o = np.argsort(-sub)
                arg[i] = k[o[0]]; margin[i] = sub[o[0]] - (sub[o[1]] if len(o) > 1 else sub[o[0]])
            if (s // 2048) % 20 == 0: print(f'  {s}/{a.n}', flush=True)
    match = arg == tgt
    print(f'\nmodel {a.model}\nrows {a.n}  argmax-match (within stored top-5 set): {100*match.mean():.2f}%')
    print(f'{"stratum":26s} {"rows":>7s} {"match%":>7s} {"top>=.6 match%":>14s} {"top<.35 match%":>14s} {"mean top":>8s}')
    for s_ in sorted(set(st.tolist()), key=lambda x: -(st == x).sum()):
        m = st == s_; hi = m & (top >= 0.6); lo = m & (top < 0.35)
        print(f'{s_:26s} {m.sum():7d} {100*match[m].mean():7.2f} {100*match[hi].mean() if hi.any() else float("nan"):14.2f} '
              f'{100*match[lo].mean() if lo.any() else float("nan"):14.2f} {top[m].mean():8.3f}')
    print('\nby target top-share bin (all strata):')
    for lo_, hi_ in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .8), (.8, 1.01)):
        m = (top >= lo_) & (top < hi_)
        if m.any(): print(f'  top in [{lo_:.1f},{hi_:.1f}): rows {m.sum():7d} ({100*m.mean():5.1f}%)  match {100*match[m].mean():6.2f}%  student margin P50 {np.median(margin[m]):.2f}')
    print(f'\nstudent logit margin (top1-top2 within set) P10/P50/P90: {np.percentile(margin,10):.2f}/{np.median(margin):.2f}/{np.percentile(margin,90):.2f}')


if __name__ == '__main__':
    main()
