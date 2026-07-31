"""fp16 (deployment-protocol) re-diagnostics + judge exports for dagger1 R2.

Per ChatGPT review of docs/small128_dagger1_postmortem_for_review.md:
  1. Recompute absorption in fp16, corrections defined vs the fp16 vh1 argmax.
  2. Export a ROW-LEVEL judge set: actual dagger1 correction rows (the exact
     stored fp16 actions used as labels), stratified by teacher gap x student
     (vh1) top-2 logit margin.
  3. Export a THIRD-ACTION judge set: rows where the trained model now plays
     neither vh1's nor the teacher's move -> judge third vs vh1's action.

    python -m alphatrain.scripts.diag_dagger1_fp16 \
        --trained alphatrain/data/small128_dagger1_e3_s400.pt
"""
import argparse
import struct

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model

DATA = 'alphatrain/inference_cpp/data'


def fp16_argmax_margin(net, ds, n, dev, batch=2048):
    """fp16 legal argmax + top-2 raw-logit margin per state."""
    arg = np.full(n, -1, dtype=np.int64)
    margin = np.zeros(n, dtype=np.float32)
    for s in range(0, n, batch):
        e = min(s + batch, n)
        obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                 next_col=ds.next_col[s:e], n_next=ds.n_next[s:e])
        with torch.no_grad():
            lg = net(obs.to(torch.float16)).float().cpu().numpy()
        bd = ds.boards[s:e].cpu().numpy().astype(np.int8)
        for i in range(e - s):
            k, fi, _ = _legal_priors_jit(bd[i], lg[i], 2)
            if k == 0:
                continue
            cand = fi[:min(k, 2)]
            lgv = lg[i][cand]
            top = int(cand[int(np.argmax(lgv))])
            arg[s + i] = top
            margin[s + i] = float(np.max(lgv) - np.min(lgv)) if len(cand) > 1 else 99.0
        if (s // batch) % 10 == 0:
            print(f'  {e:,}/{n:,}', flush=True)
    return arg, margin


def write_bin(path, rows, boards, next_pos, next_col, n_next, alt_mv, base_mv, aux):
    with open(path, 'wb') as f:
        f.write(b'CLRJ')
        f.write(struct.pack('<i', len(rows)))
        for j, i in enumerate(rows):
            f.write(boards[i].numpy().astype(np.int8).tobytes())
            nn = int(n_next[i])
            f.write(struct.pack('<i', nn))
            for t in range(3):
                f.write(struct.pack('<iii', int(next_pos[i, t, 0]),
                                    int(next_pos[i, t, 1]), int(next_col[i, t])))
            f.write(struct.pack('<iif', int(alt_mv[j]), int(base_mv[j]),
                                float(aux[j])))
    print(f'wrote {path}: {len(rows)} states', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trained',
                   default='alphatrain/data/small128_dagger1_e3_s400.pt')
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--corpus', default='alphatrain/data/dagger_v1.pt')
    p.add_argument('--meta', default='alphatrain/data/dagger_v1_states_meta.npz')
    p.add_argument('--device', default='mps')
    p.add_argument('--n-rowjudge', type=int, default=150, help='per stratum cell')
    p.add_argument('--n-third', type=int, default=300)
    p.add_argument('--seed', type=int, default=1)
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(a.seed)

    meta = np.load(a.meta)
    dis, gap = meta['disagree'], meta['gap']
    s_mv, t_mv = meta['student_move'], meta['teacher_move']
    corpus = torch.load(a.corpus, map_location='cpu', weights_only=False)
    boards, next_pos = corpus['boards'], corpus['next_pos']
    next_col, n_next = corpus['next_col'], corpus['n_next']
    pol_idx = corpus['pol_indices'].numpy()  # teacher top-5 (labels)
    n = boards.shape[0]
    ds = TensorDatasetGPU(a.corpus, augment=False, color_augment=False,
                          augment_factor=1, device=a.device)

    print('vh1 fp16 pass (argmax + top-2 margin)...', flush=True)
    net, _ = load_model(a.base, dev, fp16=True)
    v_arg, v_margin = fp16_argmax_margin(net, ds, n, dev)
    del net
    print('trained fp16 pass...', flush=True)
    net, _ = load_model(a.trained, dev, fp16=True)
    tr_arg, _ = fp16_argmax_margin(net, ds, n, dev)
    del net

    conf = dis & (gap >= 0.5)
    c = np.where(conf)[0]
    print(f'\n=== fp16 protocol, {len(c):,} confident correction rows ===')
    print(f'recorded action == vh1 fp16 argmax : '
          f'{100 * (v_arg[c] == s_mv[c]).mean():.1f}%')
    print(f'vh1 already plays teacher move     : '
          f'{100 * (v_arg[c] == t_mv[c]).mean():.1f}%')
    absorb = (tr_arg[c] == t_mv[c])
    keep = (tr_arg[c] == v_arg[c]) & ~absorb
    third = ~absorb & ~keep
    in5 = np.array([tr_arg[i] in pol_idx[i] for i in c])
    print(f'trained: absorb {100 * absorb.mean():.1f}%  keep-vh1 '
          f'{100 * keep.mean():.1f}%  third {100 * third.mean():.1f}% '
          f'(third in teacher top-5: '
          f'{100 * in5[third].mean():.1f}%)')

    # ---- row-level judge export: gap x student-margin cells ----
    med = np.median(v_margin[c])
    print(f'\nvh1 top-2 margin on correction rows: median {med:.3f}')
    cells = {
        'g1_msmall': conf & (gap >= 1.0) & (v_margin < med),
        'g1_mlarge': conf & (gap >= 1.0) & (v_margin >= med),
        'gmid_msmall': conf & (gap < 1.0) & (v_margin < med),
        'gmid_mlarge': conf & (gap < 1.0) & (v_margin >= med),
        'glow': dis & (gap < 0.5),
    }
    rows, cell_names = [], []
    for name, m in cells.items():
        idx = np.where(m)[0]
        take = rng.choice(idx, min(a.n_rowjudge, len(idx)), replace=False)
        rows += list(take)
        cell_names += [name] * len(take)
        print(f'  {name:12s}: pool {len(idx):6,} -> {len(take)}')
    write_bin(f'{DATA}/rowjudge_states.bin', rows, boards, next_pos, next_col,
              n_next, t_mv[rows], s_mv[rows], gap[rows])
    with open(f'{DATA}/rowjudge_meta.csv', 'w') as f:
        f.write('state,original_seed,label,turns,gap\n')
        for j, i in enumerate(rows):
            f.write(f'{j},{meta["seed"][i]},{cell_names[j]},{meta["turn"][i]},'
                    f'{gap[i]:.4f}\n')

    # ---- third-action judge export: trained third vs vh1's action ----
    tc = c[third]
    take = rng.choice(tc, min(a.n_third, len(tc)), replace=False)
    write_bin(f'{DATA}/thirdjudge_states.bin', take, boards, next_pos, next_col,
              n_next, tr_arg[take], s_mv[take], gap[take])
    with open(f'{DATA}/thirdjudge_meta.csv', 'w') as f:
        f.write('state,original_seed,label,turns,gap\n')
        for j, i in enumerate(take):
            f.write(f'{j},{meta["seed"][i]},third,{meta["turn"][i]},'
                    f'{gap[i]:.4f}\n')
    np.savez('alphatrain/data/dagger_v1_fp16_diag.npz',
             v_arg=v_arg, v_margin=v_margin, tr_arg=tr_arg)
    print('done', flush=True)


if __name__ == '__main__':
    main()
