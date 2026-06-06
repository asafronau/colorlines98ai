"""Closed-loop ENSEMBLE variance test (ChatGPT-recommended gate for the determinized teacher).

Single-determinization closed-loop is biased (each node freezes one spawn -> the search conditions
on luck; meanTV ~0.63 vs scalar). Question: is it high-variance-but-AVERAGEABLE? Replicate each root
into M of the K parallel closed-loop slots (each slot draws its own spawn realization -> M
independent determinizations, batched in K for free). Then:
  - intra-ensemble: do the M determinizations agree on the top move? mean pairwise TV (the spread).
  - ENSEMBLE (mean of the M visit dists) vs scalar/open-loop reference: argmax + TV.
Read (ChatGPT): ensemble TV -> ~0.27 (open-loop's) => closed-loop viable WITH M-batching;
ensemble TV still > 0.45 => closed-loop semantics are wrong for this teacher.

    PYTHONPATH=. python scripts/test_closed_ensemble.py --device cuda --n 8 --m 8 --sims 4800 \
        --model /content/drive/MyDrive/alphatrain/pillar3b_epoch_20.pt
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'


def _load_model(model_path, dev, fp16):
    from alphatrain.model import PolicyNet
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    in_ch = state['stem.0.weight'].shape[1]
    nb = sum(1 for k in state if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    ch = state['stem.0.weight'].shape[0]
    state = {k: v for k, v in state.items() if not k.startswith('value_')}
    net = PolicyNet(in_channels=in_ch, num_blocks=nb, channels=ch)
    net.load_state_dict(state); net.train(False); net = net.to(dev)
    if fp16 and dev.type in ('mps', 'cuda'):
        net = net.half()
    return net


def _roots(death_glob, n, depths=(30, 45, 60, 75)):
    out = []
    for f in sorted(glob.glob(death_glob)):
        g = json.load(open(f)); nf = len(g['frames'])
        for d in depths:
            i = (nf - 1) - d
            if i < 0:
                continue
            fr = g['frames'][i]
            if fr.get('chosen_move') is None:
                continue
            out.append((g['seed'], d, fr))
            if len(out) >= n:
                return out
    return out


def _tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL)
    p.add_argument('--death-glob', default='crisis/death_games/death_*.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--n', type=int, default=8, help='number of roots')
    p.add_argument('--m', type=int, default=8, help='determinizations per root')
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--top-k', type=int, default=300)
    p.add_argument('--mcts-seeds', type=int, default=3)
    a = p.parse_args()
    dev = torch.device(a.device)
    net = _load_model(a.model, dev, fp16=(dev.type != 'cpu'))
    dtype = next(net.parameters()).dtype
    d = np.load(FV)
    fv = (d['coefs'].astype(np.float32), d['means'].astype(np.float32),
          d['stds'].astype(np.float32), float(d['bias']))

    roots = _roots(a.death_glob, a.n)
    n = len(roots)
    M = a.m
    print(f"roots={n} M={M} (K={n*M}) sims={a.sims}", flush=True)

    # build K=n*M batch: each root replicated M times (each slot draws its own spawn realization)
    K = n * M
    boards = np.zeros((K, 9, 9), dtype=np.int8)
    npos = np.zeros((K, 3, 2), dtype=np.int8); ncol = np.zeros((K, 3), dtype=np.int8)
    nn = np.zeros(K, dtype=np.int8)
    for ri, (_, _, fr) in enumerate(roots):
        b = np.array(fr['board'], dtype=np.int8)
        nb = fr['next_balls']; m = min(3, len(nb))
        for j in range(M):
            k = ri * M + j
            boards[k] = b; nn[k] = m
            for i in range(m):
                (pr, pc), col = nb[i]
                npos[k, i] = (pr, pc); ncol[k, i] = col

    from alphatrain.batched_mcts_closed import batched_search_closed
    import time
    t0 = time.perf_counter()
    dist = batched_search_closed(net, dev, dtype, boards, npos, ncol, nn, fv,
                                 sims=a.sims, top_k=a.top_k, seed=0)   # [K,6561]
    print(f"  closed-loop ensemble search ({K} trees): {time.perf_counter()-t0:.0f}s", flush=True)
    dist = dist.reshape(n, M, -1)

    # scalar reference (3-seed averaged), per root
    from alphatrain.mcts import MCTS
    from game.board import ColorLinesGame
    mcts = MCTS(net=net, device=dev, num_simulations=a.sims, c_puct=2.5, top_k=a.top_k,
                batch_size=8, feature_weights_path=FV, q_weight=2.0)

    print("\n root            | intra-ens: agree  meanTV | ENSEMBLE vs scalar: argmax  TV", flush=True)
    ens_tvs, ens_agree, intra_tvs, intra_agree = [], 0, [], []
    for ri, (sd, dep, fr) in enumerate(roots):
        sv = np.zeros(6561)
        for s in range(a.mcts_seeds):
            g = ColorLinesGame()
            g.reset(board=np.array(fr['board'], dtype=np.int8),
                    next_balls=[(tuple(pp), int(cc)) for pp, cc in fr['next_balls']])
            g.score, g.turns = 0, int(fr['turn']) + s * 1_000_003
            _, pt = mcts.search(g, temperature=0.0, dirichlet_alpha=0.3,
                                dirichlet_weight=0.25, return_policy=True)
            sv += np.asarray(pt)
        sv = sv / sv.sum()
        ens = dist[ri].mean(axis=0); ens = ens / ens.sum()
        # intra-ensemble spread
        tops = [int(dist[ri, j].argmax()) for j in range(M)]
        modal = max(set(tops), key=tops.count)
        ia = tops.count(modal) / M
        intra_agree.append(ia)
        ptv = np.mean([_tv(dist[ri, x], dist[ri, y])
                       for x in range(M) for y in range(x + 1, M)]) if M > 1 else 0.0
        intra_tvs.append(ptv)
        # ensemble vs scalar
        etv = _tv(ens, sv); ens_tvs.append(etv)
        ok = int(ens.argmax()) == int(sv.argmax()); ens_agree += ok
        print(f"  {sd}:D-{dep:<3} | agree {ia:4.2f}  TV {ptv:.3f} | "
              f"{'AGREE' if ok else 'DIFFER'}  TV {etv:.3f}", flush=True)

    print(f"\n=== intra-ensemble: mean modal-agree {np.mean(intra_agree):.2f}, "
          f"mean pairwise TV {np.mean(intra_tvs):.3f} ===", flush=True)
    print(f"=== ENSEMBLE vs scalar: argmax {ens_agree}/{n}, meanTV {np.mean(ens_tvs):.3f} ===",
          flush=True)
    et = np.mean(ens_tvs)
    verdict = ("VIABLE with M-batching (ensemble TV near open-loop ~0.27)" if et < 0.33 else
               "MARGINAL — more M may help" if et < 0.45 else
               "WRONG semantics for this teacher (ensemble still > 0.45)")
    print(f"VERDICT: ensemble TV {et:.3f} -> {verdict}", flush=True)


if __name__ == '__main__':
    main()
