"""Validate batched_search vs scalar MCTS on real band states.

Not bit-identical (no virtual loss + fp16 batch + independent spawn draws); gate on
argmax agreement + low TV of the root visit distribution. --smoke runs tiny (no scalar)
to catch crashes / sanity-check the distribution first.

    PYTHONPATH=. python scripts/validate_batched_mcts.py --smoke
    PYTHONPATH=. python scripts/validate_batched_mcts.py --sims 4800 --n 8
"""
import os, sys, time, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'


def _states(death_glob, n, depths):
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


def _pack(states):
    """band frames -> batched arrays (boards, next_pos, next_col, next_n)."""
    K = len(states)
    boards = np.zeros((K, 9, 9), dtype=np.int8)
    npos = np.zeros((K, 3, 2), dtype=np.int8)
    ncol = np.zeros((K, 3), dtype=np.int8)
    nn = np.zeros(K, dtype=np.int8)
    for k, (_, _, fr) in enumerate(states):
        boards[k] = np.array(fr['board'], dtype=np.int8)
        nb = fr['next_balls']
        nn[k] = min(3, len(nb))
        for i in range(int(nn[k])):
            (pr, pc), col = nb[i]
            npos[k, i, 0] = pr; npos[k, i, 1] = pc; ncol[k, i] = col
    return boards, npos, ncol, nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL)
    p.add_argument('--death-glob', default='crisis/death_games/death_*.json')
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--top-k', type=int, default=300)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--mcts-seeds', type=int, default=3)
    p.add_argument('--device', default='mps')
    p.add_argument('--smoke', action='store_true', help='tiny run, no scalar compare')
    p.add_argument('--gpu', action='store_true', help='use batched_mcts_gpu (torch) search')
    a = p.parse_args()

    from alphatrain.evaluate import load_model
    from alphatrain.batched_mcts import batched_search
    if a.gpu:
        from alphatrain.batched_mcts_gpu import batched_search_gpu
        def batched_search(net, dev, dtype, b, p_, c, n, fv, rng, sims, top_k):  # noqa
            return batched_search_gpu(net, dev, dtype, b, p_, c, n, fv, sims=sims, top_k=top_k)
    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=(dev.type != 'cpu'))
    dtype = next(net.parameters()).dtype
    d = np.load(FV)
    fv = (d['coefs'].astype(np.float32), d['means'].astype(np.float32),
          d['stds'].astype(np.float32), float(d['bias']))

    if a.smoke:
        states = _states(a.death_glob, 4, depths=(30, 50))
        boards, npos, ncol, nn = _pack(states)
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        dist = batched_search(net, dev, dtype, boards, npos, ncol, nn, fv, rng,
                              sims=200, top_k=a.top_k)
        dt = time.perf_counter() - t0
        print(f"smoke: K={len(states)} sims=200 -> {dt:.1f}s", flush=True)
        for k, (sd, dep, _) in enumerate(states):
            nz = (dist[k] > 0).sum()
            top = int(dist[k].argmax())
            print(f"  {sd}:D-{dep}  sum={dist[k].sum():.4f}  nonzero={nz}  "
                  f"top={top}({dist[k][top]:.3f})", flush=True)
        assert np.allclose(dist.sum(axis=1), 1.0, atol=1e-6), "dist not normalized"
        print("smoke OK", flush=True)
        return

    # full: scalar reference (3-seed averaged) vs batched
    from alphatrain.mcts import MCTS
    from game.board import ColorLinesGame
    mcts = MCTS(net=net, device=dev, num_simulations=a.sims, c_puct=2.5,
                top_k=a.top_k, batch_size=8, feature_weights_path=FV, q_weight=2.0)
    states = _states(a.death_glob, a.n, depths=(30, 45, 60, 75))
    boards, npos, ncol, nn = _pack(states)

    # scalar reference
    def scalar_visits(fr):
        vs = np.zeros(6561)
        for s in range(a.mcts_seeds):
            g = ColorLinesGame()
            g.reset(board=np.array(fr['board'], dtype=np.int8),
                    next_balls=[(tuple(pp), int(cc)) for pp, cc in fr['next_balls']])
            g.score, g.turns = 0, int(fr['turn']) + s * 1_000_003
            _, pt = mcts.search(g, temperature=0.0, dirichlet_alpha=0.3,
                                dirichlet_weight=0.25, return_policy=True)
            vs += np.asarray(pt)
        return vs / vs.sum()

    print(f"validating batched vs scalar: {len(states)} states, sims={a.sims}\n", flush=True)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    bdist = batched_search(net, dev, dtype, boards, npos, ncol, nn, fv, rng,
                           sims=a.sims, top_k=a.top_k)
    tb = time.perf_counter() - t0
    agree = 0; tvs = []
    t0 = time.perf_counter()
    for k, (sd, dep, fr) in enumerate(states):
        sv = scalar_visits(fr)
        ib, isv = int(bdist[k].argmax()), int(sv.argmax())
        ok = ib == isv; agree += ok
        tv = 0.5 * np.abs(bdist[k] - sv).sum()
        tvs.append(tv)
        print(f"  {sd}:D-{dep}  {'AGREE' if ok else 'DIFFER'}  "
              f"batched_top={ib}({bdist[k][ib]:.2f}) scalar_top={isv}({sv[isv]:.2f}) TV={tv:.3f}",
              flush=True)
    ts = time.perf_counter() - t0
    print(f"\n=== agreement {agree}/{len(states)}  meanTV={np.mean(tvs):.3f} ===", flush=True)
    print(f"  batched (all {len(states)} at once): {tb:.0f}s | "
          f"scalar ({len(states)}x{a.mcts_seeds} seq): {ts:.0f}s", flush=True)


if __name__ == '__main__':
    main()
