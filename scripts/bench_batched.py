"""Benchmark batched_search throughput at various K -> per-tree time vs scalar (~4.5s/tree)."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ks', default='16,32,64')
    p.add_argument('--sims', type=int, default=300)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    from alphatrain.evaluate import load_model
    from alphatrain.batched_mcts import batched_search
    from scripts.validate_batched_mcts import _states, _pack
    dev = torch.device(a.device)
    net, _ = load_model(MODEL, dev, fp16=(dev.type != 'cpu'))
    dtype = next(net.parameters()).dtype
    d = np.load(FV)
    fv = (d['coefs'].astype(np.float32), d['means'].astype(np.float32),
          d['stds'].astype(np.float32), float(d['bias']))
    base = _states('crisis/death_games/death_*.json', 8, depths=(30, 45, 60, 75))
    boards8, npos8, ncol8, nn8 = _pack(base)

    print(f"sims={a.sims}  (scalar ~4.5s/tree @4800 => ~{4.5*a.sims/4800:.2f}s/tree @{a.sims})", flush=True)
    for K in [int(x) for x in a.ks.split(',')]:
        reps = (K + 7) // 8
        boards = np.tile(boards8, (reps, 1, 1))[:K]
        npos = np.tile(npos8, (reps, 1, 1))[:K]
        ncol = np.tile(ncol8, (reps, 1))[:K]
        nn = np.tile(nn8, reps)[:K]
        batched_search(net, dev, dtype, boards, npos, ncol, nn, fv,
                       np.random.default_rng(1), sims=20)  # warm
        t0 = time.perf_counter()
        batched_search(net, dev, dtype, boards, npos, ncol, nn, fv,
                       np.random.default_rng(0), sims=a.sims)
        dt = time.perf_counter() - t0
        per_tree = dt / K
        ext4800 = per_tree * 4800 / a.sims
        print(f"  K={K:4d}: {dt:6.1f}s total | {per_tree*1000:6.0f}ms/tree | "
              f"~{ext4800:5.2f}s/tree @4800  ({4.5/ext4800:.1f}x vs scalar)", flush=True)


if __name__ == '__main__':
    main()
