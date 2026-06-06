"""cProfile batched_search to find what to vectorize first (Stage 2b/2c)."""
import os, sys, cProfile, pstats, io, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'


def main():
    from alphatrain.evaluate import load_model
    from alphatrain.batched_mcts import batched_search
    from scripts.validate_batched_mcts import _states, _pack
    dev = torch.device('mps')
    net, _ = load_model(MODEL, dev, fp16=True)
    dtype = next(net.parameters()).dtype
    d = np.load(FV)
    fv = (d['coefs'].astype(np.float32), d['means'].astype(np.float32),
          d['stds'].astype(np.float32), float(d['bias']))
    states = _states('crisis/death_games/death_*.json', 16, depths=(30, 45, 60, 75))
    boards, npos, ncol, nn = _pack(states)
    rng = np.random.default_rng(0)
    # warm
    batched_search(net, dev, dtype, boards, npos, ncol, nn, fv, np.random.default_rng(1), sims=20)
    pr = cProfile.Profile(); pr.enable()
    batched_search(net, dev, dtype, boards, npos, ncol, nn, fv, rng, sims=300)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(16)
    print(f"=== batched_search K={len(states)} sims=300 ===")
    print('\n'.join(l for l in s.getvalue().splitlines()
                    if l.strip() and 'function calls' not in l)[:2600])


if __name__ == '__main__':
    main()
