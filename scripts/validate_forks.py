"""Re-validate specific (frame, policy-move, alt-move) forks under the NEW
relative catastrophe metric (died-within-H), via the batched engine, fp16, R
paired seeds + paired bootstrap CI. These forks were originally found with the
OLD absolute-score metric; this confirms they hold under the corrected one.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain.evaluate import load_model
from scripts.batched_rollout import batched_rollout

# (game seed, frame, policy_move, alt_move)
FORKS = [
    (21517, 175, ((6, 7), (4, 6)), ((5, 6), (4, 7))),
    (21295, 177, ((1, 2), (6, 5)), ((4, 8), (6, 5))),
    (21585, 181, ((4, 6), (4, 5)), ((5, 5), (7, 2))),
    (21585, 186, ((5, 6), (4, 5)), ((5, 6), (5, 5))),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--R', type=int, default=500)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--fp16', action='store_true')
    a = p.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=a.fp16)
    dtype = next(net.parameters()).dtype
    rng = np.random.default_rng(0)

    print(f"R={a.R} paired seeds, horizon={a.horizon}, catastrophe=died-within-H, "
          f"fp{'16' if a.fp16 else '32'}\n", flush=True)
    print(f"{'seed':>6} {'frame':>5} {'policyMove':>12} {'altMove':>12} "
          f"{'pol%':>6} {'alt%':>6} {'Δpp':>6} {'95% CI':>14} {'verdict':>10}",
          flush=True)
    print('-' * 88, flush=True)

    for seed, frame, pol, alt in FORKS:
        d = json.load(open(f'alphatrain/data/death_games/death_{seed}.json'))
        fr = d['frames'][frame]
        anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
                  'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
        jobs = [(anchor, pol, s) for s in range(a.R)] + \
               [(anchor, alt, s) for s in range(a.R)]
        res = batched_rollout(net, dev, dtype, jobs, a.horizon, batch=a.batch)
        polc = np.array([r['died'] for r in res[:a.R]], float)
        altc = np.array([r['died'] for r in res[a.R:]], float)
        gap = 100 * (polc.mean() - altc.mean())
        boot = np.array([100 * (polc[ix].mean() - altc[ix].mean())
                         for ix in (rng.integers(0, a.R, a.R) for _ in range(2000))])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = 'REAL' if lo > 0 else ('reversed' if hi < 0 else 'n.s.')
        print(f"{seed:>6} {frame:>5} {str(pol):>12} {str(alt):>12} "
              f"{100*polc.mean():>6.1f} {100*altc.mean():>6.1f} {gap:>6.1f} "
              f"[{lo:>5.1f},{hi:>5.1f}] {verdict:>10}", flush=True)


if __name__ == '__main__':
    main()
