"""Does override QUALITY rise with MCTS sim depth? (predicts whether crisis_v16@2400/3600 helps)

Self-play/crisis games store, per move, cand_moves + cand_visits + cand_prior. An OVERRIDE =
argmax(visits) != argmax(prior) (MCTS picked a different move than the policy). At LOW sims
the override is prior-dominated NOISE; at HIGH sims it's a genuine correction. This measures
the genuine RATE by rollout: play pillar3f greedily from A=MCTS-top vs B=policy-top over R
common-RNG rollouts to death; A beats B => genuine.

Run on existing data at different sims to see the trend (400 selfplay, 600-800 crisis_v15),
then on a new crisis_v16@2400/3600 sample to confirm higher sims -> higher genuine rate.

    PYTHONPATH=. python scripts/verify_game_overrides.py --glob 'data/selfplay_v15/*' --n 60
    PYTHONPATH=. python scripts/verify_game_overrides.py --glob 'data/crisis_v15/*'  --n 60
"""
import os, sys, glob, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from scripts.batched_rollout import batched_rollout, _decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--glob', required=True, help="game JSON glob (data/selfplay_v15/* etc)")
    p.add_argument('--model', default='alphatrain/data/pillar3f.pt')
    p.add_argument('--n', type=int, default=60, help='override moves to test')
    p.add_argument('--R', type=int, default=64)
    p.add_argument('--horizon', type=int, default=800)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--device', default='mps')
    p.add_argument('--seed-base', type=int, default=808000)
    p.add_argument('--max-games', type=int, default=400)
    p.add_argument('--first-moves', type=int, default=0,
                   help='Only consider overrides in the first N moves of each game '
                        '(the crisis region near the rewind point). 0 = all moves.')
    a = p.parse_args()

    dev = torch.device(a.device)
    net, _ = load_model(a.model, dev, fp16=True)
    dtype = next(net.parameters()).dtype

    # collect override moves: argmax(visits) != argmax(prior)
    ovr = []
    files = sorted(glob.glob(a.glob))[:a.max_games]
    for f in files:
        try: g = json.load(open(f))
        except Exception: continue
        for mi, m in enumerate(g.get('moves', [])):
            if a.first_moves and mi >= a.first_moves:
                break
            cm, cv, cp = m.get('cand_moves'), m.get('cand_visits'), m.get('cand_prior')
            if not cm or not cv or not cp or len(cm) < 2:
                continue
            ai, bi = int(np.argmax(cv)), int(np.argmax(cp))
            if cm[ai] == cm[bi]:
                continue
            nb = [((nb_['row'], nb_['col']), nb_['color']) for nb_ in m['next_balls']]
            ovr.append({'board': m['board'], 'next_balls': nb,
                        'A': int(cm[ai]), 'B': int(cm[bi]),
                        'vis_share': cv[ai] / max(sum(cv), 1e-9)})
        if len(ovr) >= a.n * 4:
            break
    if not ovr:
        print("no overrides found"); return
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ovr), size=min(a.n, len(ovr)), replace=False)
    sample = [ovr[i] for i in idx]
    print(f"{a.glob}: {len(sample)} overrides (of {len(ovr)} found), R={a.R} common-RNG "
          f"rollouts, horizon {a.horizon}, pillar3f greedy", flush=True)

    jobs, meta = [], []
    for k, o in enumerate(sample):
        anchor = {'board': o['board'], 'next_balls': o['next_balls'], 'score': 0, 'turn': 100}
        for tag, mv in (('A', _decode(o['A'])), ('B', _decode(o['B']))):
            for s in range(a.R):
                jobs.append((anchor, mv, a.seed_base + s)); meta.append((k, tag))
    res = batched_rollout(net, dev, dtype, jobs, a.horizon, batch=a.batch)

    per = {}
    for (k, tag), r in zip(meta, res):
        per.setdefault(k, {'A': [], 'B': []})[tag].append(r)
    A_win = A_safe = n = 0
    dturns, dcat = [], []
    for k, dd in per.items():
        At = np.array([r['turns'] for r in dd['A']]); Ad = np.array([r['died'] for r in dd['A']])
        Bt = np.array([r['turns'] for r in dd['B']]); Bd = np.array([r['died'] for r in dd['B']])
        n += 1
        dt = At.mean() - Bt.mean(); dc = 100*(Ad.mean() - Bd.mean())
        dturns.append(dt); dcat.append(dc)
        A_win += dt > 0; A_safe += dc < 0
    dturns = np.array(dturns)
    print(f"\n=== {a.glob} ===", flush=True)
    print(f"A (MCTS-top) beats B (policy-top) on survival: {A_win}/{n} = {100*A_win/n:.0f}%", flush=True)
    print(f"A lower catastrophe than B:                     {A_safe}/{n} = {100*A_safe/n:.0f}%", flush=True)
    print(f"mean turns delta (A-B): mean {dturns.mean():+.0f} median {np.median(dturns):+.0f}", flush=True)
    print(f"(compare: 4800-sim forks = 67% / +25 turns; >67% here => higher sims help)", flush=True)


if __name__ == '__main__':
    main()
