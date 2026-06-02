"""Profile batched_rollout to find the real bottleneck before optimizing.

Reports (a) wall-clock turns/s, (b) a macro-phase breakdown (obs build / GPU
forward+sync / game-logic), and (c) cProfile top functions. This tells us
whether to parallelize across cores (CPU-bound game logic) or pipeline the
GPU (sync-bound) — rather than guessing.

    PYTHONPATH=. python scripts/profile_rollout.py --game alphatrain/data/death_games/death_50084.json
"""
import os
import sys
import time
import json
import argparse
import cProfile
import pstats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
from scripts.batched_rollout import restore, _decode, _result, _Slot


def timed_batched_rollout(net, device, dtype, jobs, horizon, batch, prof):
    """A copy of batched_rollout with per-macro-phase timers (prof dict)."""
    results = [None] * len(jobs)
    nxt = 0

    def make_slot():
        nonlocal nxt
        while nxt < len(jobs):
            j, nxt = nxt, nxt + 1
            anchor, first_move, seed = jobs[j]
            g = restore(anchor, seed)
            if not g.move(*first_move)['valid']:
                results[j] = {'died': True, 'turns': 0, 'score': int(g.score),
                              'illegal': True}
                continue
            base = g.turns
            if g.game_over:
                results[j] = _result(g, base)
                continue
            return _Slot(j, g, base)
        return None

    t = time.perf_counter
    t0 = make_slot
    slots = [s for s in (make_slot() for _ in range(batch)) if s is not None]
    while slots:
        a = t()
        obs = np.stack([_build_obs_for_game(s.game) for s in slots])
        b = t()
        with torch.no_grad():
            logits = net(torch.from_numpy(obs).to(device, dtype)).float().cpu().numpy()
        c = t()
        survivors, finalized = [], 0
        for i, s in enumerate(slots):
            priors = _get_legal_priors_flat(s.game.board, logits[i], 30)
            if not priors or not s.game.move(
                    *_decode(max(priors.items(), key=lambda x: x[1])[0]))['valid']:
                results[s.job_idx] = _result(s.game, s.base)
                finalized += 1
            elif s.game.game_over or (s.game.turns - s.base) >= horizon:
                results[s.job_idx] = _result(s.game, s.base)
                finalized += 1
            else:
                survivors.append(s)
        d = t()
        for _ in range(finalized):
            ns = make_slot()
            if ns is not None:
                survivors.append(ns)
        slots = survivors
        e = t()
        prof['obs'] += b - a
        prof['fwd'] += c - b
        prof['logic'] += d - c
        prof['refill'] += e - d
        prof['turns'] += len(slots) + finalized
    return results


def build_jobs(net, dev, dtype, game_path, depth, n_cand, seeds):
    d = json.load(open(game_path))
    fr = d['frames'][len(d['frames']) - 1 - depth]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    g0 = restore(anchor, 0)
    obs = torch.from_numpy(_build_obs_for_game(g0)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g0.board, logits, 64)
    cand = [_decode(m) for m, _ in sorted(pri.items(), key=lambda x: -x[1])[:n_cand]]
    return [(anchor, c, s) for c in cand for s in range(seeds)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='alphatrain/data/death_games/death_50084.json')
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--depth', type=int, default=30)
    p.add_argument('--n-cand', type=int, default=10)
    p.add_argument('--seeds', type=int, default=100)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--batch', type=int, default=128)
    a = p.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=True)
    dtype = next(net.parameters()).dtype
    jobs = build_jobs(net, dev, dtype, a.game, a.depth, a.n_cand, a.seeds)
    print(f"device={dev} dtype={dtype} jobs={len(jobs)} batch={a.batch} "
          f"horizon={a.horizon}\n", flush=True)

    # warmup (numba JIT + MPS graph)
    _ = timed_batched_rollout(net, dev, dtype, jobs[:a.batch], 30,
                              a.batch, {'obs': 0, 'fwd': 0, 'logic': 0,
                                        'refill': 0, 'turns': 0})

    prof = {'obs': 0.0, 'fwd': 0.0, 'logic': 0.0, 'refill': 0.0, 'turns': 0}
    t0 = time.perf_counter()
    timed_batched_rollout(net, dev, dtype, jobs, a.horizon, a.batch, prof)
    wall = time.perf_counter() - t0
    tot = prof['obs'] + prof['fwd'] + prof['logic'] + prof['refill']
    print(f"WALL {wall:.1f}s  turns {prof['turns']}  "
          f"{prof['turns']/wall:.0f} turns/s\n")
    print(f"  obs build   {prof['obs']:6.1f}s  {100*prof['obs']/tot:4.0f}%")
    print(f"  GPU fwd+sync{prof['fwd']:6.1f}s  {100*prof['fwd']/tot:4.0f}%")
    print(f"  game logic  {prof['logic']:6.1f}s  {100*prof['logic']/tot:4.0f}%")
    print(f"  refill      {prof['refill']:6.1f}s  {100*prof['refill']/tot:4.0f}%")

    print("\n=== cProfile (top 15 by cumulative) ===")
    pr = cProfile.Profile()
    pr.enable()
    timed_batched_rollout(net, dev, dtype, jobs, a.horizon, a.batch,
                          {'obs': 0, 'fwd': 0, 'logic': 0, 'refill': 0, 'turns': 0})
    pr.disable()
    st = pstats.Stats(pr).sort_stats('cumulative')
    st.print_stats(15)


if __name__ == '__main__':
    main()
