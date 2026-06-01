"""Single-process batched rollout engine for crisis mining.

Holds B independent rollouts in flight, does ONE GPU forward per step over the
whole batch, and refills a slot from the job queue whenever a rollout finishes.
This pushes the GPU into the compute-bound regime (where fp16 actually delivers
its ~2x) and amortizes the per-forward overhead — far more throughput than the
1-obs-per-worker server path, with no multiprocessing/shared-memory/None-drain.

A "job" = (anchor, first_move, seed): restore the board, seed the RNG, play the
forced candidate move, then play POLICY argmax for `horizon` turns.
catastrophe = died within `horizon` (anchor-relative; valid at any game length).

`naive_rollout` is the one-at-a-time reference. In fp32 (batch-invariant) the
batched engine must reproduce it bit-for-bit — that's the correctness test
(run this file directly).

Usage (self-test):
    PYTHONPATH=. python scripts/batched_rollout.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame
from game.rng import SimpleRng
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat


def _decode(flat):
    s, t = flat // 81, flat % 81
    return ((s // 9, s % 9), (t // 9, t % 9))


def restore(anchor, seed):
    """Fresh game at the anchor board/next_balls/score/turn, RNG seeded."""
    g = ColorLinesGame()
    g.reset(board=np.array(anchor['board'], dtype=np.int8),
            next_balls=[(tuple(p), int(c)) for p, c in anchor['next_balls']])
    g.score = int(anchor['score'])
    g.turns = int(anchor['turn'])
    g.rng = SimpleRng(int(seed))
    return g


def _policy_move(net, device, dtype, game):
    """One policy forward (batch 1) → argmax legal move, or None."""
    obs = torch.from_numpy(_build_obs_for_game(game)).unsqueeze(0).to(device, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    priors = _get_legal_priors_flat(game.board, logits, 30)
    if not priors:
        return None
    best = max(priors.items(), key=lambda x: x[1])[0]
    return _decode(best)


def _result(g, base, illegal=False):
    return {'died': bool(g.game_over), 'turns': int(g.turns - base),
            'score': int(g.score), 'illegal': illegal}


def naive_rollout(net, device, dtype, anchor, first_move, seed, horizon):
    """Reference: one rollout, one obs per forward. (anchor + move + seed)."""
    g = restore(anchor, seed)
    res = g.move(*first_move)
    base = g.turns                      # branch point = state after candidate move
    if not res['valid']:
        return {'died': True, 'turns': 0, 'score': int(g.score), 'illegal': True}
    while not g.game_over and (g.turns - base) < horizon:
        mv = _policy_move(net, device, dtype, g)
        if mv is None:
            break                       # no legal move: stop; died = g.game_over
        if not g.move(*mv)['valid']:
            break
    return _result(g, base)


class _Slot:
    __slots__ = ('job_idx', 'game', 'base', 'done', 'result')

    def __init__(self, job_idx, game, base):
        self.job_idx = job_idx
        self.game = game
        self.base = base
        self.done = False
        self.result = None


def batched_rollout(net, device, dtype, jobs, horizon, batch=128):
    """jobs: list of (anchor, first_move, seed). Returns results aligned to jobs.

    Mirrors naive_rollout's control flow exactly, but evaluates all live slots
    in one GPU forward per step and refills finished slots from the job queue.
    """
    results = [None] * len(jobs)
    nxt = 0

    def make_slot():
        """Next live slot: restore + candidate move; resolve dead-on-arrival."""
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
            if g.game_over:                    # candidate move itself ended it
                results[j] = _result(g, base)
                continue
            return _Slot(j, g, base)
        return None

    slots = [s for s in (make_slot() for _ in range(batch)) if s is not None]
    while slots:
        # one batched forward over all (active) slots
        obs = np.stack([_build_obs_for_game(s.game) for s in slots])
        with torch.no_grad():
            logits = net(torch.from_numpy(obs).to(device, dtype)).float().cpu().numpy()
        survivors, finalized = [], 0
        for i, s in enumerate(slots):
            priors = _get_legal_priors_flat(s.game.board, logits[i], 30)
            if not priors or not s.game.move(
                    *_decode(max(priors.items(), key=lambda x: x[1])[0]))['valid']:
                results[s.job_idx] = _result(s.game, s.base)   # stop; died = game_over
                finalized += 1
            elif s.game.game_over or (s.game.turns - s.base) >= horizon:
                results[s.job_idx] = _result(s.game, s.base)   # while-condition fails next round
                finalized += 1
            else:
                survivors.append(s)
        for _ in range(finalized):                         # refill from job queue
            ns = make_slot()
            if ns is not None:
                survivors.append(ns)
        slots = survivors
    return results


# ── self-test: batched must reproduce naive bit-for-bit in fp32 ──
def _selftest():
    import json, argparse, time
    from alphatrain.evaluate import load_model
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='alphatrain/data/death_games/death_21585.json')
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--depth', type=int, default=30)
    p.add_argument('--n-cand', type=int, default=6)
    p.add_argument('--seeds', type=int, default=40)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--fp16', action='store_true',
                   help='Throughput bench in fp16 (skips the naive bit-match, '
                        'which only holds in batch-invariant fp32).')
    a = p.parse_args()
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=a.fp16)
    dtype = next(net.parameters()).dtype

    d = json.load(open(a.game))
    fr = d['frames'][len(d['frames']) - 1 - a.depth]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    g0 = restore(anchor, 0)
    smask = g0.get_source_mask()
    obs = torch.from_numpy(_build_obs_for_game(g0)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g0.board, logits, 64)
    cand = [_decode(m) for m, _ in sorted(pri.items(), key=lambda x: -x[1])[:a.n_cand]]
    jobs = [(anchor, c, s) for c in cand for s in range(a.seeds)]
    print(f"self-test: {len(cand)} candidates x {a.seeds} seeds = {len(jobs)} jobs, "
          f"horizon {a.horizon}, fp32", flush=True)

    t0 = time.time()
    bat = batched_rollout(net, dev, dtype, jobs, a.horizon, batch=a.batch)
    t_bat = time.time() - t0
    turns = sum(b['turns'] for b in bat)
    print(f"batched ({'fp16' if a.fp16 else 'fp32'}, batch {a.batch}): {t_bat:.1f}s, "
          f"{turns} rollout-turns = {turns/max(t_bat,1e-9):.0f} turns/s "
          f"(server@bs16 was ~3100 fp32 / ~4100 fp16)", flush=True)
    if not a.fp16:
        t0 = time.time()
        naive = [naive_rollout(net, dev, dtype, *j, a.horizon) for j in jobs]
        t_naive = time.time() - t0
        mism = sum(1 for n, b in zip(naive, bat)
                   if (n['died'], n['turns']) != (b['died'], b['turns']))
        print(f"naive {t_naive:.0f}s ({t_naive/max(t_bat,1e-9):.1f}x slower) "
              f"| mismatches: {mism}/{len(jobs)}", flush=True)
    # per-candidate catastrophe (died-within-horizon) rate
    for ci, c in enumerate(cand):
        idx = range(ci * a.seeds, (ci + 1) * a.seeds)
        br = 100 * np.mean([bat[i]['died'] for i in idx])
        extra = (f"  naive {100*np.mean([naive[i]['died'] for i in idx]):5.1f}%"
                 if not a.fp16 else "")
        print(f"  {str(c):>14}: batched {br:5.1f}%{extra}", flush=True)
    if not a.fp16:
        print("REPRODUCES" if mism == 0 else "MISMATCH — engine bug", flush=True)


if __name__ == '__main__':
    _selftest()
