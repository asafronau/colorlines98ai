"""Hybrid NN+heuristic player: does fixing move quality raise the floor?

Per turn: NN policy gives a probability over legal moves. Among moves within
margin delta of the NN top-1 probability, pick the one the CMA-ES heuristic
(fast_heuristic._evaluate_move_with_next) scores highest. delta=0 reproduces
pure-NN argmax (baseline). Larger delta lets the heuristic override the NN
among moves the NN still considers plausible.

Motivation (worst-game autopsy, seed 835): the NN puts high mass on
heuristically-dead moves (e.g. (0,0)->(2,8) heur 1.38, NN 0.206) and ~0 on
constructive ones. Rollout-outcome judges can't see this (continuation
confound). The only way to test if it matters for the FLOOR is to play a
uniformly-better-move policy and eval real games.

Sweeps delta over a grid, reports floor stats (P5, <1000 rate) + mean per delta.
CPU fp32 → deterministic and reproducible.

Usage:
    PYTHONPATH=. python scripts/hybrid_eval.py \\
        --model alphatrain/data/pillar3b_epoch_20.pt \\
        --n-seeds 1000 --max-turns 500 --workers 12 \\
        --deltas 0.0 0.10 0.30
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, set_start_method

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.fast_heuristic import _evaluate_move_with_next
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation

_W_NET = None
_W_DEVICE = None
_W_DTYPE = None


def _init_worker(model_path):
    global _W_NET, _W_DEVICE, _W_DTYPE
    _W_DEVICE = torch.device('cpu')
    _W_NET, _ = load_model(model_path, _W_DEVICE, fp16=False)
    _W_DTYPE = next(_W_NET.parameters()).dtype


def _legal_probs(game):
    """Return list of (move, prob) over ALL legal moves, softmax of logits."""
    board = game.board
    nr = np.zeros(3, dtype=np.intp); nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    obs = torch.from_numpy(build_observation(board, nr, nc, ncol, nn)
                           ).unsqueeze(0).to(_W_DEVICE, _W_DTYPE)
    with torch.no_grad():
        logits = _W_NET(obs)[0].float().numpy()
    src = game.get_source_mask()
    moves, vals = [], []
    for sr in range(9):
        for sc in range(9):
            if src[sr, sc] == 0:
                continue
            tm = game.get_target_mask((sr, sc))
            for tr in range(9):
                for tc in range(9):
                    if tm[tr, tc] > 0:
                        moves.append(((sr, sc), (tr, tc)))
                        vals.append(logits[(sr * 9 + sc) * 81 + tr * 9 + tc])
    if not moves:
        return []
    a = np.array(vals, dtype=np.float64); a -= a.max()
    p = np.exp(a); p /= p.sum()
    return list(zip(moves, p))


def _hybrid_move(game, delta):
    """NN top-1 if delta==0; else heuristic-best among NN-plausible moves."""
    lp = _legal_probs(game)
    if not lp:
        return None
    top1_p = max(p for _, p in lp)
    if delta <= 0.0:
        return max(lp, key=lambda x: x[1])[0]
    # Candidate set: moves within delta of top-1 probability
    cands = [(m, p) for m, p in lp if p >= top1_p - delta]
    board = game.board
    nr = np.zeros(3, dtype=np.intp); nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    best_m, best_key = None, None
    for m, p in cands:
        (sr, sc), (tr, tc) = m
        color = int(board[sr, sc])
        h = _evaluate_move_with_next(board.copy(), sr, sc, tr, tc, color,
                                     nr, nc, ncol, nn)
        key = (h, p)  # heuristic first, NN prob as tiebreak
        if best_key is None or key > best_key:
            best_key, best_m = key, m
    return best_m


def _play(args):
    seed, delta, max_turns = args
    g = ColorLinesGame(seed=seed)
    g.reset()
    while not g.game_over and g.turns < max_turns:
        mv = _hybrid_move(g, delta)
        if mv is None:
            g.game_over = True
            break
        r = g.move(*mv)
        if not r['valid']:
            g.game_over = True
            break
    return (seed, delta, int(g.score), int(g.turns), bool(g.game_over))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--n-seeds', type=int, default=1000)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--max-turns', type=int, default=500)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--deltas', type=float, nargs='+',
                   default=[0.0, 0.10, 0.30])
    p.add_argument('--out', default='alphatrain/data/hybrid_eval.json')
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    units = [(s, d, args.max_turns) for d in args.deltas for s in seeds]
    print(f"Hybrid eval: {len(seeds)} seeds x {len(args.deltas)} deltas "
          f"= {len(units)} games, max_turns={args.max_turns}", flush=True)

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    results = {d: [] for d in args.deltas}
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.model,)) as pool:
        n = 0
        for seed, delta, score, turns, over in pool.imap_unordered(
                _play, units, chunksize=4):
            results[delta].append((seed, score, turns, over))
            n += 1
            if n % 200 == 0:
                el = time.time() - t0
                print(f"  [{n}/{len(units)}] {el:.0f}s "
                      f"eta={el/n*(len(units)-n):.0f}s", flush=True)

    print(f"\n{'delta':>6} | {'mean':>7} {'P5':>6} {'P10':>6} {'P25':>7} "
          f"{'<1000':>7} {'<500':>6} {'min':>5} {'capped%':>7}", flush=True)
    print('-' * 72, flush=True)
    summary = {}
    for d in args.deltas:
        sc = np.array([r[1] for r in results[d]])
        tn = np.array([r[2] for r in results[d]])
        capped = 100 * np.mean(tn >= args.max_turns)
        summary[d] = {
            'mean': float(sc.mean()), 'p5': float(np.percentile(sc, 5)),
            'p10': float(np.percentile(sc, 10)),
            'p25': float(np.percentile(sc, 25)),
            'lt1000': float(100 * np.mean(sc < 1000)),
            'lt500': float(100 * np.mean(sc < 500)),
            'min': int(sc.min()), 'capped_pct': float(capped),
        }
        s = summary[d]
        print(f"{d:>6.2f} | {s['mean']:>7.0f} {s['p5']:>6.0f} "
              f"{s['p10']:>6.0f} {s['p25']:>7.0f} {s['lt1000']:>6.1f}% "
              f"{s['lt500']:>5.1f}% {s['min']:>5} {s['capped_pct']:>6.1f}%",
              flush=True)

    with open(args.out, 'w') as f:
        json.dump({'config': vars(args), 'summary': summary,
                   'results': {str(d): results[d] for d in args.deltas}}, f)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == '__main__':
    main()
