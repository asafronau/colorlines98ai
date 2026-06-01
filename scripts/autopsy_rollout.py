"""Rollout-judge a specific decision in a recorded game.

Restores the exact board state at a given frame of a worst_game.json, then
runs R common-RNG rollouts (policy-argmax continuation, horizon H) for the
policy's actual move AND a set of alternative moves. Reports floor metrics
per candidate so we can tell whether a visually-suspicious move actually
costs anything (the image-3 lesson: looked wrong, rollouts said fine).

Usage:
    PYTHONPATH=. python scripts/autopsy_rollout.py \\
        --game alphatrain/data/worst_game.json --frame 10 \\
        --alt 0,0,7,4 --alt 0,0,7,7 --alt 3,1,0,7 \\
        --R 64 --H 100
(--alt is sr,sc,tr,tc; the policy's own move is always included.)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from multiprocessing import Pool, set_start_method

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from scripts.mine_stationary_counterfactuals import (
    policy_argmax, restore_game, largest_empty_component)
from alphatrain.evaluate import load_model

_W_NET = None
_W_DEVICE = None
_W_DTYPE = None


def _init_worker(model_path, device_str):
    global _W_NET, _W_DEVICE, _W_DTYPE
    _W_DEVICE = torch.device('cpu')  # cpu fp32: deterministic, reproducible
    _W_NET, _ = load_model(model_path, _W_DEVICE, fp16=False)
    _W_DTYPE = next(_W_NET.parameters()).dtype


def _rollout(anchor, first_move, rollout_seed, H):
    from game.rng import SimpleRng
    g = restore_game(anchor)
    g.rng = SimpleRng(rollout_seed)
    start = g.score
    res = g.move(*first_move)
    if not res['valid']:
        return None
    died = False
    for _ in range(H - 1):
        if g.game_over:
            died = True
            break
        mv = policy_argmax(_W_NET, _W_DEVICE, _W_DTYPE, g)
        if mv is None:
            died = True
            break
        r = g.move(*mv)
        if not r['valid']:
            died = True
            break
    if g.game_over:
        died = True
    return {
        'score_gained': int(g.score - start),
        'died': died,
        'final_empties': int((g.board == 0).sum()),
        'final_lec': largest_empty_component(g.board),
        'turns': int(g.turns - anchor['turn']),
    }


def _work(args):
    anchor, label, move, rs, H = args
    return (label, _rollout(anchor, tuple(move), rs, H))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='alphatrain/data/worst_game.json')
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--alt', action='append', default=[],
                   help='Alternative move sr,sc,tr,tc (repeatable).')
    p.add_argument('--R', type=int, default=64)
    p.add_argument('--H', type=int, default=100)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()

    d = json.load(open(args.game))
    fr = d['frames'][args.frame]
    anchor = {
        'board': fr['board'],
        'next_balls': fr['next_balls'],
        'score': fr.get('score_before', fr['score']),
        'turn': fr['turn'],
    }
    pol_move = tuple(map(tuple, fr['chosen_move']))
    print(f"Game seed={d['seed']} frame={args.frame} turn={fr['turn']} "
          f"score={anchor['score']} empties={fr['empties']} "
          f"lec={fr['lec']} ncomp={fr['n_components']}", flush=True)
    print(f"Policy move: {pol_move}  (prob {fr['top_k'][0]['prob']:.3f})",
          flush=True)

    # Candidate set: policy move + alternatives
    candidates = {'policy': [list(pol_move[0]), list(pol_move[1])]}
    for a in args.alt:
        sr, sc, tr, tc = (int(x) for x in a.split(','))
        candidates[f'alt({sr},{sc})->({tr},{tc})'] = [[sr, sc], [tr, tc]]

    units = []
    for label, mv in candidates.items():
        for rs in range(args.R):
            units.append((anchor, label, mv, rs, args.H))

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    by_label = {k: [] for k in candidates}
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.model, 'cpu')) as pool:
        for label, res in pool.imap_unordered(_work, units, chunksize=4):
            by_label[label].append(res)

    print(f"\n{'candidate':>26} | {'valid':>5} {'die%':>6} {'mean_sc':>8} "
          f"{'p10_sc':>7} {'mean_emp':>8} {'p10_emp':>7}", flush=True)
    print('-' * 84, flush=True)
    for label in candidates:
        rs = [r for r in by_label[label] if r is not None]
        if not rs:
            print(f"{label:>26} |  INVALID move (unreachable)", flush=True)
            continue
        die = 100 * np.mean([r['died'] for r in rs])
        sc = np.array([r['score_gained'] for r in rs])
        emp = np.array([r['final_empties'] for r in rs])
        print(f"{label:>26} | {len(rs):>5} {die:>6.1f} {sc.mean():>8.1f} "
              f"{np.percentile(sc,10):>7.1f} {emp.mean():>8.1f} "
              f"{np.percentile(emp,10):>7.1f}", flush=True)


if __name__ == '__main__':
    main()
