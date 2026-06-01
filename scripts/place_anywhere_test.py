"""Place-anywhere test: is a specific move harmful?

Take ONE ball (the source of a frame's chosen move), place it on EVERY legal
target, resume policy-only under COMMON RNG, and rank the outcomes. If the
policy's actual target ranks near the bottom, that move was actively harmful.

Multi-horizon score capture (turns 20/50/100/200 after the branch) shows
whether any tempo cost exists short-term and gets absorbed, or persists.

Usage:
    PYTHONPATH=. python scripts/place_anywhere_test.py \\
        --game alphatrain/data/worst_game.json --frame 10 \\
        --R 96 --H 200 --workers 12
"""
from __future__ import annotations
import argparse, json, os, sys
from multiprocessing import Pool, set_start_method
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.rng import SimpleRng
from alphatrain.evaluate import load_model
from scripts.mine_stationary_counterfactuals import (
    policy_argmax, restore_game, largest_empty_component)

_W_NET = _W_DEVICE = _W_DTYPE = None
CHECKPOINTS = (20, 50, 100, 200)


def _init_worker(model_path):
    global _W_NET, _W_DEVICE, _W_DTYPE
    _W_DEVICE = torch.device('cpu')
    _W_NET, _ = load_model(model_path, _W_DEVICE, fp16=False)
    _W_DTYPE = next(_W_NET.parameters()).dtype


def _rollout(anchor, first_move, rollout_seed, H):
    g = restore_game(anchor)
    g.rng = SimpleRng(rollout_seed)
    start = g.score
    res = g.move(*first_move)
    if not res['valid']:
        return None
    score_at = {}
    died_at = {}
    died = False
    steps = 1
    score_at[1] = g.score - start
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
        steps += 1
        if steps in CHECKPOINTS:
            score_at[steps] = g.score - start
    if g.game_over:
        died = True
    # Fill checkpoints not reached (game ended) with final score + died flag
    final = g.score - start
    out = {'died': died, 'final_score': final,
           'final_empties': int((g.board == 0).sum()),
           'final_lec': largest_empty_component(g.board), 'steps': steps}
    for c in CHECKPOINTS:
        out[f'score@{c}'] = score_at.get(c, final)
        out[f'died@{c}'] = died and steps <= c
    return out


def _work(args):
    anchor, label, move, rs, H = args
    return (label, _rollout(anchor, tuple(move), rs, H))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='alphatrain/data/worst_game.json')
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--R', type=int, default=96)
    p.add_argument('--H', type=int, default=200)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()

    d = json.load(open(args.game))
    fr = d['frames'][args.frame]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    pol_move = tuple(map(tuple, fr['chosen_move']))
    src = pol_move[0]

    # Enumerate ALL legal targets for the source ball
    g = restore_game(anchor)
    smask = g.get_source_mask()
    if smask[src] == 0:
        raise SystemExit(f"Source {src} is not a legal source at frame {args.frame}")
    tmask = g.get_target_mask(src)
    targets = [(tr, tc) for tr in range(9) for tc in range(9) if tmask[tr, tc] > 0]
    color = int(np.array(fr['board'])[src[0]][src[1]])
    print(f"seed={d['seed']} frame={args.frame} turn={fr['turn']} "
          f"score={anchor['score']} empties={fr['empties']} lec={fr['lec']}",
          flush=True)
    print(f"Source ball {src} (color {color}); {len(targets)} legal targets; "
          f"policy chose {pol_move[1]}", flush=True)

    units = []
    for t in targets:
        for rs in range(args.R):
            units.append((anchor, t, [list(src), list(t)], rs, args.H))

    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    by = {t: [] for t in targets}
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.model,)) as pool:
        for label, res in pool.imap_unordered(_work, units, chunksize=4):
            by[label].append(res)

    # Per-target score@200 distribution — percentiles (robust to lucky tails)
    rows = []
    for t in targets:
        rs = [r for r in by[t] if r is not None]
        if not rs:
            continue
        s200 = np.array([r['score@200'] for r in rs])
        die200 = 100 * np.mean([r['died@200'] for r in rs])
        rows.append({
            't': t, 'n': len(rs), 'die200': die200,
            'p10': np.percentile(s200, 10), 'p25': np.percentile(s200, 25),
            'p50': np.percentile(s200, 50), 'p75': np.percentile(s200, 75),
            'mean': s200.mean(),
        })

    # Rank by MEDIAN score@200 (robust; user: mean is tail-inflated)
    rows.sort(key=lambda x: -x['p50'])
    print(f"\n{'rank':>4} {'target':>8} {'die%':>5} {'P10':>6} {'P25':>6} "
          f"{'P50':>6} {'P75':>6} {'mean':>6}", flush=True)
    print('-' * 58, flush=True)
    pol_rank = None
    for i, r in enumerate(rows):
        mark = ''
        if r['t'] == pol_move[1]:
            mark = '  <== POLICY PICK'
            pol_rank = i + 1
        print(f"{i+1:>4} {str(r['t']):>8} {r['die200']:>5.1f} {r['p10']:>6.0f} "
              f"{r['p25']:>6.0f} {r['p50']:>6.0f} {r['p75']:>6.0f} "
              f"{r['mean']:>6.0f}{mark}", flush=True)

    # Summary by each percentile: where does the policy pick rank?
    polr = next(r for r in rows if r['t'] == pol_move[1])
    print(f"\nPolicy pick {pol_move[1]} ranking among {len(rows)} placements:",
          flush=True)
    for pk in ('p10', 'p25', 'p50', 'p75'):
        vals = sorted((r[pk] for r in rows), reverse=True)
        rk = vals.index(polr[pk]) + 1
        print(f"  by {pk.upper():>3}: rank {rk}/{len(rows)}  "
              f"(policy={polr[pk]:.0f}, best={vals[0]:.0f}, "
              f"median={np.median(vals):.0f}, worst={vals[-1]:.0f})", flush=True)
    print(f"  die@200: policy={polr['die200']:.1f}%  "
          f"range=[{min(r['die200'] for r in rows):.1f}, "
          f"{max(r['die200'] for r in rows):.1f}]%", flush=True)


if __name__ == '__main__':
    main()
