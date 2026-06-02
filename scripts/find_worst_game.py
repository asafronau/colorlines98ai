"""Find and record pillar3b's worst NATURAL-DEATH game over a seed sweep.

Two phases:
  1. Scan: play `n_seeds` games with policy argmax, capped at `max_turns`.
     Record only (seed, score, turns, game_over) — lightweight, parallel.
  2. Pick worst: among games that DIED NATURALLY (game_over=True AND
     turns < max_turns — i.e. not cap-truncated), pick the lowest score.
  3. Replay that one seed with FULL per-turn recording (board, next_balls,
     score, chosen move, top-K policy candidates+logits, empties, LEC,
     n_components) → JSON for GUI replay and offline analysis.

The cap matters: a game that hits max_turns wasn't a failure, it's a strong
game we truncated. We only want games the policy actually killed.

Usage:
    PYTHONPATH=. python scripts/find_worst_game.py \\
        --model alphatrain/data/pillar3b_epoch_20.pt \\
        --n-seeds 10000 --max-turns 500 --workers 12 \\
        --out alphatrain/data/worst_game.json
"""
from __future__ import annotations
import argparse
import collections
import json
import os
import sys
import time
from multiprocessing import Pool, set_start_method

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from scripts.mine_stationary_counterfactuals import (
    policy_topk, policy_argmax, largest_empty_component)


def count_empty_components(board):
    """Return number of connected empty components (4-connectivity)."""
    visited = np.zeros_like(board, dtype=bool)
    n = 0
    for r0 in range(9):
        for c0 in range(9):
            if board[r0, c0] != 0 or visited[r0, c0]:
                continue
            n += 1
            stack = [(r0, c0)]
            visited[r0, c0] = True
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr, nc] and board[nr, nc] == 0):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return n


# ── Scan workers ────────────────────────────────────────────────────────────
_W_NET = None
_W_DEVICE = None
_W_DTYPE = None


def _init_worker(model_path, device_str):
    global _W_NET, _W_DEVICE, _W_DTYPE
    if device_str == 'cuda' and torch.cuda.is_available():
        _W_DEVICE = torch.device('cuda')
    elif device_str == 'mps' and torch.backends.mps.is_available():
        _W_DEVICE = torch.device('mps')
    else:
        _W_DEVICE = torch.device('cpu')
    _W_NET, _ = load_model(model_path, _W_DEVICE,
                            fp16=(_W_DEVICE.type != 'cpu'))
    _W_DTYPE = next(_W_NET.parameters()).dtype


def _scan_one(args):
    seed, max_turns = args
    g = ColorLinesGame(seed=seed)
    g.reset()
    while not g.game_over and g.turns < max_turns:
        mv = policy_argmax(_W_NET, _W_DEVICE, _W_DTYPE, g)
        if mv is None:
            g.game_over = True
            break
        r = g.move(*mv)
        if not r['valid']:
            g.game_over = True
            break
    return (seed, int(g.score), int(g.turns), bool(g.game_over))


def record_game(net, device, net_dtype, seed, max_turns, top_k=10, tail=0):
    """Replay a seed, recording per-turn debug info.

    tail>0 keeps only the LAST `tail` frames — all we need, since mining rewinds
    at most ~45 plies. This keeps a natural-death recording tiny (a few dozen
    frames) no matter how long the game runs (could be 70k+ turns).
    """
    g = ColorLinesGame(seed=seed)
    g.reset()
    frames = collections.deque(maxlen=tail) if tail else []
    t0 = time.time()
    while not g.game_over and g.turns < max_turns:
        board = g.board.copy()
        next_balls = [[list(rc), int(c)] for rc, c in g.next_balls]
        empties = int((board == 0).sum())
        lec = largest_empty_component(board)
        n_comp = count_empty_components(board)
        topk = policy_topk(net, device, net_dtype, g, k=top_k)
        if not topk:
            frames.append({
                'turn': int(g.turns), 'score': int(g.score),
                'board': board.tolist(), 'next_balls': next_balls,
                'empties': empties, 'lec': lec, 'n_components': n_comp,
                'chosen_move': None, 'top_k': [], 'result': None,
                'no_legal_move': True,
            })
            g.game_over = True
            break
        chosen = topk[0][0]
        result = g.move(*chosen)
        frames.append({
            'turn': int(g.turns) - 1 if result['valid'] else int(g.turns),
            'score_before': int(g.score) - int(result.get('score', 0)),
            'score': int(g.score),
            'board': board.tolist(),
            'next_balls': next_balls,
            'empties': empties,
            'lec': lec,
            'n_components': n_comp,
            'chosen_move': [list(chosen[0]), list(chosen[1])],
            'top_k': [
                {'move': [list(m[0]), list(m[1])], 'prob': float(p)}
                for m, p in topk
            ],
            'result': {
                'valid': bool(result['valid']),
                'cleared': int(result.get('cleared', 0)),
                'score': int(result.get('score', 0)),
            },
        })
        if not result['valid']:
            g.game_over = True
            break
        if g.turns % 2000 == 0:
            print(f"    ...recording turn {g.turns}, score {g.score} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return list(frames), int(g.score), int(g.turns), bool(g.game_over)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--n-seeds', type=int, default=10000)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--max-turns', type=int, default=500)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--record-tail', type=int, default=0,
                   help='Keep only the last N recorded frames (0=all). Mining '
                        'rewinds <=~45 plies, so ~60 keeps a natural-death '
                        'death-game file tiny regardless of game length.')
    p.add_argument('--out', default='alphatrain/data/worst_game.json')
    p.add_argument('--scan-out', default='alphatrain/data/worst_game_scan.json',
                   help='Where to dump the full scan table (all seeds).')
    p.add_argument('--record-seed', type=int, default=None,
                   help='Skip the scan: record this single seed (e.g. the '
                        'worst seed reported by eval_parallel) and exit. '
                        'For exact reproduction, use the SAME --device the '
                        'scan used (batched fp16 on MPS diverges from CPU).')
    args = p.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ── Single-seed record mode (no scan) ──
    if args.record_seed is not None:
        device = torch.device('cpu')
        if args.device == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        elif args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        net, _ = load_model(args.model, device, fp16=(device.type != 'cpu'))
        net_dtype = next(net.parameters()).dtype
        print(f"\n=== Recording seed {args.record_seed} "
              f"(device={device.type}, dtype={net_dtype}) ===", flush=True)
        frames, fscore, fturns, fdied = record_game(
            net, device, net_dtype, args.record_seed, args.max_turns,
            top_k=args.top_k, tail=args.record_tail)
        rec = {
            'seed': args.record_seed, 'model': args.model,
            'final_score': fscore, 'final_turns': fturns, 'died': fdied,
            'max_turns': args.max_turns, 'frames': frames,
        }
        with open(args.out, 'w') as f:
            json.dump(rec, f)
        sz = os.path.getsize(args.out) / 1e6
        print(f"  score={fscore} turns={fturns} died={fdied}", flush=True)
        print(f"  recorded {len(frames)} frames → {args.out} ({sz:.1f} MB)",
              flush=True)
        if fturns >= args.max_turns:
            print(f"  NOTE: hit max_turns cap ({args.max_turns}) — this game "
                  f"did NOT die naturally. Raise --max-turns to see the end.",
                  flush=True)
        return

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    units = [(s, args.max_turns) for s in seeds]

    print(f"\n=== Scan: {len(seeds)} seeds, max_turns={args.max_turns} ===",
          flush=True)
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    results = []
    t0 = time.time()
    with Pool(processes=args.workers,
              initializer=_init_worker,
              initargs=(args.model, args.device)) as pool:
        n_done = 0
        for res in pool.imap_unordered(_scan_one, units, chunksize=8):
            results.append(res)
            n_done += 1
            if n_done % 200 == 0:
                el = time.time() - t0
                eta = el / n_done * (len(seeds) - n_done)
                print(f"  [{n_done}/{len(seeds)}] {el:.0f}s eta={eta:.0f}s",
                      flush=True)
    print(f"  scan done: {time.time()-t0:.0f}s", flush=True)

    # Classify
    scores = np.array([r[1] for r in results])
    turns = np.array([r[2] for r in results])
    over = np.array([r[3] for r in results])
    natural_death = over & (turns < args.max_turns)
    capped = turns >= args.max_turns
    print(f"\n=== Outcomes ===", flush=True)
    print(f"  natural deaths (turns<{args.max_turns}): "
          f"{natural_death.sum()} ({100*natural_death.mean():.1f}%)",
          flush=True)
    print(f"  capped at {args.max_turns}: {capped.sum()} "
          f"({100*capped.mean():.1f}%)", flush=True)
    print(f"  score: min={scores.min()} median={np.median(scores):.0f} "
          f"max={scores.max()} mean={scores.mean():.0f}", flush=True)

    # Save scan table
    with open(args.scan_out, 'w') as f:
        json.dump({
            'config': vars(args),
            'results': [
                {'seed': s, 'score': sc, 'turns': t, 'game_over': o}
                for (s, sc, t, o) in results
            ],
        }, f)
    print(f"  scan table → {args.scan_out}", flush=True)

    if natural_death.sum() == 0:
        print("\nNo natural deaths under the cap — nothing to record.",
              flush=True)
        return

    # Worst natural death (lowest score)
    nd_idx = np.where(natural_death)[0]
    worst_local = nd_idx[np.argmin(scores[nd_idx])]
    worst_seed, worst_score, worst_turns, _ = results[worst_local]
    print(f"\n=== Worst natural death ===", flush=True)
    print(f"  seed={worst_seed} score={worst_score} turns={worst_turns}",
          flush=True)

    # Show a few worst for context
    order = nd_idx[np.argsort(scores[nd_idx])][:10]
    print(f"  10 worst natural deaths (seed: score @ turns):", flush=True)
    for i in order:
        s, sc, t, _ = results[i]
        print(f"    {s}: {sc} @ {t}", flush=True)

    # Record the worst game fully
    print(f"\n=== Recording seed {worst_seed} ===", flush=True)
    device = torch.device(args.device if args.device != 'cpu' else 'cpu')
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    net, _ = load_model(args.model, device, fp16=(device.type != 'cpu'))
    net_dtype = next(net.parameters()).dtype
    frames, fscore, fturns, fdied = record_game(
        net, device, net_dtype, worst_seed, args.max_turns, top_k=args.top_k)
    # Sanity: recorded replay must reproduce the scan score/turns
    if fscore != worst_score or fturns != worst_turns:
        print(f"  WARNING: replay mismatch! scan=({worst_score},{worst_turns}) "
              f"replay=({fscore},{fturns}). fp16/threading nondeterminism?",
              flush=True)

    rec = {
        'seed': worst_seed,
        'model': args.model,
        'final_score': fscore,
        'final_turns': fturns,
        'died': fdied,
        'max_turns': args.max_turns,
        'frames': frames,
    }
    with open(args.out, 'w') as f:
        json.dump(rec, f)
    sz = os.path.getsize(args.out) / 1e6
    print(f"  recorded {len(frames)} frames → {args.out} ({sz:.1f} MB)",
          flush=True)


if __name__ == '__main__':
    main()
