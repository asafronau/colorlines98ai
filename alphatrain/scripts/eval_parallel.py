"""Parallel evaluation: policy vs MCTS on same seeds, multi-worker.

Runs N games per seed on CPU workers. Each worker loads its own model.

Usage:
    python -m alphatrain.scripts.eval_parallel \
        --model alphatrain/data/alphatrain_td_best.pt \
        --seeds 42 43 44 45 46 --games-per-seed 5 --simulations 400
"""

import os
import time
import argparse
import numpy as np
import torch
from multiprocessing import Pool, cpu_count

from game.board import ColorLinesGame


# ── Worker functions (run in separate processes, CPU only) ──────────

def _init_worker(model_path, num_sims, c_puct, top_k):
    """Initialize model in each worker process."""
    global _net, _max_score, _num_sims, _c_puct, _top_k
    from alphatrain.evaluate import load_model
    _net, _max_score = load_model(model_path, torch.device('cpu'))
    _num_sims = num_sims
    _c_puct = c_puct
    _top_k = top_k


def _play_policy(seed):
    """Play one game with greedy policy."""
    from alphatrain.evaluate import make_policy_player, play_game
    player = make_policy_player(_net, torch.device('cpu'))
    result = play_game(player, seed=seed)
    return seed, result['score'], result['turns']


def _play_mcts(seed):
    """Play one game with MCTS."""
    from alphatrain.mcts import make_mcts_player
    from alphatrain.evaluate import play_game
    player = make_mcts_player(
        _net, torch.device('cpu'), max_score=_max_score,
        num_simulations=_num_sims, c_puct=_c_puct, top_k=_top_k)
    t0 = time.time()
    result = play_game(player, seed=seed)
    elapsed = time.time() - t0
    print(f"  MCTS seed={seed}: score={result['score']}, "
          f"turns={result['turns']}, {elapsed:.0f}s", flush=True)
    return seed, result['score'], result['turns']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/alphatrain_td_best.pt')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    p.add_argument('--games-per-seed', type=int, default=5)
    p.add_argument('--simulations', type=int, default=400)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--num-workers', type=int, default=0,
                   help='Number of workers (0=auto-detect)')
    p.add_argument('--policy-only', action='store_true')
    p.add_argument('--mcts-only', action='store_true')
    args = p.parse_args()

    n_workers = args.num_workers or cpu_count()
    seeds = args.seeds
    n_per = args.games_per_seed
    total = len(seeds) * n_per

    print(f"Evaluation: {len(seeds)} seeds × {n_per} games = {total} games", flush=True)
    print(f"Workers: {n_workers} (CPU), Model: {args.model}", flush=True)
    print(f"MCTS: {args.simulations} sims, c_puct={args.c_puct}, top_k={args.top_k}",
          flush=True)

    # Build task list: each seed repeated n_per times
    task_seeds = []
    for s in seeds:
        task_seeds.extend([s] * n_per)

    # ── Policy evaluation ──
    if not args.mcts_only:
        print(f"\n{'='*60}", flush=True)
        print(f"Policy player ({total} games, {n_workers} workers)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(args.model, args.simulations, args.c_puct, args.top_k)) as pool:
            pol_results = pool.map(_play_policy, task_seeds)
        pol_time = time.time() - t0
        print(f"Policy done: {pol_time:.1f}s ({pol_time/total:.1f}s/game)", flush=True)

    # ── MCTS evaluation ──
    if not args.policy_only:
        print(f"\n{'='*60}", flush=True)
        print(f"MCTS player ({total} games, {n_workers} workers, "
              f"{args.simulations} sims)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        with Pool(n_workers, initializer=_init_worker,
                  initargs=(args.model, args.simulations, args.c_puct, args.top_k)) as pool:
            mcts_results = pool.map(_play_mcts, task_seeds)
        mcts_time = time.time() - t0
        print(f"MCTS done: {mcts_time:.1f}s ({mcts_time/total:.1f}s/game)", flush=True)

    # ── Results table ──
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {n_per} games per seed, averaged", flush=True)
    print(f"{'='*60}\n", flush=True)

    def _stats(results, seeds_list, n_per):
        """Compute per-seed and overall stats."""
        by_seed = {}
        for seed, score, turns in results:
            by_seed.setdefault(seed, []).append(score)
        return by_seed

    header = f"{'Seed':>6}"
    if not args.mcts_only:
        header += f" | {'Policy Mean':>11} {'Med':>5} {'Min':>5} {'Max':>5}"
    if not args.policy_only:
        header += f" | {'MCTS Mean':>10} {'Med':>5} {'Min':>5} {'Max':>5}"
    if not args.mcts_only and not args.policy_only:
        header += f" | {'Change':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    pol_all = []
    mcts_all = []
    for s in seeds:
        row = f"{s:>6}"
        if not args.mcts_only:
            pol_by_seed = _stats(pol_results, seeds, n_per)
            ps = pol_by_seed[s]
            pol_all.extend(ps)
            row += (f" | {np.mean(ps):>11.0f} {np.median(ps):>5.0f} "
                    f"{np.min(ps):>5} {np.max(ps):>5}")
        if not args.policy_only:
            mcts_by_seed = _stats(mcts_results, seeds, n_per)
            ms = mcts_by_seed[s]
            mcts_all.extend(ms)
            row += (f" | {np.mean(ms):>10.0f} {np.median(ms):>5.0f} "
                    f"{np.min(ms):>5} {np.max(ms):>5}")
        if not args.mcts_only and not args.policy_only:
            pct = (np.mean(ms) / np.mean(ps) - 1) * 100 if np.mean(ps) > 0 else 0
            row += f" | {pct:>+6.0f}%"
        print(row, flush=True)

    print("-" * len(header), flush=True)
    row = f"{'ALL':>6}"
    if not args.mcts_only:
        row += (f" | {np.mean(pol_all):>11.0f} {np.median(pol_all):>5.0f} "
                f"{np.min(pol_all):>5} {np.max(pol_all):>5}")
    if not args.policy_only:
        row += (f" | {np.mean(mcts_all):>10.0f} {np.median(mcts_all):>5.0f} "
                f"{np.min(mcts_all):>5} {np.max(mcts_all):>5}")
    if not args.mcts_only and not args.policy_only:
        pct = (np.mean(mcts_all) / np.mean(pol_all) - 1) * 100 if np.mean(pol_all) > 0 else 0
        row += f" | {pct:>+6.0f}%"
    print(row, flush=True)

    print(f"\n{total} games per player", flush=True)


if __name__ == '__main__':
    main()
