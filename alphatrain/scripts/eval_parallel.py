"""Evaluation: policy vs MCTS on same seeds, parallel on CPU.

Each worker runs one game at a time with single-threaded PyTorch inference.
18 workers = 18 games in parallel.

Usage:
    python -m alphatrain.scripts.eval_parallel \
        --model alphatrain/data/alphatrain_td_best.pt \
        --seeds 42 43 44 45 46 --games-per-seed 5 --simulations 400
"""

import time
import argparse
import numpy as np
import torch
from multiprocessing import Pool, cpu_count


# ── Worker functions (one thread per worker, no contention) ─────────

def _init_worker(model_path, num_sims, c_puct, top_k, batch_size):
    """Initialize model in each worker process."""
    global _net, _max_score, _num_sims, _c_puct, _top_k, _batch_size
    # Critical: 1 thread per worker to avoid contention across processes
    torch.set_num_threads(1)
    from alphatrain.evaluate import load_model
    _net, _max_score = load_model(model_path, torch.device('cpu'))
    _num_sims = num_sims
    _c_puct = c_puct
    _top_k = top_k
    _batch_size = batch_size


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
        num_simulations=_num_sims, c_puct=_c_puct, top_k=_top_k,
        batch_size=_batch_size)
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
    p.add_argument('--batch-size', type=int, default=1,
                   help='MCTS batch size (1 for CPU, 16 for GPU)')
    p.add_argument('--num-workers', type=int, default=0,
                   help='Number of workers (0=auto-detect CPU count)')
    p.add_argument('--policy-only', action='store_true')
    p.add_argument('--mcts-only', action='store_true')
    args = p.parse_args()

    n_workers = args.num_workers or cpu_count()
    seeds = args.seeds
    n_per = args.games_per_seed
    total = len(seeds) * n_per

    print(f"Evaluation: {len(seeds)} seeds × {n_per} games = {total} games",
          flush=True)
    print(f"Workers: {n_workers} (CPU, 1 thread each)", flush=True)
    print(f"Model: {args.model}", flush=True)
    if not args.policy_only:
        print(f"MCTS: {args.simulations} sims, c_puct={args.c_puct}, "
              f"top_k={args.top_k}, batch_size={args.batch_size}", flush=True)

    # Build task list: each seed repeated n_per times
    task_seeds = []
    for s in seeds:
        task_seeds.extend([s] * n_per)

    init_args = (args.model, args.simulations, args.c_puct,
                 args.top_k, args.batch_size)

    # ── Policy evaluation ──
    pol_results = []
    if not args.mcts_only:
        print(f"\n{'='*60}", flush=True)
        print(f"Policy player ({total} games, {n_workers} workers)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        with Pool(n_workers, initializer=_init_worker,
                  initargs=init_args) as pool:
            pol_results = pool.map(_play_policy, task_seeds)
        pol_time = time.time() - t0
        print(f"Policy done: {pol_time:.1f}s ({pol_time/total:.1f}s/game)",
              flush=True)

    # ── MCTS evaluation ──
    mcts_results = []
    if not args.policy_only:
        print(f"\n{'='*60}", flush=True)
        print(f"MCTS player ({total} games, {n_workers} workers, "
              f"{args.simulations} sims)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        with Pool(n_workers, initializer=_init_worker,
                  initargs=init_args) as pool:
            mcts_results = pool.map(_play_mcts, task_seeds)
        mcts_time = time.time() - t0
        print(f"MCTS done: {mcts_time:.0f}s ({mcts_time/total:.0f}s/game)",
              flush=True)

    # ── Results table ──
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {n_per} games per seed", flush=True)
    print(f"{'='*60}\n", flush=True)

    def _by_seed(results):
        d = {}
        for seed, score, turns in results:
            d.setdefault(seed, []).append(score)
        return d

    header = f"{'Seed':>6}"
    if not args.mcts_only:
        header += f" | {'Pol Mean':>8} {'Med':>5} {'Min':>5} {'Max':>5}"
    if not args.policy_only:
        header += f" | {'MCTS Mean':>9} {'Med':>5} {'Min':>5} {'Max':>5}"
    if not args.mcts_only and not args.policy_only:
        header += f" | {'Chg':>5}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    pol_by_seed = _by_seed(pol_results) if pol_results else {}
    mcts_by_seed = _by_seed(mcts_results) if mcts_results else {}
    pol_all = []
    mcts_all = []

    for s in seeds:
        row = f"{s:>6}"
        if not args.mcts_only:
            ps = pol_by_seed[s]
            pol_all.extend(ps)
            row += (f" | {np.mean(ps):>8.0f} {np.median(ps):>5.0f} "
                    f"{np.min(ps):>5} {np.max(ps):>5}")
        if not args.policy_only:
            ms = mcts_by_seed[s]
            mcts_all.extend(ms)
            row += (f" | {np.mean(ms):>9.0f} {np.median(ms):>5.0f} "
                    f"{np.min(ms):>5} {np.max(ms):>5}")
        if not args.mcts_only and not args.policy_only:
            pm = np.mean(pol_by_seed[s])
            mm = np.mean(mcts_by_seed[s])
            pct = (mm / pm - 1) * 100 if pm > 0 else 0
            row += f" | {pct:>+4.0f}%"
        print(row, flush=True)

    print("-" * len(header), flush=True)
    row = f"{'ALL':>6}"
    if not args.mcts_only:
        row += (f" | {np.mean(pol_all):>8.0f} {np.median(pol_all):>5.0f} "
                f"{np.min(pol_all):>5} {np.max(pol_all):>5}")
    if not args.policy_only:
        row += (f" | {np.mean(mcts_all):>9.0f} {np.median(mcts_all):>5.0f} "
                f"{np.min(mcts_all):>5} {np.max(mcts_all):>5}")
    if not args.mcts_only and not args.policy_only:
        pct = (np.mean(mcts_all) / np.mean(pol_all) - 1) * 100 if np.mean(pol_all) > 0 else 0
        row += f" | {pct:>+4.0f}%"
    print(row, flush=True)
    print(flush=True)


if __name__ == '__main__':
    main()
