"""Rewind-from-death miner (Gate 1 of docs/floor_aware_policy_plan.md).

Given a recorded death game, walk BACKWARD from the death. At each rewind depth
d (board d moves before the final move), enumerate the top-K policy candidate
moves and floor-evaluate each over R common-RNG rollouts to death (floor-capped).
This is the avoidability curve: are catastrophes move-avoidable, and at what
horizon? Catastrophes self-identify, so rollouts target death trajectories only.

Two phases (winner's-curse guard — the (2,8) lesson):
  1. DISCOVERY (seeds 0..R-1): score all top-K candidates; per depth find the
     floor-best vs the policy's move; flag a fork if best beats policy by
     >= --flag-threshold pp catastrophe (and best != policy, not sealed).
  2. HELD-OUT (fresh seeds R..2R-1): re-evaluate ONLY the policy move + the
     flagged floor-best, paired bootstrap. Keep the fork only if it still wins
     with CI excluding 0 ("REAL"); else it was a selection artifact ("curse").

fp32 + common-RNG, same verified machinery as resume_eval_parallel.

Usage (smoke):
    PYTHONPATH=. python scripts/rewind_from_death.py \\
        --game alphatrain/data/worst_game.json \\
        --depths 5 10 15 20 25 30 35 40 45 50 --top-k 5 --R 100 \\
        --max-turns 1000 --workers 16 --fp32
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
import torch
from multiprocessing import Process, Queue as MPQueue


def _decode(flat):
    s, t = flat // 81, flat % 81
    return ((s // 9, s % 9), (t // 9, t % 9))


def _next_arrays(next_balls):
    """Anchor next-balls (JSON [[r,c],color] form) -> intp arrays for the
    feature-value evaluator."""
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(next_balls), 3)
    for i in range(nn):
        p, c = next_balls[i]
        nr[i], nc[i], ncol[i] = int(p[0]), int(p[1]), int(c)
    return nr, nc, ncol, nn


def _ser(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


# ── Worker: anchors + per-depth candidate moves via GPU server ──────────

def _rewind_worker(slot_id, task_queue, result_queue,
                   obs_shm_name, pol_shm_name, val_shm_name,
                   num_workers, max_batch, request_queue, response_queue,
                   anchors, candidates, max_turns):
    torch.set_num_threads(1)
    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame
    from game.rng import SimpleRng

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)
    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)
    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf,
                             request_queue, response_queue)

    prepped = [(np.array(a['board'], dtype=np.int8),
                [(tuple(p), int(c)) for p, c in a['next_balls']],
                int(a['score']), int(a['turn'])) for a in anchors]

    while True:
        task = task_queue.get()
        if task is None:
            break
        di, mi, seed = task
        board0, nb0, score0, turn0 = prepped[di]
        src, tgt = candidates[di][mi]
        game = ColorLinesGame()
        game.reset(board=board0, next_balls=list(nb0))
        game.score = score0
        game.turns = turn0
        game.rng = SimpleRng(seed)
        res = game.move(tuple(src), tuple(tgt))
        if not res['valid']:
            result_queue.put((di, mi, seed, -1, 0, True))
            continue
        while not game.game_over and game.turns - turn0 < max_turns:
            obs_np = _build_obs_for_game(game)
            pol_np, _ = client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, 30)
            if not priors:
                break
            best = max(priors.items(), key=lambda x: x[1])[0]
            r = game.move(*_decode(best))
            if not r['valid']:
                break
        result_queue.put((di, mi, seed, int(game.score),
                          int(game.turns - turn0), bool(game.game_over)))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='alphatrain/data/worst_game.json')
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--depths', type=int, nargs='+',
                   default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    p.add_argument('--top-k', type=int, default=12,
                   help='Policy top-K candidate moves.')
    p.add_argument('--fv-weights',
                   default='alphatrain/data/feature_value_weights_2y_nb.npz',
                   help='Feature-value (survival) weights — net-widener that '
                        'adds survival-good moves the policy under-ranks. MUST '
                        'be a 27-feature (with-next-ball) fit; the 18-feature '
                        'files are incompatible with _evaluate_features_linear.')
    p.add_argument('--fv-k', type=int, default=18,
                   help='Add the top-N legal moves by feature-value survival '
                        '(afterstate, move+clear pre-spawn). 0 disables.')
    p.add_argument('--R', type=int, default=100,
                   help='Discovery rollouts per candidate.')
    p.add_argument('--held-r', type=int, default=0,
                   help='Held-out re-eval rollouts for flagged forks '
                        '(0 = same as --R). Fresh seed range.')
    p.add_argument('--flag-threshold', type=float, default=5.0,
                   help='Discovery catastrophe gap (pp) to flag a fork.')
    p.add_argument('--max-turns', type=int, default=300,
                   help='Catastrophe horizon H: roll out H turns from the '
                        'rewind point; catastrophe = DIED within H. '
                        'Anchor-relative, so it is valid at any game length '
                        '(absolute-score thresholds were blind on high-score '
                        'long games — die@turn6005 already had ~12k points).')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--fp32', action='store_true')
    p.add_argument('--device', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    held_r = args.held_r or args.R

    d = json.load(open(args.game))
    frames = d['frames']
    death_idx = len(frames) - 1
    device_str = args.device or ('mps' if torch.backends.mps.is_available()
                                 else 'cuda' if torch.cuda.is_available()
                                 else 'cpu')

    # ── Build anchors + top-K candidate moves per depth (local fp32 model) ──
    from alphatrain.evaluate import load_model
    from alphatrain.mcts import (_build_obs_for_game, _get_legal_priors_flat,
                                 _evaluate_features_linear)
    from game.board import ColorLinesGame, _clear_lines_at
    dev = torch.device('cpu')
    net, _ = load_model(args.model, dev, fp16=False)
    dtype = next(net.parameters()).dtype
    fvw = np.load(args.fv_weights)
    fv_coefs = fvw['coefs'].astype(np.float32)
    fv_means = fvw['means'].astype(np.float32)
    fv_stds = fvw['stds'].astype(np.float32)
    fv_bias = float(fvw['bias'])
    if args.fv_k > 0 and fv_coefs.shape[0] != 27:
        raise SystemExit(
            f"--fv-weights {args.fv_weights} has {fv_coefs.shape[0]} coefs; "
            f"_evaluate_features_linear needs 27 (with-next-ball fit). Use "
            f"alphatrain/data/feature_value_weights_2y_nb.npz.")

    anchors, candidates, pol_idx, depth_meta = [], [], [], []
    for depth in args.depths:
        idx = death_idx - depth
        if idx < 0:
            continue
        fr = frames[idx]
        anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
                  'score': fr.get('score_before', fr['score']),
                  'turn': fr['turn']}
        g = ColorLinesGame()
        g.reset(board=np.array(anchor['board'], dtype=np.int8),
                next_balls=[(tuple(pp), int(c)) for pp, c in anchor['next_balls']])
        obs = _build_obs_for_game(g)
        with torch.no_grad():
            logits = net(torch.from_numpy(obs).unsqueeze(0).to(dev, dtype))[0]
        priors = _get_legal_priors_flat(g.board, logits.float().numpy(), 64)
        topk = sorted(priors.items(), key=lambda x: -x[1])[:args.top_k]
        cand = [_decode(m) for m, _ in topk]
        n_pol = len(cand)
        # Feature-value net-widener: rank ALL legal moves by predicted survival
        # of the afterstate (move + line-clear, pre-spawn); add the top fv_k.
        # Catches survival-good moves the policy ranks near-zero (e.g. the
        # original (7,4) was policy-rank 85). The rollout still decides.
        n_fv_added = 0
        if args.fv_k > 0:
            nr, nc, ncol, nn = _next_arrays(anchor['next_balls'])
            board0 = np.array(anchor['board'], dtype=np.int8)
            fv_scored = []
            for (sr, sc), (tr, tc) in g.get_legal_moves():
                aft = board0.copy()
                col = aft[sr, sc]
                aft[sr, sc] = 0
                aft[tr, tc] = col
                _clear_lines_at(aft, int(tr), int(tc))
                v = _evaluate_features_linear(aft, nr, nc, ncol, nn,
                                              fv_coefs, fv_means, fv_stds, fv_bias)
                fv_scored.append((float(v), ((sr, sc), (tr, tc))))
            fv_scored.sort(key=lambda x: -x[0])
            for _, mv in fv_scored[:args.fv_k]:
                if mv not in cand:
                    cand.append(mv)
                    n_fv_added += 1
        pol_move = tuple(map(tuple, fr['chosen_move']))
        if pol_move not in cand:
            cand.append(pol_move)
        anchors.append(anchor)
        candidates.append(cand)
        pol_idx.append(cand.index(pol_move))
        depth_meta.append({'depth': depth, 'frame': idx, 'turn': fr['turn'],
                           'empties': fr.get('empties'), 'lec': fr.get('lec'),
                           'pol_move': list(map(list, pol_move)),
                           'pol_top1_p': float(topk[0][1]), 'n_pol': n_pol,
                           'n_fv_added': n_fv_added, 'n_cand': len(cand)})

    out = args.out or (f"logs/rewind_{os.path.basename(args.game).split('.')[0]}"
                       f"_k{args.top_k}_R{args.R}_t{args.max_turns}"
                       f"{'_fp32' if args.fp32 else ''}.json")
    avg_cand = int(np.mean([m['n_cand'] for m in depth_meta])) if depth_meta else 0
    print(f"seed={d['seed']} death@turn{frames[death_idx]['turn']+1} "
          f"score={d['final_score']}; {len(anchors)} depths x ~{avg_cand} cand "
          f"(pol{args.top_k}+fv{args.fv_k}); R={args.R} disc + {held_r} held; "
          f"flag>={args.flag_threshold}pp; catastrophe=died-within-"
          f"{args.max_turns}t; fp{'32' if args.fp32 else '16'} -> {out}",
          flush=True)

    from alphatrain.inference_server import InferenceServer
    nw = max(args.workers, 4)
    server = InferenceServer(args.model, nw, device=device_str,
                             max_batch_per_worker=args.batch_size,
                             fp16=not args.fp32)
    server.start()
    tq, rq = MPQueue(), MPQueue()
    procs = []
    for i in range(nw):
        pr = Process(target=_rewind_worker,
                     args=(i, tq, rq, server._obs_shm.name, server._pol_shm.name,
                           server._val_shm.name, nw, args.batch_size,
                           server.request_queue, server.response_queues[i],
                           anchors, candidates, args.max_turns))
        pr.start()
        procs.append(pr)

    def feed_collect(tasks, label):
        for t in tasks:
            tq.put(t)
        out_rows, t0 = [], time.time()
        every = max(50, len(tasks) // 20)
        for i in range(len(tasks)):
            out_rows.append(rq.get(timeout=14400))
            if (i + 1) % every == 0 or (i + 1) == len(tasks):
                el = time.time() - t0
                print(f"  [{label} {i+1}/{len(tasks)}] {el:.0f}s "
                      f"(ETA {el/(i+1)*(len(tasks)-i-1):.0f}s)", flush=True)
        return out_rows

    def catrate(died_dict):
        # catastrophe rate = % of rollouts that DIED within the horizon
        # (anchor-relative; valid at any game length, unlike absolute score).
        a = np.array(list(died_dict.values()), dtype=float)
        return float(100.0 * a.mean()) if a.size else float('nan')

    # ── Phase 1: discovery ──
    disc_tasks = [(di, mi, s) for di in range(len(anchors))
                  for mi in range(len(candidates[di])) for s in range(args.R)]
    print(f"\nDiscovery: {len(disc_tasks)} games", flush=True)
    disc = feed_collect(disc_tasks, 'disc')
    disc_sc = {}
    for di, mi, seed, score, turns, died in disc:
        if score >= 0:                       # score<0 = illegal move sentinel
            disc_sc.setdefault((di, mi), {})[seed] = bool(died)

    flagged, depth_rows = [], []
    for di, dm in enumerate(depth_meta):
        cands = [(mi, catrate(disc_sc.get((di, mi), {})))
                 for mi in range(len(candidates[di])) if (di, mi) in disc_sc]
        if not cands:
            continue
        pol_cat = dict(cands).get(pol_idx[di], float('nan'))
        best_mi, best_cat = min(cands, key=lambda x: x[1])
        gap = pol_cat - best_cat
        sealed = min(c for _, c in cands) > 60.0
        flag = (not sealed) and (best_mi != pol_idx[di]) and (gap >= args.flag_threshold)
        # Per-candidate (move, catastrophe-rate, n) — the TEACHER labels; with
        # board+next_balls this depth_row is a self-contained training example
        # (board, next_balls, [(move, P(catastrophe))]).
        cand_rates = [[list(map(list, candidates[di][mi])),
                       catrate(disc_sc[(di, mi)]), len(disc_sc[(di, mi)])]
                      for mi in range(len(candidates[di])) if (di, mi) in disc_sc]
        depth_rows.append({'di': di, **dm, 'pol_cat': pol_cat, 'best_mi': best_mi,
                           'best_cat': best_cat, 'disc_gap': gap, 'sealed': sealed,
                           'flag': flag,
                           'best_move': list(map(list, candidates[di][best_mi])),
                           'board': anchors[di]['board'],
                           'next_balls': anchors[di]['next_balls'],
                           'cand_rates': cand_rates})
        if flag:
            flagged.append((di, best_mi))

    # ── Phase 2: held-out re-eval of flagged forks (fresh seeds) ──
    held_rows = {}
    if flagged:
        held_tasks = [(di, mi, s) for di, best_mi in flagged
                      for mi in (pol_idx[di], best_mi)
                      for s in range(args.R, args.R + held_r)]
        print(f"\nHeld-out: {len(flagged)} flagged forks, {len(held_tasks)} games",
              flush=True)
        held = feed_collect(held_tasks, 'held')
        hsc = {}
        for di, mi, seed, score, turns, died in held:
            if score >= 0:
                hsc.setdefault((di, mi), {})[seed] = bool(died)
        rng = np.random.default_rng(0)
        for di, best_mi in flagged:
            pol_d, best_d = hsc.get((di, pol_idx[di]), {}), hsc.get((di, best_mi), {})
            seeds = sorted(set(pol_d) & set(best_d))
            if not seeds:
                continue
            polc = np.array([pol_d[s] for s in seeds], float)   # died flags
            bestc = np.array([best_d[s] for s in seeds], float)
            n = len(seeds)
            boot = np.empty(2000)
            for b in range(2000):
                idx = rng.integers(0, n, n)
                boot[b] = 100.0 * (polc[idx].mean() - bestc[idx].mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            held_rows[(di, best_mi)] = {
                'gap': float(100.0 * (polc.mean() - bestc.mean())),
                'pol_cat': float(100.0 * polc.mean()),
                'best_cat': float(100.0 * bestc.mean()),
                'lo': float(lo), 'hi': float(hi), 'n': n, 'survives': bool(lo > 0)}

    for _ in range(nw):
        tq.put(None)
    for pr in procs:
        pr.join(timeout=10)
    server.shutdown()

    # ── Report ──
    print(f"\n{'depth':>5} {'turn':>5} {'empt':>4} {'polMove':>10} "
          f"{'pol%':>6} {'best%':>6} {'discΔ':>6}  {'bestMove':>10} "
          f"{'verdict':>22}", flush=True)
    print('-' * 92, flush=True)
    for r in depth_rows:
        pm, bm = f"{tuple(map(tuple, r['pol_move']))}", f"{tuple(map(tuple, r['best_move']))}"
        if r['sealed']:
            verdict = 'sealed'
        elif not r['flag']:
            verdict = 'neutral'
        else:
            h = held_rows.get((r['di'], r['best_mi']))
            if h is None:
                verdict = 'flag(no-held)'
            elif h['survives']:
                verdict = f"REAL Δ{h['gap']:.0f}[{h['lo']:.0f},{h['hi']:.0f}]"
            else:
                verdict = f"curse Δ{h['gap']:.0f}[{h['lo']:.0f},{h['hi']:.0f}]"
        print(f"{r['depth']:>5} {r['turn']:>5} {str(r['empties']):>4} {pm:>10} "
              f"{r['pol_cat']:>6.1f} {r['best_cat']:>6.1f} {r['disc_gap']:>6.1f}  "
              f"{bm:>10} {verdict:>22}", flush=True)

    n_real = sum(1 for k, h in held_rows.items() if h['survives'])
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump({'meta': {'game': args.game, 'seed': d['seed'],
                        'final_score': d['final_score'], 'R': args.R,
                        'held_r': held_r, 'top_k': args.top_k,
                        'flag_threshold': args.flag_threshold,
                        'cat_horizon': args.max_turns, 'fp32': args.fp32},
               'depth_rows': depth_rows,
               'held': {f"{k[0]}_{k[1]}": v for k, v in held_rows.items()}},
              open(out, 'w'), default=_ser)
    print(f"\n{n_real} REAL forks (survived held-out), "
          f"{sum(1 for r in depth_rows if r['flag']) - n_real} winner's-curse, "
          f"{sum(1 for r in depth_rows if r['sealed'])} sealed, "
          f"{len(depth_rows)} depths. Wrote {out}", flush=True)


if __name__ == '__main__':
    main()
