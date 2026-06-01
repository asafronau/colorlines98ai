"""Resume-from-position parallel eval (eval_parallel architecture).

Takes ONE board position (a frame of a recorded game), relocates a single
source ball to every legal target, and resumes POLICY-ONLY play to natural
death under N common-RNG seeds per placement. Reports the full final-score
distribution (P1/P5/P10/P25/P50/P75/P90, died%, where a reference score lands)
per placement, so we can validate whether the policy's chosen target is worse
than relocating that same ball elsewhere.

Crucially this runs to NATURAL DEATH (capped at --max-turns), not a short
horizon: since score == survival (~2.0 pts/turn), the floor only shows up when
games are allowed to die. We do NOT move any other ball — only the one the
policy moved.

Architecture mirrors alphatrain.scripts.eval_parallel: one GPU InferenceServer
process does all NN inference via shared memory; N CPU workers simulate games
and pull (target, seed) tasks off a queue.

Usage:
    # calibrate on the policy's own pick (1 placement, many seeds, run to death)
    PYTHONPATH=. python scripts/resume_eval_parallel.py \\
        --game alphatrain/data/worst_game.json --frame 10 \\
        --targets policy --num-seeds 200 --max-turns 4000 --workers 16

    # full run: every legal placement
    PYTHONPATH=. python scripts/resume_eval_parallel.py \\
        --game alphatrain/data/worst_game.json --frame 10 \\
        --targets all --num-seeds 256 --max-turns 600 --workers 16 --fp32
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

# Sentinel "seed" meaning: inject seed-835's ACTUAL turn-N RNG state (captured
# by replaying the recorded moves), not a fresh SimpleRng. This is both the
# resume-fidelity proof (must reproduce the recorded death) and the single most
# relevant paired sample (the real situation, counterfactual move).
REAL_SEED = -987654321


def _replay_capture(d, frame):
    """Replay the recorded game's own moves from turn 0 under its real seed.

    Returns (rng_state_at_frame, board_at_frame, final_score, final_turn).
    rng_state is the picklable PCG64 state at the moment the policy chose the
    frame-`frame` move — injecting it + replaying that move reproduces the real
    continuation. Also returns the full-replay final score as a mechanics check.
    """
    from game.board import ColorLinesGame
    g = ColorLinesGame(seed=d['seed'])
    g.reset()
    state_at = None
    board_at = None
    for i, fr in enumerate(d['frames']):
        if i == frame:
            state_at = dict(g.rng._gen.bit_generator.state)
            board_at = g.board.copy()
        mv = fr.get('chosen_move')
        if mv is None:
            break
        res = g.move((int(mv[0][0]), int(mv[0][1])),
                     (int(mv[1][0]), int(mv[1][1])))
        if not res['valid']:
            break
    return state_at, board_at, int(g.score), int(g.turns)


# ── Worker: resume-from-anchor policy player via GPU server ─────────────

def _resume_worker(slot_id, task_queue, result_queue,
                   obs_shm_name, pol_shm_name, val_shm_name,
                   num_workers, max_batch, request_queue, response_queue,
                   anchor, source, max_turns, real_state):
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

    board0 = np.array(anchor['board'], dtype=np.int8)
    nb0 = [(tuple(p), int(c)) for p, c in anchor['next_balls']]
    score0 = int(anchor['score'])
    turn0 = int(anchor['turn'])
    src = tuple(source)

    while True:
        task = task_queue.get()
        if task is None:
            break
        target, seed = task
        game = ColorLinesGame()
        game.reset(board=board0, next_balls=list(nb0))
        game.score = score0
        game.turns = turn0
        if seed == REAL_SEED:
            game.rng = SimpleRng(0)
            game.rng._gen.bit_generator.state = real_state
        else:
            game.rng = SimpleRng(seed)

        res = game.move(src, tuple(target))
        if not res['valid']:
            result_queue.put((target, seed, -1, 0, True))  # illegal placement
            continue

        while not game.game_over and game.turns - turn0 < max_turns:
            obs_np = _build_obs_for_game(game)
            pol_np, _ = client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, 30)
            if not priors:
                break
            best = max(priors.items(), key=lambda x: x[1])[0]
            sf, tf = best // 81, best % 81
            r = game.move((sf // 9, sf % 9), (tf // 9, tf % 9))
            if not r['valid']:
                break
        died = bool(game.game_over)
        result_queue.put((target, seed, int(game.score),
                          int(game.turns - turn0), died))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


# ── Main ────────────────────────────────────────────────────────────

def _enumerate_targets(anchor, source):
    from game.board import ColorLinesGame
    g = ColorLinesGame()
    g.reset(board=np.array(anchor['board'], dtype=np.int8),
            next_balls=[(tuple(p), int(c)) for p, c in anchor['next_balls']])
    smask = g.get_source_mask()
    if smask[source] == 0:
        raise SystemExit(f"Source {source} is not a legal source at this frame")
    tmask = g.get_target_mask(source)
    return [(tr, tc) for tr in range(9) for tc in range(9) if tmask[tr, tc] > 0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game', default='alphatrain/data/worst_game.json')
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--targets', default='all',
                   help="'all' (every legal target), 'policy' (only the "
                        "policy's chosen target), or 'r,c r,c ...'.")
    p.add_argument('--num-seeds', type=int, default=256,
                   help='Common-RNG replicates per placement (seeds 0..N-1, '
                        'shared across placements for paired comparison).')
    p.add_argument('--max-turns', type=int, default=600,
                   help='Cap on turns played AFTER the branch (natural death '
                        'before then is recorded). Low cap captures the floor '
                        '(early deaths) cheaply; raise to study survivors.')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--fp32', action='store_true',
                   help='Deterministic fp32 inference (batch-invariant). '
                        'Default fp16 (faster, batch-noisy).')
    p.add_argument('--ref-score', type=int, default=215,
                   help='Reference score to locate in the distribution '
                        '(default 215 = seed-835 natural death).')
    p.add_argument('--device', default=None)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    d = json.load(open(args.game))
    fr = d['frames'][args.frame]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    pol_move = tuple(map(tuple, fr['chosen_move']))
    source = pol_move[0]
    pol_target = pol_move[1]
    color = int(np.array(fr['board'])[source[0]][source[1]])

    # Capture seed-835's ACTUAL turn-`frame` RNG state by replaying its own
    # recorded moves. This both proves resume fidelity (the policy continuation
    # under this state must reproduce the recorded death) and supplies the
    # single most relevant paired sample: the real situation, alternate move.
    real_state, board_at, rep_score, rep_turn = _replay_capture(d, args.frame)
    anchor_board = np.array(anchor['board'], dtype=np.int8)
    if board_at is None or not np.array_equal(board_at, anchor_board):
        raise SystemExit(f"VERIFY FAILED: replayed frame-{args.frame} board "
                         f"does not match the recorded anchor board.")

    all_targets = _enumerate_targets(anchor, source)
    if args.targets == 'all':
        targets = all_targets
    elif args.targets == 'policy':
        targets = [pol_target]
    else:
        targets = [tuple(int(x) for x in tok.split(','))
                   for tok in args.targets.split()]
    targets = [t for t in targets if t in set(all_targets)]

    if args.device:
        device_str = args.device
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    elif torch.cuda.is_available():
        device_str = 'cuda'
    else:
        device_str = 'cpu'

    out = args.out or (f"logs/resume_f{args.frame}_"
                       f"{args.targets if args.targets != 'all' else 'all'}_"
                       f"n{args.num_seeds}_t{args.max_turns}"
                       f"{'_fp32' if args.fp32 else ''}.json")

    print(f"seed={d['seed']} frame={args.frame} turn={fr['turn']} "
          f"score={anchor['score']} empties={fr.get('empties','?')} "
          f"lec={fr.get('lec','?')}", flush=True)
    print(f"Source ball {source} (color {color}); policy chose {pol_target}; "
          f"{len(targets)} placement(s) x {args.num_seeds} seeds = "
          f"{len(targets)*args.num_seeds} games", flush=True)
    print(f"device={device_str} fp{'32' if args.fp32 else '16'} "
          f"max_turns={args.max_turns} workers={args.workers}  -> {out}",
          flush=True)
    print(f"VERIFY mechanics: recorded-move replay -> score={rep_score} "
          f"turn={rep_turn} (recorded final {d['final_score']}@"
          f"{d['final_turns']})  "
          f"{'OK' if rep_score == d['final_score'] else 'MISMATCH!'}",
          flush=True)
    print(f"VERIFY anchor: replayed frame-{args.frame} board matches the "
          f"recorded anchor; turn-{args.frame} RNG state captured for the "
          f"real-RNG counterfactual.", flush=True)

    # Tasks: common-RNG seeds 0..N-1 (paired across placements) PLUS one
    # REAL_SEED task per placement = seed-835's actual turn-N RNG state.
    tasks = []
    for t in targets:
        tasks.append((t, REAL_SEED))
        for s in range(args.num_seeds):
            tasks.append((t, s))
    total = len(tasks)

    from alphatrain.inference_server import InferenceServer
    n_workers = max(args.workers, 4)
    server = InferenceServer(args.model, n_workers, device=device_str,
                             max_batch_per_worker=args.batch_size,
                             fp16=not args.fp32)
    server.start()

    task_queue = MPQueue()
    for t in tasks:
        task_queue.put(t)
    for _ in range(n_workers):
        task_queue.put(None)
    result_queue = MPQueue()

    procs = []
    for i in range(n_workers):
        pr = Process(target=_resume_worker,
                     args=(i, task_queue, result_queue,
                           server._obs_shm.name, server._pol_shm.name,
                           server._val_shm.name, n_workers, args.batch_size,
                           server.request_queue, server.response_queues[i],
                           anchor, source, args.max_turns, real_state))
        pr.start()
        procs.append(pr)

    t0 = time.time()
    raw = []
    every = max(25, total // 40)
    try:
        for i in range(total):
            raw.append(result_queue.get(timeout=14400))
            if (i + 1) % every == 0 or (i + 1) == total:
                el = time.time() - t0
                eta = el / (i + 1) * (total - i - 1)
                done = [r for r in raw if r[2] >= 0]
                dr = 100.0 * np.mean([r[4] for r in done]) if done else 0.0
                print(f"  [{i+1}/{total}] {el:.0f}s (ETA {eta:.0f}s) "
                      f"died={dr:.0f}%", flush=True)
    finally:
        for pr in procs:
            pr.join(timeout=10)
        server.shutdown()
    print(f"Done: {time.time()-t0:.0f}s", flush=True)

    # ── Aggregate per placement (REAL_SEED held out as the counterfactual) ──
    by = {t: [] for t in targets}
    real = {}  # target -> (score, turns, died) under seed-835's real RNG
    illegal = 0
    for tr_tc, seed, score, turns, died in raw:
        t = tuple(tr_tc)
        if score < 0:
            illegal += 1
            continue
        if seed == REAL_SEED:
            real[t] = (score, turns, died)
            continue
        by[t].append((score, turns, died))

    rows = []
    for t in targets:
        recs = by[t]
        if not recs:
            continue
        sc = np.array([r[0] for r in recs])
        tn = np.array([r[1] for r in recs])
        dd = np.array([r[2] for r in recs], dtype=bool)
        died_turns = tn[dd]
        rows.append({
            't': list(t), 'n': len(recs),
            'died_pct': 100.0 * dd.mean(),
            'med_death_turn': float(np.median(died_turns)) if dd.any() else None,
            'P1': float(np.percentile(sc, 1)), 'P5': float(np.percentile(sc, 5)),
            'P10': float(np.percentile(sc, 10)),
            'P25': float(np.percentile(sc, 25)),
            'P50': float(np.percentile(sc, 50)),
            'P75': float(np.percentile(sc, 75)),
            'P90': float(np.percentile(sc, 90)),
            'mean': float(sc.mean()),
            'med_turns': float(np.median(tn)),
            'frac_le_ref': float(100.0 * (sc <= args.ref_score).mean()),
            'frac_le_500': float(100.0 * (sc <= 500).mean()),
            'frac_le_1000': float(100.0 * (sc <= 1000).mean()),
            'real': list(real[t]) if t in real else None,
        })

    # Rank by P10 (floor-focused; the user's question is about the floor).
    rows.sort(key=lambda x: -x['P10'])

    summary = {
        'meta': {
            'game': args.game, 'seed': d['seed'], 'frame': args.frame,
            'turn': fr['turn'], 'source': list(source),
            'policy_target': list(pol_target), 'color': color,
            'num_seeds': args.num_seeds, 'max_turns': args.max_turns,
            'fp32': args.fp32, 'ref_score': args.ref_score,
            'n_placements': len(rows), 'illegal_tasks': illegal,
        },
        'per_target': rows,
        # [tr, tc, seed, score, turns, died] — seed kept for paired analysis.
        'raw': [[int(x[0][0]), int(x[0][1]), int(x[1]), int(x[2]),
                 int(x[3]), bool(x[4])] for x in raw],
    }
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f)

    # ── Print table ──
    print(f"\n{'rk':>3} {'target':>8} {'died%':>6} {'dT':>5} {'P1':>6} "
          f"{'P5':>6} {'P10':>6} {'P25':>6} {'P50':>6} {'P75':>6} "
          f"{'mean':>6} {'<=ref%':>7}", flush=True)
    print('-' * 86, flush=True)
    for i, r in enumerate(rows):
        mark = '  <== POLICY' if tuple(r['t']) == pol_target else ''
        dt = f"{r['med_death_turn']:.0f}" if r['med_death_turn'] else '-'
        print(f"{i+1:>3} {str(tuple(r['t'])):>8} {r['died_pct']:>6.1f} "
              f"{dt:>5} {r['P1']:>6.0f} {r['P5']:>6.0f} {r['P10']:>6.0f} "
              f"{r['P25']:>6.0f} {r['P50']:>6.0f} {r['P75']:>6.0f} "
              f"{r['mean']:>6.0f} {r['frac_le_ref']:>7.1f}{mark}", flush=True)

    # ── Real-RNG counterfactual: identical RNG that killed (2,8) at 215 ──
    print(f"\nREAL-RNG continuation (seed-835's actual turn-{args.frame} state, "
          f"the exact RNG that produced {d['final_score']}@{d['final_turns']}):",
          flush=True)
    for t in targets:
        if t not in real:
            continue
        sc_, tu_, dd_ = real[t]
        tag = ''
        if t == pol_target:
            ok = (sc_ == d['final_score'])
            tag = (f"  <== POLICY; reproduces recorded {d['final_score']}"
                   f"@{d['final_turns']} {'OK' if ok else 'MISMATCH!'}")
        print(f"  {str(t):>8}: score={sc_:>6} turns={tu_:>5} "
              f"died={dd_}{tag}", flush=True)

    # Single-placement detail: full distribution shape + catastrophe rates.
    if len(rows) == 1:
        recs = by[tuple(rows[0]['t'])]
        sc = np.array([r[0] for r in recs])
        tn = np.array([r[1] for r in recs])
        dd = np.array([r[2] for r in recs], dtype=bool)
        edges = [0, args.ref_score, 500, 1000, 2000, 5000, 10000, 15000,
                 args.max_turns * 2 * 5]
        labels = ['0..ref', 'ref..500', '500..1k', '1k..2k', '2k..5k',
                  '5k..10k', '10k..15k', '15k+']
        print(f"\nScore distribution for {tuple(rows[0]['t'])} "
              f"({len(sc)} seeds, fp{'32' if args.fp32 else '16'}, "
              f"no cap@{args.max_turns}t):", flush=True)
        for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
            c = int(((sc > lo) & (sc <= hi)).sum())
            bar = '#' * int(round(50 * c / len(sc)))
            print(f"  {lab:>9} ({lo:>5}-{hi:<6}]: {c:>4} "
                  f"({100*c/len(sc):>5.1f}%) {bar}", flush=True)
        print(f"  died(before {args.max_turns}t)={100*dd.mean():.1f}%  "
              f"turns: P10={np.percentile(tn,10):.0f} "
              f"P50={np.percentile(tn,50):.0f} P90={np.percentile(tn,90):.0f}",
              flush=True)
        print(f"  CATASTROPHE rates: <=215={100*(sc<=215).mean():.1f}%  "
              f"<=500={100*(sc<=500).mean():.1f}%  "
              f"<=1000={100*(sc<=1000).mean():.1f}%  "
              f"<=2000={100*(sc<=2000).mean():.1f}%", flush=True)
        print(f"  score: min={sc.min():.0f} P50={np.percentile(sc,50):.0f} "
              f"mean={sc.mean():.0f} max={sc.max():.0f}", flush=True)

    if pol_target in [tuple(r['t']) for r in rows]:
        pr = next(r for r in rows if tuple(r['t']) == pol_target)
        print(f"\nPolicy pick {pol_target} among {len(rows)} placements:",
              flush=True)
        for pk in ('P1', 'P5', 'P10', 'P25', 'P50', 'mean'):
            vals = sorted((r[pk] for r in rows), reverse=True)
            rk = vals.index(pr[pk]) + 1
            print(f"  by {pk:>4}: rank {rk}/{len(rows)}  "
                  f"(policy={pr[pk]:.0f}, best={vals[0]:.0f}, "
                  f"worst={vals[-1]:.0f})", flush=True)
        print(f"  died%: policy={pr['died_pct']:.1f}  "
              f"frac<=ref({args.ref_score}): policy={pr['frac_le_ref']:.1f}%  "
              f"range=[{min(r['frac_le_ref'] for r in rows):.1f}, "
              f"{max(r['frac_le_ref'] for r in rows):.1f}]%", flush=True)

    print(f"\nWrote {out}", flush=True)


if __name__ == '__main__':
    main()
