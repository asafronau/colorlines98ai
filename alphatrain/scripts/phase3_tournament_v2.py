"""Phase 3 v2 — tournament with MCTS teachers + judgment.

Same anchor pool as v1 (505 anchors from selfplay/crisis/oracle_disagree).
Five teachers per anchor:
  1. 2z policy top-1
  2. v9 policy top-1
  3. 2z + value_head_v12_v12targets MCTS top-1 (loaded from picks file)
  4. 2y2 + value_head_v11 MCTS top-1 (loaded from picks file)
  5. Rollout oracle (top-K from 2z, K=32 judging, pick highest cap_rate)

Each unique candidate move is judged with K=32 common-RNG rollouts at H=300.
Regret = oracle_best - teacher_pick measured in cap_rate and mean_turns.

Usage:
    python -m alphatrain.scripts.phase3_tournament_v2 \\
        --model-2z alphatrain/data/pillar2z_epoch_19.pt \\
        --model-v9 alphatrain/data/policy_dagger_v9.pt \\
        --phase1 alphatrain/data/phase1_oracle.pt \\
        --picks-2z-mcts alphatrain/data/mcts_picks_2z_v12targets.pt \\
        --picks-2y2-mcts alphatrain/data/mcts_picks_2y2_v11.pt \\
        --crisis-dir data/crisis_v12 --selfplay-dir data/selfplay_v12 \\
        --n-per-source 200 --margin-threshold 0.15 --sample-seed 42 \\
        --top-k 6 --k-rollouts 32 --horizon 300 \\
        --workers 16 --batch-size 8 \\
        --device mps \\
        --output alphatrain/data/phase3_tournament_v2.pt
"""

import argparse
import glob
import json
import os
import time
from random import Random
import numpy as np
import torch
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def build_anchor_pool(crisis_dir, selfplay_dir, phase1_path,
                       n_per_source, margin_threshold, seed):
    """Same logic as collect_mcts_picks.load_anchors(tournament_pool)."""
    def sample_dirs(dirs, n, label, rng):
        files = []
        for d in dirs:
            files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
        out = []
        while len(out) < n:
            f = rng.choice(files)
            try:
                with open(f) as fp:
                    g = json.load(fp)
            except Exception:
                continue
            mv = g.get('moves', [])
            if not mv:
                continue
            mi = rng.randint(0, len(mv) - 1)
            m = mv[mi]
            out.append({
                'board': np.asarray(m['board'], dtype=np.int8),
                'next_balls': [((int(nb['row']), int(nb['col'])),
                                 int(nb['color']))
                                for nb in m['next_balls']],
                'num_next': int(m['num_next']),
                'turn_origin': mi,
                'source_label': label,
            })
        return out

    rng = Random(seed)
    a = sample_dirs([selfplay_dir], n_per_source, 'selfplay', rng)
    b = sample_dirs([crisis_dir], n_per_source, 'crisis', rng)

    d = []
    if phase1_path:
        data = torch.load(phase1_path, weights_only=False)
        for r in data['results']:
            pm = r['per_move']
            if len(pm) < 2:
                continue
            sm = sorted(pm.items(), key=lambda kv: kv[1]['rank'])[:6]
            qs = np.array([mv['cap_rate'] for _, mv in sm])
            if qs.max() - qs[0] >= margin_threshold:
                d.append({
                    'board': r['anchor_board'],
                    'next_balls': r['anchor_next_balls'],
                    'num_next': r['anchor_n_next'],
                    'turn_origin': r.get('turn_origin', 0),
                    'source_label': 'oracle_disagree',
                })
        rng2 = Random(0); rng2.shuffle(d)
        d = d[:n_per_source]

    anchors = a + b + d
    for i, x in enumerate(anchors):
        x['id'] = i
    return anchors


def _worker(slot_id, anchor_queue, result_queue,
            obs_shm_2z, pol_shm_2z, val_shm_2z,
            obs_shm_v9, pol_shm_v9, val_shm_v9,
            num_workers, max_batch,
            req_q_2z, resp_q_2z, req_q_v9, resp_q_v9,
            top_k_moves, k_rollouts, horizon,
            mcts_picks_2z_mcts, mcts_picks_2y2_mcts,
            afterstate_seed_base, continuation_seed_base):
    torch.set_num_threads(1)
    from alphatrain.inference_server import InferenceClient
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    o2 = SharedMemory(name=obs_shm_2z); p2 = SharedMemory(name=pol_shm_2z); v2 = SharedMemory(name=val_shm_2z)
    o9 = SharedMemory(name=obs_shm_v9); p9 = SharedMemory(name=pol_shm_v9); v9_ = SharedMemory(name=val_shm_v9)
    N, B = num_workers, max_batch
    ob_2z = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=o2.buf)
    po_2z = np.ndarray((N, B, 6561), dtype=np.float32, buffer=p2.buf)
    va_2z = np.ndarray((N, B), dtype=np.float32, buffer=v2.buf)
    ob_v9 = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=o9.buf)
    po_v9 = np.ndarray((N, B, 6561), dtype=np.float32, buffer=p9.buf)
    va_v9 = np.ndarray((N, B), dtype=np.float32, buffer=v9_.buf)
    c_2z = InferenceClient(slot_id, ob_2z, po_2z, va_2z, req_q_2z, resp_q_2z)
    c_v9 = InferenceClient(slot_id, ob_v9, po_v9, va_v9, req_q_v9, resp_q_v9)

    def policy_2z(g):
        return c_2z.evaluate(_build_obs_for_game(g))[0]
    def policy_v9(g):
        return c_v9.evaluate(_build_obs_for_game(g))[0]

    def realize(anchor, mv, seed):
        g = ColorLinesGame(seed=seed)
        g.reset(board=anchor['board'].copy(),
                next_balls=list(anchor['next_balls']))
        g.turns = anchor['turn_origin']
        sr, sc = mv // 81 // 9, mv // 81 % 9
        tr, tc = mv % 81 // 9, mv % 81 % 9
        r = g.move((sr, sc), (tr, tc))
        if not r['valid']:
            return None
        return {'board': g.board.copy(), 'next_balls': list(g.next_balls),
                'turns': g.turns, 'score_at_after': g.score,
                'game_over': g.game_over}

    def cont(after, seed):
        if after['game_over']:
            return {'cap_hit': False,
                    'score_gain_after': int(after['score_at_after']),
                    'turns_after': 0}
        g = ColorLinesGame(seed=seed)
        g.reset(board=after['board'].copy(),
                next_balls=list(after['next_balls']))
        g.turns = after['turns']
        s0 = g.score
        end = g.turns + horizon
        while not g.game_over and g.turns < end:
            pol = policy_2z(g)
            pr = _get_legal_priors_flat(g.board, pol, 30)
            if not pr:
                break
            b = max(pr.items(), key=lambda x: x[1])[0]
            r = g.move((b // 81 // 9, b // 81 % 9), (b % 81 // 9, b % 81 % 9))
            if not r['valid']:
                break
        surv = (not g.game_over) and (g.turns >= end)
        return {'cap_hit': bool(surv),
                'score_gain_after': int(after['score_at_after']
                                         + (g.score - s0)),
                'turns_after': int(g.turns - after['turns'])}

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break

        ag = ColorLinesGame(seed=0)
        ag.reset(board=anchor['board'].copy(),
                 next_balls=list(anchor['next_balls']))
        ag.turns = anchor['turn_origin']
        if ag.game_over:
            result_queue.put({'anchor_id': anchor['id'], 'skipped': 'game_over'})
            continue

        # 2z top-K + v9 top-1
        pol = policy_2z(ag)
        pr = _get_legal_priors_flat(ag.board, pol, top_k_moves)
        if not pr or len(pr) < 2:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'fewer_moves'})
            continue
        top_2z = sorted(pr.items(), key=lambda x: -x[1])[:top_k_moves]
        move_2z = int(top_2z[0][0])
        pol_v = policy_v9(ag)
        pr_v9 = _get_legal_priors_flat(ag.board, pol_v, top_k_moves)
        move_v9 = int(sorted(pr_v9.items(),
                              key=lambda x: -x[1])[0][0]) if pr_v9 else move_2z

        move_2z_mcts = mcts_picks_2z_mcts.get(anchor['id'])
        move_2y2_mcts = mcts_picks_2y2_mcts.get(anchor['id'])

        candidates = set([int(m) for m, _ in top_2z])
        candidates.add(move_2z)
        candidates.add(move_v9)
        if move_2z_mcts is not None:
            candidates.add(int(move_2z_mcts))
        if move_2y2_mcts is not None:
            candidates.add(int(move_2y2_mcts))

        after_seed = afterstate_seed_base + anchor['id'] * 7919
        cont_base = continuation_seed_base + anchor['id'] * 7919 * 1000
        per_move = {}
        for mv in candidates:
            af = realize(anchor, mv, after_seed)
            if af is None:
                continue
            outcomes = [cont(af, cont_base + k) for k in range(k_rollouts)]
            per_move[int(mv)] = {
                'cap_rate': sum(o['cap_hit'] for o in outcomes) / len(outcomes),
                'mean_turns': float(np.mean([o['turns_after']
                                              for o in outcomes])),
                'mean_score': float(np.mean([o['score_gain_after']
                                              for o in outcomes])),
            }

        top_2z_moves = [int(m) for m, _ in top_2z]
        in_pm = [m for m in top_2z_moves if m in per_move]
        oracle_pick = max(in_pm,
                          key=lambda m: per_move[m]['cap_rate']) if in_pm \
            else move_2z

        result_queue.put({
            'anchor_id': anchor['id'],
            'source_label': anchor['source_label'],
            'turn_origin': anchor['turn_origin'],
            'move_2z': move_2z,
            'move_v9': move_v9,
            'move_2z_mcts': int(move_2z_mcts) if move_2z_mcts is not None else None,
            'move_2y2_mcts': int(move_2y2_mcts) if move_2y2_mcts is not None else None,
            'move_oracle': int(oracle_pick),
            'top_2z_moves': top_2z_moves,
            'per_move': per_move,
        })

    o2.close(); p2.close(); v2.close()
    o9.close(); p9.close(); v9_.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-2z', required=True)
    p.add_argument('--model-v9', required=True)
    p.add_argument('--phase1', required=True)
    p.add_argument('--picks-2z-mcts', required=True)
    p.add_argument('--picks-2y2-mcts', required=True)
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--n-per-source', type=int, default=200)
    p.add_argument('--margin-threshold', type=float, default=0.15)
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--k-rollouts', type=int, default=32)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--afterstate-seed-base', type=int, default=5000000)
    p.add_argument('--continuation-seed-base', type=int, default=6000000)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    print(f"Building anchor pool ({args.n_per_source} per source)...",
          flush=True)
    anchors = build_anchor_pool(
        args.crisis_dir, args.selfplay_dir, args.phase1,
        args.n_per_source, args.margin_threshold, args.sample_seed)
    print(f"  total: {len(anchors)}", flush=True)

    print(f"Loading MCTS picks...", flush=True)
    picks_2z_mcts = torch.load(args.picks_2z_mcts,
                                weights_only=False)['picks']
    picks_2y2_mcts = torch.load(args.picks_2y2_mcts,
                                 weights_only=False)['picks']
    print(f"  2z_mcts: {len(picks_2z_mcts)}  2y2_mcts: {len(picks_2y2_mcts)}",
          flush=True)

    from alphatrain.inference_server import InferenceServer
    print(f"Starting 2z server...", flush=True)
    s_2z = InferenceServer(args.model_2z, args.workers, device=args.device,
                            max_batch_per_worker=args.batch_size)
    s_2z.start()
    print(f"Starting v9 server...", flush=True)
    s_v9 = InferenceServer(args.model_v9, args.workers, device=args.device,
                            max_batch_per_worker=args.batch_size)
    s_v9.start()

    anchor_queue = MPQueue()
    for a in anchors:
        anchor_queue.put(a)
    for _ in range(args.workers):
        anchor_queue.put(None)
    result_queue = MPQueue()

    workers = []
    for i in range(args.workers):
        proc = Process(target=_worker,
                       args=(i, anchor_queue, result_queue,
                             s_2z._obs_shm.name, s_2z._pol_shm.name,
                             s_2z._val_shm.name,
                             s_v9._obs_shm.name, s_v9._pol_shm.name,
                             s_v9._val_shm.name,
                             args.workers, args.batch_size,
                             s_2z.request_queue, s_2z.response_queues[i],
                             s_v9.request_queue, s_v9.response_queues[i],
                             args.top_k, args.k_rollouts, args.horizon,
                             picks_2z_mcts, picks_2y2_mcts,
                             args.afterstate_seed_base,
                             args.continuation_seed_base))
        proc.start()
        workers.append(proc)

    t0 = time.time()
    results = []
    skipped = 0
    for i in range(len(anchors)):
        try:
            r = result_queue.get(timeout=1800)
        except Exception:
            print(f"Timeout at {i}", flush=True)
            break
        if 'skipped' in r:
            skipped += 1
        else:
            results.append(r)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(anchors) - i - 1)
            print(f"  [{i+1}/{len(anchors)}] kept={len(results)} "
                  f"skip={skipped} {elapsed:.0f}s ETA {eta:.0f}s", flush=True)

    for w in workers:
        w.join(timeout=10)
    s_2z.shutdown(); s_v9.shutdown()
    print(f"\nDone: {len(results)} judged in {time.time()-t0:.0f}s",
          flush=True)

    # ── Regret table per teacher × source ──
    print(f"\n=== REGRET TABLE (mean cap_rate vs oracle-best) ===",
          flush=True)
    teachers = ['move_2z', 'move_v9', 'move_2z_mcts',
                'move_2y2_mcts', 'move_oracle']
    sources = sorted(set(r['source_label'] for r in results))
    for src in sources:
        rs = [r for r in results if r['source_label'] == src]
        print(f"\n  {src} (n={len(rs)}):", flush=True)
        for t in teachers:
            rc = []; rt = []
            for r in rs:
                pm = r['per_move']
                if not pm:
                    continue
                mv = r.get(t)
                if mv is None or mv not in pm:
                    continue
                best_cap = max(v['cap_rate'] for v in pm.values())
                best_turns = max(v['mean_turns'] for v in pm.values())
                rc.append(best_cap - pm[mv]['cap_rate'])
                rt.append(best_turns - pm[mv]['mean_turns'])
            n = len(rc)
            if n == 0:
                print(f"    {t}: no data", flush=True)
                continue
            rc_arr = np.array(rc); rt_arr = np.array(rt)
            print(f"    {t}: "
                  f"Δcap={rc_arr.mean():.4f}±{rc_arr.std()/np.sqrt(n):.4f}  "
                  f"Δturns={rt_arr.mean():.2f}±{rt_arr.std()/np.sqrt(n):.2f}  "
                  f"(n={n})", flush=True)

    # Head-to-head matrix for crisis specifically
    print(f"\n=== HEAD-TO-HEAD on CRISIS anchors (cap_rate) ===", flush=True)
    crisis_rs = [r for r in results if r['source_label'] == 'crisis']
    for i, ta in enumerate(teachers):
        for tb in teachers[i+1:]:
            wins_a = wins_b = ties = 0
            for r in crisis_rs:
                pm = r['per_move']
                ma = r.get(ta); mb = r.get(tb)
                if ma is None or mb is None: continue
                if ma not in pm or mb not in pm: continue
                ca = pm[ma]['cap_rate']; cb = pm[mb]['cap_rate']
                if ca > cb: wins_a += 1
                elif cb > ca: wins_b += 1
                else: ties += 1
            tot = wins_a + wins_b + ties
            if tot > 0:
                print(f"  {ta} vs {tb}: "
                      f"{wins_a}W / {wins_b}L / {ties}T  "
                      f"({100*wins_a/tot:.0f}% vs {100*wins_b/tot:.0f}%)",
                      flush=True)

    torch.save({'args': vars(args), 'results': results}, args.output)
    print(f"\nSaved {args.output}", flush=True)


if __name__ == '__main__':
    main()
