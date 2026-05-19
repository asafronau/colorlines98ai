"""Judge source C (first-divergence) anchors: for each anchor, run K=32
common-RNG rollouts on the 2z move, the v9 move, and the rollout-oracle
top-1 pick from 2z's top-K. Compute regret per teacher.

This answers: when 2z and v9 first differ in identical-seed play, who's
actually right? If v9's pick has higher regret than 2z's, the fine-tune
broke real decisions. If v9 matches/beats 2z, the regression must come
from downstream distribution shift, not the divergence point itself.

Usage:
    python -m alphatrain.scripts.judge_source_c \\
        --model-2z alphatrain/data/pillar2z_epoch_19.pt \\
        --source-c alphatrain/data/source_c_first_divergence.pt \\
        --k-rollouts 32 --horizon 300 --top-k 6 \\
        --workers 16 --batch-size 8 \\
        --device mps \\
        --output alphatrain/data/source_c_judged.pt
"""

import argparse
import time
import numpy as np
import torch
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def _worker(slot_id, anchor_queue, result_queue,
            obs_shm_name, pol_shm_name, val_shm_name,
            num_workers, max_batch,
            req_q, resp_q,
            top_k_moves, k_rollouts, horizon,
            afterstate_seed_base, continuation_seed_base):
    torch.set_num_threads(1)
    from alphatrain.inference_server import InferenceClient
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)
    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, 6561), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)
    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf, req_q, resp_q)

    def policy_forward(game):
        obs = _build_obs_for_game(game)
        pol, _ = client.evaluate(obs)
        return pol

    def realize_afterstate(anchor, move_action, afterstate_seed):
        game = ColorLinesGame(seed=afterstate_seed)
        game.reset(board=anchor['board'].copy(),
                   next_balls=list(anchor['next_balls']))
        game.turns = anchor['turn_origin']
        sr = move_action // 81 // 9
        sc = move_action // 81 % 9
        tr = move_action % 81 // 9
        tc = move_action % 81 % 9
        r = game.move((sr, sc), (tr, tc))
        if not r['valid']:
            return None
        return {
            'board': game.board.copy(),
            'next_balls': list(game.next_balls),
            'turns': game.turns,
            'score_at_after': game.score,
            'game_over': game.game_over,
        }

    def continue_from_afterstate(afterstate, continuation_seed):
        if afterstate['game_over']:
            return {'cap_hit': False,
                    'score_gain_after': int(afterstate['score_at_after']),
                    'turns_after': 0}
        game = ColorLinesGame(seed=continuation_seed)
        game.reset(board=afterstate['board'].copy(),
                   next_balls=list(afterstate['next_balls']))
        game.turns = afterstate['turns']
        score_start = game.score
        target_end = game.turns + horizon
        while not game.game_over and game.turns < target_end:
            pol = policy_forward(game)
            priors = _get_legal_priors_flat(game.board, pol, 30)
            if not priors:
                break
            best = max(priors.items(), key=lambda x: x[1])[0]
            r = game.move((best // 81 // 9, best // 81 % 9),
                          (best % 81 // 9, best % 81 % 9))
            if not r['valid']:
                break
        survived = (not game.game_over) and (game.turns >= target_end)
        return {
            'cap_hit': bool(survived),
            'score_gain_after': int(afterstate['score_at_after']
                                    + (game.score - score_start)),
            'turns_after': int(game.turns - afterstate['turns']),
        }

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break

        ag = ColorLinesGame(seed=0)
        ag.reset(board=anchor['board'].copy(),
                 next_balls=list(anchor['next_balls']))
        ag.turns = anchor['turn_origin']
        if ag.game_over:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'game_over'})
            continue

        # Get 2z top-K for oracle pick
        pol = policy_forward(ag)
        priors = _get_legal_priors_flat(ag.board, pol, top_k_moves)
        top_2z = sorted(priors.items(), key=lambda x: -x[1])[:top_k_moves] \
            if priors else []
        top_2z_moves = [int(m) for m, _ in top_2z]

        # Candidates: 2z's move (anchor's move_a), v9's move (move_b),
        # plus 2z's top-K so we can find oracle pick.
        candidates = set([int(anchor['move_a']), int(anchor['move_b'])])
        candidates.update(top_2z_moves)

        anchor_after_seed = afterstate_seed_base + anchor['id'] * 7919
        anchor_cont_base = continuation_seed_base + anchor['id'] * 7919 * 1000
        per_move = {}
        for mv in candidates:
            af = realize_afterstate(anchor, mv, anchor_after_seed)
            if af is None:
                continue
            outcomes = [continue_from_afterstate(af, anchor_cont_base + k)
                         for k in range(k_rollouts)]
            per_move[int(mv)] = {
                'cap_rate': sum(o['cap_hit'] for o in outcomes) / len(outcomes),
                'mean_turns': float(np.mean([o['turns_after']
                                              for o in outcomes])),
                'mean_score': float(np.mean([o['score_gain_after']
                                              for o in outcomes])),
            }

        # Oracle pick = best cap_rate among 2z top-K (only).
        cands_in_top = [m for m in top_2z_moves if m in per_move]
        oracle_pick = max(cands_in_top,
                          key=lambda m: per_move[m]['cap_rate']) \
            if cands_in_top else int(anchor['move_a'])

        result_queue.put({
            'anchor_id': anchor['id'],
            'seed_origin': anchor['seed_origin'],
            'turn_origin': anchor['turn_origin'],
            'move_2z': int(anchor['move_a']),
            'move_v9': int(anchor['move_b']),
            'move_oracle': int(oracle_pick),
            'top_2z_moves': top_2z_moves,
            'per_move': per_move,
        })

    obs_shm.close(); pol_shm.close(); val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-2z', required=True)
    p.add_argument('--source-c', required=True)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--k-rollouts', type=int, default=32)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--afterstate-seed-base', type=int, default=7000000)
    p.add_argument('--continuation-seed-base', type=int, default=8000000)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    print(f"Loading source C from {args.source_c}...", flush=True)
    sc = torch.load(args.source_c, weights_only=False)
    anchors = sc['anchors']
    print(f"  {len(anchors)} first-divergence anchors", flush=True)
    if not anchors:
        print(f"No anchors. Aborting.", flush=True)
        return

    from alphatrain.inference_server import InferenceServer
    server = InferenceServer(args.model_2z, args.workers, device=args.device,
                              max_batch_per_worker=args.batch_size)
    server.start()

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
                             server._obs_shm.name, server._pol_shm.name,
                             server._val_shm.name,
                             args.workers, args.batch_size,
                             server.request_queue,
                             server.response_queues[i],
                             args.top_k, args.k_rollouts, args.horizon,
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
    server.shutdown()
    print(f"\nDone: {len(results)} judged in {time.time()-t0:.0f}s",
          flush=True)

    # Regret table
    print(f"\n=== SOURCE C REGRET (first-divergence states) ===", flush=True)
    teachers = ['move_2z', 'move_v9', 'move_oracle']
    for teacher in teachers:
        regrets_cap = []
        regrets_turns = []
        for r in results:
            pm = r['per_move']
            if not pm:
                continue
            oracle_best_cap = max(v['cap_rate'] for v in pm.values())
            oracle_best_turns = max(v['mean_turns'] for v in pm.values())
            t_move = r[teacher]
            if t_move not in pm:
                continue
            regrets_cap.append(oracle_best_cap - pm[t_move]['cap_rate'])
            regrets_turns.append(oracle_best_turns - pm[t_move]['mean_turns'])
        n = len(regrets_cap)
        if n == 0:
            continue
        rc = np.array(regrets_cap)
        rt = np.array(regrets_turns)
        print(f"  {teacher}: "
              f"Δcap_rate={rc.mean():.4f}±{rc.std()/np.sqrt(n):.4f}  "
              f"Δmean_turns={rt.mean():.2f}±{rt.std()/np.sqrt(n):.2f}  "
              f"(n={n})", flush=True)

    # Head-to-head: 2z vs v9 directly
    print(f"\n=== HEAD-TO-HEAD: 2z move vs v9 move on the same anchor ===",
          flush=True)
    wins_2z = 0; wins_v9 = 0; ties = 0; total_diff_cap = 0.0
    for r in results:
        pm = r['per_move']
        m_2z, m_v9 = r['move_2z'], r['move_v9']
        if m_2z not in pm or m_v9 not in pm:
            continue
        c_2z = pm[m_2z]['cap_rate']
        c_v9 = pm[m_v9]['cap_rate']
        if c_2z > c_v9: wins_2z += 1
        elif c_v9 > c_2z: wins_v9 += 1
        else: ties += 1
        total_diff_cap += (c_v9 - c_2z)
    total = wins_2z + wins_v9 + ties
    if total > 0:
        print(f"  2z wins: {wins_2z} ({100*wins_2z/total:.1f}%)", flush=True)
        print(f"  v9 wins: {wins_v9} ({100*wins_v9/total:.1f}%)", flush=True)
        print(f"  ties:    {ties} ({100*ties/total:.1f}%)", flush=True)
        print(f"  mean Δcap (v9 - 2z): {total_diff_cap/total:+.4f}", flush=True)

    torch.save({'args': vars(args), 'results': results}, args.output)
    print(f"\nSaved {args.output}", flush=True)


if __name__ == '__main__':
    main()
