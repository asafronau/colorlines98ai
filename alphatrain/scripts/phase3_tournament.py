"""Phase 3 — teacher tournament (causal audit per ChatGPT 2026-05-15).

For each anchor in a stratified pool, collect candidate moves from multiple
teachers, then judge each UNIQUE candidate move with K=32 common-RNG
rollouts at H=300. Report regret per (anchor_source, teacher).

This version: 3 policy-only teachers + rollout oracle. MCTS teachers added
later if v1 results justify the extra complexity.

Teachers:
  1. 2z policy top-1
  2. v9 policy top-1 (DAgger-trained)
  3. Rollout oracle (top-K from 2z policy, K=32 rollouts each, pick best
     cap_rate). Note this is what Phase 1 produced.

Anchor sources (200 each):
  A. Normal 2z policy states (selfplay_v12)
  B. 2z crisis/failure states (crisis_v12)
  D. High-margin oracle-disagreement states (filtered from phase1_oracle.pt)

Output:
  - per-anchor: candidates from each teacher, K=32 outcomes per unique move
  - regret table: per (source, teacher), mean regret in cap_rate and turns

Usage:
    python -m alphatrain.scripts.phase3_tournament \\
        --model-2z alphatrain/data/pillar2z_epoch_19.pt \\
        --model-v9 alphatrain/data/policy_dagger_v9.pt \\
        --phase1 alphatrain/data/phase1_oracle.pt \\
        --crisis-dir data/crisis_v12 --selfplay-dir data/selfplay_v12 \\
        --n-per-source 200 \\
        --k-rollouts 32 --horizon 300 --top-k 6 \\
        --workers 16 --batch-size 8 \\
        --device mps \\
        --output alphatrain/data/phase3_tournament.pt
"""

import argparse
import hashlib
import json
import glob
import os
import time
from random import Random
import numpy as np
import torch
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def sample_states_from_jsons(dirs, n, rng, label):
    files = []
    for d in dirs:
        files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
    out = []
    while len(out) < n:
        f = rng.choice(files)
        try:
            with open(f) as fp:
                game = json.load(fp)
        except Exception:
            continue
        moves = game.get('moves', [])
        if not moves:
            continue
        mi = rng.randint(0, len(moves) - 1)
        m = moves[mi]
        out.append({
            'board': np.asarray(m['board'], dtype=np.int8),
            'next_balls': [((int(nb['row']), int(nb['col'])), int(nb['color']))
                            for nb in m['next_balls']],
            'num_next': int(m['num_next']),
            'source_label': label,
            'turn_origin': mi,
            'seed_origin': int(game.get('seed', 0)),
        })
    return out


def select_high_margin_from_phase1(phase1_path, margin_threshold, n):
    """Source D: anchors where rollout oracle disagrees with policy by big margin."""
    data = torch.load(phase1_path, weights_only=False)
    out = []
    for r in data['results']:
        pm = r['per_move']
        if len(pm) < 2:
            continue
        sorted_moves = sorted(pm.items(), key=lambda kv: kv[1]['rank'])[:6]
        qs = np.array([mv['cap_rate'] for _, mv in sorted_moves])
        margin = qs.max() - qs[0]
        if margin >= margin_threshold:
            out.append({
                'board': r['anchor_board'],
                'next_balls': r['anchor_next_balls'],
                'num_next': r['anchor_n_next'],
                'source_label': 'oracle_disagree',
                'turn_origin': r.get('turn_origin', 0),
                'seed_origin': r.get('source_file', 'phase1'),
                # Embed precomputed teacher info to save work later.
                'phase1_per_move': {
                    int(mv): {
                        'cap_rate': p['cap_rate'],
                        'mean_turns': p['mean_turns'],
                        'mean_score': p['mean_score'],
                        'prior': p['prior'],
                        'rank': p['rank'],
                    } for mv, p in r['per_move'].items()
                },
            })
    rng = Random(0)
    rng.shuffle(out)
    return out[:n]


def _worker(slot_id, anchor_queue, result_queue,
            obs_shm_name, pol_shm_name, val_shm_name,
            num_workers, max_batch,
            req_queue_2z, resp_queue_2z,
            req_queue_v9, resp_queue_v9,
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
    # Two banks of shared memory: one per server.
    # Layout: [server_idx, worker_idx, batch_idx, ...]
    obs_buf_2z = np.ndarray((N, B, 18, 9, 9), dtype=np.float32,
                              buffer=obs_shm.buf, offset=0)
    obs_buf_v9 = np.ndarray((N, B, 18, 9, 9), dtype=np.float32,
                              buffer=obs_shm.buf,
                              offset=N * B * 18 * 9 * 9 * 4)
    pol_buf_2z = np.ndarray((N, B, 6561), dtype=np.float32,
                              buffer=pol_shm.buf, offset=0)
    pol_buf_v9 = np.ndarray((N, B, 6561), dtype=np.float32,
                              buffer=pol_shm.buf,
                              offset=N * B * 6561 * 4)
    val_buf_2z = np.ndarray((N, B), dtype=np.float32,
                              buffer=val_shm.buf, offset=0)
    val_buf_v9 = np.ndarray((N, B), dtype=np.float32,
                              buffer=val_shm.buf,
                              offset=N * B * 4)

    client_2z = InferenceClient(slot_id, obs_buf_2z, pol_buf_2z, val_buf_2z,
                                 req_queue_2z, resp_queue_2z)
    client_v9 = InferenceClient(slot_id, obs_buf_v9, pol_buf_v9, val_buf_v9,
                                 req_queue_v9, resp_queue_v9)

    def policy_2z(game):
        obs = _build_obs_for_game(game)
        pol, _ = client_2z.evaluate(obs)
        return pol

    def policy_v9(game):
        obs = _build_obs_for_game(game)
        pol, _ = client_v9.evaluate(obs)
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
        """Continue from afterstate using 2z policy."""
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
            pol = policy_2z(game)
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

        # ── Gather candidate moves from each teacher ──
        # 2z top-K and top-1
        pol_2z_v = policy_2z(ag)
        priors_2z = _get_legal_priors_flat(ag.board, pol_2z_v, top_k_moves)
        if not priors_2z or len(priors_2z) < 2:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'fewer_moves'})
            continue
        top_2z = sorted(priors_2z.items(), key=lambda x: -x[1])[:top_k_moves]
        move_2z = top_2z[0][0]

        # v9 top-1
        pol_v9_v = policy_v9(ag)
        priors_v9 = _get_legal_priors_flat(ag.board, pol_v9_v, top_k_moves)
        move_v9 = sorted(priors_v9.items(), key=lambda x: -x[1])[0][0] \
            if priors_v9 else move_2z

        # Candidates = union of (2z top-K, v9 top-1). The oracle's pick is
        # determined later as max(cap_rate over 2z top-K).
        candidates = {int(m): {'rank_2z': r + 1, 'prior_2z': p}
                       for r, (m, p) in enumerate(top_2z)}
        candidates.setdefault(int(move_v9), {'rank_2z': -1, 'prior_2z': 0.0})

        # ── Run K=32 common-RNG rollouts per UNIQUE candidate ──
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
                'mean_turns': float(np.mean([o['turns_after'] for o in outcomes])),
                'mean_score': float(np.mean([o['score_gain_after'] for o in outcomes])),
                'std_turns': float(np.std([o['turns_after'] for o in outcomes])),
                **candidates[int(mv)],
            }

        # Oracle pick = move with highest cap_rate among 2z top-K.
        top_2z_moves = [int(m) for m, _ in top_2z]
        oracle_pick = max(top_2z_moves,
                          key=lambda m: per_move.get(m, {'cap_rate': -1})['cap_rate'])

        result_queue.put({
            'anchor_id': anchor['id'],
            'anchor_board': anchor['board'],
            'source_label': anchor['source_label'],
            'turn_origin': anchor['turn_origin'],
            'move_2z': int(move_2z),
            'move_v9': int(move_v9),
            'move_oracle': int(oracle_pick),
            'top_2z_moves': top_2z_moves,
            'per_move': per_move,
        })

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-2z', required=True)
    p.add_argument('--model-v9', required=True)
    p.add_argument('--phase1', required=True,
                   help='phase1_oracle.pt — source for oracle-disagreement anchors.')
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--n-per-source', type=int, default=200)
    p.add_argument('--top-k', type=int, default=6)
    p.add_argument('--k-rollouts', type=int, default=32)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--margin-threshold', type=float, default=0.15)
    p.add_argument('--afterstate-seed-base', type=int, default=5000000)
    p.add_argument('--continuation-seed-base', type=int, default=6000000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    rng = Random(args.seed)

    # ── Build anchor pool ──
    print(f"Building anchor pool ({args.n_per_source} per source)...",
          flush=True)
    src_a = sample_states_from_jsons([args.selfplay_dir],
                                       args.n_per_source, rng, 'selfplay')
    src_b = sample_states_from_jsons([args.crisis_dir],
                                       args.n_per_source, rng, 'crisis')
    src_d = select_high_margin_from_phase1(
        args.phase1, args.margin_threshold, args.n_per_source)
    anchors = src_a + src_b + src_d
    for i, a in enumerate(anchors):
        a['id'] = i
    print(f"  sources: A={len(src_a)} (selfplay), "
          f"B={len(src_b)} (crisis), D={len(src_d)} (oracle-disagree)",
          flush=True)
    print(f"  total: {len(anchors)} anchors", flush=True)

    # ── Two inference servers (one per model) ──
    from alphatrain.inference_server import InferenceServer

    print(f"\nStarting 2z inference server ({args.model_2z})...", flush=True)
    server_2z = InferenceServer(args.model_2z, args.workers,
                                  device=args.device,
                                  max_batch_per_worker=args.batch_size)
    server_2z.start()

    print(f"Starting v9 inference server ({args.model_v9})...", flush=True)
    server_v9 = InferenceServer(args.model_v9, args.workers,
                                  device=args.device,
                                  max_batch_per_worker=args.batch_size)
    server_v9.start()

    # We need to give workers access to BOTH servers' shared memory. The
    # existing InferenceServer uses its own SHM names. Workers can be set up
    # to talk to both by passing both queue/SHM references.
    # Simpler approach: use ONE custom SHM that fits both, and pass req/resp
    # queues for both. But InferenceServer pre-allocates its own SHM.
    # Hack: pass both SHM names + queues to worker.

    anchor_queue = MPQueue()
    for a in anchors:
        anchor_queue.put(a)
    for _ in range(args.workers):
        anchor_queue.put(None)
    result_queue = MPQueue()

    # Build workers. To avoid the dual-SHM complication, just use 2z server's
    # SHM for the worker's primary connection and call v9 via its server's
    # queues. Simplest: the worker uses 2z server for both 2z and v9 calls
    # by switching the SHM buffer pointer per call. This requires the
    # InferenceClient to be set up twice with different SHMs.

    # For simplicity, we'll have the worker independently call each server
    # with its own InferenceClient. SHMs are separate. The worker function
    # signature handles this via two sets of buffers.

    workers = []
    for i in range(args.workers):
        # Pass each server's SHM names + queues.
        proc = Process(target=_worker_dual_server,
                       args=(i, anchor_queue, result_queue,
                             server_2z._obs_shm.name, server_2z._pol_shm.name,
                             server_2z._val_shm.name,
                             server_v9._obs_shm.name, server_v9._pol_shm.name,
                             server_v9._val_shm.name,
                             args.workers, args.batch_size,
                             server_2z.request_queue,
                             server_2z.response_queues[i],
                             server_v9.request_queue,
                             server_v9.response_queues[i],
                             args.top_k, args.k_rollouts, args.horizon,
                             args.afterstate_seed_base,
                             args.continuation_seed_base))
        proc.start()
        workers.append(proc)

    # Collect
    t0 = time.time()
    results = []
    skipped = 0
    for i in range(len(anchors)):
        try:
            r = result_queue.get(timeout=1800)
        except Exception:
            print(f"Result timeout at {i}", flush=True)
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
    server_2z.shutdown()
    server_v9.shutdown()
    print(f"\nDone: kept={len(results)} skipped={skipped} "
          f"in {time.time()-t0:.0f}s", flush=True)

    # ── Compute regret table ──
    print(f"\n=== REGRET TABLE (mean cap_rate vs oracle-best) ===",
          flush=True)
    teachers = ['move_2z', 'move_v9', 'move_oracle']
    sources_seen = sorted(set(r['source_label'] for r in results))

    for src in sources_seen:
        rs = [r for r in results if r['source_label'] == src]
        if not rs:
            continue
        print(f"\n  {src} (n={len(rs)}):", flush=True)
        for teacher in teachers:
            regrets_cap = []
            regrets_turns = []
            for r in rs:
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
            print(f"    {teacher}: "
                  f"Δcap_rate={np.mean(regrets_cap):.3f}±{np.std(regrets_cap)/np.sqrt(max(len(regrets_cap),1)):.3f}  "
                  f"Δmean_turns={np.mean(regrets_turns):.1f}±{np.std(regrets_turns)/np.sqrt(max(len(regrets_turns),1)):.1f}  "
                  f"(n={len(regrets_cap)})", flush=True)

    torch.save({'args': vars(args), 'results': results}, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)


def _worker_dual_server(slot_id, anchor_queue, result_queue,
                         obs_shm_2z, pol_shm_2z, val_shm_2z,
                         obs_shm_v9, pol_shm_v9, val_shm_v9,
                         num_workers, max_batch,
                         req_q_2z, resp_q_2z,
                         req_q_v9, resp_q_v9,
                         top_k_moves, k_rollouts, horizon,
                         afterstate_seed_base, continuation_seed_base):
    """Worker connecting to two separate inference servers."""
    torch.set_num_threads(1)
    from alphatrain.inference_server import InferenceClient
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    o_shm_2z = SharedMemory(name=obs_shm_2z)
    p_shm_2z = SharedMemory(name=pol_shm_2z)
    v_shm_2z = SharedMemory(name=val_shm_2z)
    o_shm_v9 = SharedMemory(name=obs_shm_v9)
    p_shm_v9 = SharedMemory(name=pol_shm_v9)
    v_shm_v9 = SharedMemory(name=val_shm_v9)

    N, B = num_workers, max_batch
    obs_buf_2z = np.ndarray((N, B, 18, 9, 9), dtype=np.float32,
                              buffer=o_shm_2z.buf)
    pol_buf_2z = np.ndarray((N, B, 6561), dtype=np.float32,
                              buffer=p_shm_2z.buf)
    val_buf_2z = np.ndarray((N, B), dtype=np.float32, buffer=v_shm_2z.buf)
    obs_buf_v9 = np.ndarray((N, B, 18, 9, 9), dtype=np.float32,
                              buffer=o_shm_v9.buf)
    pol_buf_v9 = np.ndarray((N, B, 6561), dtype=np.float32,
                              buffer=p_shm_v9.buf)
    val_buf_v9 = np.ndarray((N, B), dtype=np.float32, buffer=v_shm_v9.buf)

    client_2z = InferenceClient(slot_id, obs_buf_2z, pol_buf_2z, val_buf_2z,
                                  req_q_2z, resp_q_2z)
    client_v9 = InferenceClient(slot_id, obs_buf_v9, pol_buf_v9, val_buf_v9,
                                  req_q_v9, resp_q_v9)

    def policy_2z(game):
        obs = _build_obs_for_game(game)
        pol, _ = client_2z.evaluate(obs)
        return pol

    def policy_v9(game):
        obs = _build_obs_for_game(game)
        pol, _ = client_v9.evaluate(obs)
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
            pol = policy_2z(game)
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

        pol_2z_v = policy_2z(ag)
        priors_2z = _get_legal_priors_flat(ag.board, pol_2z_v, top_k_moves)
        if not priors_2z or len(priors_2z) < 2:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'fewer_moves'})
            continue
        top_2z = sorted(priors_2z.items(), key=lambda x: -x[1])[:top_k_moves]
        move_2z = top_2z[0][0]

        pol_v9_v = policy_v9(ag)
        priors_v9 = _get_legal_priors_flat(ag.board, pol_v9_v, top_k_moves)
        move_v9 = sorted(priors_v9.items(), key=lambda x: -x[1])[0][0] \
            if priors_v9 else move_2z

        candidates = {int(m): {'rank_2z': r + 1, 'prior_2z': float(p)}
                       for r, (m, p) in enumerate(top_2z)}
        candidates.setdefault(
            int(move_v9), {'rank_2z': -1, 'prior_2z': 0.0})

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
                **candidates[int(mv)],
            }

        top_2z_moves = [int(m) for m, _ in top_2z]
        moves_in_pm = [m for m in top_2z_moves if m in per_move]
        oracle_pick = max(moves_in_pm,
                          key=lambda m: per_move[m]['cap_rate']) \
            if moves_in_pm else int(move_2z)

        result_queue.put({
            'anchor_id': anchor['id'],
            'anchor_board': anchor['board'],
            'source_label': anchor['source_label'],
            'turn_origin': anchor['turn_origin'],
            'move_2z': int(move_2z),
            'move_v9': int(move_v9),
            'move_oracle': int(oracle_pick),
            'top_2z_moves': top_2z_moves,
            'per_move': per_move,
        })

    o_shm_2z.close(); p_shm_2z.close(); v_shm_2z.close()
    o_shm_v9.close(); p_shm_v9.close(); v_shm_v9.close()


if __name__ == '__main__':
    main()
