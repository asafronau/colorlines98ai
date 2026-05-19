"""Stage 1 separability smoke miner — Pillar 3a-v2.

Sample candidate anchor states, run cheap shared-RNG rollouts per top-K
policy move, filter by margin to find anchors where moves actually
separate. The retained anchors flow to Stage 2 for full pairwise label
generation.

Reuses the InferenceServer + N-worker parallelism pattern from
crisis_mining.py. 16 worker processes each handle one anchor at a time;
GPU server batches policy forwards across all workers.

Usage:
    python -m alphatrain.scripts.mine_separable_anchors \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --crisis-dir data/crisis_v12 \\
        --selfplay-dir data/selfplay_v12 \\
        --num-anchors 500 \\
        --crisis-frac 0.70 --selfplay-frac 0.10 \\
        --top-moves 4 --k-rollouts 4 --horizon 150 \\
        --workers 16 --batch-size 8 \\
        --output alphatrain/data/anchors_smoke_500.pt

The remaining 20% is filled with fresh policy-probe rewinds if
--probe-rewinds-frac > 0 (default: mine inside this script).
"""

import os
import json
import glob
import time
import argparse
from random import Random
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory

# Defer heavy imports to worker setup


def sample_anchors_from_jsons(games_dirs, n, label, rng):
    """Sample n anchor states uniformly from a set of game JSON files.

    Returns list of dicts:
      {board, next_balls, turn_origin, source_label, source_file}
    """
    files = []
    for d in games_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
    if not files:
        return []

    anchors = []
    attempts = 0
    while len(anchors) < n and attempts < n * 5:
        attempts += 1
        f = rng.choice(files)
        try:
            with open(f) as fp:
                game = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        moves = game.get('moves', [])
        if not moves:
            continue
        mi = rng.randint(0, len(moves) - 1)
        m = moves[mi]
        anchors.append({
            'board': np.asarray(m['board'], dtype=np.int8),
            'next_balls': [((int(nb['row']), int(nb['col'])), int(nb['color']))
                            for nb in m['next_balls']],
            'num_next': int(m['num_next']),
            'turn_origin': mi,
            'seed_origin': int(game.get('seed', 0)),
            'source_label': label,
            'source_file': os.path.basename(f),
        })
    return anchors


def _rollout_worker(slot_id, anchor_queue, result_queue,
                    obs_shm_name, pol_shm_name, val_shm_name,
                    num_workers, max_batch,
                    request_queue, response_queue,
                    top_k_moves, k_rollouts, horizon,
                    rollout_base_seed):
    """Per-worker: pop anchors, run rollouts for top-K × K_rollouts, return."""
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
    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf,
                             request_queue, response_queue)

    def policy_forward(game):
        obs = _build_obs_for_game(game)
        pol_np, _ = client.evaluate(obs)
        return pol_np

    def do_rollout(anchor, first_action, rollout_seed):
        game = ColorLinesGame(seed=rollout_seed)
        game.reset(board=anchor['board'].copy(),
                   next_balls=list(anchor['next_balls']))
        game.turns = anchor['turn_origin']
        score_before = game.score

        sr = first_action // 81 // 9
        sc = first_action // 81 % 9
        tr = first_action % 81 // 9
        tc = first_action % 81 % 9
        r = game.move((sr, sc), (tr, tc))
        if not r['valid']:
            return {'cap_hit': False, 'score_gain': 0, 'turns': 0,
                    'invalid_first': True}
        if game.game_over:
            return {'cap_hit': False, 'score_gain': game.score - score_before,
                    'turns': game.turns - anchor['turn_origin'], 'died_at_first_spawn': True}

        start_turn = game.turns
        target_end = start_turn + horizon

        while not game.game_over and game.turns < target_end:
            pol_np = policy_forward(game)
            priors = _get_legal_priors_flat(game.board, pol_np, 30)
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
            'score_gain': int(game.score - score_before),
            'turns': int(game.turns - anchor['turn_origin']),
        }

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break

        # Get top-K moves at anchor
        anchor_game = ColorLinesGame(seed=anchor['seed_origin'])
        anchor_game.reset(board=anchor['board'].copy(),
                          next_balls=list(anchor['next_balls']))
        anchor_game.turns = anchor['turn_origin']
        if anchor_game.game_over:
            result_queue.put({'anchor_id': anchor['id'], 'skipped': 'game_over_at_anchor'})
            continue

        pol_np = policy_forward(anchor_game)
        priors = _get_legal_priors_flat(anchor_game.board, pol_np, top_k_moves)
        if not priors or len(priors) < 2:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': f'only {len(priors) if priors else 0} legal'})
            continue
        top_moves = sorted(priors.items(), key=lambda x: x[1], reverse=True)[:top_k_moves]

        # Run rollouts: per move × K_rollouts, shared base RNG
        per_move = {}
        for move_action, prior in top_moves:
            outcomes = []
            for k in range(k_rollouts):
                outcomes.append(do_rollout(anchor, move_action,
                                            rollout_base_seed + k))
            per_move[int(move_action)] = {
                'prior': float(prior),
                'outcomes': outcomes,
                'cap_rate': sum(o['cap_hit'] for o in outcomes) / len(outcomes),
                'mean_score_gain': float(np.mean([o['score_gain'] for o in outcomes])),
                'mean_turns': float(np.mean([o['turns'] for o in outcomes])),
            }

        result_queue.put({
            'anchor_id': anchor['id'],
            'source_label': anchor['source_label'],
            'per_move': per_move,
        })

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def evaluate_separability(per_move, thresholds):
    """Given per-move outcome stats, return list of (move_w, move_l, margin) pairs
    that pass any threshold."""
    pairs = []
    moves = list(per_move.keys())
    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            m_a, m_b = moves[i], moves[j]
            a = per_move[m_a]
            b = per_move[m_b]
            d_cap = a['cap_rate'] - b['cap_rate']
            d_turns = a['mean_turns'] - b['mean_turns']
            d_score = a['mean_score_gain'] - b['mean_score_gain']

            passes_cap = abs(d_cap) >= thresholds['cap_rate']
            passes_turns = abs(d_turns) >= thresholds['turns']
            passes_score = abs(d_score) >= thresholds['score']

            if passes_cap or passes_turns or passes_score:
                # winner = whichever wins on the MOST decisive metric
                # priority: cap_rate > turns > score
                if passes_cap:
                    winner = m_a if d_cap > 0 else m_b
                    margin = abs(d_cap)
                    metric = 'cap_rate'
                elif passes_turns:
                    winner = m_a if d_turns > 0 else m_b
                    margin = abs(d_turns)
                    metric = 'turns'
                else:
                    winner = m_a if d_score > 0 else m_b
                    margin = abs(d_score)
                    metric = 'score'
                loser = m_b if winner == m_a else m_a
                pairs.append({
                    'win_move': winner, 'lose_move': loser,
                    'margin': margin, 'metric': metric,
                    'd_cap': d_cap, 'd_turns': d_turns, 'd_score': d_score,
                })
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--num-anchors', type=int, default=500)
    p.add_argument('--crisis-frac', type=float, default=0.70)
    p.add_argument('--selfplay-frac', type=float, default=0.10)
    # Remaining (1 - crisis - selfplay) reserved for fresh probe rewinds (TODO).
    p.add_argument('--top-moves', type=int, default=4)
    p.add_argument('--k-rollouts', type=int, default=4)
    p.add_argument('--horizon', type=int, default=150)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--rollout-base-seed', type=int, default=100000)
    # Separability thresholds (loose for Stage 1)
    p.add_argument('--cap-rate-threshold', type=float, default=0.25)
    p.add_argument('--turns-threshold', type=float, default=25)
    p.add_argument('--score-threshold', type=float, default=150)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    rng = Random(args.sample_seed)
    n_crisis = int(round(args.num_anchors * args.crisis_frac))
    n_selfplay = int(round(args.num_anchors * args.selfplay_frac))
    n_rewinds = args.num_anchors - n_crisis - n_selfplay  # TODO: implement

    print(f"Anchor budget: {n_crisis} crisis + {n_selfplay} selfplay + "
          f"{n_rewinds} probe-rewinds (rewinds TODO; pulling from selfplay for now)",
          flush=True)

    anchors = []
    anchors += sample_anchors_from_jsons([args.crisis_dir], n_crisis, 'crisis', rng)
    sp_n = n_selfplay + n_rewinds  # fold rewinds into selfplay for v1
    anchors += sample_anchors_from_jsons([args.selfplay_dir], sp_n, 'selfplay', rng)
    for i, a in enumerate(anchors):
        a['id'] = i

    print(f"Sampled {len(anchors)} anchors", flush=True)

    # Spin up InferenceServer and workers
    from alphatrain.inference_server import InferenceServer
    server = InferenceServer(args.model, args.workers, device=args.device,
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
        proc = Process(target=_rollout_worker,
                       args=(i, anchor_queue, result_queue,
                             server._obs_shm.name, server._pol_shm.name,
                             server._val_shm.name,
                             args.workers, args.batch_size,
                             server.request_queue, server.response_queues[i],
                             args.top_moves, args.k_rollouts, args.horizon,
                             args.rollout_base_seed))
        proc.start()
        workers.append(proc)

    # Collect results
    t0 = time.time()
    results = []
    skipped = []
    for i in range(len(anchors)):
        try:
            r = result_queue.get(timeout=600)
        except Exception:
            print(f"Result timeout at {i}", flush=True)
            break
        if 'skipped' in r:
            skipped.append(r)
        else:
            results.append(r)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(anchors) - i - 1)
            print(f"  [{i+1}/{len(anchors)}] kept={len(results)} "
                  f"skip={len(skipped)} {elapsed:.0f}s ETA {eta:.0f}s",
                  flush=True)

    for p in workers:
        p.join(timeout=10)
    server.shutdown()

    print(f"\nDone rollouts: {len(results)} processed, {len(skipped)} skipped, "
          f"in {time.time()-t0:.0f}s", flush=True)

    # Separability filter
    thresholds = {
        'cap_rate': args.cap_rate_threshold,
        'turns': args.turns_threshold,
        'score': args.score_threshold,
    }

    separable_ids = set()
    all_pair_margins = []
    pairs_per_anchor = {}
    bucket_sep = {'crisis': [0, 0], 'selfplay': [0, 0]}  # [n_total, n_sep]

    for r in results:
        anchor_id = r['anchor_id']
        label = r['source_label']
        bucket_sep[label][0] += 1
        pairs = evaluate_separability(r['per_move'], thresholds)
        if pairs:
            separable_ids.add(anchor_id)
            pairs_per_anchor[anchor_id] = pairs
            bucket_sep[label][1] += 1
            for pr in pairs:
                all_pair_margins.append((pr['metric'], pr['margin']))

    sep_frac = len(separable_ids) / max(len(results), 1)
    print(f"\n=== SEPARABILITY REPORT ===", flush=True)
    print(f"Separable anchors: {len(separable_ids)}/{len(results)} "
          f"({100*sep_frac:.1f}%)", flush=True)
    for lab, (tot, sep) in bucket_sep.items():
        pct = 100 * sep / max(tot, 1)
        print(f"  {lab}: {sep}/{tot} ({pct:.1f}%)", flush=True)

    # Per-metric breakdown
    by_metric = {}
    for metric, margin in all_pair_margins:
        by_metric.setdefault(metric, []).append(margin)
    print(f"\nPairs accepted by metric:", flush=True)
    for metric, margins in by_metric.items():
        ms = np.array(margins)
        print(f"  {metric}: n={len(ms)}  mean={ms.mean():.3f}  "
              f"P50={np.median(ms):.3f}  P90={np.percentile(ms,90):.3f}",
              flush=True)

    # Save output
    out = {
        'args': vars(args),
        'anchors': anchors,           # keep all for stage 2 / inspection
        'results': results,           # per-anchor per-move outcomes
        'skipped': skipped,
        'separable_ids': sorted(separable_ids),
        'pairs_per_anchor': pairs_per_anchor,
        'thresholds': thresholds,
        'separability_fraction': sep_frac,
    }
    torch.save(out, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)

    # Decision gate hint
    print(f"\n=== DECISION GATE ===", flush=True)
    if sep_frac >= 0.20:
        print(f"PASS: {100*sep_frac:.1f}% >= 20% threshold. Proceed to Stage 2.",
              flush=True)
    else:
        print(f"FAIL: {100*sep_frac:.1f}% < 20% threshold.", flush=True)
        print(f"  → rollout judge is too weak OR anchors not crisis enough.",
              flush=True)
        print(f"  → DO NOT scale to Stage 2 without diagnosing.", flush=True)


if __name__ == '__main__':
    main()
