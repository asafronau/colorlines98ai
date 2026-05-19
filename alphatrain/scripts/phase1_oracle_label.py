"""Phase 1 — Oracle labeling for DAgger.

Per-anchor, capture rich per-move quality stats: top-K policy moves, each
move's realized afterstate, K independent common-RNG continuations, and
aggregate (cap_rate, mean_turns, mean_score). Saves full per-move data for
Phase 2 (DAgger policy retraining).

Differences from pilot_clean_pairwise.py:
  - top-K = 6 (was 4) — wider candidate space
  - K = 32 continuations (was 16) — half the label variance
  - Common-RNG seeds shared across sibling moves (fix from 2026-05-15)
  - Saves FULL per-move stats, not just pairwise winners
  - Optional turn-bucket stratification

Output: per-anchor record with all top-K moves' quality vectors.

Usage:
    python -m alphatrain.scripts.phase1_oracle_label \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --crisis-dir data/crisis_v12 \\
        --selfplay-dir data/selfplay_v12 \\
        --num-anchors 2000 --crisis-frac 0.5 --selfplay-frac 0.5 \\
        --top-moves 6 --k-continuations 32 --horizon 300 \\
        --workers 16 --batch-size 8 \\
        --output alphatrain/data/phase1_oracle.pt
"""

import os
import json
import glob
import time
import argparse
from random import Random
import numpy as np
import torch
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def sample_anchors_from_jsons(games_dirs, n, label, rng):
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


def _worker(slot_id, anchor_queue, result_queue,
            obs_shm_name, pol_shm_name, val_shm_name,
            num_workers, max_batch,
            request_queue, response_queue,
            top_k_moves, k_continuations, horizon,
            afterstate_seed_base, continuation_seed_base):
    torch.set_num_threads(1)
    from alphatrain.inference_server import InferenceClient
    from alphatrain.mcts import _get_legal_priors_flat
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

    from alphatrain.mcts import _build_obs_for_game

    def policy_forward(game):
        obs = _build_obs_for_game(game)
        pol_np, _ = client.evaluate(obs)
        return pol_np

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
            'score_gain_after': int(afterstate['score_at_after']
                                    + (game.score - score_start)),
            'turns_after': int(game.turns - afterstate['turns']),
        }

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break

        anchor_game = ColorLinesGame(seed=anchor['seed_origin'])
        anchor_game.reset(board=anchor['board'].copy(),
                          next_balls=list(anchor['next_balls']))
        anchor_game.turns = anchor['turn_origin']
        if anchor_game.game_over:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'game_over'})
            continue
        pol_np = policy_forward(anchor_game)
        priors = _get_legal_priors_flat(anchor_game.board, pol_np, top_k_moves)
        if not priors or len(priors) < 2:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'fewer_moves'})
            continue
        top_moves = sorted(priors.items(), key=lambda x: x[1],
                           reverse=True)[:top_k_moves]

        # Common-RNG seeds shared across sibling moves under this anchor.
        anchor_after_seed = afterstate_seed_base + anchor['id'] * 7919
        anchor_cont_base = continuation_seed_base + anchor['id'] * 7919 * 1000

        per_move = {}
        for rank, (move_action, prior) in enumerate(top_moves, start=1):
            af = realize_afterstate(anchor, move_action, anchor_after_seed)
            if af is None:
                continue
            outcomes = []
            for k in range(k_continuations):
                cs = anchor_cont_base + k
                outcomes.append(continue_from_afterstate(af, cs))
            per_move[int(move_action)] = {
                'prior': float(prior),
                'rank': int(rank),
                'afterstate_board': af['board'],
                'afterstate_next_balls': af['next_balls'],
                'afterstate_game_over': af['game_over'],
                'afterstate_score_at_after': int(af['score_at_after']),
                'cap_rate': sum(o['cap_hit'] for o in outcomes) / len(outcomes),
                'mean_turns': float(np.mean([o['turns_after']
                                              for o in outcomes])),
                'mean_score': float(np.mean([o['score_gain_after']
                                              for o in outcomes])),
                'std_turns': float(np.std([o['turns_after']
                                            for o in outcomes])),
                'std_score': float(np.std([o['score_gain_after']
                                            for o in outcomes])),
                'outcomes': outcomes,  # raw, for stability re-checks
            }

        result_queue.put({
            'anchor_id': anchor['id'],
            'anchor_board': anchor['board'],
            'anchor_next_balls': anchor['next_balls'],
            'anchor_n_next': anchor['num_next'],
            'turn_origin': anchor['turn_origin'],
            'source_label': anchor['source_label'],
            'source_file': anchor['source_file'],
            'per_move': per_move,
        })

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--num-anchors', type=int, default=2000)
    p.add_argument('--crisis-frac', type=float, default=0.5)
    p.add_argument('--selfplay-frac', type=float, default=0.5)
    p.add_argument('--top-moves', type=int, default=6)
    p.add_argument('--k-continuations', type=int, default=32)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--sample-seed', type=int, default=137)
    p.add_argument('--afterstate-seed-base', type=int, default=2000000)
    p.add_argument('--continuation-seed-base', type=int, default=3000000)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    rng = Random(args.sample_seed)
    n_crisis = int(round(args.num_anchors * args.crisis_frac))
    n_selfplay = args.num_anchors - n_crisis
    print(f"Anchor budget: {n_crisis} crisis + {n_selfplay} selfplay",
          flush=True)

    anchors = []
    anchors += sample_anchors_from_jsons([args.crisis_dir], n_crisis,
                                          'crisis', rng)
    anchors += sample_anchors_from_jsons([args.selfplay_dir], n_selfplay,
                                          'selfplay', rng)
    for i, a in enumerate(anchors):
        a['id'] = i
    print(f"Sampled {len(anchors)} anchors", flush=True)

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
        proc = Process(target=_worker,
                       args=(i, anchor_queue, result_queue,
                             server._obs_shm.name, server._pol_shm.name,
                             server._val_shm.name,
                             args.workers, args.batch_size,
                             server.request_queue, server.response_queues[i],
                             args.top_moves, args.k_continuations, args.horizon,
                             args.afterstate_seed_base,
                             args.continuation_seed_base))
        proc.start()
        workers.append(proc)

    t0 = time.time()
    results = []
    skipped = 0
    for i in range(len(anchors)):
        try:
            r = result_queue.get(timeout=1200)
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
                  f"skip={skipped} {elapsed:.0f}s ETA {eta:.0f}s",
                  flush=True)

    for w in workers:
        w.join(timeout=10)
    server.shutdown()
    print(f"\nDone: kept={len(results)} skipped={skipped} "
          f"in {time.time()-t0:.0f}s", flush=True)

    # Save full results
    out = {
        'args': vars(args),
        'results': results,
    }
    torch.save(out, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)

    # Quick audit
    n_top1_loses_to_best = 0
    n_anchors_with_data = 0
    best_margin_cap = []
    best_margin_turns = []
    for r in results:
        pm = r['per_move']
        if len(pm) < 2:
            continue
        n_anchors_with_data += 1
        moves = sorted(pm.items(), key=lambda x: -x[1]['prior'])  # by prior desc
        top1 = moves[0][1]
        best_move = max(pm.items(), key=lambda x: x[1]['cap_rate'])
        if best_move[1]['prior'] < top1['prior']:
            n_top1_loses_to_best += 1
            best_margin_cap.append(best_move[1]['cap_rate'] - top1['cap_rate'])
        best_turns_move = max(pm.items(), key=lambda x: x[1]['mean_turns'])
        if best_turns_move[1]['prior'] < top1['prior']:
            best_margin_turns.append(best_turns_move[1]['mean_turns']
                                      - top1['mean_turns'])
    print(f"\n=== Quick audit ===", flush=True)
    print(f"Anchors with >=2 moves resolved: {n_anchors_with_data}",
          flush=True)
    print(f"P(best-cap_rate-move not policy_top_1): "
          f"{n_top1_loses_to_best}/{n_anchors_with_data} = "
          f"{100*n_top1_loses_to_best/max(n_anchors_with_data,1):.1f}%",
          flush=True)
    if best_margin_cap:
        print(f"  Δcap_rate when oracle disagrees: "
              f"mean={np.mean(best_margin_cap):.3f}  "
              f"median={np.median(best_margin_cap):.3f}", flush=True)
    if best_margin_turns:
        print(f"  Δmean_turns when oracle disagrees: "
              f"mean={np.mean(best_margin_turns):.1f}  "
              f"median={np.median(best_margin_turns):.1f}", flush=True)


if __name__ == '__main__':
    main()
