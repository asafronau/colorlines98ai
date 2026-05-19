"""Stage 2 — build pairwise training dataset from Stage 1's separable anchors.

For each separable anchor (output of mine_separable_anchors.py):
  1. Compute top-K candidate moves from raw policy.
  2. For each move: K=8 rollouts × H=300 turns, shared RNG, policy-only.
  3. Capture the AFTERSTATE (board + next_balls after move+spawn) — what the
     value head will see at MCTS leaf time.
  4. Apply tighter Stage 2 threshold per ChatGPT: Δcap_rate ≥ 0.375 OR
     Δturns ≥ 50 OR Δscore ≥ 300. Discard near-ties (they poison the head).
  5. Output pairwise training tensor.

Reuses InferenceServer + N-worker pattern.

Usage:
    python -m alphatrain.scripts.build_pairwise_dataset \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --stage1 alphatrain/data/anchors_stage1_5000.pt \\
        --top-moves 4 --k-rollouts 8 --horizon 300 \\
        --workers 16 --batch-size 8 \\
        --output alphatrain/data/pairwise_v12.pt
"""

import os
import time
import argparse
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def _stage2_worker(slot_id, anchor_queue, result_queue,
                   obs_shm_name, pol_shm_name, val_shm_name,
                   num_workers, max_batch,
                   request_queue, response_queue,
                   top_k_moves, k_rollouts, horizon,
                   rollout_base_seed):
    """Per-worker: pop anchor, run K=8/H=300 rollouts on top-K moves, capture afterstates."""
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

    def get_afterstate_and_outcome(anchor, first_action, rollout_seed):
        """Apply first_action to anchor; capture afterstate; rollout H more turns.
        Returns: (afterstate_board, afterstate_next_balls, afterstate_n_next,
                  outcome_dict).
        """
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
            return None, None, None, {'cap_hit': False, 'score_gain': 0,
                                       'turns': 0, 'invalid_first': True}

        # Capture afterstate (post-move-and-spawn)
        afterstate_board = game.board.copy()
        afterstate_next_balls = list(game.next_balls)
        afterstate_n_next = len(game.next_balls)

        if game.game_over:
            return (afterstate_board, afterstate_next_balls, afterstate_n_next,
                    {'cap_hit': False, 'score_gain': game.score - score_before,
                     'turns': game.turns - anchor['turn_origin']})

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
        return (afterstate_board, afterstate_next_balls, afterstate_n_next,
                {'cap_hit': bool(survived),
                 'score_gain': int(game.score - score_before),
                 'turns': int(game.turns - anchor['turn_origin'])})

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break

        # Get top-K moves (recompute via policy)
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
            result_queue.put({'anchor_id': anchor['id'], 'skipped': 'fewer moves'})
            continue
        top_moves = sorted(priors.items(), key=lambda x: x[1],
                           reverse=True)[:top_k_moves]

        per_move = {}
        for move_action, prior in top_moves:
            outcomes = []
            afterstate = None
            for k in range(k_rollouts):
                ab, an, ann, outcome = get_afterstate_and_outcome(
                    anchor, move_action, rollout_base_seed + k)
                outcomes.append(outcome)
                if afterstate is None and ab is not None:
                    # Snapshot first rollout's afterstate (deterministic given
                    # rollout_seed; same across K rollouts for same move).
                    afterstate = (ab, an, ann)
            if afterstate is None:
                continue
            per_move[int(move_action)] = {
                'prior': float(prior),
                'afterstate': afterstate,  # (board, next_balls, n_next)
                'cap_rate': sum(o['cap_hit'] for o in outcomes) / len(outcomes),
                'mean_score_gain': float(np.mean([o['score_gain']
                                                    for o in outcomes])),
                'mean_turns': float(np.mean([o['turns'] for o in outcomes])),
            }

        result_queue.put({
            'anchor_id': anchor['id'],
            'anchor_board': anchor['board'],
            'anchor_next_balls': anchor['next_balls'],
            'anchor_n_next': anchor['num_next'],
            'source_label': anchor['source_label'],
            'per_move': per_move,
        })

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def filter_pairs(per_move, thresholds):
    """Apply Stage 2 tight filter. Returns list of accepted pair dicts."""
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

            if not (passes_cap or passes_turns or passes_score):
                continue
            # priority: cap > turns > score
            if passes_cap:
                winner_a = d_cap > 0
                margin = abs(d_cap)
                metric = 'cap_rate'
            elif passes_turns:
                winner_a = d_turns > 0
                margin = abs(d_turns)
                metric = 'turns'
            else:
                winner_a = d_score > 0
                margin = abs(d_score)
                metric = 'score'
            win_move = m_a if winner_a else m_b
            lose_move = m_b if winner_a else m_a
            pairs.append({
                'win_move': win_move, 'lose_move': lose_move,
                'margin': float(margin), 'metric': metric,
                'd_cap': float(d_cap), 'd_turns': float(d_turns),
                'd_score': float(d_score),
            })
    return pairs


def next_balls_to_arrays(next_balls):
    npos = np.zeros((3, 2), dtype=np.int8)
    ncol = np.zeros(3, dtype=np.int8)
    for k, item in enumerate(next_balls[:3]):
        # item could be ((r,c), col) tuple or [(r,c), col] list
        pos, col = item[0], item[1]
        npos[k, 0] = pos[0]
        npos[k, 1] = pos[1]
        ncol[k] = col
    return npos, ncol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--stage1', required=True,
                   help='Stage 1 output .pt (e.g., anchors_stage1_5000.pt)')
    p.add_argument('--top-moves', type=int, default=4)
    p.add_argument('--k-rollouts', type=int, default=8)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--rollout-base-seed', type=int, default=200000)
    p.add_argument('--cap-rate-threshold', type=float, default=0.375)
    p.add_argument('--turns-threshold', type=float, default=50)
    p.add_argument('--score-threshold', type=float, default=300)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    print(f"Loading Stage 1 {args.stage1}...", flush=True)
    stage1 = torch.load(args.stage1, weights_only=False)
    all_anchors = stage1['anchors']
    separable_ids = set(stage1['separable_ids'])
    sep_anchors = [a for a in all_anchors if a['id'] in separable_ids]
    print(f"Stage 1: {len(all_anchors)} anchors total, "
          f"{len(sep_anchors)} separable", flush=True)

    # Spin up server + workers
    from alphatrain.inference_server import InferenceServer
    server = InferenceServer(args.model, args.workers, device=args.device,
                             max_batch_per_worker=args.batch_size)
    server.start()

    anchor_queue = MPQueue()
    for a in sep_anchors:
        anchor_queue.put(a)
    for _ in range(args.workers):
        anchor_queue.put(None)
    result_queue = MPQueue()

    workers = []
    for i in range(args.workers):
        proc = Process(target=_stage2_worker,
                       args=(i, anchor_queue, result_queue,
                             server._obs_shm.name, server._pol_shm.name,
                             server._val_shm.name,
                             args.workers, args.batch_size,
                             server.request_queue, server.response_queues[i],
                             args.top_moves, args.k_rollouts, args.horizon,
                             args.rollout_base_seed))
        proc.start()
        workers.append(proc)

    # Collect
    t0 = time.time()
    results = []
    skipped = 0
    for i in range(len(sep_anchors)):
        try:
            r = result_queue.get(timeout=900)
        except Exception:
            print(f"Result timeout at {i}", flush=True)
            break
        if 'skipped' in r:
            skipped += 1
        else:
            results.append(r)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sep_anchors) - i - 1)
            print(f"  [{i+1}/{len(sep_anchors)}] kept={len(results)} "
                  f"skip={skipped} {elapsed:.0f}s ETA {eta:.0f}s", flush=True)

    for p_ in workers:
        p_.join(timeout=10)
    server.shutdown()
    print(f"\nDone rollouts: {len(results)} processed, {skipped} skipped, "
          f"in {time.time()-t0:.0f}s", flush=True)

    # Filter and build tensor
    thresholds = {
        'cap_rate': args.cap_rate_threshold,
        'turns': args.turns_threshold,
        'score': args.score_threshold,
    }

    anchor_boards = []
    anchor_next_pos = []
    anchor_next_col = []
    anchor_n_next = []
    win_boards = []
    win_next_pos = []
    win_next_col = []
    win_n_next = []
    lose_boards = []
    lose_next_pos = []
    lose_next_col = []
    lose_n_next = []
    margins = []
    metrics = []
    source_labels = []

    n_pairs_total = 0
    n_pairs_accepted = 0
    by_metric = {}
    by_source = {}

    for r in results:
        pairs = filter_pairs(r['per_move'], thresholds)
        n_pairs_total += len(r['per_move']) * (len(r['per_move']) - 1) // 2
        n_pairs_accepted += len(pairs)
        if pairs:
            by_source.setdefault(r['source_label'], 0)
            by_source[r['source_label']] += len(pairs)
        for pr in pairs:
            by_metric.setdefault(pr['metric'], []).append(pr['margin'])

            anchor_boards.append(r['anchor_board'])
            an_np, an_nc = next_balls_to_arrays(r['anchor_next_balls'])
            anchor_next_pos.append(an_np)
            anchor_next_col.append(an_nc)
            anchor_n_next.append(r['anchor_n_next'])

            w_b, w_nb, w_nn = r['per_move'][pr['win_move']]['afterstate']
            win_boards.append(w_b)
            w_np, w_nc = next_balls_to_arrays(w_nb)
            win_next_pos.append(w_np)
            win_next_col.append(w_nc)
            win_n_next.append(w_nn)

            l_b, l_nb, l_nn = r['per_move'][pr['lose_move']]['afterstate']
            lose_boards.append(l_b)
            l_np, l_nc = next_balls_to_arrays(l_nb)
            lose_next_pos.append(l_np)
            lose_next_col.append(l_nc)
            lose_n_next.append(l_nn)

            margins.append(pr['margin'])
            metrics.append(pr['metric'])
            source_labels.append(r['source_label'])

    print(f"\nPair filter results:", flush=True)
    print(f"  Total candidate pairs: {n_pairs_total}", flush=True)
    print(f"  Accepted (Stage 2 tight): {n_pairs_accepted} "
          f"({100*n_pairs_accepted/max(n_pairs_total,1):.1f}%)", flush=True)
    print(f"\nBy metric:", flush=True)
    for m, marg in by_metric.items():
        mn = np.array(marg)
        print(f"  {m}: n={len(mn)}  mean={mn.mean():.3f}  "
              f"P50={np.median(mn):.3f}  P90={np.percentile(mn,90):.3f}",
              flush=True)
    print(f"\nBy source:", flush=True)
    for lab, n in by_source.items():
        print(f"  {lab}: {n} pairs", flush=True)

    if n_pairs_accepted == 0:
        print("\nERROR: 0 pairs passed. Lower thresholds or check Stage 2 inputs.",
              flush=True)
        return

    # Convert source_labels to int for tensor friendliness
    label_to_int = {lab: i for i, lab in enumerate(sorted(set(source_labels)))}
    src_ints = np.array([label_to_int[l] for l in source_labels], dtype=np.int8)

    out = {
        'anchor_boards': torch.tensor(np.stack(anchor_boards), dtype=torch.int8),
        'anchor_next_pos': torch.tensor(np.stack(anchor_next_pos), dtype=torch.int8),
        'anchor_next_col': torch.tensor(np.stack(anchor_next_col), dtype=torch.int8),
        'anchor_n_next': torch.tensor(anchor_n_next, dtype=torch.int8),
        'win_boards': torch.tensor(np.stack(win_boards), dtype=torch.int8),
        'win_next_pos': torch.tensor(np.stack(win_next_pos), dtype=torch.int8),
        'win_next_col': torch.tensor(np.stack(win_next_col), dtype=torch.int8),
        'win_n_next': torch.tensor(win_n_next, dtype=torch.int8),
        'lose_boards': torch.tensor(np.stack(lose_boards), dtype=torch.int8),
        'lose_next_pos': torch.tensor(np.stack(lose_next_pos), dtype=torch.int8),
        'lose_next_col': torch.tensor(np.stack(lose_next_col), dtype=torch.int8),
        'lose_n_next': torch.tensor(lose_n_next, dtype=torch.int8),
        'margins': torch.tensor(margins, dtype=torch.float32),
        'metric_names': [str(m) for m in metrics],
        'source_labels': torch.tensor(src_ints, dtype=torch.int8),
        'source_label_map': label_to_int,
        'args': vars(args),
    }
    torch.save(out, args.output)
    print(f"\nSaved {args.output} ({os.path.getsize(args.output)/1e6:.0f} MB)",
          flush=True)


if __name__ == '__main__':
    main()
