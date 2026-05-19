"""Pillar 3a-v3 pilot: clean pairwise with stable-label verification.

Fixes ChatGPT's 4 bugs in build_pairwise_dataset.py / train_ranking_head.py:

  1. Input/label mismatch: each move now has ONE deterministic realized
     afterstate; K independent continuations from THAT afterstate produce the
     label. The head sees the same afterstate the label was computed on.
  2. Selection bias: split K=16 continuations into halves A/B; keep only
     pairs where both halves AGREE on winner under tight threshold.
  3. Margin scale: deferred to training (use --unweighted in trainer).
  4. Terminal-V calibration: deferred — matters for MCTS A/B, not for
     "is the signal there at all".

Usage:
    python -m alphatrain.scripts.pilot_clean_pairwise \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --stage1 alphatrain/data/anchors_stage1_20000.pt \\
        --num-anchors 200 --top-moves 4 --k-continuations 16 --horizon 200 \\
        --workers 16 --batch-size 8 \\
        --output alphatrain/data/pairwise_pilot.pt
"""

import os
import time
import argparse
from random import Random
import numpy as np
import torch
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def _pilot_worker(slot_id, anchor_queue, result_queue,
                  obs_shm_name, pol_shm_name, val_shm_name,
                  num_workers, max_batch,
                  request_queue, response_queue,
                  top_k_moves, k_continuations, horizon,
                  afterstate_seed_base, continuation_seed_base):
    """Per-worker: realize ONE afterstate per move, then K continuations from it."""
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

    def realize_afterstate(anchor, move_action, afterstate_seed):
        """Apply move_action with deterministic spawn-RNG. Return afterstate
        (board, next_balls, turns, score_at_after, game_over) or None if invalid.
        """
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
        """K=1 continuation from a fixed afterstate. Returns outcome dict.

        score_gain_after includes the first move's line-clear score
        (afterstate['score_at_after']) plus all subsequent gains, so the
        score metric reflects total move quality, not just post-afterstate.
        """
        if afterstate['game_over']:
            return {'cap_hit': False,
                    'score_gain_after': int(afterstate['score_at_after']),
                    'turns_after': 0}
        game = ColorLinesGame(seed=continuation_seed)
        game.reset(board=afterstate['board'].copy(),
                   next_balls=list(afterstate['next_balls']))
        game.turns = afterstate['turns']
        score_start = game.score  # after reset, may be 0
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
            'score_gain_after': int(game.score - score_start),
            'turns_after': int(game.turns - afterstate['turns']),
        }

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break

        # Top-K policy moves at anchor
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

        per_move = {}
        # Common-RNG seeds shared across sibling moves under this anchor so
        # outcome differences reflect move choice, not seed luck. (Per ChatGPT
        # review 2026-05-15: prior version mixed move_action into the seed,
        # giving each sibling its own RNG stream — added avoidable variance.)
        anchor_after_seed = afterstate_seed_base + anchor['id'] * 7919
        anchor_cont_base = continuation_seed_base + anchor['id'] * 7919 * 1000
        for move_action, prior in top_moves:
            # ONE realized afterstate per move (anchor-shared seed).
            af = realize_afterstate(anchor, move_action, anchor_after_seed)
            if af is None:
                continue

            # K continuations from the realized afterstate — seeds SHARED
            # across sibling moves (common-RNG variance reduction).
            outcomes = []
            for k in range(k_continuations):
                cs = anchor_cont_base + k
                outcomes.append(continue_from_afterstate(af, cs))

            per_move[int(move_action)] = {
                'prior': float(prior),
                'afterstate': af,
                'outcomes': outcomes,
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


def half_stats(outcomes_half):
    """Aggregate cap_rate / mean_turns / mean_score from a half of outcomes."""
    n = len(outcomes_half)
    if n == 0:
        return {'cap_rate': 0.0, 'mean_turns': 0.0, 'mean_score': 0.0}
    return {
        'cap_rate': sum(o['cap_hit'] for o in outcomes_half) / n,
        'mean_turns': float(np.mean([o['turns_after'] for o in outcomes_half])),
        'mean_score': float(np.mean([o['score_gain_after'] for o in outcomes_half])),
    }


def decide_pair(stats_a, stats_b, thresholds):
    """Apply tight threshold to (move_a stats, move_b stats). Returns
    (winner_a_is_winner: bool|None, metric: str, margin: float). None=no pass.
    """
    d_cap = stats_a['cap_rate'] - stats_b['cap_rate']
    d_turns = stats_a['mean_turns'] - stats_b['mean_turns']
    d_score = stats_a['mean_score'] - stats_b['mean_score']
    if abs(d_cap) >= thresholds['cap_rate']:
        return (d_cap > 0, 'cap_rate', abs(d_cap))
    if abs(d_turns) >= thresholds['turns']:
        return (d_turns > 0, 'turns', abs(d_turns))
    if abs(d_score) >= thresholds['score']:
        return (d_score > 0, 'score', abs(d_score))
    return (None, None, 0.0)


def next_balls_to_arrays(next_balls):
    npos = np.zeros((3, 2), dtype=np.int8)
    ncol = np.zeros(3, dtype=np.int8)
    for k, item in enumerate(next_balls[:3]):
        pos, col = item[0], item[1]
        npos[k, 0] = pos[0]
        npos[k, 1] = pos[1]
        ncol[k] = col
    return npos, ncol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--stage1', required=True,
                   help='Stage 1 output .pt — we sample separable anchors from it')
    p.add_argument('--num-anchors', type=int, default=200)
    p.add_argument('--top-moves', type=int, default=4)
    p.add_argument('--k-continuations', type=int, default=16,
                   help='Total continuations per realized afterstate. '
                        'Split into halves A/B for relabel verification.')
    p.add_argument('--horizon', type=int, default=200)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--afterstate-seed-base', type=int, default=900000)
    p.add_argument('--continuation-seed-base', type=int, default=1500000)
    p.add_argument('--cap-rate-threshold', type=float, default=0.375)
    p.add_argument('--turns-threshold', type=float, default=40)
    p.add_argument('--score-threshold', type=float, default=300)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    if args.k_continuations % 2 != 0:
        raise SystemExit("--k-continuations must be even (split into halves)")

    print(f"Loading Stage 1 {args.stage1}...", flush=True)
    stage1 = torch.load(args.stage1, weights_only=False)
    all_anchors = stage1['anchors']
    separable_ids = set(stage1['separable_ids'])
    sep_anchors = [a for a in all_anchors if a['id'] in separable_ids]
    print(f"  separable in Stage 1: {len(sep_anchors)}", flush=True)

    rng = Random(args.sample_seed)
    if len(sep_anchors) > args.num_anchors:
        sep_anchors = rng.sample(sep_anchors, args.num_anchors)
    print(f"  pilot anchors: {len(sep_anchors)}", flush=True)

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
        proc = Process(target=_pilot_worker,
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
    for i in range(len(sep_anchors)):
        try:
            r = result_queue.get(timeout=600)
        except Exception:
            print(f"Result timeout at {i}", flush=True)
            break
        if 'skipped' in r:
            skipped += 1
        else:
            results.append(r)
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(sep_anchors) - i - 1)
            print(f"  [{i+1}/{len(sep_anchors)}] kept={len(results)} "
                  f"skip={skipped} {elapsed:.0f}s ETA {eta:.0f}s", flush=True)

    for w in workers:
        w.join(timeout=10)
    server.shutdown()
    print(f"\nRollouts done: kept={len(results)} skipped={skipped} "
          f"in {time.time()-t0:.0f}s", flush=True)

    # Build stable-pair dataset
    thresholds = {
        'cap_rate': args.cap_rate_threshold,
        'turns': args.turns_threshold,
        'score': args.score_threshold,
    }
    half = args.k_continuations // 2

    n_candidate_pairs = 0
    n_pass_full = 0
    n_pass_a = 0
    n_pass_b = 0
    n_pass_both = 0
    n_agree_winner_given_both = 0
    n_stable = 0

    by_metric_full = {}
    by_metric_stable = {}

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
    margins_full = []
    metrics_stable = []
    source_labels = []

    for r in results:
        moves = list(r['per_move'].keys())
        for i in range(len(moves)):
            for j in range(i + 1, len(moves)):
                m_a, m_b = moves[i], moves[j]
                a_outs = r['per_move'][m_a]['outcomes']
                b_outs = r['per_move'][m_b]['outcomes']
                if len(a_outs) < args.k_continuations or len(b_outs) < args.k_continuations:
                    continue
                n_candidate_pairs += 1

                # Full label
                stats_a_full = half_stats(a_outs)
                stats_b_full = half_stats(b_outs)
                win_full, metric_full, margin_full = decide_pair(
                    stats_a_full, stats_b_full, thresholds)
                full_pass = (win_full is not None)

                # Half A and Half B labels (independent halves)
                stats_a_half_a = half_stats(a_outs[:half])
                stats_b_half_a = half_stats(b_outs[:half])
                stats_a_half_b = half_stats(a_outs[half:])
                stats_b_half_b = half_stats(b_outs[half:])
                win_ha, _, _ = decide_pair(stats_a_half_a, stats_b_half_a, thresholds)
                win_hb, _, _ = decide_pair(stats_a_half_b, stats_b_half_b, thresholds)

                if full_pass:
                    n_pass_full += 1
                    by_metric_full[metric_full] = by_metric_full.get(metric_full, 0) + 1
                if win_ha is not None:
                    n_pass_a += 1
                if win_hb is not None:
                    n_pass_b += 1
                if win_ha is not None and win_hb is not None:
                    n_pass_both += 1
                    if win_ha == win_hb:
                        n_agree_winner_given_both += 1

                # Stable definition: full pass AND both halves agree on winner
                # AND both halves agree with full label.
                if not full_pass:
                    continue
                if win_ha is None or win_hb is None:
                    continue
                if win_ha != win_full or win_hb != win_full:
                    continue

                n_stable += 1
                by_metric_stable[metric_full] = by_metric_stable.get(metric_full, 0) + 1

                win_m = m_a if win_full else m_b
                lose_m = m_b if win_full else m_a
                w_af = r['per_move'][win_m]['afterstate']
                l_af = r['per_move'][lose_m]['afterstate']

                anchor_boards.append(r['anchor_board'])
                an_np, an_nc = next_balls_to_arrays(r['anchor_next_balls'])
                anchor_next_pos.append(an_np)
                anchor_next_col.append(an_nc)
                anchor_n_next.append(r['anchor_n_next'])

                win_boards.append(w_af['board'])
                w_np, w_nc = next_balls_to_arrays(w_af['next_balls'])
                win_next_pos.append(w_np)
                win_next_col.append(w_nc)
                win_n_next.append(len(w_af['next_balls']))

                lose_boards.append(l_af['board'])
                l_np, l_nc = next_balls_to_arrays(l_af['next_balls'])
                lose_next_pos.append(l_np)
                lose_next_col.append(l_nc)
                lose_n_next.append(len(l_af['next_balls']))

                margins_full.append(margin_full)
                metrics_stable.append(metric_full)
                source_labels.append(r['source_label'])

    print(f"\n=== STABLE-PAIR REPORT ===", flush=True)
    print(f"Candidate pairs (any anchor, any pair of top-K moves): "
          f"{n_candidate_pairs}", flush=True)
    print(f"Pass FULL threshold: {n_pass_full} "
          f"({100*n_pass_full/max(n_candidate_pairs,1):.1f}%)", flush=True)
    print(f"Pass HALF-A threshold: {n_pass_a}", flush=True)
    print(f"Pass HALF-B threshold: {n_pass_b}", flush=True)
    print(f"Pass BOTH halves: {n_pass_both}", flush=True)
    if n_pass_both > 0:
        agree_rate = 100 * n_agree_winner_given_both / n_pass_both
        print(f"  → winner agreement | both pass: {n_agree_winner_given_both}"
              f"/{n_pass_both} ({agree_rate:.1f}%)", flush=True)
        print(f"  (chance is 50%; >75% = signal, <60% = mostly noise)",
              flush=True)
    print(f"STABLE (full + both halves agree with full): {n_stable}",
          flush=True)
    print(f"\nFull-label metric mix: {by_metric_full}", flush=True)
    print(f"Stable-label metric mix: {by_metric_stable}", flush=True)

    if n_stable == 0:
        print("\nERROR: 0 stable pairs. Labels are too noisy under tight "
              "thresholds — confirms ChatGPT/Gemini diagnosis.", flush=True)
        # still save metadata so we know what we ran
        torch.save({'args': vars(args),
                    'results_meta': {
                        'n_results': len(results),
                        'n_candidate_pairs': n_candidate_pairs,
                        'n_pass_full': n_pass_full,
                        'n_pass_a': n_pass_a, 'n_pass_b': n_pass_b,
                        'n_pass_both': n_pass_both,
                        'n_agree_winner_given_both': n_agree_winner_given_both,
                        'n_stable': 0,
                    }}, args.output)
        return

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
        'margins': torch.tensor(margins_full, dtype=torch.float32),
        'metric_names': [str(m) for m in metrics_stable],
        'source_labels': torch.tensor(src_ints, dtype=torch.int8),
        'source_label_map': label_to_int,
        'args': vars(args),
        'audit': {
            'n_candidate_pairs': n_candidate_pairs,
            'n_pass_full': n_pass_full,
            'n_pass_a': n_pass_a,
            'n_pass_b': n_pass_b,
            'n_pass_both': n_pass_both,
            'n_agree_winner_given_both': n_agree_winner_given_both,
            'n_stable': n_stable,
            'by_metric_full': by_metric_full,
            'by_metric_stable': by_metric_stable,
        },
    }
    torch.save(out, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
