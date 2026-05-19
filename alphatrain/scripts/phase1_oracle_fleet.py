"""Phase 1 oracle mining — fleet-vectorized version for CUDA (Colab L4/H100).

Replaces the multiprocessing 16-worker design with a single-process fleet of
M=512 concurrent rollouts advanced in lockstep. Each step:
  1. Build M observations (one per active rollout)
  2. ONE GPU forward pass at bs=M  (vs old bs=16)
  3. CPU: pick legal argmax per game, advance via game.move()
  4. Refill dead slots from the work queue

Expected speedup over M5 MAX 16-worker:
  - Per-step bs jumps 16 → 512 (32×)
  - L4 fp16 is faster than MPS fp16 (~3-5×)
  - Combined: ~5-10× total throughput.

Output format matches phase1_oracle_label.py so downstream tooling works
unchanged. Can be loaded as `phase1_oracle.pt`.

Colab usage:
    !python -m alphatrain.scripts.phase1_oracle_fleet \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --crisis-dir data/crisis_v12 --selfplay-dir data/selfplay_v12 \\
        --num-anchors 10000 --crisis-frac 1.0 --selfplay-frac 0.0 \\
        --top-moves 6 --k-continuations 32 --horizon 300 \\
        --fleet-size 512 \\
        --device cuda \\
        --output alphatrain/data/phase1_oracle_v3.pt
"""

import os
import json
import glob
import time
import argparse
from random import Random
import numpy as np
import torch
import torch.nn.functional as F


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--num-anchors', type=int, default=10000)
    p.add_argument('--crisis-frac', type=float, default=1.0)
    p.add_argument('--selfplay-frac', type=float, default=0.0)
    p.add_argument('--top-moves', type=int, default=6)
    p.add_argument('--k-continuations', type=int, default=32)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--fleet-size', type=int, default=512,
                   help='Number of concurrent rollouts in the fleet. '
                        'Larger = bigger GPU batch but more memory.')
    p.add_argument('--device', default='cuda')
    p.add_argument('--sample-seed', type=int, default=2026)
    p.add_argument('--afterstate-seed-base', type=int, default=9000000)
    p.add_argument('--continuation-seed-base', type=int, default=10000000)
    p.add_argument('--checkpoint-every', type=int, default=500,
                   help='Save partial results every N anchors processed. '
                        '0 to disable.')
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

    device = torch.device(args.device)
    fp16 = (args.device.startswith('cuda'))

    from alphatrain.evaluate import load_model
    from alphatrain.observation import build_observation
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    print(f"Loading {args.model}...", flush=True)
    net, _ = load_model(args.model, device, fp16=fp16, jit_trace=False)
    net.train(False)

    @torch.inference_mode()
    def batched_forward(obs_batch_np):
        """Forward pass on a batch of observations. Returns softmax probs."""
        x = torch.from_numpy(obs_batch_np).to(
            device, dtype=torch.float16 if fp16 else torch.float32)
        logits = net(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        pol = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return pol

    # ── Phase A: per-anchor top-K moves (batched) ──
    print(f"\nPhase A: computing top-{args.top_moves} moves for "
          f"{len(anchors)} anchors...", flush=True)
    t0 = time.time()
    chunk = max(args.fleet_size, 256)
    anchor_topk = {}  # anchor_id -> list of (move, prior) sorted desc
    anchor_skipped = set()

    for i in range(0, len(anchors), chunk):
        sub = anchors[i:i + chunk]
        obs_list = []
        valid_idx = []
        for j, a in enumerate(sub):
            game = ColorLinesGame(seed=a['seed_origin'])
            game.reset(board=a['board'].copy(),
                       next_balls=list(a['next_balls']))
            game.turns = a['turn_origin']
            if game.game_over:
                anchor_skipped.add(a['id'])
                continue
            obs_list.append(_build_obs_for_game(game))
            valid_idx.append(j)
        if not obs_list:
            continue
        obs_batch = np.stack(obs_list)
        pol_batch = batched_forward(obs_batch)
        for k, j in enumerate(valid_idx):
            a = sub[j]
            priors = _get_legal_priors_flat(
                a['board'].astype(np.int8), pol_batch[k], args.top_moves)
            if not priors or len(priors) < 2:
                anchor_skipped.add(a['id'])
                continue
            top = sorted(priors.items(), key=lambda x: -x[1])[:args.top_moves]
            anchor_topk[a['id']] = top
        if (i + chunk) % (chunk * 4) == 0:
            print(f"  topK [{min(i+chunk, len(anchors))}/{len(anchors)}] "
                  f"in {time.time()-t0:.0f}s", flush=True)
    print(f"  Phase A done in {time.time()-t0:.0f}s. "
          f"valid={len(anchor_topk)} skipped={len(anchor_skipped)}",
          flush=True)

    # ── Phase B: realize afterstates per (anchor, move) ──
    print(f"\nPhase B: realizing afterstates...", flush=True)
    t0 = time.time()
    afterstates = {}  # (anchor_id, move_action) -> {board, next_balls, ...}
    for a in anchors:
        if a['id'] not in anchor_topk:
            continue
        anchor_after_seed = (args.afterstate_seed_base
                              + a['id'] * 7919)
        for move, prior in anchor_topk[a['id']]:
            game = ColorLinesGame(seed=anchor_after_seed)
            game.reset(board=a['board'].copy(),
                       next_balls=list(a['next_balls']))
            game.turns = a['turn_origin']
            sr = move // 81 // 9; sc = move // 81 % 9
            tr = move % 81 // 9; tc = move % 81 % 9
            r = game.move((sr, sc), (tr, tc))
            if not r['valid']:
                continue
            afterstates[(a['id'], int(move))] = {
                'board': game.board.copy(),
                'next_balls': list(game.next_balls),
                'turns': game.turns,
                'score_at_after': int(game.score),
                'game_over': game.game_over,
                'prior': float(prior),
            }
    print(f"  Phase B done in {time.time()-t0:.0f}s. "
          f"afterstates={len(afterstates)}", flush=True)

    # ── Phase C: build task queue (per (anchor, move, k)) ──
    print(f"\nPhase C: building task queue...", flush=True)
    tasks = []  # list of dicts
    for (aid, move), af in afterstates.items():
        # Get this anchor's continuation_seed_base (shared across moves)
        cont_base = (args.continuation_seed_base
                      + aid * 7919 * 1000)
        for k in range(args.k_continuations):
            tasks.append({
                'anchor_id': aid,
                'move': move,
                'k': k,
                'afterstate': af,
                'seed': cont_base + k,
            })
    print(f"  total rollouts queued: {len(tasks):,}", flush=True)

    # ── Phase D: fleet-vectorized rollout (JIT batched step) ──
    print(f"\nPhase D: fleet rollout (M={args.fleet_size}, H={args.horizon}) "
          f"with JIT batched step...", flush=True)

    from alphatrain.scripts.fleet_jit import (
        step_fleet_jit, build_obs_fleet_jit)

    # Per-move outcome aggregation
    per_move_outcomes = {}  # (aid, move) -> list of {cap_hit, score_gain, turns}

    # Pre-record dead-on-spawn (game_over at afterstate)
    dead_at_spawn = [t for t in tasks if t['afterstate']['game_over']]
    for t in dead_at_spawn:
        key = (t['anchor_id'], t['move'])
        per_move_outcomes.setdefault(key, []).append({
            'cap_hit': False,
            'score_gain_after': int(t['afterstate']['score_at_after']),
            'turns_after': 0,
        })
    tasks = [t for t in tasks if not t['afterstate']['game_over']]
    queue_len = len(tasks)
    print(f"  dead-on-spawn: {len(dead_at_spawn)}, live tasks: {queue_len}",
          flush=True)

    M = args.fleet_size
    # Per-slot SoA state (only `args.fleet_size` slots; refill from queue)
    s_boards = np.zeros((M, 9, 9), dtype=np.int8)
    s_next_pos = np.zeros((M, 3, 2), dtype=np.int8)
    s_next_col = np.zeros((M, 3), dtype=np.int8)
    s_n_next = np.zeros(M, dtype=np.int8)
    s_scores = np.zeros(M, dtype=np.int32)
    s_turns = np.zeros(M, dtype=np.int32)
    s_game_overs = np.zeros(M, dtype=np.bool_)
    s_completion = np.zeros(M, dtype=np.int8)
    s_rng_states = np.zeros(M, dtype=np.uint64)
    s_target_end = np.zeros(M, dtype=np.int32)
    s_afterstate_turns = np.zeros(M, dtype=np.int32)
    s_afterstate_score = np.zeros(M, dtype=np.int32)
    s_task_idx = np.full(M, -1, dtype=np.int64)
    s_active = np.zeros(M, dtype=np.bool_)

    def load_task_into_slot(slot, task_idx):
        t = tasks[task_idx]
        af = t['afterstate']
        s_boards[slot] = af['board']
        # Pack next_balls into slot arrays
        for k in range(3):
            if k < len(af['next_balls']):
                pos, col = af['next_balls'][k]
                s_next_pos[slot, k, 0] = pos[0]
                s_next_pos[slot, k, 1] = pos[1]
                s_next_col[slot, k] = col
            else:
                s_next_pos[slot, k, 0] = 0
                s_next_pos[slot, k, 1] = 0
                s_next_col[slot, k] = 0
        s_n_next[slot] = min(len(af['next_balls']), 3)
        s_scores[slot] = 0
        s_turns[slot] = af['turns']
        s_game_overs[slot] = False
        s_completion[slot] = 0
        s_rng_states[slot] = np.uint64(t['seed'])
        s_target_end[slot] = af['turns'] + args.horizon
        s_afterstate_turns[slot] = af['turns']
        s_afterstate_score[slot] = af['score_at_after']
        s_task_idx[slot] = task_idx
        s_active[slot] = True

    queue_idx = 0
    # Initial fill
    for slot in range(M):
        if queue_idx < queue_len:
            load_task_into_slot(slot, queue_idx)
            queue_idx += 1

    obs_buf_np = np.empty((M, 18, 9, 9), dtype=np.float32)
    t0 = time.time()
    n_steps = 0
    n_completed = 0
    last_log_t = t0

    while s_active.any():
        # Compact: which slots are active?
        active_idx = np.where(s_active)[0]
        n_active = len(active_idx)

        # Build observations for ACTIVE slots only (still bs=N for GPU forward)
        # We pass the full M-sized obs_buf but only fill active slots, then
        # forward only those rows for GPU efficiency.
        active_boards = s_boards[active_idx]
        active_next_pos = s_next_pos[active_idx]
        active_next_col = s_next_col[active_idx]
        active_n_next = s_n_next[active_idx]
        obs_active = np.empty((n_active, 18, 9, 9), dtype=np.float32)
        build_obs_fleet_jit(active_boards, active_next_pos,
                             active_next_col, active_n_next, obs_active)

        # GPU forward
        pol_active = batched_forward(obs_active)  # (n_active, 6561)

        # Scatter pol back into full M-sized array for step_fleet_jit
        # Actually we can call step_fleet_jit on active subset directly.
        # Use a contiguous active view; step_fleet_jit mutates active arrays;
        # then write back.
        active_scores = s_scores[active_idx].copy()
        active_turns = s_turns[active_idx].copy()
        active_game_overs = s_game_overs[active_idx].copy()
        active_completion = np.zeros(n_active, dtype=np.int8)
        active_rng = s_rng_states[active_idx].copy()

        step_fleet_jit(active_boards, active_next_pos, active_next_col,
                        active_n_next, active_scores, active_turns,
                        active_game_overs, active_completion,
                        pol_active, active_rng)

        # Write active state back to slots
        s_boards[active_idx] = active_boards
        s_next_pos[active_idx] = active_next_pos
        s_next_col[active_idx] = active_next_col
        s_n_next[active_idx] = active_n_next
        s_scores[active_idx] = active_scores
        s_turns[active_idx] = active_turns
        s_game_overs[active_idx] = active_game_overs
        s_rng_states[active_idx] = active_rng

        # Determine which slots completed (died or capped)
        for ai, slot in enumerate(active_idx):
            died = (active_completion[ai] == 1) or s_game_overs[slot]
            capped = (not died) and (s_turns[slot] >= s_target_end[slot])
            if died or capped:
                t = tasks[s_task_idx[slot]]
                outcome = {
                    'cap_hit': bool(capped),
                    'score_gain_after': int(s_afterstate_score[slot]
                                             + s_scores[slot]),
                    'turns_after': int(s_turns[slot]
                                        - s_afterstate_turns[slot]),
                }
                key = (t['anchor_id'], t['move'])
                per_move_outcomes.setdefault(key, []).append(outcome)
                n_completed += 1
                # Refill the slot from queue if available
                if queue_idx < queue_len:
                    load_task_into_slot(slot, queue_idx)
                    queue_idx += 1
                else:
                    s_active[slot] = False

        n_steps += 1
        now = time.time()
        if now - last_log_t > 15.0:
            elapsed = now - t0
            rate_rollouts = n_completed / max(elapsed, 1e-3)
            queued_remaining = (queue_len - queue_idx) + int(s_active.sum())
            eta = queued_remaining / max(rate_rollouts, 1e-3)
            print(f"  step {n_steps} | active={n_active} | "
                  f"completed={n_completed}/{queue_len} | "
                  f"queued={queued_remaining} | "
                  f"rate={rate_rollouts:.1f} rollouts/sec | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)
            last_log_t = now

    print(f"\n  Phase D done in {time.time()-t0:.0f}s. "
          f"completed={n_completed} rollouts in {n_steps} steps", flush=True)

    # ── Phase E: aggregate per-anchor results ──
    print(f"\nPhase E: aggregating results...", flush=True)
    results = []
    for a in anchors:
        if a['id'] not in anchor_topk:
            continue
        per_move = {}
        for move, prior in anchor_topk[a['id']]:
            key = (a['id'], int(move))
            outcomes = per_move_outcomes.get(key, [])
            if not outcomes:
                continue
            af = afterstates.get(key)
            cap_rate = sum(o['cap_hit'] for o in outcomes) / len(outcomes)
            mean_turns = float(np.mean([o['turns_after'] for o in outcomes]))
            mean_score = float(np.mean([o['score_gain_after']
                                          for o in outcomes]))
            std_turns = float(np.std([o['turns_after'] for o in outcomes]))
            std_score = float(np.std([o['score_gain_after']
                                        for o in outcomes]))
            per_move[int(move)] = {
                'prior': af['prior'],
                'rank': len(per_move) + 1,
                'afterstate_board': af['board'],
                'afterstate_next_balls': af['next_balls'],
                'afterstate_game_over': af['game_over'],
                'afterstate_score_at_after': af['score_at_after'],
                'cap_rate': cap_rate,
                'mean_turns': mean_turns,
                'mean_score': mean_score,
                'std_turns': std_turns,
                'std_score': std_score,
                'outcomes': outcomes,
            }
        if not per_move:
            continue
        results.append({
            'anchor_id': a['id'],
            'anchor_board': a['board'],
            'anchor_next_balls': a['next_balls'],
            'anchor_n_next': a['num_next'],
            'turn_origin': a['turn_origin'],
            'source_label': a['source_label'],
            'source_file': a['source_file'],
            'per_move': per_move,
        })

    print(f"  results: {len(results)} anchors with per-move data",
          flush=True)

    # ── Save ──
    out = {
        'args': vars(args),
        'results': results,
    }
    torch.save(out, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.0f} MB)", flush=True)

    # Quick audit
    print(f"\n=== Quick audit ===", flush=True)
    n_top1_loses = 0
    margins_cap = []
    margins_turns = []
    for r in results:
        pm = r['per_move']
        if len(pm) < 2:
            continue
        sorted_mv = sorted(pm.items(), key=lambda kv: kv[1]['rank'])
        policy_top1 = sorted_mv[0][1]
        best_cap_mv = max(pm.values(), key=lambda v: v['cap_rate'])
        if best_cap_mv['cap_rate'] > policy_top1['cap_rate']:
            n_top1_loses += 1
            margins_cap.append(best_cap_mv['cap_rate']
                                - policy_top1['cap_rate'])
            margins_turns.append(best_cap_mv['mean_turns']
                                  - policy_top1['mean_turns'])
    n_eligible = sum(1 for r in results if len(r['per_move']) >= 2)
    if n_eligible:
        print(f"P(oracle-best != policy_top_1): "
              f"{n_top1_loses}/{n_eligible} = "
              f"{100*n_top1_loses/n_eligible:.1f}%", flush=True)
        if margins_cap:
            print(f"  mean Δcap_rate when oracle wins: "
                  f"{np.mean(margins_cap):.3f}  "
                  f"median={np.median(margins_cap):.3f}", flush=True)
            print(f"  mean Δmean_turns when oracle wins: "
                  f"{np.mean(margins_turns):.1f}", flush=True)


if __name__ == '__main__':
    main()
