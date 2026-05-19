"""Phase 1 oracle mining — FULLY GPU-vectorized.

Every step keeps state on GPU as torch tensors:
  - boards, next_pos, next_col, n_next, scores, turns, game_overs
  - per-slot: afterstate_turns, afterstate_score, target_end, task_idx

Per step the only Python work is:
  1. Generate random tensors (rand_score, rand_color) — torch.rand on GPU
  2. Call gpu_step → mutates state in place
  3. Identify completed slots (small mask op + .nonzero())
  4. For each completed slot: record outcome to CPU dict, refill from queue

No Python loop over fleet on the hot path. Should saturate the GPU.

Output format matches phase1_oracle_label.py.

Usage:
    python -m alphatrain.scripts.phase1_oracle_fleet_gpu \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --crisis-dir data/crisis_v12 --selfplay-dir data/selfplay_v12 \\
        --num-anchors 10000 --crisis-frac 1.0 --selfplay-frac 0.0 \\
        --top-moves 6 --k-continuations 32 --horizon 300 \\
        --fleet-size 1024 \\
        --device cuda \\
        --output alphatrain/data/phase1_oracle_gpu.pt
"""

import os
import json
import glob
import time
import argparse
from random import Random
import numpy as np
import torch


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
    p.add_argument('--fleet-size', type=int, default=1024)
    p.add_argument('--compile', action='store_true',
                   help='Wrap pre/post in torch.compile(reduce-overhead) for '
                        'cudagraph capture.')
    p.add_argument('--fixed-horizon', action='store_true', default=True,
                   help='Batched mode: load M tasks, run H steps with NO per-'
                        'step CPU sync, collect all M outcomes at end, refill '
                        'M tasks, repeat. Eliminates per-step sync from '
                        'completion detection. (Default: on.)')
    p.add_argument('--continuous-refill', action='store_false',
                   dest='fixed_horizon',
                   help='Disable fixed-horizon; use per-step refill (legacy).')
    p.add_argument('--device', default='cuda')
    p.add_argument('--sample-seed', type=int, default=2027)
    p.add_argument('--afterstate-seed-base', type=int, default=9000000)
    p.add_argument('--continuation-seed-base', type=int, default=10000000)
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
    from alphatrain.scripts.fleet_gpu import (
        gpu_step, gpu_step_functional, gpu_build_obs, gpu_legal_argmax,
        gpu_label_components)

    use_functional = args.compile  # cudagraph-compatible path

    # ONE compiled function for the entire per-step body: pre + forward +
    # post. This puts the network call INSIDE the compile boundary, so
    # intermediate tensors (labels, obs, logits, pol) never cross cudagraph
    # boundaries. cudagraph capture can then replay the whole step with no
    # per-op kernel launch overhead.
    def _full_step(boards, next_pos, next_col, n_next, scores, turns,
                    game_overs, rand_score, rand_color):
        labels = gpu_label_components(boards)
        obs = gpu_build_obs(boards, next_pos, next_col, n_next, labels=labels)
        if fp16:
            obs = obs.half()
        logits = net(obs)
        if isinstance(logits, tuple):
            logits = logits[0]
        pol = torch.softmax(logits.float(), dim=-1)
        return gpu_step_functional(boards, next_pos, next_col, n_next,
                                     scores, turns, game_overs,
                                     pol, rand_score, rand_color,
                                     labels=labels)

    if use_functional:
        full_step_fn = torch.compile(_full_step,
                                       mode='reduce-overhead',
                                       dynamic=False)
        print("  compiling full_step (label+obs+forward+step) as ONE "
              "cudagraph (first call will be slow ~30-60s)", flush=True)
    else:
        full_step_fn = _full_step
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    print(f"Loading {args.model}...", flush=True)
    net, _ = load_model(args.model, device, fp16=fp16, jit_trace=False)
    net.train(False)

    # ── Phase A: per-anchor top-K moves (batched) ──
    print(f"\nPhase A: top-{args.top_moves} moves for {len(anchors)} anchors...",
          flush=True)
    t0 = time.time()
    chunk = max(args.fleet_size, 512)
    anchor_topk = {}
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
        x = torch.from_numpy(obs_batch).to(
            device, dtype=torch.float16 if fp16 else torch.float32)
        with torch.inference_mode():
            logits = net(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            pol_batch = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        for k, j in enumerate(valid_idx):
            a = sub[j]
            priors = _get_legal_priors_flat(
                a['board'].astype(np.int8), pol_batch[k], args.top_moves)
            if not priors or len(priors) < 2:
                anchor_skipped.add(a['id'])
                continue
            top = sorted(priors.items(), key=lambda x: -x[1])[:args.top_moves]
            anchor_topk[a['id']] = top
    print(f"  Phase A done in {time.time()-t0:.0f}s. "
          f"valid={len(anchor_topk)} skipped={len(anchor_skipped)}",
          flush=True)

    # ── Phase B: realize afterstates ──
    print(f"\nPhase B: realizing afterstates...", flush=True)
    t0 = time.time()
    afterstates = {}
    for a in anchors:
        if a['id'] not in anchor_topk:
            continue
        anchor_after_seed = args.afterstate_seed_base + a['id'] * 7919
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

    # ── Phase C: build task queue ──
    print(f"\nPhase C: building task queue...", flush=True)
    tasks = []
    for (aid, move), af in afterstates.items():
        cont_base = args.continuation_seed_base + aid * 7919 * 1000
        for k in range(args.k_continuations):
            tasks.append({
                'anchor_id': aid,
                'move': move,
                'k': k,
                'afterstate': af,
                'seed': cont_base + k,
            })
    print(f"  total rollouts queued: {len(tasks):,}", flush=True)

    # Pre-record dead-on-spawn
    per_move_outcomes = {}
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

    # ── Phase D: GPU-native fleet rollout ──
    M = args.fleet_size
    print(f"\nPhase D: GPU fleet rollout (M={M}, H={args.horizon})...",
          flush=True)

    # All fleet state on GPU
    s_boards = torch.zeros(M, 9, 9, dtype=torch.long, device=device)
    s_next_pos = torch.zeros(M, 3, 2, dtype=torch.int8, device=device)
    s_next_col = torch.zeros(M, 3, dtype=torch.int8, device=device)
    s_n_next = torch.zeros(M, dtype=torch.int8, device=device)
    s_scores = torch.zeros(M, dtype=torch.long, device=device)
    s_turns = torch.zeros(M, dtype=torch.long, device=device)
    s_game_overs = torch.ones(M, dtype=torch.bool, device=device)  # start all dead
    s_target_end = torch.zeros(M, dtype=torch.long, device=device)
    s_afterstate_turns = torch.zeros(M, dtype=torch.long, device=device)
    s_afterstate_score = torch.zeros(M, dtype=torch.long, device=device)
    s_task_idx = np.full(M, -1, dtype=np.int64)  # CPU array — small, infrequent access

    def load_task_into_slot(slot, task_idx):
        t = tasks[task_idx]
        af = t['afterstate']
        s_boards[slot] = torch.from_numpy(
            af['board'].astype(np.int64)).to(device)
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
        s_target_end[slot] = af['turns'] + args.horizon
        s_afterstate_turns[slot] = af['turns']
        s_afterstate_score[slot] = af['score_at_after']
        s_task_idx[slot] = task_idx

    queue_idx = 0
    H = args.horizon
    t0 = time.time()
    n_completed = 0
    last_log_t = t0
    n_batches = 0

    def step_once():
        """One step of the fleet, NO per-step CPU sync.

        Under --compile this calls a SINGLE cudagraph-captured function
        containing label+obs+forward+step. All intermediates (labels, obs,
        logits) stay inside the graph — no boundary crossings.
        """
        nonlocal s_boards, s_next_pos, s_next_col, s_n_next
        nonlocal s_scores, s_turns, s_game_overs
        rand_score = torch.rand(M, 81, device=device)
        rand_color = torch.randint(1, 8, (M, 3), device=device,
                                     dtype=torch.long)

        if use_functional:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.inference_mode():
                (b, np_, nc, nn_, sc, tu, go, _) = full_step_fn(
                    s_boards, s_next_pos, s_next_col, s_n_next,
                    s_scores, s_turns, s_game_overs,
                    rand_score, rand_color)
            # Clone out of the cudagraph + inference-mode pools so we can
            # in-place modify between batches (load_task_into_slot,
            # s_game_overs[:] = True, etc).
            s_boards = b.clone()
            s_next_pos = np_.clone()
            s_next_col = nc.clone()
            s_n_next = nn_.clone()
            s_scores = sc.clone()
            s_turns = tu.clone()
            s_game_overs = go.clone()
        else:
            # Eager path: compute labels + obs + forward eagerly, mutate state
            with torch.inference_mode():
                labels = gpu_label_components(s_boards)
                obs = gpu_build_obs(s_boards, s_next_pos, s_next_col,
                                      s_n_next, labels=labels)
                if fp16:
                    obs = obs.half()
                logits = net(obs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pol = torch.softmax(logits.float(), dim=-1)
            gpu_step(s_boards, s_next_pos, s_next_col, s_n_next,
                     s_scores, s_turns, s_game_overs,
                     pol, rand_score, rand_color, labels=labels)

    while queue_idx < queue_len or not s_game_overs.all():
        # ── BATCH START: load M tasks, run H steps, collect outcomes ──
        batch_start_time = time.time()
        if args.fixed_horizon:
            # Reset fleet: clear game_overs, load M tasks (or skip slots
            # where queue exhausted — they'll game_over immediately and be
            # ignored in collection).
            s_game_overs[:] = True  # all dead initially → step is no-op
            n_loaded_this_batch = 0
            for slot in range(M):
                if queue_idx < queue_len:
                    load_task_into_slot(slot, queue_idx)
                    queue_idx += 1
                    n_loaded_this_batch += 1
            if n_loaded_this_batch == 0:
                break

            # Run H steps with NO per-step sync. Dead lanes (game_over=True)
            # are skipped inside gpu_step internally.
            for step in range(H):
                step_once()

            # ── Collect outcomes for this batch (one big sync) ──
            t_collect = time.time()
            scores_cpu = s_scores[:n_loaded_this_batch].cpu().tolist()
            turns_cpu = s_turns[:n_loaded_this_batch].cpu().tolist()
            ats_cpu = s_afterstate_turns[:n_loaded_this_batch].cpu().tolist()
            ass_cpu = s_afterstate_score[:n_loaded_this_batch].cpu().tolist()
            te_cpu = s_target_end[:n_loaded_this_batch].cpu().tolist()
            go_cpu = s_game_overs[:n_loaded_this_batch].cpu().tolist()

            for slot in range(n_loaded_this_batch):
                t = tasks[s_task_idx[slot]]
                # Game over with turns < target_end means it died
                survived = (turns_cpu[slot] >= te_cpu[slot])
                outcome = {
                    'cap_hit': bool(survived),
                    'score_gain_after': int(ass_cpu[slot] + scores_cpu[slot]),
                    'turns_after': int(turns_cpu[slot] - ats_cpu[slot]),
                }
                key = (t['anchor_id'], t['move'])
                per_move_outcomes.setdefault(key, []).append(outcome)
                n_completed += 1

            n_batches += 1
            elapsed = time.time() - t0
            rate = n_completed / max(elapsed, 1e-3)
            queued_remaining = queue_len - queue_idx
            eta = queued_remaining / max(rate, 1e-3)
            print(f"  batch {n_batches} | M_loaded={n_loaded_this_batch} | "
                  f"completed={n_completed}/{queue_len} | "
                  f"queued={queued_remaining} | "
                  f"rate={rate:.1f} rollouts/sec | batch_time="
                  f"{time.time()-batch_start_time:.1f}s | ETA={eta:.0f}s",
                  flush=True)
            continue

        # ── LEGACY continuous-refill mode (kept for diagnostic) ──
        # Initial fill (once)
        if queue_idx == 0:
            for slot in range(M):
                if queue_idx < queue_len:
                    load_task_into_slot(slot, queue_idx)
                    queue_idx += 1

        active_mask = ~s_game_overs
        if not active_mask.any():
            break
        step_once()
        # Cap detection + completion + refill (per-step sync — the bottleneck)
        capped = (s_turns >= s_target_end) & active_mask
        s_game_overs |= capped
        completed = capped | (s_game_overs & active_mask)
        if completed.any():
            comp_idx = completed.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            scores_cpu = s_scores[completed].cpu().tolist()
            turns_cpu = s_turns[completed].cpu().tolist()
            ats = s_afterstate_turns[completed].cpu().tolist()
            ass = s_afterstate_score[completed].cpu().tolist()
            te = s_target_end[completed].cpu().tolist()
            for i, slot in enumerate(comp_idx):
                t = tasks[s_task_idx[slot]]
                survived = (turns_cpu[i] >= te[i])
                outcome = {
                    'cap_hit': bool(survived),
                    'score_gain_after': int(ass[i] + scores_cpu[i]),
                    'turns_after': int(turns_cpu[i] - ats[i]),
                }
                key = (t['anchor_id'], t['move'])
                per_move_outcomes.setdefault(key, []).append(outcome)
                n_completed += 1
                if queue_idx < queue_len:
                    load_task_into_slot(slot, queue_idx)
                    queue_idx += 1
        now = time.time()
        if now - last_log_t > 15.0:
            elapsed = now - t0
            rate = n_completed / max(elapsed, 1e-3)
            queued_remaining = (queue_len - queue_idx) + int((~s_game_overs).sum().item())
            eta = queued_remaining / max(rate, 1e-3)
            print(f"  legacy step | completed={n_completed}/{queue_len} | "
                  f"queued={queued_remaining} | rate={rate:.1f} rollouts/sec | "
                  f"elapsed={elapsed:.0f}s | ETA={eta:.0f}s", flush=True)
            last_log_t = now

    print(f"\n  Phase D done in {time.time()-t0:.0f}s. "
          f"completed={n_completed} rollouts in {n_batches} batches",
          flush=True)

    # ── Phase E: aggregate ──
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
            mean_turns = float(np.mean([o['turns_after']
                                         for o in outcomes]))
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
    print(f"  results: {len(results)} anchors", flush=True)

    out = {'args': vars(args), 'results': results}
    torch.save(out, args.output)
    print(f"\nSaved {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)

    # Quick audit
    n_top1_loses = 0
    margins_cap = []
    margins_turns = []
    for r in results:
        pm = r['per_move']
        if len(pm) < 2:
            continue
        sorted_mv = sorted(pm.items(), key=lambda kv: kv[1]['rank'])
        policy_top1 = sorted_mv[0][1]
        best_cap = max(pm.values(), key=lambda v: v['cap_rate'])
        if best_cap['cap_rate'] > policy_top1['cap_rate']:
            n_top1_loses += 1
            margins_cap.append(best_cap['cap_rate'] - policy_top1['cap_rate'])
            margins_turns.append(best_cap['mean_turns']
                                  - policy_top1['mean_turns'])
    n_elig = sum(1 for r in results if len(r['per_move']) >= 2)
    if n_elig:
        print(f"\nP(oracle-best != policy_top_1): "
              f"{n_top1_loses}/{n_elig} = {100*n_top1_loses/n_elig:.1f}%",
              flush=True)
        if margins_cap:
            print(f"  mean Δcap_rate: {np.mean(margins_cap):.3f}", flush=True)
            print(f"  mean Δmean_turns: {np.mean(margins_turns):.1f}",
                  flush=True)


if __name__ == '__main__':
    main()
