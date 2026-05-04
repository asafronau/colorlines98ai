"""Crisis Mining: find hard positions with policy, solve with deep search.

Phase 1: Play policy-only (0 sims) on all seeds — instant, finds crisis positions.
Phase 2: Dispatch replay tasks to parallel GPU workers via InferenceServer.

Usage:
    python -m alphatrain.scripts.crisis_mining \
        --model alphatrain/data/pillar2u_epoch_8.pt \
        --seed-start 100000 --seed-end 101000 \
        --recovery-turns 25 --recovery-sims 2000 \
        --prevention-turns 75 --prevention-sims 1600 \
        --continue-turns 1000 \
        --device mps --workers 16 --batch-size 64 \
        --save-dir data/crisis_v1
"""

import os
import re
import time
import argparse
import json
import numpy as np
import torch
import torch.multiprocessing as mp

# Use 'spawn' so CUDA can be used in main process (probes) AND workers
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass  # already set

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def play_policy_only(net, device, seed, fp16=False, max_turns=5000):
    """Play with greedy policy (0 sims). Returns list of snapshots."""
    game = ColorLinesGame(seed=seed)
    game.reset()

    snapshots = []
    turn = 0

    while not game.game_over and turn < max_turns:
        snapshots.append({
            'board': game.board.copy(),
            'next_balls': list(game.next_balls),
            'turn': turn,
            'score': game.score,
            'empty': int(np.sum(game.board == 0)),
        })

        obs_np = _build_obs_for_game(game)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        if fp16:
            obs = obs.half()
        with torch.inference_mode():
            out = net(obs)
            pol_logits = out[0] if isinstance(out, tuple) else out
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol_np, 30)

        if not priors:
            break

        best_action = max(priors.items(), key=lambda x: x[1])[0]
        src_flat = best_action // 81
        tgt_flat = best_action % 81
        game.move(
            (src_flat // 9, src_flat % 9),
            (tgt_flat // 9, tgt_flat % 9))
        turn += 1

    return snapshots, game.score, turn


def replay_from_snapshot(mcts, snapshot, replay_seed, num_sims,
                         continue_turns, max_turns=5000,
                         feature_weights_path=None):
    """Replay from a saved board position with MCTS search."""
    game = ColorLinesGame(seed=replay_seed)
    game.reset(
        board=snapshot['board'],
        next_balls=snapshot['next_balls'])
    game.score = snapshot['score']
    game.turns = snapshot['turn']

    mcts_replay = MCTS(
        net=mcts.net, device=mcts.device, max_score=mcts.max_score,
        num_simulations=num_sims, c_puct=mcts.c_puct,
        top_k=mcts.top_k, batch_size=mcts.batch_size,
        inference_client=mcts.inference_client,
        feature_weights_path=feature_weights_path)

    moves_data = []
    t0 = time.time()
    turn = 0
    capped = False
    turns_limit = min(continue_turns, max_turns - snapshot['turn'])

    while not game.game_over and turn < turns_limit:
        board_snapshot = game.board.copy().tolist()
        nb = game.next_balls
        next_balls_json = [
            {'row': int(pc[0][0]), 'col': int(pc[0][1]), 'color': int(pc[1])}
            for pc in nb
        ]

        result = mcts_replay.search(
            game, temperature=0.0,
            dirichlet_alpha=0.3, dirichlet_weight=0.25,
            return_policy=True)

        if result[0] is None:
            break

        action, policy_target = result

        top_indices = np.argsort(policy_target)[::-1][:5]
        top_moves = []
        top_scores = []
        for idx in top_indices:
            if policy_target[idx] <= 0:
                break
            flat = int(idx)
            top_moves.append({
                'sr': int(flat // 81 // 9), 'sc': int(flat // 81 % 9),
                'tr': int(flat % 81 // 9), 'tc': int(flat % 81 % 9),
            })
            top_scores.append(float(np.log(policy_target[idx] + 1e-8)))

        chosen_src, chosen_tgt = action
        moves_data.append({
            'board': board_snapshot,
            'next_balls': next_balls_json,
            'num_next': len(nb),
            'chosen_move': {
                'sr': int(chosen_src[0]), 'sc': int(chosen_src[1]),
                'tr': int(chosen_tgt[0]), 'tc': int(chosen_tgt[1]),
            },
            'top_moves': top_moves,
            'top_scores': top_scores,
        })

        move_result = game.move(action[0], action[1])
        if not move_result['valid']:
            break
        turn += 1

    if turn >= turns_limit and not game.game_over:
        capped = True

    return {
        'seed': replay_seed,
        'original_seed': snapshot.get('original_seed', replay_seed),
        'score': game.score,
        'turns': snapshot['turn'] + turn,
        'replay_from_turn': snapshot['turn'],
        'replay_sims': num_sims,
        'moves': moves_data,
        'capped': capped,
        # Match selfplay convention: capped games carry bootstrap_value=0.0
        # explicitly. V10+ data is policy-only training; downstream value
        # consumers must pass --policy-only-data to acknowledge.
        'bootstrap_value': 0.0,
        'time': time.time() - t0,
    }


def _probe_worker(slot_id, seed_queue, task_queue, progress_queue,
                  obs_shm_name, pol_shm_name, val_shm_name,
                  num_workers, max_batch,
                  request_queue, response_queue,
                  policy_max_turns, recovery_turns, recovery_sims,
                  prevention_turns, prevention_sims,
                  existing_keys):
    """Phase 1 worker: pull seed, play policy-only via inference server,
    push replay tasks for the dying probes back to main.

    Each forward pass is batch=1 inside this worker, but the server gathers
    across all `num_workers` slots, so the GPU sees avg batch ~num_workers.
    Restores Phase 1 to roughly the same per-eval throughput as Phase 2.
    """
    torch.set_num_threads(1)

    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)
    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, 6561), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf,
                             request_queue, response_queue)
    existing_keys = set(existing_keys)

    while True:
        seed = seed_queue.get()
        if seed is None:
            break

        # Play policy-only via server.
        game = ColorLinesGame(seed=seed)
        game.reset()
        snapshots = []
        turn = 0
        died = False

        while turn < policy_max_turns and not game.game_over:
            snapshots.append({
                'board': game.board.copy(),
                'next_balls': list(game.next_balls),
                'turn': turn,
                'score': game.score,
                'empty': int(np.sum(game.board == 0)),
            })
            obs_np = _build_obs_for_game(game)
            pol_np, _ = client.evaluate(obs_np)
            priors = _get_legal_priors_flat(game.board, pol_np, 30)
            if not priors:
                died = True
                break
            best = max(priors.items(), key=lambda x: x[1])[0]
            sf = best // 81
            tf = best % 81
            game.move((sf // 9, sf % 9), (tf // 9, tf % 9))
            turn += 1

        if game.game_over and not died:
            died = True

        if not died:
            progress_queue.put((seed, 'survived', 0, 0))
            continue

        # Build replay tasks for this dying seed.
        tasks_pushed = 0
        skipped_existing = 0
        for label, rewind, sims in [
            ('recovery', recovery_turns, recovery_sims),
            ('prevention', prevention_turns, prevention_sims),
        ]:
            rewind_idx = max(0, turn - rewind)
            if rewind_idx >= len(snapshots):
                continue
            if (seed, label) in existing_keys:
                skipped_existing += 1
                continue
            snapshot = dict(snapshots[rewind_idx])
            snapshot['original_seed'] = seed
            replay_seed = seed * 37 + rewind
            task_queue.put((snapshot, replay_seed, sims, label))
            tasks_pushed += 1

        progress_queue.put((seed, 'died', tasks_pushed, skipped_existing))

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def _replay_worker(slot_id, task_queue, result_queue,
                    obs_shm_name, pol_shm_name, val_shm_name,
                    num_workers, max_batch,
                    request_queue, response_queue,
                    batch_size, max_score, continue_turns, max_turns,
                    feature_weights_path=None):
    """Persistent worker for GPU server mode replay."""
    torch.set_num_threads(1)

    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient

    N, B = num_workers, max_batch
    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)

    obs_buf = np.ndarray((N, B, 18, 9, 9), dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, 6561), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)

    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf,
                             request_queue, response_queue)

    while True:
        task = task_queue.get()
        if task is None:
            break

        snapshot, replay_seed, num_sims, label = task

        mcts = MCTS(inference_client=client, max_score=max_score,
                    num_simulations=num_sims, batch_size=batch_size,
                    top_k=30, c_puct=2.5,
                    feature_weights_path=feature_weights_path)

        result = replay_from_snapshot(
            mcts, snapshot, replay_seed, num_sims,
            continue_turns, max_turns,
            feature_weights_path=feature_weights_path)
        result['label'] = label
        result['original_seed'] = snapshot.get('original_seed', replay_seed)

        cap = " [CAP]" if result.get('capped') else ""
        survived = result['turns'] - result['replay_from_turn']
        print(f"  [w{slot_id}] seed={result['original_seed']} {label}: "
              f"{result['score']}pts ({survived}t, {result['time']:.0f}s){cap}",
              flush=True)

        result_queue.put(result)

    obs_shm.close()
    pol_shm.close()
    val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed-start', type=int, default=100000)
    p.add_argument('--seed-end', type=int, default=100100)
    p.add_argument('--recovery-turns', type=int, default=25,
                   help='Rewind N turns before death for recovery replay')
    p.add_argument('--recovery-sims', type=int, default=2000,
                   help='Sims for recovery replay')
    p.add_argument('--prevention-turns', type=int, default=50,
                   help='Rewind N turns before death for prevention replay')
    p.add_argument('--prevention-sims', type=int, default=1600,
                   help='Sims for prevention replay')
    p.add_argument('--continue-turns', type=int, default=500,
                   help='Max turns to play from rewind point')
    p.add_argument('--max-turns', type=int, default=5000,
                   help='Hard cap on total game length for replay games '
                        '(snapshot turn + replay turns ≤ max_turns).')
    p.add_argument('--policy-max-turns', type=int, default=None,
                   help='Cap on the policy-only probe game length. Probes '
                        'beyond this are treated as survivors and produce '
                        'no replay tasks. Default: falls back to --max-turns.')
    p.add_argument('--device', default=None)
    p.add_argument('--workers', type=int, default=1,
                   help='Parallel workers (1=serial, >1=GPU server)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--save-dir', default='data/crisis_v1')
    p.add_argument('--feature-value-weights', default=None,
                   help='Path to feature_value_weights.npz. When set, MCTS '
                        'replaces the NN value head with the linear feature '
                        'evaluator. Required for value-quality crisis mining.')
    p.add_argument('--compile', action='store_true',
                   help='Use torch.compile(mode=reduce-overhead) in the GPU '
                        'inference server. CUDA only; ignored elsewhere.')
    args = p.parse_args()

    if args.device:
        device_str = args.device
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device_str = 'mps'
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device_str = 'cuda'
        device = torch.device('cuda')
    else:
        device_str = 'cpu'
        device = torch.device('cpu')

    # Fail-fast: policy_only models need an MCTS value source.
    ckpt_meta = torch.load(args.model, map_location='cpu', weights_only=False)
    is_policy_only_model = ckpt_meta.get(
        'policy_only', 'value_fc2.weight' not in ckpt_meta['model'])
    del ckpt_meta
    if is_policy_only_model and not args.feature_value_weights:
        raise SystemExit(
            f"Model '{args.model}' is policy_only but no value source. "
            f"Pass --feature-value-weights for crisis MCTS replays.")

    n_seeds = args.seed_end - args.seed_start
    print(f"Crisis Mining: {n_seeds} seeds ({args.seed_start}-{args.seed_end})",
          flush=True)
    print(f"Recovery: rw{args.recovery_turns} @ {args.recovery_sims} sims",
          flush=True)
    print(f"Prevention: rw{args.prevention_turns} @ {args.prevention_sims} sims",
          flush=True)
    print(f"Continue: {args.continue_turns} turns | Device: {device} | "
          f"Workers: {args.workers}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)

    # Resume support: filenames already in the save dir tell us which
    # (seed, label) replays are done and should be skipped.
    existing_files = set(os.listdir(args.save_dir))
    already_done = len([f for f in existing_files if f.endswith('.json')])
    if already_done > 0:
        print(f"Resuming: {already_done} replays already in {args.save_dir}",
              flush=True)

    _resume_re = re.compile(r'game_seed(\d+)_(\w+)_score\d+\.json')
    existing_keys = set()
    for f in existing_files:
        m = _resume_re.match(f)
        if m:
            existing_keys.add((int(m.group(1)), m.group(2)))

    policy_max_turns = (args.policy_max_turns
                        if args.policy_max_turns is not None
                        else args.max_turns)

    # Pull model max_score from checkpoint (used by both paths to
    # configure MCTS / server).
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    srv_max_score = float(ckpt.get('max_score', 30000.0))
    del ckpt

    if args.workers <= 1:
        # ── Serial mode: load model in main, do everything sequentially ──
        net, _ = load_model(args.model, device,
                            fp16=(device_str != 'cpu'),
                            jit_trace=True)
        fp16 = False
        try:
            fp16 = next(net.parameters()).dtype == torch.float16
        except (StopIteration, AttributeError):
            pass

        print(f"\n=== Phase 1: Policy probes (cap {policy_max_turns}t) ===",
              flush=True)
        t0 = time.time()
        replay_tasks = []
        skipped = 0
        died = 0
        already_skipped = 0

        for seed in range(args.seed_start, args.seed_end):
            snapshots, pol_score, pol_turns = play_policy_only(
                net, device, seed, fp16=fp16, max_turns=policy_max_turns)
            if pol_turns >= policy_max_turns:
                skipped += 1
                continue
            died += 1
            for label, rewind, sims in [
                ('recovery', args.recovery_turns, args.recovery_sims),
                ('prevention', args.prevention_turns, args.prevention_sims),
            ]:
                rewind_idx = max(0, pol_turns - rewind)
                if rewind_idx >= len(snapshots):
                    continue
                if (seed, label) in existing_keys:
                    already_skipped += 1
                    continue
                snapshot = snapshots[rewind_idx]
                snapshot['original_seed'] = seed
                replay_seed = seed * 37 + rewind
                replay_tasks.append((snapshot, replay_seed, sims, label))
            if (died + skipped) % 100 == 0:
                print(f"  Probed {died + skipped}/{n_seeds}: "
                      f"{died} died, {skipped} survived, "
                      f"{len(replay_tasks)} replay tasks", flush=True)

        probe_time = time.time() - t0
        if already_skipped > 0:
            print(f"  ({already_skipped} replays already done, skipped)",
                  flush=True)
        print(f"Phase 1 done: {died} died, {skipped} survived, "
              f"{len(replay_tasks)} replay tasks ({probe_time:.0f}s)",
              flush=True)

        if not replay_tasks:
            print("No replay tasks — all seeds survived!", flush=True)
            return

        # Phase 2: serial replays
        print(f"\n=== Phase 2: {len(replay_tasks)} replays (serial) ===",
              flush=True)
        t1 = time.time()
        total_replays = 0
        total_states = 0
        mcts = MCTS(net, device, max_score=srv_max_score,
                    num_simulations=args.recovery_sims,
                    batch_size=args.batch_size,
                    top_k=30, c_puct=2.5,
                    feature_weights_path=args.feature_value_weights)
        for ti, (snapshot, replay_seed, sims, label) in enumerate(replay_tasks):
            result = replay_from_snapshot(
                mcts, snapshot, replay_seed, sims,
                args.continue_turns, args.max_turns,
                feature_weights_path=args.feature_value_weights)
            orig_seed = snapshot.get('original_seed', replay_seed)
            fname = f"game_seed{orig_seed}_{label}_score{result['score']}.json"
            with open(os.path.join(args.save_dir, fname), 'w') as f:
                json.dump(result, f)
            total_replays += 1
            total_states += len(result['moves'])
            cap = " [CAP]" if result.get('capped') else ""
            survived = result['turns'] - snapshot['turn']
            elapsed = time.time() - t1
            eta = elapsed / (ti + 1) * (len(replay_tasks) - ti - 1)
            print(f"  [{ti+1}/{len(replay_tasks)}] seed={orig_seed} {label}: "
                  f"{result['score']}pts ({survived}t, {result['time']:.0f}s)"
                  f"{cap} ETA {eta/60:.0f}m", flush=True)

    else:
        # ── Server mode: parallelize BOTH phases through one server ──
        from alphatrain.inference_server import InferenceServer
        MPQueue = mp.Queue
        Process = mp.Process

        server = InferenceServer(args.model, args.workers,
                                 device=device_str,
                                 max_batch_per_worker=args.batch_size,
                                 use_compile=args.compile)
        server.start()
        try:
            # Phase 1: parallel policy probes via the server
            print(f"\n=== Phase 1: Policy probes "
                  f"(cap {policy_max_turns}t, {args.workers} workers) ===",
                  flush=True)
            t0 = time.time()

            seed_queue = MPQueue()
            for s in range(args.seed_start, args.seed_end):
                seed_queue.put(s)
            for _ in range(args.workers):
                seed_queue.put(None)

            probe_task_queue = MPQueue()
            progress_queue = MPQueue()

            probe_workers = []
            for i in range(args.workers):
                proc = Process(
                    target=_probe_worker,
                    args=(i, seed_queue, probe_task_queue, progress_queue,
                          server._obs_shm.name, server._pol_shm.name,
                          server._val_shm.name,
                          args.workers, args.batch_size,
                          server.request_queue, server.response_queues[i],
                          policy_max_turns,
                          args.recovery_turns, args.recovery_sims,
                          args.prevention_turns, args.prevention_sims,
                          list(existing_keys)))
                proc.start()
                probe_workers.append(proc)

            died = 0
            skipped = 0
            already_skipped = 0
            for _ in range(n_seeds):
                _seed, status, n_pushed, n_skipped = progress_queue.get(
                    timeout=3600)
                if status == 'died':
                    died += 1
                else:
                    skipped += 1
                already_skipped += n_skipped
                done_so_far = died + skipped
                if done_so_far % 100 == 0:
                    print(f"  Probed {done_so_far}/{n_seeds}: "
                          f"{died} died, {skipped} survived "
                          f"({(time.time()-t0):.0f}s)", flush=True)

            for proc in probe_workers:
                proc.join(timeout=30)

            # Drain replay tasks pushed by the probe workers.
            replay_tasks = []
            while True:
                try:
                    replay_tasks.append(probe_task_queue.get_nowait())
                except Exception:
                    break

            probe_time = time.time() - t0
            if already_skipped > 0:
                print(f"  ({already_skipped} replays already done, skipped)",
                      flush=True)
            print(f"Phase 1 done: {died} died, {skipped} survived, "
                  f"{len(replay_tasks)} replay tasks ({probe_time:.0f}s)",
                  flush=True)

            if not replay_tasks:
                print("No replay tasks — all seeds survived!", flush=True)
                return

            # Phase 2: parallel replays via the same server
            print(f"\n=== Phase 2: {len(replay_tasks)} replays "
                  f"({args.workers} workers) ===", flush=True)
            t1 = time.time()
            total_replays = 0
            total_states = 0

            task_queue = MPQueue()
            for task in replay_tasks:
                task_queue.put(task)
            for _ in range(args.workers):
                task_queue.put(None)

            result_queue = MPQueue()
            replay_workers = []
            for i in range(args.workers):
                proc = Process(
                    target=_replay_worker,
                    args=(i, task_queue, result_queue,
                          server._obs_shm.name, server._pol_shm.name,
                          server._val_shm.name,
                          args.workers, args.batch_size,
                          server.request_queue, server.response_queues[i],
                          args.batch_size, srv_max_score,
                          args.continue_turns, args.max_turns,
                          args.feature_value_weights))
                proc.start()
                replay_workers.append(proc)

            for ti in range(len(replay_tasks)):
                result = result_queue.get(timeout=7200)
                orig_seed = result.get('original_seed', result['seed'])
                label = result.get('label', 'replay')
                fname = f"game_seed{orig_seed}_{label}_score{result['score']}.json"
                with open(os.path.join(args.save_dir, fname), 'w') as f:
                    json.dump(result, f)
                total_replays += 1
                total_states += len(result['moves'])
                elapsed = time.time() - t1
                eta = elapsed / (ti + 1) * (len(replay_tasks) - ti - 1)
                if (ti + 1) % 10 == 0:
                    print(f"  [{ti+1}/{len(replay_tasks)}] "
                          f"{total_states:,} states, "
                          f"ETA {eta/60:.0f}m", flush=True)

            for proc in replay_workers:
                proc.join(timeout=5)
        finally:
            server.shutdown()

    elapsed = time.time() - t0
    print(f"\nDone: {total_replays} replays, {total_states:,} states, "
          f"{skipped} skipped in {elapsed/60:.0f}m", flush=True)


if __name__ == '__main__':
    main()
