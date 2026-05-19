"""Collect MCTS top-move picks for a set of anchors.

For each anchor, runs MCTS once (N sims) with the specified (backbone,
value_head, q_weight) and records the move MCTS picks. Used by Tournament v2
to compare MCTS teachers (2z+v12targets, 2y2+v11, etc.) against raw policy
and rollout oracle.

The anchors input can be:
  - phase3_tournament.pt (505 anchors from sources A/B/D)
  - source_c_first_divergence.pt (300 anchors from source C)
Both share the (board, next_balls, num_next, turn_origin) fields.

Output: dict {anchor_id: move_flat_action} saved as .pt.

Usage:
    python -m alphatrain.scripts.collect_mcts_picks \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --value-head alphatrain/data/value_head_v12_v12targets.pt \\
        --anchors alphatrain/data/phase3_tournament.pt \\
        --anchors-format tournament \\
        --simulations 100 --q-weight 2.0 --top-k 30 \\
        --workers 16 --batch-size 8 \\
        --device mps \\
        --output alphatrain/data/mcts_picks_2z_v12targets.pt
"""

import argparse
import time
import numpy as np
import torch
from multiprocessing import Queue as MPQueue, Process
from multiprocessing.shared_memory import SharedMemory


def _worker(slot_id, anchor_queue, result_queue,
            obs_shm_name, pol_shm_name, val_shm_name,
            num_workers, max_batch, req_q, resp_q,
            num_sims, c_puct, top_k, max_score, q_weight,
            value_head_path):
    torch.set_num_threads(1)
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE
    from alphatrain.mcts import MCTS
    from game.board import ColorLinesGame

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)
    N, B = num_workers, max_batch
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)
    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf, req_q, resp_q)

    mcts = MCTS(inference_client=client, max_score=max_score,
                num_simulations=num_sims, c_puct=c_puct, top_k=top_k,
                batch_size=max_batch, q_weight=q_weight,
                value_head_path=value_head_path)

    while True:
        anchor = anchor_queue.get()
        if anchor is None:
            break
        game = ColorLinesGame(seed=0)
        game.reset(board=np.asarray(anchor['board'], dtype=np.int8).copy(),
                   next_balls=list(anchor['next_balls']))
        game.turns = anchor['turn_origin']
        if game.game_over:
            result_queue.put({'anchor_id': anchor['id'], 'skipped': 'game_over'})
            continue
        try:
            move = mcts.search(game)
        except Exception as e:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': f'mcts_error: {e}'})
            continue
        if move is None:
            result_queue.put({'anchor_id': anchor['id'],
                              'skipped': 'no_move'})
            continue
        sr, sc = move[0]
        tr, tc = move[1]
        flat_action = sr * 81 * 9 + sc * 81 + tr * 9 + tc
        result_queue.put({'anchor_id': anchor['id'],
                          'move': int(flat_action)})

    obs_shm.close(); pol_shm.close(); val_shm.close()


def load_anchors(path, fmt, **kwargs):
    """Load anchors from various source formats.

    fmt='source_c': load from source_c_first_divergence.pt
    fmt='tournament_pool': resample from selfplay_v12 / crisis_v12 / phase1
        using same seeds as phase3_tournament.py to match its anchor pool.
    """
    if fmt == 'source_c':
        data = torch.load(path, weights_only=False)
        out = []
        for a in data['anchors']:
            out.append({
                'id': a['id'],
                'board': a['board'],
                'next_balls': a['next_balls'],
                'num_next': a['num_next'],
                'turn_origin': a['turn_origin'],
                'source_label': a['source_label'],
            })
        return out
    elif fmt == 'tournament_pool':
        from random import Random
        import glob, json, os
        crisis_dir = kwargs.get('crisis_dir', 'data/crisis_v12')
        selfplay_dir = kwargs.get('selfplay_dir', 'data/selfplay_v12')
        phase1_path = kwargs.get('phase1_path')
        n_per_source = kwargs.get('n_per_source', 200)
        margin_threshold = kwargs.get('margin_threshold', 0.15)
        seed = kwargs.get('seed', 42)

        def sample_from_dirs(dirs, n, label, rng):
            files = []
            for d in dirs:
                files.extend(sorted(glob.glob(
                    os.path.join(d, 'game_seed*.json'))))
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
                    'next_balls': [((int(nb['row']), int(nb['col'])),
                                     int(nb['color']))
                                    for nb in m['next_balls']],
                    'num_next': int(m['num_next']),
                    'turn_origin': mi,
                    'source_label': label,
                })
            return out

        rng = Random(seed)
        src_a = sample_from_dirs([selfplay_dir], n_per_source, 'selfplay', rng)
        src_b = sample_from_dirs([crisis_dir], n_per_source, 'crisis', rng)

        # Source D: high-margin from phase1
        src_d = []
        if phase1_path:
            data = torch.load(phase1_path, weights_only=False)
            for r in data['results']:
                pm = r['per_move']
                if len(pm) < 2:
                    continue
                sorted_moves = sorted(pm.items(),
                                       key=lambda kv: kv[1]['rank'])[:6]
                qs = np.array([mv['cap_rate'] for _, mv in sorted_moves])
                margin = qs.max() - qs[0]
                if margin >= margin_threshold:
                    src_d.append({
                        'board': r['anchor_board'],
                        'next_balls': r['anchor_next_balls'],
                        'num_next': r['anchor_n_next'],
                        'turn_origin': r.get('turn_origin', 0),
                        'source_label': 'oracle_disagree',
                    })
            rng2 = Random(0)
            rng2.shuffle(src_d)
            src_d = src_d[:n_per_source]

        anchors = src_a + src_b + src_d
        for i, a in enumerate(anchors):
            a['id'] = i
        return anchors
    else:
        raise ValueError(f"Unknown format: {fmt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--value-head', required=True)
    p.add_argument('--anchors', required=False,
                   help='Source file (for source_c) or unused (for tournament_pool).')
    p.add_argument('--anchors-format',
                   choices=['source_c', 'tournament_pool'],
                   required=True)
    p.add_argument('--phase1-path',
                   default='alphatrain/data/phase1_oracle.pt',
                   help='For tournament_pool: phase1 file for source D.')
    p.add_argument('--crisis-dir', default='data/crisis_v12')
    p.add_argument('--selfplay-dir', default='data/selfplay_v12')
    p.add_argument('--n-per-source', type=int, default=200)
    p.add_argument('--margin-threshold', type=float, default=0.15)
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--simulations', type=int, default=100)
    p.add_argument('--q-weight', type=float, default=2.0)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    print(f"Loading anchors (fmt={args.anchors_format})...", flush=True)
    anchors = load_anchors(args.anchors, args.anchors_format,
                            phase1_path=args.phase1_path,
                            crisis_dir=args.crisis_dir,
                            selfplay_dir=args.selfplay_dir,
                            n_per_source=args.n_per_source,
                            margin_threshold=args.margin_threshold,
                            seed=args.sample_seed)
    print(f"  {len(anchors)} anchors", flush=True)

    from alphatrain.inference_server import InferenceServer
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    max_score = float(ckpt.get('max_score', 6050.0))
    del ckpt

    print(f"Starting inference server "
          f"({args.model} + {args.value_head})...", flush=True)
    server = InferenceServer(args.model, args.workers, device=args.device,
                              max_batch_per_worker=args.batch_size,
                              value_head_path=args.value_head)
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
                             args.simulations, args.c_puct, args.top_k,
                             max_score, args.q_weight, args.value_head))
        proc.start()
        workers.append(proc)

    t0 = time.time()
    picks = {}
    skipped = 0
    for i in range(len(anchors)):
        try:
            r = result_queue.get(timeout=600)
        except Exception:
            print(f"Timeout at {i}", flush=True)
            break
        if 'skipped' in r:
            skipped += 1
        else:
            picks[r['anchor_id']] = r['move']
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(anchors) - i - 1)
            print(f"  [{i+1}/{len(anchors)}] kept={len(picks)} "
                  f"skip={skipped} {elapsed:.0f}s ETA {eta:.0f}s", flush=True)

    for w in workers:
        w.join(timeout=10)
    server.shutdown()
    print(f"\nDone: {len(picks)} picks in {time.time()-t0:.0f}s", flush=True)
    torch.save({'args': vars(args), 'picks': picks}, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == '__main__':
    main()
