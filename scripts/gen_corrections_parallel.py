"""Parallel MCTS-corrections generator -- reuses the selfplay/eval inference-server
architecture (one GPU process batches NN forwards across ALL workers; N CPU worker
processes each run MCTS). Saturates the GPU and uses every core.

Job: rewind each policy death game's crisis band (D-15..D-85); at each state run
WIDENED multi-seed MCTS@4800; keep ONLY corrections (MCTS top != policy move); the
soft visit distribution is the target. Per-game output (resumable + shard-friendly).

The QUEUE carries seeds (like selfplay), NOT 45k board-states -- each worker reads
its own crisis/death_games/death_<seed>.json and does the rewind+MCTS, so the queue
stays tiny and startup is instant.

    PYTHONPATH=. python scripts/gen_corrections_parallel.py --workers 16
"""
# Single-thread BLAS/OpenMP/numba BEFORE numpy/torch (spawn re-imports per worker).
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys, glob, json, time, argparse, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue

FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'
MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'


def _flat(m):
    return (m[0][0] * 9 + m[0][1]) * 81 + (m[1][0] * 9 + m[1][1])


def _crisis_worker(slot_id, seed_queue, result_queue,
                   obs_shm_name, pol_shm_name, val_shm_name, N, B,
                   request_queue, response_queue,
                   sims, top_k, q_weight, fv_weights, mcts_seeds, topk_visits,
                   death_dir, lo, hi, out_dir):
    torch.set_num_threads(1)
    from multiprocessing.shared_memory import SharedMemory
    from alphatrain.inference_server import InferenceClient, OBS_SHAPE, POL_SIZE
    from alphatrain.mcts import MCTS
    from game.board import ColorLinesGame

    obs_shm = SharedMemory(name=obs_shm_name)
    pol_shm = SharedMemory(name=pol_shm_name)
    val_shm = SharedMemory(name=val_shm_name)
    obs_buf = np.ndarray((N, B) + OBS_SHAPE, dtype=np.float32, buffer=obs_shm.buf)
    pol_buf = np.ndarray((N, B, POL_SIZE), dtype=np.float32, buffer=pol_shm.buf)
    val_buf = np.ndarray((N, B), dtype=np.float32, buffer=val_shm.buf)
    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf,
                             request_queue, response_queue)
    mcts = MCTS(inference_client=client, num_simulations=sims, c_puct=2.5,
                top_k=top_k, batch_size=B, feature_weights_path=fv_weights,
                q_weight=q_weight)

    while True:
        seed = seed_queue.get()
        if seed is None:
            break
        g = json.load(open(os.path.join(death_dir, f'death_{seed}.json')))
        nf = len(g['frames'])
        corr, n_states = [], 0
        for i, fr in enumerate(g['frames']):
            depth = (nf - 1) - i
            if not (lo <= depth <= hi) or fr.get('chosen_move') is None:
                continue
            n_states += 1
            pol_idx = _flat(fr['chosen_move'])
            visit_sum = np.zeros(6561, dtype=np.float64)
            for s in range(mcts_seeds):
                game = ColorLinesGame()
                game.reset(board=np.array(fr['board'], dtype=np.int8),
                           next_balls=[(tuple(p), int(c)) for p, c in fr['next_balls']])
                game.score, game.turns = 0, int(fr['turn']) + s * 1_000_003
                _, pt = mcts.search(game, temperature=0.0, dirichlet_alpha=0.3,
                                    dirichlet_weight=0.25, return_policy=True)
                visit_sum += np.asarray(pt, dtype=np.float64)
            visits = visit_sum / visit_sum.sum()
            top_idx = int(visits.argmax())
            if top_idx != pol_idx:
                order = np.argsort(-visits)[:topk_visits]
                corr.append({'seed': seed, 'turn': int(fr['turn']), 'depth': depth,
                             'board': fr['board'], 'next_balls': fr['next_balls'],
                             'pol_idx': pol_idx, 'pol_share': float(visits[pol_idx]),
                             'mcts_top_idx': top_idx, 'mcts_top_share': float(visits[top_idx]),
                             'visits': [[int(j), float(visits[j])] for j in order if visits[j] > 0]})
        json.dump({'seed': seed, 'n_band_states': n_states, 'corrections': corr},
                  open(os.path.join(out_dir, f'corr_{seed}.json'), 'w'), default=float)
        result_queue.put((seed, len(corr), n_states))
    obs_shm.close(); pol_shm.close(); val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--death-glob', default='crisis/death_games/death_*.json')
    p.add_argument('--lo', type=int, default=15)
    p.add_argument('--hi', type=int, default=85)
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--mcts-seeds', type=int, default=3)
    p.add_argument('--topk-visits', type=int, default=20)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--top-k', type=int, default=300)
    p.add_argument('--q-weight', type=float, default=2.0)
    p.add_argument('--max-games', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--max-final-score', type=int, default=0,
                   help='Only mine games whose final_score <= this (0=off). Use to '
                        'target the floor: mine the early-death games whose crisis '
                        'corrections the mid-game-dominated corpus has saturated on.')
    p.add_argument('--out-dir', default='crisis/corrections')
    p.add_argument('--model', default=MODEL,
                   help='Policy whose prior drives MCTS. MUST match the model that '
                        'recorded the death games (chosen_move), else corrections are '
                        'measured against the wrong policy.')
    a = p.parse_args()

    from alphatrain.inference_server import InferenceServer
    os.makedirs(a.out_dir, exist_ok=True)
    death_dir = os.path.dirname(a.death_glob)
    device_str = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')
    files = sorted(glob.glob(a.death_glob))
    if a.shards > 1:
        files = [f for k, f in enumerate(files) if k % a.shards == a.shard]
    if a.max_final_score:
        kept, scanned = [], 0
        for f in files:
            try:
                fs = json.load(open(f)).get('final_score', 1 << 30)
            except Exception:
                continue
            scanned += 1
            if fs <= a.max_final_score:
                kept.append(f)
        print(f"floor filter: {len(kept)}/{scanned} games with final_score<="
              f"{a.max_final_score}", flush=True)
        files = kept
    if a.max_games:
        files = files[:a.max_games]
    seeds = []
    for f in files:
        sd = int(re.search(r'_(\d+)\.json$', f).group(1))
        if not os.path.exists(os.path.join(a.out_dir, f'corr_{sd}.json')):   # resume
            seeds.append(sd)
    if not seeds:
        print("No games to do (all corr_*.json present)."); return
    print(f"device={device_str} workers={a.workers} | {len(seeds)} games "
          f"(band D-{a.lo}..D-{a.hi}, MCTS@{a.sims} x{a.mcts_seeds}, top_k={a.top_k}) "
          f"-> {a.out_dir}/", flush=True)

    print(f"policy model: {a.model}", flush=True)
    server = InferenceServer(a.model, a.workers, device=device_str,
                             max_batch_per_worker=a.batch_size, use_compile=False,
                             value_head_path=None)
    server.start()
    seed_q, res_q = MPQueue(), MPQueue()
    for s in seeds:
        seed_q.put(s)
    for _ in range(a.workers):
        seed_q.put(None)
    procs = []
    for i in range(a.workers):
        pr = Process(target=_crisis_worker,
                     args=(i, seed_q, res_q, server._obs_shm.name,
                           server._pol_shm.name, server._val_shm.name,
                           a.workers, a.batch_size, server.request_queue,
                           server.response_queues[i], a.sims, a.top_k,
                           a.q_weight, FV, a.mcts_seeds, a.topk_visits, death_dir,
                           a.lo, a.hi, a.out_dir))
        pr.start(); procs.append(pr)

    n_corr, n_states, t0 = 0, 0, time.time()
    try:
        for k in range(len(seeds)):
            seed, nc, ns = res_q.get(timeout=14400)
            n_corr += nc; n_states += ns
            el = time.time() - t0
            eta = el / (k + 1) * (len(seeds) - k - 1)
            print(f"  [{k+1}/{len(seeds)}] seed {seed}: {nc}/{ns} corrections "
                  f"(total {n_corr}/{n_states}={100*n_corr/max(n_states,1):.0f}%, "
                  f"{el/60:.1f}m, ETA {eta/60:.0f}m)", flush=True)
    finally:
        for pr in procs:
            pr.join(timeout=30)
        server.shutdown()
    print(f"\nDONE: {n_corr} corrections from {n_states} band states over "
          f"{len(seeds)} games in {(time.time()-t0)/60:.1f}m. -> {a.out_dir}/", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
