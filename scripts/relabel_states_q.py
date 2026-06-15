"""Relabel existing self-play/crisis states with CLEAN root Q + prior (the Gumbel corpus).

Instead of regenerating self-play to record Q (days), re-search a SAMPLED subset of the
states we already have. Two wins over a full regen:
  * cheaper — re-search ~stride⁻¹ of the states (no probe phase, no game-gen loop), the
    trunk distillation needs coverage not all 9.7M states.
  * cleaner — re-search with NO Dirichlet, so the recorded prior + root Q are the
    noise-free label-side ingredients the Gumbel target wants ("separate exploration from
    labels"). The states still come from the Dirichlet-influenced self-play distribution.

One clean MCTS@400 per sampled state already integrates over spawn stochasticity (each of
the 400 sims rolls its own spawns), so no determinization loop is needed.

Reuses the InferenceServer + N MCTS-worker architecture (one GPU process batches forwards).
Output = relabeled game JSONs (selfplay schema: moves with cand_moves/cand_visits/cand_prior/
cand_q/root_value/q_min/q_max) → feed to build_expert_v2_tensor --policy-only-data.

    PYTHONPATH=. python scripts/relabel_states_q.py \\
        --in-dirs data/selfplay_v14 data/crisis_v14_s600 \\
        --model alphatrain/data/pillar3f.pt \\
        --value-head-path alphatrain/data/value_head_pillar3f.pt --q-weight 2.0 \\
        --sims 400 --stride 4 --workers 16 --out-dir data/relabel_v15
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys, glob, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue


def _relabel_worker(slot_id, file_q, res_q, obs_shm_name, pol_shm_name, val_shm_name,
                    N, B, request_q, response_q, sims, top_k, q_weight,
                    value_head_path, fv_weights, stride, top_k_save, out_dir):
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
    client = InferenceClient(slot_id, obs_buf, pol_buf, val_buf, request_q, response_q)
    mcts = MCTS(inference_client=client, num_simulations=sims, c_puct=2.5,
                top_k=top_k, batch_size=B, value_head_path=value_head_path,
                feature_weights_path=fv_weights, q_weight=q_weight)

    while True:
        fpath = file_q.get()
        if fpath is None:
            break
        try:
            g = json.load(open(fpath))
        except Exception:
            res_q.put((os.path.basename(fpath), 0)); continue
        moves = g.get('moves', [])
        out_moves = []
        for mi in range(0, len(moves), stride):
            mv = moves[mi]
            board = np.array(mv['board'], dtype=np.int8)
            nb = [((int(b['row']), int(b['col'])), int(b['color']))
                  for b in mv['next_balls']]
            game = ColorLinesGame()
            game.reset(board=board, next_balls=nb)
            game.score, game.turns = 0, int(mv.get('turn', mi))
            # CLEAN re-search: no Dirichlet → noise-free prior + root Q.
            mcts.search(game, temperature=0.0, dirichlet_alpha=0.0,
                        dirichlet_weight=0.0, return_policy=True)
            rec = mcts.last_root_record(top_k=top_k_save)
            if rec is None:
                continue
            out = {'board': mv['board'], 'next_balls': mv['next_balls'],
                   'num_next': mv.get('num_next', len(nb)),
                   'chosen_move': mv['chosen_move']}
            out.update(rec)
            out_moves.append(out)
        json.dump({'seed': g.get('seed'), 'moves': out_moves,
                   'capped': g.get('capped', False)},
                  open(os.path.join(out_dir, os.path.basename(fpath)), 'w'),
                  default=float)
        res_q.put((os.path.basename(fpath), len(out_moves)))
    obs_shm.close(); pol_shm.close(); val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in-dirs', nargs='+', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--value-head-path', default=None)
    p.add_argument('--feature-value-weights', default=None)
    p.add_argument('--q-weight', type=float, default=2.0)
    p.add_argument('--sims', type=int, default=400)
    p.add_argument('--stride', type=int, default=4,
                   help='Re-search every Nth state of each game (4 = 25%% coverage).')
    p.add_argument('--top-k-save', type=int, default=15)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--max-games', type=int, default=0)
    p.add_argument('--out-dir', default='data/relabel_v15')
    p.add_argument('--compile', action='store_true')
    a = p.parse_args()
    if not (a.value_head_path or a.feature_value_weights):
        raise SystemExit("pass --value-head-path or --feature-value-weights")

    from alphatrain.inference_server import InferenceServer
    os.makedirs(a.out_dir, exist_ok=True)
    device_str = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')

    files = []
    for d in a.in_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, 'game_seed*.json'))))
    # resume: skip games already relabeled
    done = set(os.listdir(a.out_dir))
    files = [f for f in files if os.path.basename(f) not in done]
    if a.max_games:
        files = files[:a.max_games]
    if not files:
        print("Nothing to do (all relabeled)."); return
    print(f"device={device_str} workers={a.workers} | {len(files)} games "
          f"(stride {a.stride}, MCTS@{a.sims} CLEAN no-Dirichlet) -> {a.out_dir}/",
          flush=True)

    server = InferenceServer(a.model, a.workers, device=device_str,
                             max_batch_per_worker=a.batch_size,
                             use_compile=a.compile,
                             value_head_path=a.value_head_path)
    server.start()
    file_q, res_q = MPQueue(), MPQueue()
    for f in files:
        file_q.put(f)
    for _ in range(a.workers):
        file_q.put(None)
    procs = []
    for i in range(a.workers):
        pr = Process(target=_relabel_worker,
                     args=(i, file_q, res_q, server._obs_shm.name,
                           server._pol_shm.name, server._val_shm.name,
                           a.workers, a.batch_size, server.request_queue,
                           server.response_queues[i], a.sims, a.top_k, a.q_weight,
                           a.value_head_path, a.feature_value_weights, a.stride,
                           a.top_k_save, a.out_dir))
        pr.start(); procs.append(pr)

    n_states, t0 = 0, time.time()
    try:
        for k in range(len(files)):
            name, ns = res_q.get(timeout=14400)
            n_states += ns
            if (k + 1) % 25 == 0:
                el = time.time() - t0
                eta = el / (k + 1) * (len(files) - k - 1)
                print(f"  [{k+1}/{len(files)}] {n_states:,} states "
                      f"({n_states/max(el,1):.0f} st/s, {el/60:.0f}m, "
                      f"ETA {eta/60:.0f}m)", flush=True)
    finally:
        for pr in procs:
            pr.join(timeout=30)
        server.shutdown()
    print(f"\nDONE: {n_states:,} relabeled states from {len(files)} games "
          f"in {(time.time()-t0)/60:.1f}m -> {a.out_dir}/ "
          f"(build: build_expert_v2_tensor --games-dir {a.out_dir} --policy-only-data)",
          flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
