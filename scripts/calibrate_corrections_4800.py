"""Calibrate the 400-sim completed-Q corrections against a deep widened MCTS@4800.

The Gumbel target flips the policy on the ~2.7% "correction" states (400-sim Q says a
well-visited move beats the prior's confident pick). But the prior is ~33x confident and
~40% of greedy-mined forks have historically been phantoms at low sims
([[project_mcts_teacher_validated]]). Before we trust the 400-sim corrections enough to
override the prior, re-search a sample of them with a DEEP widened MCTS@4800 (same
pillar3f + value head, 12x sims + Dirichlet widening = the firefighting gold standard) and
ask: does the deep search agree with the correction?

Per correction state we know (from the recorded data):
    p_arg_move = the prior's confident pick (the move the policy would play)
    q_arg_move = the 400-sim completed-Q best well-visited move (our distillation target)
Deep MCTS@4800 then yields a visit distribution; we classify:
    CONFIRM   deep visit-best == q_arg_move          (correction is real)
    PHANTOM   deep visit-best == p_arg_move          (prior was right; 400 was noise)
    OTHER     deep visit-best is a third move
plus the softer directional check: does deep rank q_arg_move above p_arg_move by visits?

High CONFIRM (+ deep prefers q over p) => trust tau=0.02. High PHANTOM => soften tau.

    PYTHONPATH=. python scripts/calibrate_corrections_4800.py \\
        --model alphatrain/data/pillar3f.pt \\
        --value-head-path alphatrain/data/value_head_pillar3f.pt \\
        --n-states 300 --sims 4800 --mcts-seeds 2 --workers 14
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

import sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import multiprocessing as mp
from multiprocessing import Process, Queue as MPQueue

TENSOR = 'alphatrain/data/v15_pillar3f_slim.pt'


def _sample_corrections(tensor, n_states, seed, visit_floor, kappa, spread_gate):
    """Return a list of correction-state dicts (board, next_balls, q/p moves)."""
    from alphatrain.gumbel import completed_q_target, NEG_INF
    d = torch.load(tensor, weights_only=True)
    N = d['cand_idx'].shape[0]
    cv, cp, cq = d['cand_visit'], d['cand_prior'], d['cand_q']
    nnz, rv, cidx = d['cand_nnz'], d['root_value'], d['cand_idx']
    # is_correction over all states (chunked to bound memory)
    flags = []
    for s in range(0, N, 1_000_000):
        e = min(s + 1_000_000, N)
        _, _, _, corr, _ = completed_q_target(
            cv[s:e], cp[s:e], cq[s:e], nnz[s:e], rv[s:e],
            visit_floor=visit_floor, kappa=kappa, spread_gate=spread_gate, tau=0.02)
        flags.append(corr)
    is_corr = torch.cat(flags)
    corr_idx = torch.where(is_corr)[0]
    g = torch.Generator().manual_seed(seed)
    pick = corr_idx[torch.randperm(len(corr_idx), generator=g)[:n_states]]
    print(f"  {len(corr_idx):,} correction states; sampling {len(pick)}", flush=True)

    # recompute well-visited q_arg / prior p_arg slots, map to move indices
    out = []
    rootv = rv[pick].unsqueeze(1)
    compq = (cv[pick] * cq[pick] + kappa * rootv) / (cv[pick] + kappa)
    ar = torch.arange(cv.shape[1]).unsqueeze(0)
    valid = ar < nnz[pick].unsqueeze(1)
    well = valid & (cv[pick] >= visit_floor)
    cqw = torch.where(well, compq, torch.full_like(compq, NEG_INF))
    pw = torch.where(well, cp[pick], torch.full_like(cp[pick], NEG_INF))
    q_arg = cqw.argmax(1); p_arg = pw.argmax(1)
    for k, i in enumerate(pick.tolist()):
        n = int(d['n_next'][i])
        nb = [[[int(d['next_pos'][i, j, 0]), int(d['next_pos'][i, j, 1])],
               int(d['next_col'][i, j])] for j in range(n)]
        out.append({
            'board': d['boards'][i].numpy().tolist(),
            'next_balls': nb,
            'q_move': int(cidx[i, q_arg[k]]),
            'p_move': int(cidx[i, p_arg[k]]),
            'q400': float(cq[i, q_arg[k]]), 'p_q400': float(cq[i, p_arg[k]]),
        })
    return out


def _worker(slot_id, state_q, res_q, obs_shm_name, pol_shm_name, val_shm_name,
            N, B, request_q, response_q, sims, top_k, q_weight,
            value_head_path, fv_weights, mcts_seeds, dir_alpha, dir_weight):
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
        st = state_q.get()
        if st is None:
            break
        board = np.array(st['board'], dtype=np.int8)
        nb = [((int(p[0]), int(p[1])), int(c)) for p, c in st['next_balls']]
        visits = np.zeros(6561, dtype=np.float64)
        vc = np.zeros(6561, dtype=np.float64); vs = np.zeros(6561, dtype=np.float64)
        for s in range(mcts_seeds):
            game = ColorLinesGame()
            game.reset(board=board.copy(), next_balls=list(nb))
            game.score, game.turns = 0, s * 1_000_003
            _, pt = mcts.search(game, temperature=0.0, dirichlet_alpha=dir_alpha,
                                dirichlet_weight=dir_weight, return_policy=True)
            visits += np.asarray(pt, dtype=np.float64)
            for act, child in mcts._last_root.children.items():
                if child.visit_count > 0:
                    vc[act] += child.visit_count
                    vs[act] += child.value_sum
        deep_best = int(visits.argmax())
        qm, pm = st['q_move'], st['p_move']
        res_q.put({
            'deep_best': deep_best, 'q_move': qm, 'p_move': pm,
            'confirm': deep_best == qm, 'phantom': deep_best == pm,
            'deep_v_q': float(visits[qm]), 'deep_v_p': float(visits[pm]),
            'deep_q_q': float(vs[qm] / vc[qm]) if vc[qm] > 0 else None,
            'deep_q_p': float(vs[pm] / vc[pm]) if vc[pm] > 0 else None,
            'prefers_q': float(visits[qm]) > float(visits[pm]),
        })
    obs_shm.close(); pol_shm.close(); val_shm.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', default=TENSOR)
    p.add_argument('--model', required=True)
    p.add_argument('--value-head-path', default=None)
    p.add_argument('--feature-value-weights', default=None)
    p.add_argument('--q-weight', type=float, default=2.0)
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--mcts-seeds', type=int, default=2)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--dir-alpha', type=float, default=0.3)
    p.add_argument('--dir-weight', type=float, default=0.25)
    p.add_argument('--n-states', type=int, default=300)
    p.add_argument('--workers', type=int, default=14)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--visit-floor', type=float, default=20.0)
    p.add_argument('--kappa', type=float, default=15.0)
    p.add_argument('--spread-gate', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='logs/calib_4800.json')
    a = p.parse_args()
    if not (a.value_head_path or a.feature_value_weights):
        raise SystemExit("pass --value-head-path or --feature-value-weights")

    from alphatrain.inference_server import InferenceServer
    device_str = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"sampling correction states from {a.tensor} ...", flush=True)
    states = _sample_corrections(a.tensor, a.n_states, a.seed,
                                 a.visit_floor, a.kappa, a.spread_gate)
    print(f"device={device_str} workers={a.workers} | deep MCTS@{a.sims} x{a.mcts_seeds} "
          f"(widened a={a.dir_alpha}/w={a.dir_weight}, top_k={a.top_k}) on {len(states)} "
          f"corrections", flush=True)

    server = InferenceServer(a.model, a.workers, device=device_str,
                             max_batch_per_worker=a.batch_size, use_compile=False,
                             value_head_path=a.value_head_path)
    server.start()
    state_q, res_q = MPQueue(), MPQueue()
    for st in states:
        state_q.put(st)
    for _ in range(a.workers):
        state_q.put(None)
    procs = []
    for i in range(a.workers):
        pr = Process(target=_worker,
                     args=(i, state_q, res_q, server._obs_shm.name,
                           server._pol_shm.name, server._val_shm.name,
                           a.workers, a.batch_size, server.request_queue,
                           server.response_queues[i], a.sims, a.top_k, a.q_weight,
                           a.value_head_path, a.feature_value_weights, a.mcts_seeds,
                           a.dir_alpha, a.dir_weight))
        pr.start(); procs.append(pr)

    results, t0 = [], time.time()
    try:
        for k in range(len(states)):
            results.append(res_q.get(timeout=14400))
            if (k + 1) % 25 == 0:
                el = time.time() - t0
                conf = 100 * np.mean([r['confirm'] for r in results])
                print(f"  [{k+1}/{len(states)}] confirm so far {conf:.0f}% "
                      f"({el/60:.1f}m, {(k+1)/el:.2f} st/s)", flush=True)
    finally:
        for pr in procs:
            pr.join(timeout=30)
        server.shutdown()

    n = len(results)
    confirm = np.mean([r['confirm'] for r in results])
    phantom = np.mean([r['phantom'] for r in results])
    other = 1 - confirm - phantom
    prefers_q = np.mean([r['prefers_q'] for r in results])
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump({'n': n, 'confirm': confirm, 'phantom': phantom, 'other': other,
               'prefers_q_over_p': prefers_q, 'results': results},
              open(a.out, 'w'), default=float)
    print(f"\n=== CALIBRATION ({n} corrections, deep MCTS@{a.sims}x{a.mcts_seeds}) ===")
    print(f"  CONFIRM (deep best == 400 q-correction): {100*confirm:.1f}%")
    print(f"  PHANTOM (deep best == prior's pick)    : {100*phantom:.1f}%")
    print(f"  OTHER   (deep best == a third move)    : {100*other:.1f}%")
    print(f"  deep prefers q over p (by visits)      : {100*prefers_q:.1f}%")
    print(f"  -> high CONFIRM + prefers_q: trust tau=0.02. high PHANTOM: soften tau.")
    print(f"  wrote {a.out} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
