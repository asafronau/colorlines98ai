"""Parallel rollout: shard rollout jobs across N worker processes sharing the GPU.

The single-process batched engine is sync-bound — it blocks on `.cpu()` every
step while the GPU sits ~83% idle (see scripts/profile_rollout.py). Running N
workers lets their syncs/CPU-logic interleave so the GPU fills up. Each worker
loads the model once (persistent Pool) and runs the validated single-process
batched_rollout on a shard; results recombine in original order.

Run directly to benchmark single vs parallel and pick a worker count:
    PYTHONPATH=. python scripts/parallel_rollout.py --workers 6
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import multiprocessing as mp

_W = {}


def _init(model_path, device_str, fp16, use_compile=False):
    import torch
    from alphatrain.evaluate import load_model
    torch.set_num_threads(1)
    dev = torch.device(device_str)
    net, _ = load_model(model_path, dev, fp16=fp16)
    if use_compile and dev.type == 'cuda':   # cuda-only; MPS compile unsupported
        net = torch.compile(net, mode='reduce-overhead')
    _W['net'] = net
    _W['dev'] = dev
    _W['dtype'] = next(net.parameters()).dtype


def _chunk(args):
    jobs, horizon, batch = args
    from scripts.batched_rollout import batched_rollout
    return batched_rollout(_W['net'], _W['dev'], _W['dtype'], jobs, horizon, batch)


class RolloutPool:
    """Persistent pool of rollout workers (model loaded once per worker)."""

    def __init__(self, model_path, device_str='mps', fp16=True, n_workers=6,
                 use_compile=False):
        ctx = mp.get_context('spawn')
        self.n = n_workers
        self.pool = ctx.Pool(n_workers, initializer=_init,
                             initargs=(model_path, device_str, fp16, use_compile))

    def run(self, jobs, horizon, batch=128):
        """Shard jobs across workers, return results in original order."""
        if not jobs:
            return []
        # Split into ~4x more chunks than workers for dynamic load balancing;
        # pool.map preserves input order, so concatenation restores job order.
        nchunks = min(len(jobs), self.n * 4)
        bounds = [round(i * len(jobs) / nchunks) for i in range(nchunks + 1)]
        chunks = [jobs[bounds[i]:bounds[i + 1]] for i in range(nchunks)]
        chunks = [c for c in chunks if c]
        out = self.pool.map(_chunk, [(c, horizon, batch) for c in chunks])
        return [r for chunk_res in out for r in chunk_res]

    def close(self):
        self.pool.close()
        self.pool.join()


# ── benchmark ──
def _bench():
    import time
    import json
    import argparse
    import torch
    from alphatrain.evaluate import load_model
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from scripts.batched_rollout import batched_rollout, restore, _decode

    p = argparse.ArgumentParser()
    p.add_argument('--game', default=None)
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--depth', type=int, default=30)
    p.add_argument('--n-cand', type=int, default=10)
    p.add_argument('--seeds', type=int, default=200)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args()

    import glob
    game = a.game or sorted(glob.glob('alphatrain/data/death_games/death_*.json'))[0]
    dev = torch.device('mps' if torch.backends.mps.is_available()
                       else 'cuda' if torch.cuda.is_available() else 'cpu')
    net, _ = load_model(a.model, dev, fp16=True)
    dtype = next(net.parameters()).dtype
    d = json.load(open(game))
    fr = d['frames'][len(d['frames']) - 1 - a.depth]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    g0 = restore(anchor, 0)
    obs = torch.from_numpy(_build_obs_for_game(g0)).unsqueeze(0).to(dev, dtype)
    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g0.board, logits, 64)
    cand = [_decode(m) for m, _ in sorted(pri.items(), key=lambda x: -x[1])[:a.n_cand]]
    jobs = [(anchor, c, s) for c in cand for s in range(a.seeds)]
    print(f"game={os.path.basename(game)} jobs={len(jobs)} batch={a.batch} "
          f"workers={a.workers} device={dev}\n", flush=True)

    # single process baseline
    _ = batched_rollout(net, dev, dtype, jobs[:a.batch], 30, a.batch)  # warmup
    t = time.perf_counter()
    r1 = batched_rollout(net, dev, dtype, jobs, a.horizon, a.batch)
    t_single = time.perf_counter() - t
    turns1 = sum(x['turns'] for x in r1)
    print(f"SINGLE : {t_single:5.1f}s  {turns1} turns  {turns1/t_single:.0f} turns/s",
          flush=True)

    # parallel
    t = time.perf_counter()
    pool = RolloutPool(a.model, str(dev), True, a.workers)
    t_init = time.perf_counter() - t
    t = time.perf_counter()
    rN = pool.run(jobs, a.horizon, a.batch)
    t_par = time.perf_counter() - t
    pool.close()
    turnsN = sum(x['turns'] for x in rN if x)
    print(f"PARALLEL: {t_par:5.1f}s  {turnsN} turns  {turnsN/t_par:.0f} turns/s "
          f"(+{t_init:.0f}s one-time pool init)", flush=True)
    print(f"\nSPEEDUP: {t_single/t_par:.2f}x  (workers={a.workers})", flush=True)
    # agreement check (died counts should match closely; fp16 batch noise ok)
    d1 = sum(x['died'] for x in r1)
    dN = sum(x['died'] for x in rN if x)
    print(f"died: single {d1} / parallel {dN} (close = correct)", flush=True)


if __name__ == '__main__':
    _bench()
