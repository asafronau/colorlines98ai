"""CUDA-Graph capture feasibility spike (ChatGPT-recommended, time-boxed).

Question: the descent is eager-dispatch-bound (~46k tiny ops/sim, GPU ~13% utilized). Does
CUDA-graph replay cut that dispatch wall? We capture the SYNC-FREE engine bundle that is the bulk
of a descent step — connected-components (label_components_sv) + reachability + the 4 line-clears
(the GPU cost inside apply_move) + a PUCT-style argmax — and compare eager vs graph-replay over N
iterations. (apply_move_t itself has internal bool(.any()) syncs that break capture; the spawn
logic is excluded — the clears are its dominant GPU cost and ARE capturable.)

Decision rule (ChatGPT): graph replay drops near GPU/CUDA time -> port feature_value + go open-loop
graph. No material help / brittle -> pivot to closed-loop node-board caching + W_internal.

    PYTHONPATH=. python scripts/bench_cuda_graph.py            # auto device (real test = cuda/L4)
    PYTHONPATH=. python scripts/bench_cuda_graph.py --k 256 --iters 400
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain import batched_engine_gpu as beg

BOARD, W = 9, 300


def _device(arg):
    if arg:
        return torch.device(arg)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def _make_step(boards, src, tgt, rows, cols, prior, visits, vsum, nodevis, out_best):
    """One descent-step's worth of sync-free engine ops, writing argmax into out_best (in place)."""
    def step():
        labels = beg.label_components_sv(boards, iters=8)
        reach = beg.reachable_many_t(labels, src, tgt)                  # [K,W]
        for j in range(4):                                             # move-clear + 3 spawn-clears
            beg.clear_lines_at_t(boards, rows[j], cols[j])
        # PUCT-style score over W children (all persistent buffers)
        q = vsum / visits.clamp(min=1.0)
        sqrt_n = torch.sqrt(nodevis).unsqueeze(1)
        u = 2.5 * prior * sqrt_n / (1.0 + visits)
        score = torch.where(reach, 2.0 * q + u, torch.full_like(q, -1e30))
        out_best.copy_(score.argmax(dim=1))
    return step


PROBE_NAMES = ['label_cc', 'build_obs', 'net_forward', 'legal_priors', 'apply_move_det',
               'scatter_add', 'topk']


def _probe_one(dev, K, name):
    """Capture ONE component in this (isolated) process. Prints OK/FAILED. A CUDA capture failure
    corrupts the context, so each component must run in its own process (see _probe)."""
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = False
    from alphatrain.model import PolicyNet
    from alphatrain import batched_engine_gpu as beg
    rng = np.random.default_rng(0)
    bnp = np.where(rng.random((K, 9, 9)) < 0.55, rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
    boards = torch.from_numpy(bnp.astype(np.int64)).to(dev)
    npos = torch.zeros((K, 3, 2), dtype=torch.int64, device=dev)
    ncol = torch.ones((K, 3), dtype=torch.int64, device=dev)
    nn = torch.full((K,), 3, dtype=torch.int64, device=dev)
    src = torch.zeros((K, 2), dtype=torch.int64, device=dev)
    tgt = torch.zeros((K, 2), dtype=torch.int64, device=dev)
    if name in ('net_forward', 'legal_priors'):
        net = PolicyNet(in_channels=18, num_blocks=10, channels=256).to(dev).half(); net.train(False)
        obs = beg.build_observation_t(boards, npos, ncol, nn).half()
        logits = net(obs).float()
    tree = torch.zeros((K, 200 * 300), device=dev)
    idxb = torch.randint(0, 200 * 300, (K, 64), device=dev); valb = torch.ones((K, 64), device=dev)
    big = torch.rand((K, 6561), device=dev)
    fns = {
        'label_cc':       lambda: beg.label_components_sv(boards, iters=8),
        'build_obs':      lambda: beg.build_observation_t(boards, npos, ncol, nn),
        'net_forward':    lambda: net(beg.build_observation_t(boards, npos, ncol, nn).half()),
        'legal_priors':   lambda: beg.legal_priors_t(boards, logits, 300),
        'apply_move_det': lambda: beg.apply_move_nosync_t(boards.clone(), npos, ncol, nn, src, tgt,
                                                          deterministic=True),
        'scatter_add':    lambda: tree.scatter_add_(1, idxb, valb),
        'topk':           lambda: big.topk(300, dim=1),
    }
    fn = fns[name]
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay(); torch.cuda.synchronize()
    print(f"  {name:16s} CAPTURE OK", flush=True)


def _probe(dev, K):
    """Run each component's capture in its OWN subprocess (isolation)."""
    import subprocess
    self = os.path.abspath(__file__)
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.dirname(self)),
               CUDA_LAUNCH_BLOCKING='1')
    for name in PROBE_NAMES:
        r = subprocess.run([sys.executable, self, '--probe-one', name, '--k', str(K),
                            '--device', dev.type], capture_output=True, text=True, env=env)
        out = (r.stdout + r.stderr).strip().splitlines()
        ok = any('CAPTURE OK' in l for l in out)
        if ok:
            print(f"  {name:16s} CAPTURE OK", flush=True)
        else:
            err = [l for l in out if ('Error' in l or 'error' in l or 'FAILED' in l)]
            print(f"  {name:16s} FAILED: {(err[-1] if err else out[-1] if out else '?')[:140]}",
                  flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default=None)
    p.add_argument('--k', type=int, default=256)
    p.add_argument('--iters', type=int, default=400)
    p.add_argument('--probe', action='store_true', help='isolate which component breaks capture')
    p.add_argument('--probe-one', default=None, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.probe_one:
        _probe_one(_device(a.device), a.k, a.probe_one)
        return
    if a.probe:
        dev = _device(a.device)
        print(f"device={dev.type} K={a.k} -- per-component capture probe (each in own process)",
              flush=True)
        if dev.type != 'cuda':
            print("  (capture only on cuda)", flush=True); return
        _probe(dev, a.k)
        return
    dev = _device(a.device)
    K = a.k
    print(f"device={dev.type} K={K} iters={a.iters} W={W}", flush=True)
    if dev.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    rng = np.random.default_rng(0)
    bnp = np.where(rng.random((K, 9, 9)) < 0.55, rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
    boards = torch.from_numpy(bnp.astype(np.int32)).to(dev)
    # persistent buffers
    src = torch.randint(0, 9, (K, W, 2), device=dev)
    tgt = torch.randint(0, 9, (K, W, 2), device=dev)
    rows = [torch.randint(0, 9, (K,), device=dev) for _ in range(4)]
    cols = [torch.randint(0, 9, (K,), device=dev) for _ in range(4)]
    prior = torch.rand(K, W, device=dev)
    visits = torch.randint(0, 5, (K, W), device=dev).float()
    vsum = torch.rand(K, W, device=dev)
    nodevis = torch.randint(1, 50, (K,), device=dev).float()
    out_best = torch.zeros(K, dtype=torch.int64, device=dev)
    step = _make_step(boards, src, tgt, rows, cols, prior, visits, vsum, nodevis, out_best)

    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else \
           (lambda: torch.mps.synchronize()) if dev.type == 'mps' else (lambda: None)

    # eager timing
    for _ in range(5):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        step()
    sync()
    eager = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"\neager step:  {eager:6.3f} ms/step", flush=True)

    if dev.type != 'cuda':
        print("  (CUDA-graph capture only on cuda — eager path validated; run on L4 for the verdict)",
              flush=True)
        return

    # CUDA-graph capture
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
    except Exception as e:
        print(f"  CAPTURE FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        print("  => full-sim open-loop graph likely brittle; lean closed-loop. (see decision rule)",
              flush=True)
        return

    for _ in range(5):
        g.replay()
    sync()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        g.replay()
    sync()
    graph = (time.perf_counter() - t0) / a.iters * 1e3
    print(f"graph replay: {graph:6.3f} ms/step   => {eager/graph:.1f}x vs eager", flush=True)
    print(f"\nVERDICT: graph capture {'WORKS' if eager/graph > 2 else 'does NOT materially help'} "
          f"on the engine bundle.", flush=True)
    print(f"  A real sim runs ~30-60 of these steps; if the {eager/graph:.1f}x holds for the full "
          f"per-sim capture, the ~20.8s K=256/sims=100 wall would drop toward the ~2.7s GPU floor.",
          flush=True)
    print("  -> if WORKS: port board_features_with_next to GPU + capture the full sim (open-loop).",
          flush=True)
    print("  -> if not:   pivot to closed-loop node-board caching + W_internal=64.", flush=True)


if __name__ == '__main__':
    main()
