"""L4/CUDA throughput benchmark for the FULL batched_search_gpu (lever #3 — the real proof).

Self-contained: uses a RANDOMLY-INITIALIZED net of the production architecture (throughput is
architecture-bound, not weight-bound; and batched-vs-scalar argmax agreement holds for any fixed
net), so no model checkpoint or crisis JSONs are needed. Synthetic crisis-like band states.

Headline number = trees/sec for the full search at the mining sims count, compared to the live
M5 scalar miner (16 workers ~= 3.56 trees/s). Prints each K as soon as it finishes (no blind wait).

    PYTHONPATH=. python scripts/bench_full_search.py                 # auto device, sims=4800
    PYTHONPATH=. python scripts/bench_full_search.py --sims 1600 --ks 256,512,1024
    PYTHONPATH=. python scripts/bench_full_search.py --scalar         # also time scalar (slow)
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

M5_SCALAR_TREES_PER_S = 3.56   # live miner: 16 parallel scalar workers on the M5
FV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'alphatrain', 'data', 'feature_value_weights_2y_nb.npz')


def _device(arg):
    if arg:
        return torch.device(arg)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def _make_net(dev, dtype):
    from alphatrain.model import PolicyNet
    net = PolicyNet(in_channels=18, num_blocks=10, channels=256).to(dev)
    net.train(False)                                        # inference mode (== .eval())
    if dtype == torch.float16:
        net = net.half()
    # BN running stats default to (0,1) — fine: timing is weight-independent and both batched &
    # scalar use the SAME net, so argmax agreement still validates the descent port.
    return net


def _synth_states(K, density, seed):
    """K crisis-like band states: ~density-full boards + 3 next balls at random empty cells."""
    rng = np.random.default_rng(seed)
    boards = np.where(rng.random((K, 9, 9)) < density,
                      rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
    npos = np.zeros((K, 3, 2), dtype=np.int8)
    ncol = np.zeros((K, 3), dtype=np.int8)
    nn = np.full(K, 3, dtype=np.int8)
    for k in range(K):
        empt = np.argwhere(boards[k] == 0)
        if len(empt) < 4:                                   # guarantee a movable + 3 spawn cells
            boards[k].flat[rng.choice(81, 6, replace=False)] = 0
            empt = np.argwhere(boards[k] == 0)
        pick = empt[rng.choice(len(empt), 3, replace=False)]
        npos[k] = pick
        ncol[k] = rng.integers(1, 8, 3)
    return boards, npos, ncol, nn


def _load_fv():
    d = np.load(FV_PATH)
    return (d['coefs'].astype(np.float32), d['means'].astype(np.float32),
            d['stds'].astype(np.float32), float(d['bias']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default=None)
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--top-k', type=int, default=300)
    p.add_argument('--ks', default='128,256,512,1024')
    p.add_argument('--density', type=float, default=0.62)
    p.add_argument('--ablate', default=None, choices=[None, 'descent_only', 'no_net'],
                   help='isolate cost: descent_only (no leaf/sync) or no_net (skip forward)')
    p.add_argument('--nn-bench', action='store_true', help='isolated NN-forward timing at each K')
    p.add_argument('--profile', action='store_true', help='torch.profiler trace (short, K=ks[0])')
    p.add_argument('--scalar', action='store_true', help='also time scalar MCTS (slow on Colab CPU)')
    p.add_argument('--graph', action='store_true',
                   help='full-sim CUDA-graph spike: eager vs captured wall (fake value)')
    p.add_argument('--block', action='store_true',
                   help='block-capture spike: sweep block sizes, captured descent block + early-exit')
    p.add_argument('--block-sizes', default='4,8,16')
    p.add_argument('--max-depth', type=int, default=64)
    a = p.parse_args()

    dev = _device(a.device)
    dtype = torch.float16 if dev.type != 'cpu' else torch.float32
    ks = [int(x) for x in a.ks.split(',')]
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else \
           (lambda: torch.mps.synchronize()) if dev.type == 'mps' else (lambda: None)
    print(f"device={dev.type} dtype={dtype} sims={a.sims} top_k={a.top_k} ks={ks}", flush=True)
    if dev.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB", flush=True)

    from alphatrain.batched_mcts_gpu import batched_search_gpu
    net = _make_net(dev, dtype)
    fv = _load_fv()

    if a.block:
        from alphatrain.batched_mcts_gpu import batched_search_gpu_block
        bss = [int(x) for x in a.block_sizes.split(',')]
        print(f"--- block-capture spike (FAKE value), max_depth={a.max_depth}, sims-honest vs M5 "
              f"(@sims4800) ---", flush=True)
        for K in ks:
            b, pp, c, n = _synth_states(K, a.density, K)
            for B in bss:
                args = (net, dev, dtype, b, pp, c, n)
                kw = dict(sims=a.sims, top_k=a.top_k, max_depth=a.max_depth, block_size=B)
                batched_search_gpu_block(*args, use_graph=(dev.type == 'cuda'),
                                         **dict(kw, sims=4)); sync()       # warm/capture
                try:
                    t0 = time.perf_counter()
                    batched_search_gpu_block(*args, use_graph=(dev.type == 'cuda'), **kw); sync()
                    wall = time.perf_counter() - t0
                except RuntimeError as e:
                    print(f"  K={K:>5} B={B:>2}: FAILED: {str(e)[:80]}", flush=True); continue
                tps = K / wall
                tps4800 = tps * (a.sims / 4800.0)
                print(f"  K={K:>5} B={B:>2}: {wall:6.1f}s | @sims{a.sims} {tps:6.2f} tr/s | "
                      f"@sims4800 {tps4800:5.2f} = {tps4800/M5_SCALAR_TREES_PER_S:.2f}x M5", flush=True)
        return

    if a.graph:
        from alphatrain.batched_mcts_gpu import batched_search_gpu_graph
        print("--- full-sim CUDA-graph spike (FAKE value): eager vs captured wall ---", flush=True)
        for K in ks:
            b, pp, c, n = _synth_states(K, a.density, K)
            args = (net, dev, dtype, b, pp, c, n)
            kw = dict(sims=a.sims, top_k=a.top_k)
            batched_search_gpu_graph(*args, use_graph=False, **dict(kw, sims=4)); sync()  # warm numba/kernels
            t0 = time.perf_counter()
            batched_search_gpu_graph(*args, use_graph=False, **kw); sync()
            eager = time.perf_counter() - t0
            try:
                t0 = time.perf_counter()
                batched_search_gpu_graph(*args, use_graph=True, **kw); sync()
                graph = time.perf_counter() - t0
                sp = f"{eager/graph:.1f}x" if graph > 0 else "-"
                # sims-HONEST: M5's 3.56 trees/s is at the mining sims=4800; scale this run's
                # trees/s to 4800 before comparing (wall is ~linear in sims).
                tps4800 = (K / graph) * (a.sims / 4800.0)
                print(f"  K={K:>5}: eager {eager:6.1f}s | graph {graph:6.1f}s | {sp} vs eager | "
                      f"graph@sims{a.sims} {K/graph:6.2f} tr/s | @sims4800 {tps4800:5.2f} tr/s = "
                      f"{tps4800/M5_SCALAR_TREES_PER_S:.2f}x M5", flush=True)
            except RuntimeError as e:
                print(f"  K={K:>5}: eager {eager:6.1f}s | CAPTURE FAILED: {str(e)[:90]}", flush=True)
        return

    # isolated NN-forward timing: full-search NN cost = sims x this (a direct measurement, not <1s by assertion)
    if a.nn_bench:
        from alphatrain.observation import build_observation
        print("--- isolated NN forward (one batched fwd; full search does `sims` of these) ---",
              flush=True)
        for K in ks:
            b, pp, c, n = _synth_states(K, a.density, K)
            obs = np.stack([build_observation(b[k], pp[k, :, 0].astype(np.intp),
                            pp[k, :, 1].astype(np.intp), c[k].astype(np.intp), int(n[k]))
                            for k in range(K)])
            ot = torch.from_numpy(obs).to(dev, dtype)
            with torch.no_grad():
                for _ in range(5):
                    net(ot).float().cpu()
                sync(); t0 = time.perf_counter()
                for _ in range(20):
                    net(ot).float().cpu()
                sync()
            ms = (time.perf_counter() - t0) * 1e3 / 20
            print(f"  K={K:>5}: {ms:6.2f} ms/fwd  -> x{a.sims} sims = {ms*a.sims/1e3:6.1f}s of "
                  f"pure forward", flush=True)
        print(flush=True)

    # warmup (compile kernels / allocate)
    print("warmup (K=32, sims=64) ...", flush=True)
    wb, wp, wc, wn = _synth_states(32, a.density, 0)
    t0 = time.perf_counter()
    batched_search_gpu(net, dev, dtype, wb, wp, wc, wn, fv, sims=64, top_k=a.top_k, ablate=a.ablate)
    sync()
    print(f"  warmup done in {time.perf_counter()-t0:.1f}s\n", flush=True)

    if a.profile:
        K = ks[0]
        psims = 25                       # few sims: the profiler buffers EVERY op in host RAM
        b, pp, c, n = _synth_states(K, a.density, 1)
        print(f"--- torch.profiler: K={K} sims={psims} ablate={a.ablate} ---", flush=True)
        from torch.profiler import profile, ProfilerActivity
        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == 'cuda' else [])
        with profile(activities=acts) as prof:
            batched_search_gpu(net, dev, dtype, b, pp, c, n, fv, sims=psims, top_k=a.top_k,
                               ablate=a.ablate)
            sync()
        key = 'cuda_time_total' if dev.type == 'cuda' else 'cpu_time_total'
        print(prof.key_averages().table(sort_by=key, row_limit=18), flush=True)
        evs = prof.key_averages()
        print(f"  total distinct kernels/ops: {len(evs)};  total calls: "
              f"{sum(e.count for e in evs)} over {psims} sims = {sum(e.count for e in evs)/psims:.0f}/sim",
              flush=True)
        print(flush=True)

    lbl = f" [ablate={a.ablate}]" if a.ablate else ""
    print(f"{'K':>6} {'wall(s)':>9} {'trees/s':>9} {'vs M5 16w':>10}  {'peakGB':>7}{lbl}", flush=True)
    results = []
    for K in ks:
        b, pp, c, n = _synth_states(K, a.density, 100 + K)
        if dev.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        try:
            t0 = time.perf_counter()
            batched_search_gpu(net, dev, dtype, b, pp, c, n, fv, sims=a.sims, top_k=a.top_k,
                               ablate=a.ablate)
            sync()
            wall = time.perf_counter() - t0
        except RuntimeError as e:
            print(f"{K:>6}  OOM/err: {str(e)[:60]}", flush=True)
            continue
        tps = K / wall
        peak = (torch.cuda.max_memory_allocated() / 1e9) if dev.type == 'cuda' else float('nan')
        results.append((K, wall, tps))
        print(f"{K:>6} {wall:>9.1f} {tps:>9.2f} {tps/M5_SCALAR_TREES_PER_S:>9.2f}x {peak:>7.1f}",
              flush=True)

    if results:
        best = max(results, key=lambda r: r[2])
        print(f"\nBEST: K={best[0]}  {best[2]:.2f} trees/s = "
              f"{best[2]/M5_SCALAR_TREES_PER_S:.2f}x the M5 16-worker miner "
              f"({M5_SCALAR_TREES_PER_S} trees/s).", flush=True)
        print("  (One L4 process vs the whole M5 mining fleet. >1x => GPU mining is the win.)",
              flush=True)

    if a.scalar:
        print("\n--- scalar MCTS reference (n=2, same net) ---", flush=True)
        from alphatrain.mcts import MCTS
        from game.board import ColorLinesGame
        mcts = MCTS(net=net, device=dev, num_simulations=a.sims, c_puct=2.5,
                    top_k=a.top_k, batch_size=8, feature_weights_path=FV_PATH, q_weight=2.0)
        b, pp, c, n = _synth_states(2, a.density, 7)
        t0 = time.perf_counter()
        for k in range(2):
            g = ColorLinesGame()
            g.reset(board=b[k].copy(),
                    next_balls=[((int(pp[k, i, 0]), int(pp[k, i, 1])), int(c[k, i]))
                                for i in range(int(n[k]))])
            g.score, g.turns = 0, 100
            mcts.search(g, temperature=0.0)
        ts = (time.perf_counter() - t0) / 2
        print(f"  scalar: {ts:.1f}s/tree on this CPU  ({1/ts:.2f} trees/s single-thread)",
              flush=True)
        print("  (Colab CPU is weak; the meaningful scalar baseline is the M5 fleet above.)",
              flush=True)


if __name__ == '__main__':
    main()
