"""Profile the single-search MCTS hot path for crisis mining.

Mining (gen_corrections_parallel) is CPU-bound: each state runs deep WIDENED
MCTS@4800 (top_k=300), and the PUCT descent loops over every child in pure
Python — so the root's ~300 children are scanned on every one of 4800 sims.
This isolates that cost from the parallel/server/IPC layer so we can see, on
THIS machine, (a) where the CPU goes and (b) how cost scales with top_k and
batch_size. Run on M5 and on the L4 Colab and compare to pinpoint the gap.

    PYTHONPATH=. python scripts/profile_mining.py --device mps
"""
import os, sys, time, json, glob, argparse, cProfile, pstats, io, multiprocessing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'
FV = 'alphatrain/data/feature_value_weights_2y_nb.npz'


def _load_state(death_glob, depth):
    f = sorted(glob.glob(death_glob))[0]
    g = json.load(open(f))
    nf = len(g['frames'])
    i = (nf - 1) - depth
    fr = g['frames'][max(0, i)]
    return f, fr


def _make_game(fr):
    from game.board import ColorLinesGame
    game = ColorLinesGame()
    game.reset(board=np.array(fr['board'], dtype=np.int8),
               next_balls=[(tuple(p), int(c)) for p, c in fr['next_balls']])
    game.score, game.turns = 0, int(fr['turn'])
    return game


def _make_mcts(net, dev, sims, top_k, batch_size):
    from alphatrain.mcts import MCTS
    return MCTS(net=net, device=dev, num_simulations=sims, c_puct=2.5,
                top_k=top_k, batch_size=batch_size, feature_weights_path=FV,
                q_weight=2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=MODEL)
    p.add_argument('--death-glob', default='crisis/death_games/death_*.json')
    p.add_argument('--depth', type=int, default=50, help='depth-from-death (band) state')
    p.add_argument('--sims', type=int, default=4800)
    p.add_argument('--top-k', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', default='mps')
    a = p.parse_args()

    from alphatrain.evaluate import load_model
    dev = torch.device(a.device)
    print(f"=== env ===", flush=True)
    print(f"  cpu_count={multiprocessing.cpu_count()}  torch_threads={torch.get_num_threads()}"
          f"  device={a.device}", flush=True)
    try:
        import numba
        print(f"  numba={numba.__version__}", flush=True)
    except Exception as e:
        print(f"  numba MISSING: {e}", flush=True)

    net, _ = load_model(a.model, dev, fp16=(dev.type != 'cpu'))
    f, fr = _load_state(a.death_glob, a.depth)
    print(f"  state: {os.path.basename(f)} depth-{a.depth} turn={fr['turn']} "
          f"empties={int((np.array(fr['board'])==0).sum())}", flush=True)

    # Warm up numba JIT (first call compiles; don't time it).
    warm = _make_mcts(net, dev, 64, a.top_k, a.batch_size)
    warm.search(_make_game(fr), temperature=0.0)
    print("  (numba warmed)\n", flush=True)

    # === cProfile one full search at the mining config ===
    mcts = _make_mcts(net, dev, a.sims, a.top_k, a.batch_size)
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    mcts.search(_make_game(fr), temperature=0.0, dirichlet_alpha=0.3,
                dirichlet_weight=0.25, return_policy=True)
    pr.disable()
    dt = time.perf_counter() - t0
    print(f"=== search@{a.sims} top_k={a.top_k} bs={a.batch_size}: "
          f"{dt*1000:.0f}ms  ({a.sims/dt:.0f} sims/s) ===", flush=True)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14)
    print('\n'.join(l for l in s.getvalue().splitlines()
                     if l.strip() and 'function calls' not in l)[:2400], flush=True)

    # === cost curve: how top_k and batch_size drive wall time ===
    print(f"\n=== cost curve (wall ms / search, ~{3} reps) ===", flush=True)
    print(f"  {'top_k':>6} {'bs':>4} {'ms':>7} {'sims/s':>8}", flush=True)
    for tk in (a.top_k, 100, 50):
        for bs in (a.batch_size, 24):
            m = _make_mcts(net, dev, a.sims, tk, bs)
            best = 1e30
            for _ in range(3):
                t0 = time.perf_counter()
                m.search(_make_game(fr), temperature=0.0, dirichlet_alpha=0.3,
                         dirichlet_weight=0.25, return_policy=True)
                best = min(best, time.perf_counter() - t0)
            print(f"  {tk:>6} {bs:>4} {best*1000:>7.0f} {a.sims/best:>8.0f}", flush=True)


if __name__ == '__main__':
    main()
