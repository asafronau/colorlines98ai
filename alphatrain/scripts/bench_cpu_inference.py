"""Benchmark CPU inference speed for parallel MCTS planning.

Usage:
    python -m alphatrain.scripts.bench_cpu_inference
"""

import time
import torch
from alphatrain.evaluate import load_model


def main():
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    net, ms = load_model(model_path, torch.device('cpu'))
    x = torch.randn(1, 18, 9, 9)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            net(x)

    # Benchmark
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.no_grad():
            net(x)
    elapsed_ms = (time.perf_counter() - t0) / n * 1000

    print(f"CPU inference: {elapsed_ms:.1f}ms/eval", flush=True)
    print(f"MCTS move (400 sims): {elapsed_ms * 400 / 1000:.1f}s", flush=True)
    print(f"Game (300 turns): {elapsed_ms * 400 * 300 / 1000 / 60:.1f} min", flush=True)
    print(f"With 18 workers parallel: {elapsed_ms * 400 * 300 / 1000 / 60 / 18:.1f} min effective",
          flush=True)

    # MPS comparison
    if torch.backends.mps.is_available():
        net_mps, _ = load_model(model_path, torch.device('mps'))
        x_mps = x.to('mps')
        for _ in range(10):
            with torch.no_grad():
                net_mps(x_mps)
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.no_grad():
                net_mps(x_mps)
        mps_ms = (time.perf_counter() - t0) / n * 1000
        print(f"\nMPS inference: {mps_ms:.1f}ms/eval ({elapsed_ms/mps_ms:.1f}x slower on CPU)",
              flush=True)


if __name__ == '__main__':
    main()
