"""Benchmark CPU inference with different batch sizes (single-threaded).

Usage:
    python -m alphatrain.scripts.bench_cpu_batched
"""

import time
import torch
from alphatrain.evaluate import load_model


def main():
    torch.set_num_threads(1)
    net, ms = load_model('alphatrain/data/alphatrain_td_best.pt', torch.device('cpu'))

    print("CPU batch inference (1 thread):\n", flush=True)
    for bs in [1, 4, 8, 16, 32]:
        x = torch.randn(bs, 18, 9, 9)
        # warmup
        for _ in range(5):
            with torch.no_grad():
                net(x)
        # benchmark
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.no_grad():
                net(x)
        total = (time.perf_counter() - t0) / n
        per_sample = total / bs * 1000
        sims_400 = per_sample * 400
        print(f"  bs={bs:>2}: {per_sample:.2f}ms/sample, "
              f"400 sims={sims_400:.0f}ms, "
              f"game(300t)={sims_400*300/1000/60:.1f}min", flush=True)


if __name__ == '__main__':
    main()
