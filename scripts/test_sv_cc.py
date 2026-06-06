"""Validate label_components_sv (Shiloach-Vishkin O(log) CC) vs numpy reference, find the
minimum iters for full partition correctness at large K, and time it vs the 45-iter pj version.

    PYTHONPATH=. python scripts/test_sv_cc.py            # auto device
    PYTHONPATH=. python scripts/test_sv_cc.py cuda 1024,2048,4096
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain import batched_engine as be
from alphatrain import batched_engine_gpu as beg


def _dev(arg):
    if arg:
        return torch.device(arg)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def _partition_match(lab_t, ref, boards_np):
    """Fraction of trees whose empty-cell same-component relation matches the numpy reference."""
    lp = lab_t.cpu().numpy()
    K = boards_np.shape[0]
    bad = 0
    for k in range(K):
        e = boards_np[k] == 0
        a = ref[k][e]; b = lp[k][e]
        if not np.array_equal((a[:, None] == a[None, :]), (b[:, None] == b[None, :])):
            bad += 1
    return K - bad


def _boards(K, density, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((K, 9, 9)) < density,
                    rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    ks = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [512, 2048, 4096]
    dev = _dev(arg)
    print(f"device={dev.type}", flush=True)

    # Convergence: across a few densities + seeds (snake-y low-density boards are worst-case),
    # find the smallest iters giving 100% partition match at the largest K.
    print("\n--- SV convergence (min iters for 100% partition match) ---", flush=True)
    for K in ks:
        worst_needed = 0
        for density in (0.3, 0.5, 0.7):
            for seed in range(3):
                bnp = _boards(K, density, 1000 + seed)
                ref = be.label_components(bnp)
                bt = torch.from_numpy(bnp.astype(np.int32)).to(dev)
                need = None
                for it in range(1, 16):
                    ok = _partition_match(beg.label_components_sv(bt, it), ref, bnp)
                    if ok == K:
                        need = it; break
                worst_needed = max(worst_needed, need if need is not None else 99)
        flag = "OK" if worst_needed <= 12 else "HIGH"
        print(f"  K={K:5d}: needs {worst_needed} iters [{flag}]", flush=True)

    # Timing: SV@8 vs pj@45 on K-sweep (the descent-step label cost).
    print("\n--- timing: SV@8 vs pj@45 (label only) ---", flush=True)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else \
           (lambda: torch.mps.synchronize()) if dev.type == 'mps' else (lambda: None)

    def timeit(fn, n=300):
        for _ in range(8):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return (time.perf_counter() - t0) * 1e3 / n

    for K in ks:
        bnp = _boards(K, 0.5, 7)
        bt = torch.from_numpy(bnp.astype(np.int32)).to(dev)
        t_sv = timeit(lambda: beg.label_components_sv(bt, 8))
        t_pj = timeit(lambda: beg.label_components_pj(bt, 45))
        print(f"  K={K:5d}: SV@8 {t_sv:6.3f}ms | pj@45 {t_pj:6.3f}ms | {t_pj/t_sv:.2f}x", flush=True)


if __name__ == '__main__':
    main()
