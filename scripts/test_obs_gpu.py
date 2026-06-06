"""Golden test: build_observation_t (GPU) vs alphatrain.observation.build_observation (numba).

Bit-close match required on all 18 channels across densities + next-ball configs.

    PYTHONPATH=. python scripts/test_obs_gpu.py            # auto device
    PYTHONPATH=. python scripts/test_obs_gpu.py cuda
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain.observation import build_observation
from alphatrain import batched_engine_gpu as beg


def _dev(arg):
    if arg:
        return torch.device(arg)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    dev = _dev(arg)
    print(f"device={dev.type}", flush=True)
    rng = np.random.default_rng(0)

    worst = 0.0; n = 0; per_ch = np.zeros(18)
    for density in (0.2, 0.45, 0.7, 0.9):
        for seed in range(4):
            K = 64
            r = np.random.default_rng(100 + seed)
            boards = np.where(r.random((K, 9, 9)) < density,
                              r.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
            # next balls: 0..3, at random empty cells
            npos = np.zeros((K, 3, 2), dtype=np.int8)
            ncol = np.zeros((K, 3), dtype=np.int8)
            nn = np.zeros(K, dtype=np.int8)
            for k in range(K):
                e = np.argwhere(boards[k] == 0)
                m = min(3, len(e))
                nn[k] = m
                if m:
                    pick = e[r.integers(0, len(e), m)]
                    npos[k, :m] = pick
                    ncol[k, :m] = r.integers(1, 8, m)
            # reference
            ref = np.stack([build_observation(boards[k], npos[k, :, 0].astype(np.intp),
                            npos[k, :, 1].astype(np.intp), ncol[k].astype(np.intp), int(nn[k]))
                            for k in range(K)])
            # gpu
            got = beg.build_observation_t(
                torch.from_numpy(boards.astype(np.int32)).to(dev),
                torch.from_numpy(npos.astype(np.int64)).to(dev),
                torch.from_numpy(ncol.astype(np.int64)).to(dev),
                torch.from_numpy(nn.astype(np.int64)).to(dev)).cpu().numpy()
            d = np.abs(ref - got)
            worst = max(worst, d.max())
            per_ch = np.maximum(per_ch, d.reshape(K, 18, -1).max(axis=(0, 2)))
            n += K

    print(f"boards tested: {n}", flush=True)
    print(f"max abs error (all channels): {worst:.2e}", flush=True)
    bad = [int(c) for c in np.where(per_ch > 1e-5)[0]]
    if bad:
        print(f"  channels exceeding 1e-5: {bad}  (errors: {per_ch[bad]})", flush=True)
    assert worst < 1e-5, f"build_observation_t mismatch (max err {worst:.2e}, channels {bad})"
    print("BUILD_OBSERVATION_T OK", flush=True)


if __name__ == '__main__':
    main()
