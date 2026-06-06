"""Golden test: legal_priors_t (GPU, vectorized) vs mcts._legal_priors_jit (numba reference).

Validates the Stage-1 keystone. Two checks per board:
  (1) full legal SET + priors (top_k=6561): the set of legal flat-actions must match exactly, and
      the softmax priors must match (same legality + same softmax => same numbers up to fp).
  (2) top_k=300 selection: same set of selected actions (top-k tie-breaks at the boundary may differ
      slightly; we report the overlap and the prior-mass agreement, the search contract is TV not bit).

    PYTHONPATH=. python scripts/test_legal_priors_gpu.py            # auto device
    PYTHONPATH=. python scripts/test_legal_priors_gpu.py cuda
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain.mcts import _legal_priors_jit
from alphatrain import batched_engine_gpu as beg


def _dev(arg):
    if arg:
        return torch.device(arg)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def _boards(K, density, seed):
    rng = np.random.default_rng(seed)
    return np.where(rng.random((K, 9, 9)) < density,
                    rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    dev = _dev(arg)
    print(f"device={dev.type}", flush=True)
    rng = np.random.default_rng(0)

    set_bad = 0; prior_maxerr = 0.0; n = 0
    top_overlap = []; top_massdiff = []
    for density in (0.3, 0.5, 0.7, 0.85):
        for seed in range(4):
            K = 96
            bnp = _boards(K, density, 1000 + seed * 7 + int(density * 100))
            logits = rng.standard_normal((K, 6561)).astype(np.float32)
            bt = torch.from_numpy(bnp.astype(np.int32)).to(dev)
            lt = torch.from_numpy(logits).to(dev)

            # (1) full legal set + priors
            cnt_g, idx_g, pri_g = beg.legal_priors_t(bt, lt, 6561)
            cnt_g = cnt_g.cpu().numpy(); idx_g = idx_g.cpu().numpy(); pri_g = pri_g.cpu().numpy()
            # (2) top-300
            cnt3, idx3, pri3 = beg.legal_priors_t(bt, lt, 300)
            cnt3 = cnt3.cpu().numpy(); idx3 = idx3.cpu().numpy(); pri3 = pri3.cpu().numpy()

            for k in range(K):
                n += 1
                rc, ri, rp = _legal_priors_jit(bnp[k], logits[k], 6561)
                rc = int(rc)
                ref = {int(i): float(p) for i, p in zip(ri, rp)}
                gv = idx_g[k][:int(cnt_g[k])]
                gset = {int(i): float(p) for i, p in zip(gv, pri_g[k][:int(cnt_g[k])])}
                if set(ref.keys()) != set(gset.keys()):
                    set_bad += 1
                else:
                    for a in ref:
                        prior_maxerr = max(prior_maxerr, abs(ref[a] - gset[a]))

                # top-300: compare selected action sets
                rc3, ri3, rp3 = _legal_priors_jit(bnp[k], logits[k], 300)
                refset = set(int(i) for i in ri3)
                gset3 = set(int(i) for i in idx3[k][:int(cnt3[k])])
                if refset:
                    top_overlap.append(len(refset & gset3) / len(refset))
                    # prior mass on the agreed actions
                    refmap = {int(i): float(p) for i, p in zip(ri3, rp3)}
                    gmap = {int(i): float(p) for i, p in zip(idx3[k][:int(cnt3[k])], pri3[k][:int(cnt3[k])])}
                    common = refset & gset3
                    top_massdiff.append(abs(sum(refmap[a] for a in common) - sum(gmap[a] for a in common)))

    print(f"boards tested: {n}", flush=True)
    print(f"(1) full legal-SET exact match: {n - set_bad}/{n}  "
          f"{'PASS' if set_bad == 0 else 'FAIL'}", flush=True)
    print(f"(1) max prior abs-error: {prior_maxerr:.2e}  "
          f"{'PASS' if prior_maxerr < 1e-5 else 'FAIL'}", flush=True)
    print(f"(2) top-300 action overlap: mean {np.mean(top_overlap)*100:.2f}%  "
          f"min {np.min(top_overlap)*100:.2f}%", flush=True)
    print(f"(2) top-300 common-mass diff: mean {np.mean(top_massdiff):.2e}  "
          f"max {np.max(top_massdiff):.2e}", flush=True)
    assert set_bad == 0 and prior_maxerr < 1e-5, "legal_priors_t mismatch vs reference"
    print("LEGAL_PRIORS_T OK", flush=True)


if __name__ == '__main__':
    main()
