"""Random subsample of a corrections corpus — the E1 size-control corpus.

E1 tests SIZE vs COMPOSITION: a random 13.8k subsample of the 27.6k corpus has the
original corpus's size but the full 3,676-game diversity. Trained at λ=0.01 with the
mC recipe: ≈ bar (16,738) ⇒ the mC27k regression is purely size/dilution; regresses
⇒ deeper problem. Deterministic (fixed generator seed). Weights renormalized to
mean 1 over the subset (λ keeps its meaning).

    PYTHONPATH=. python scripts/subsample_corpus.py \\
        --in crisis/corrections_corpus_mm05.pt --n 13800 \\
        --out crisis/corrections_corpus_sub13k.pt
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    c = torch.load(a.inp, map_location='cpu', weights_only=False)
    N = c['boards'].size(0)
    assert a.n < N, f"subsample {a.n} >= corpus {N}"
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(N, generator=g)[:a.n]
    out = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in c.items()}
    out['weight'] = out['weight'] / out['weight'].mean().clamp(min=1e-6)
    st = dict(c['_stats'])
    st.update({'n_corrections': a.n, 'subsample_of': st.get('n_corrections', N),
               'subsample_seed': a.seed,
               'n_seeds': len(set(out['seed'].tolist()))})
    out['_stats'] = st
    torch.save(out, a.out)
    print(f"Wrote {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB): "
          f"{a.n}/{N} corrections, {st['n_seeds']} seeds, seed={a.seed}")


if __name__ == '__main__':
    main()
