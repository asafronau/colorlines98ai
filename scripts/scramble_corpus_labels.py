"""Scrambled-label control corpus (Gemini diagnostic, run in the task-arithmetic channel).

Same correction STATES, targets+weights randomly permuted between them (fixed seed). Fine-tune
θ_scrambled on this, merge at the plateau α, eval: collapse to ~base ⇒ the correction CONTENT
drives the task-arithmetic gains (the +9k median is knowledge, not perturbation); gains anyway ⇒
the base is stuck in argmax ruts and any strong OOD update helps — stop paying 4s/state for MCTS.

    PYTHONPATH=. python scripts/scramble_corpus_labels.py \\
        --in crisis/corrections_corpus_mm05.pt --out crisis/corrections_corpus_scrambled.pt
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='inp', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    c = torch.load(a.inp, map_location='cpu', weights_only=False)
    N = c['boards'].size(0)
    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(N, generator=g)
    # Permute the LABEL side (targets + weight) against the states. Seeds stay
    # with the STATES so the by-seed holdout split keeps grouping board states
    # from the same game (the split must mirror the real corpus's).
    for k in ('tgt_idx', 'tgt_prob', 'weight'):
        c[k] = c[k][perm]
    st = dict(c['_stats'])
    st['scrambled_seed'] = a.seed
    c['_stats'] = st
    torch.save(c, a.out)
    frac_self = float((torch.arange(N) == perm).float().mean())
    print(f"Wrote {a.out}: {N} states, labels permuted "
          f"(fixed points: {100*frac_self:.2f}%)")


if __name__ == '__main__':
    main()
