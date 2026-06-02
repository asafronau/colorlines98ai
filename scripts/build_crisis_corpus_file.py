"""Build the crisis-fork aux corpus ONCE and save it to a single .pt file.

Run locally (where logs/mine_*.json and the death games live), then upload the
small output .pt to Drive for the Colab trainer (train_path_b --aux-crisis-corpus
<file>). This avoids shipping the whole mine/death JSON tree to Colab and makes
the policy-move recovery (which reads death_games/) a local, one-time step.

    PYTHONPATH=. python scripts/build_crisis_corpus_file.py \\
        --out alphatrain/data/crisis_corpus_v1.pt
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from alphatrain.counterfactual import build_crisis_corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mine-glob', default='logs/mine_*.json')
    p.add_argument('--death-dir', default='alphatrain/data/death_games')
    p.add_argument('--clean-loser-margin', type=float, default=10.0)
    p.add_argument('--min-gap', type=float, default=0.0)
    p.add_argument('--out', default='alphatrain/data/crisis_corpus_v1.pt')
    a = p.parse_args()

    corpus = build_crisis_corpus(
        a.mine_glob, death_dir=a.death_dir, device='cpu',
        clean_loser_margin=a.clean_loser_margin, min_gap=a.min_gap)
    cs = corpus['_stats']
    # weights are normalized to mean 1 over the FULL corpus here; train_path_b
    # re-normalizes over the train split after the held-out seeds are removed.
    torch.save(corpus, a.out)
    sz = os.path.getsize(a.out) / 1e6
    print(f"Wrote {a.out} ({sz:.2f} MB)")
    print(f"  {cs['n_anchors']} confirmed forks from {cs['n_seeds']} seeds / "
          f"{cs['n_files']} games")
    print(f"  {cs['n_clean_pairs']} clean-loser pairs; dropped "
          f"{cs['n_unconfirmed']} unconfirmed, {cs['n_degenerate']} degenerate")
    print(f"  clean_loser_margin={a.clean_loser_margin}pp  min_gap={a.min_gap}pp")


if __name__ == '__main__':
    main()
