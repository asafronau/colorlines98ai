"""Build pairwise afterstate tensor file for ranked value training.

Usage:
    python -m alphatrain.scripts.build_pairwise_tensors
    python -m alphatrain.scripts.build_pairwise_tensors --gamma 0.99 --max-score 500
"""

import argparse
from alphatrain.dataset import precompute_pairwise_tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/alphazero_v1')
    p.add_argument('--output', default='data/alphatrain_pairwise.pt')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--max-score', type=float, default=500.0)
    args = p.parse_args()

    precompute_pairwise_tensors(
        args.data, args.output,
        gamma=args.gamma, max_score=args.max_score,
    )


if __name__ == '__main__':
    main()
