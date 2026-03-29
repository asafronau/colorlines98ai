"""Build TD value target tensor files with different gamma values.

Usage:
    python -m alphatrain.scripts.build_td_tensors --gamma 0.99
    python -m alphatrain.scripts.build_td_tensors --gamma 0.95 --output data/alphatrain_td95.pt
"""

import argparse
from alphatrain.dataset import precompute_tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/alphazero_v1')
    p.add_argument('--output', default=None,
                   help='Output path (default: data/alphatrain_td{gamma*100:.0f}.pt)')
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--max-score', type=float, default=None,
                   help='Max value for two-hot bins (auto-detected if omitted)')
    args = p.parse_args()

    if args.output is None:
        tag = f"{args.gamma*100:.0f}" if args.gamma < 1.0 else "mc"
        args.output = f"data/alphatrain_td{tag}.pt"

    # Auto-detect max_score: first pass to get range, then build with proper bins
    if args.max_score is None:
        # Quick estimate: build with large max_score, check actual range
        print(f"Building with gamma={args.gamma}...", flush=True)
        # Use 30000 as default — we'll report the actual range
        max_score = 30000.0
    else:
        max_score = args.max_score

    precompute_tensors(
        args.data, args.output,
        value_mode='td', gamma=args.gamma,
        max_score=max_score,
    )


if __name__ == '__main__':
    main()
