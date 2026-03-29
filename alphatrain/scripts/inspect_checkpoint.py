"""Inspect checkpoint keys and metadata.

Usage:
    python -m alphatrain.scripts.inspect_checkpoint
    python -m alphatrain.scripts.inspect_checkpoint --model alphatrain/data/alphatrain_td_best.pt
"""

import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/alphatrain_td_best.pt')
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    print(f"Epoch: {ckpt.get('epoch', '?')}")
    print(f"Val loss: {ckpt.get('val_loss', '?')}")
    print(f"Max score: {ckpt.get('max_score', '?')}")

    state = ckpt['model']
    print(f"\nState dict: {len(state)} keys")
    for k in sorted(state.keys())[:10]:
        print(f"  {k}: {state[k].shape}")


if __name__ == '__main__':
    main()
