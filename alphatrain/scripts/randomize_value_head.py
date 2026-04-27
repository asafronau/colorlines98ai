"""Create model variants with randomized value head weights.

Loads a checkpoint, freezes the backbone + policy head, reinitializes
only the value head with different random seeds. Saves N variants.

Usage:
    python -m alphatrain.scripts.randomize_value_head \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --seeds 0 1 2 3 4 \
        --output-dir alphatrain/data/random_heads
"""

import os
import argparse
import torch
import torch.nn as nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--output-dir', default='alphatrain/data/random_heads')
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    state_dict = ckpt['model']

    # Identify value head parameters
    value_prefixes = ('value_conv', 'value_bn', 'value_fc1', 'value_fc2')
    value_keys = [k for k in state_dict if k.startswith(value_prefixes)]
    other_keys = [k for k in state_dict if not k.startswith(value_prefixes)]

    print(f"Loaded {args.model}", flush=True)
    print(f"  Value head params: {len(value_keys)} tensors", flush=True)
    print(f"  Frozen params: {len(other_keys)} tensors", flush=True)

    # Show current value head stats
    for k in value_keys:
        t = state_dict[k]
        print(f"  {k}: shape={list(t.shape)} mean={t.float().mean():.4f} "
              f"std={t.float().std():.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seeds:
        torch.manual_seed(seed)
        new_state = dict(state_dict)  # shallow copy

        for k in value_keys:
            shape = state_dict[k].shape
            if 'weight' in k and len(shape) >= 2:
                # Conv or Linear weight
                new_state[k] = torch.empty_like(state_dict[k])
                nn.init.kaiming_normal_(new_state[k], mode='fan_out',
                                        nonlinearity='relu')
            elif 'bias' in k:
                new_state[k] = torch.zeros_like(state_dict[k])
            elif 'bn' in k and 'weight' in k:
                new_state[k] = torch.ones_like(state_dict[k])
            elif 'bn' in k and 'bias' in k:
                new_state[k] = torch.zeros_like(state_dict[k])
            elif 'running_mean' in k:
                new_state[k] = torch.zeros_like(state_dict[k])
            elif 'running_var' in k:
                new_state[k] = torch.ones_like(state_dict[k])
            elif 'num_batches_tracked' in k:
                new_state[k] = torch.zeros_like(state_dict[k])
            else:
                # Unknown — reinit random
                new_state[k] = torch.randn_like(state_dict[k])

        new_ckpt = dict(ckpt)
        new_ckpt['model'] = new_state

        path = os.path.join(args.output_dir,
                            f"pillar2w2_rhead_s{seed}.pt")
        torch.save(new_ckpt, path)

        # Quick stats on new head
        stds = [new_state[k].float().std().item() for k in value_keys
                if 'weight' in k and len(new_state[k].shape) >= 2]
        print(f"  Saved seed={seed}: {path} "
              f"(weight stds: {[f'{s:.3f}' for s in stds]})", flush=True)

    print(f"\nDone: {len(args.seeds)} variants in {args.output_dir}/",
          flush=True)


if __name__ == '__main__':
    main()
