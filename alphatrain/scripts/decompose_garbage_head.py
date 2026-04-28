"""Decompose the garbage value head into parts to find what makes it work.

Creates checkpoint variants by swapping parts of the value head:
1. Original garbage head (baseline)
2. Original projector + randomized fc2
3. Original projector + trained fc2 (needs separate training)
4. Fresh random projector + original fc2
5. Fresh random projector + random fc2

Usage:
    python -m alphatrain.scripts.decompose_garbage_head \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --output-dir alphatrain/data/decompose \
        --seeds 0 1 2
"""

import os
import argparse
import torch
import torch.nn as nn


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--output-dir', default='alphatrain/data/decompose')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = p.parse_args()

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    state = ckpt['model']
    os.makedirs(args.output_dir, exist_ok=True)

    # Identify value head layers
    projector_keys = [k for k in state
                      if k.startswith(('value_conv', 'value_bn', 'value_fc1'))]
    fc2_keys = [k for k in state if k.startswith('value_fc2')]

    print("Original value head:", flush=True)
    for k in projector_keys + fc2_keys:
        t = state[k]
        print(f"  {k}: {list(t.shape)} mean={t.float().mean():.4f} "
              f"std={t.float().std():.4f}", flush=True)

    # Save original projector and fc2
    orig_projector = {k: state[k].clone() for k in projector_keys}
    orig_fc2 = {k: state[k].clone() for k in fc2_keys}

    def save_variant(name, proj, fc2):
        new_state = dict(state)
        for k, v in proj.items():
            new_state[k] = v
        for k, v in fc2.items():
            new_state[k] = v
        new_ckpt = dict(ckpt)
        new_ckpt['model'] = new_state
        path = os.path.join(args.output_dir, f'{name}.pt')
        torch.save(new_ckpt, path)
        print(f"  Saved: {path}", flush=True)

    def random_fc2(seed):
        torch.manual_seed(seed)
        return {k: torch.randn_like(v) * v.std() for k, v in orig_fc2.items()}

    def random_projector(seed):
        torch.manual_seed(seed)
        result = {}
        for k, v in orig_projector.items():
            if 'weight' in k and len(v.shape) >= 2:
                t = torch.empty_like(v)
                nn.init.kaiming_normal_(t, mode='fan_out', nonlinearity='relu')
                result[k] = t
            elif 'bias' in k:
                result[k] = torch.zeros_like(v)
            elif 'running_mean' in k:
                result[k] = torch.zeros_like(v)
            elif 'running_var' in k:
                result[k] = torch.ones_like(v)
            elif 'num_batches_tracked' in k:
                result[k] = torch.zeros_like(v)
            elif 'weight' in k:  # BN weight
                result[k] = torch.ones_like(v)
            else:
                result[k] = torch.randn_like(v)
        return result

    # 1. Original (unchanged) — just copy
    print("\n1. Original garbage head:", flush=True)
    save_variant('original', orig_projector, orig_fc2)

    # 2. Original projector + randomized fc2 (multiple seeds)
    for seed in args.seeds:
        print(f"\n2. Original projector + random fc2 (seed={seed}):", flush=True)
        save_variant(f'orig_proj_rand_fc2_s{seed}',
                     orig_projector, random_fc2(seed))

    # 3. Fresh random projector + original fc2 (multiple seeds)
    for seed in args.seeds:
        print(f"\n3. Random projector + original fc2 (seed={seed}):", flush=True)
        save_variant(f'rand_proj_orig_fc2_s{seed}',
                     random_projector(seed), orig_fc2)

    # 4. Fresh random projector + random fc2 (multiple seeds)
    for seed in args.seeds:
        print(f"\n4. Random projector + random fc2 (seed={seed}):", flush=True)
        save_variant(f'rand_proj_rand_fc2_s{seed}',
                     random_projector(seed), random_fc2(seed + 100))

    print(f"\nDone. Variants in {args.output_dir}/", flush=True)
    print(f"\nEval commands:", flush=True)
    print(f"for f in {args.output_dir}/*.pt; do", flush=True)
    print(f'  echo "=== $(basename $f) ==="', flush=True)
    print(f"  python -m alphatrain.scripts.eval_parallel \\", flush=True)
    print(f"    --model $f --device mps --workers 16 \\", flush=True)
    print(f"    --simulations 400 --games-per-seed 1 --batch-size 64 \\", flush=True)
    print(f"    --mcts-only --terminal-value 0 --seeds $(seq 0 19)", flush=True)
    print(f"done", flush=True)


if __name__ == '__main__':
    main()
