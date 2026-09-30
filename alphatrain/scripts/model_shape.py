"""Print a PolicyNet checkpoint's trunk shape as train_path_b flags, e.g. `--num-blocks 9 --channels 64`, so shell
pipelines (scripts/flywheel_turn.sh) fine-tune any actor size without hard-coding it.

    python -m alphatrain.scripts.model_shape alphatrain/data/actor.pt
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    blocks = sum(1 for k in st if k.startswith('blocks.') and k.endswith('.conv1.weight'))
    print(f"--num-blocks {blocks} --channels {st['stem.0.weight'].shape[0]}")


if __name__ == '__main__':
    main()
