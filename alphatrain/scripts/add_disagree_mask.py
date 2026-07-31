"""Add a per-state disagree_mask to a mixed tensor: 1 where the target argmax
(over the pol_indices support) differs from the BASE policy's argmax on the
same support. Rehearsal rows (beyond n_main from mix_info) get 0 — they are
the anchor, not the learning signal.

    PYTHONPATH=. python -m alphatrain.scripts.add_disagree_mask \
        --tensor alphatrain/data/iter2_mix.pt \
        --base alphatrain/data/small128_vh1.pt \
        --output alphatrain/data/iter2_mixg.pt
"""
import argparse, sys
import numpy as np
import torch

sys.path.insert(0, '.')
from alphatrain.dataset import TensorDatasetGPU
from alphatrain.evaluate import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tensor', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch', type=int, default=2048)
    a = p.parse_args()

    data = torch.load(a.tensor, map_location='cpu', weights_only=False)
    n_total = data['boards'].shape[0]
    n_main = data.get('mix_info', {}).get('n_main', n_total)
    print(f'{n_total:,} states, computing mask on the first {n_main:,} (main)')

    dev = torch.device('mps')
    net, _ = load_model(a.base, dev, fp16=False)
    ds = TensorDatasetGPU(a.tensor, augment=False, color_augment=False,
                          augment_factor=1, device='cpu')

    mask = torch.zeros(n_total, dtype=torch.int8)
    tgt_arg_all = data['pol_values'].argmax(dim=1)
    disagree_count = 0
    with torch.inference_mode():
        for s in range(0, n_main, a.batch):
            e = min(s + a.batch, n_main)
            obs = ds._build_obs_core(ds.boards[s:e], next_pos=ds.next_pos[s:e],
                                     next_col=ds.next_col[s:e],
                                     n_next=ds.n_next[s:e]).float()
            logits = net(obs.to(dev)).float().cpu()
            sup_idx = data['pol_indices'][s:e]                    # (b, K)
            sup_logits = torch.gather(logits, 1, sup_idx)
            sup_logits[data['pol_values'][s:e] <= 0] = -1e30      # mask pads
            base_pick = sup_logits.argmax(dim=1)                  # index into K
            dis = (base_pick != tgt_arg_all[s:e]).to(torch.int8)
            mask[s:e] = dis
            disagree_count += int(dis.sum())
            if (s // a.batch) % 200 == 0:
                print(f'  {e:,}/{n_main:,}  disagreements so far {disagree_count:,}',
                      flush=True)

    data['disagree_mask'] = mask
    torch.save(data, a.output)
    print(f'disagreements: {disagree_count:,}/{n_main:,} '
          f'({100*disagree_count/n_main:.1f}% of main; rehearsal rows = 0)')
    print(f'wrote {a.output}')


if __name__ == '__main__':
    main()
