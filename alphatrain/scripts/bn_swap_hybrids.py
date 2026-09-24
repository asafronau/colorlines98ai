"""Zero-training-cost BN/weight attribution hybrids (review #8, answer 2):
  hybrid A = a3's weights + vh2's BN running stats
  hybrid B = vh2's weights + a3's BN running stats
Gameplay screens then separate parameter-learning damage from running-stat
damage in the a3 collapse.

    python -m alphatrain.scripts.bn_swap_hybrids
"""
import torch

BASE = 'alphatrain/data/small128_vh2.pt'
A3 = 'alphatrain/data/iter5a3_epoch_1.pt'
BN_KEYS = ('running_mean', 'running_var', 'num_batches_tracked')


def sd_of(p):
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def mix(weights_from, bn_from, out):
    w, b = sd_of(weights_from), sd_of(bn_from)
    sd = {k: (b[k].clone() if any(t in k for t in BN_KEYS) else v.clone())
          for k, v in w.items()}
    n_bn = sum(1 for k in sd if any(t in k for t in BN_KEYS))
    torch.save({'model': sd, 'policy_only': True,
                'hybrid': {'weights': weights_from, 'bn': bn_from}}, out)
    print(f'{out}: weights<-{weights_from}  bn<-{bn_from}  ({n_bn} BN buffers)')


mix(A3, BASE, 'alphatrain/data/hyb_a3w_vh2bn.pt')
mix(BASE, A3, 'alphatrain/data/hyb_vh2w_a3bn.pt')
