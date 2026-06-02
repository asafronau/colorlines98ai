"""Recompute (de-poison) a checkpoint's BatchNorm running stats on clean data.

If a checkpoint was trained with the aux-forward BN-contamination bug, its
WEIGHTS are fine but its BN running mean/var are poisoned by OOD crisis states
(inference uses those stats -> degraded gameplay). This resets the running
stats and recomputes them from a clean pass over the V13 main corpus, then
saves a de-poisoned checkpoint usable for a valid floor eval.

Idempotent: on an already-clean checkpoint, recomputed val ~= eval val (no harm).

    PYTHONPATH=. python scripts/recompute_bn.py \\
        --ckpt alphatrain/data/pillar3d_epoch_1.pt \\
        --out  alphatrain/data/pillar3d_epoch_1_bnfix.pt
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet
from alphatrain.train_path_b import cross_entropy_soft


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--tensor', default='alphatrain/data/v13_pillar3a.pt')
    p.add_argument('--n-recompute', type=int, default=32768)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    out = a.out or a.ckpt.replace('.pt', '_bnfix.pt')

    dev = a.device
    train_set, val_set = TensorDatasetGPU.make_train_val_split(
        a.tensor, val_split=0.01, augment=False, color_augment=False,
        augment_factor=1, device=dev, seed=42)
    val_obs, val_tgt = val_set.collate(list(range(min(4096, len(val_set)))))
    recomp_obs, _ = train_set.collate(list(range(a.n_recompute)))

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    state = ck['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    m = AlphaTrainNet(num_blocks=10, channels=256).to(dev)
    ms = m.state_dict()
    filt = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    m.load_state_dict(filt, strict=False)

    @torch.no_grad()
    def val():
        m.train(False)
        return float(cross_entropy_soft(m(val_obs), val_tgt))

    v_before = val()
    # reset + recompute BN running stats from a clean main-corpus pass
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats()
            mod.momentum = None
    m.train(True)
    with torch.no_grad():
        bs = 4096
        for i in range(0, recomp_obs.size(0), bs):
            m(recomp_obs[i:i + bs])
    v_after = val()

    poisoned = (v_before - v_after) > 0.3
    print(f"{os.path.basename(a.ckpt)}: val before={v_before:.4f} "
          f"after={v_after:.4f}  "
          f"{'WAS BN-POISONED -> fixed' if poisoned else 'was clean'}",
          flush=True)

    # save de-poisoned checkpoint (BN buffers now clean), preserve other fields
    m.train(False)
    ck_out = dict(ck)
    ck_out['model'] = m.state_dict()
    ck_out['bn_recomputed'] = True
    torch.save(ck_out, out)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()
