"""Rigorously confirm the BN-running-stat contamination hypothesis.

For each checkpoint, measure V12 val three ways on the SAME val sample:
  evalBN      : model.train(False) -> uses stored running mean/var (inference path)
  trainBN     : model.train(True)  -> uses the val batch's own stats
  recomputed  : reset running stats, forward clean MAIN data, then inference path

If the WEIGHTS are fine and only the running stats were poisoned (the aux-forward
contamination hypothesis), then for the buggy aux checkpoint:
    evalBN >> trainBN ~= recomputed ~= 2.2
For a clean checkpoint (control / pillar3b): all three ~= 2.2.

    PYTHONPATH=. python scripts/check_bn_bug.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.model import AlphaTrainNet
from alphatrain.train_path_b import cross_entropy_soft

DEV = 'cpu'   # diagnostic; keep off MPS so it doesn't fight the control run
TENSOR = 'alphatrain/data/v13_pillar3a.pt'
CKPTS = [
    ('pillar3b',    'alphatrain/data/pillar3b_epoch_20.pt'),
    ('control_ep1', 'alphatrain/data/pillar3d_control/epoch_1.pt'),
    ('verify_ep5',  'alphatrain/data/pillar3d_verify/epoch_5.pt'),
]


def load_ckpt(path):
    ck = torch.load(path, map_location=DEV, weights_only=False)
    state = ck['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    m = AlphaTrainNet(num_blocks=10, channels=256).to(DEV)
    ms = m.state_dict()
    filt = {k: v for k, v in state.items()
            if k in ms and v.shape == ms[k].shape}
    m.load_state_dict(filt, strict=False)
    return m


@torch.no_grad()
def ce(m, obs, tgt):
    return float(cross_entropy_soft(m(obs), tgt))


def reset_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.reset_running_stats()
            mod.momentum = None        # cumulative average over the recompute pass


def main():
    train_set, val_set = TensorDatasetGPU.make_train_val_split(
        TENSOR, val_split=0.01, augment=False, color_augment=False,
        augment_factor=1, device=DEV, seed=42)
    nval = min(4096, len(val_set))
    val_obs, val_tgt = val_set.collate(list(range(nval)))
    recomp_obs, _ = train_set.collate(list(range(16384)))
    print(f"val sample={nval}  recompute sample=16384  device={DEV}\n", flush=True)
    print(f"{'checkpoint':>12} {'evalBN':>9} {'trainBN':>9} {'recomputed':>11}  verdict",
          flush=True)
    print('-' * 60, flush=True)

    for name, path in CKPTS:
        if not os.path.exists(path):
            print(f"{name:>12}  (missing {path})", flush=True)
            continue
        m = load_ckpt(path)
        m.train(False)
        v_eval = ce(m, val_obs, val_tgt)
        m.train(True)
        v_train = ce(m, val_obs, val_tgt)
        reset_bn(m)
        m.train(True)
        with torch.no_grad():
            for i in range(0, recomp_obs.size(0), 4096):
                m(recomp_obs[i:i + 4096])
        m.train(False)
        v_recomp = ce(m, val_obs, val_tgt)
        bn_gap = v_eval - max(v_train, v_recomp)
        verdict = ("BN-CONTAMINATED (weights ok, running stats poisoned)"
                   if bn_gap > 0.3 else "clean")
        print(f"{name:>12} {v_eval:>9.4f} {v_train:>9.4f} {v_recomp:>11.4f}  {verdict}",
              flush=True)


if __name__ == '__main__':
    main()
