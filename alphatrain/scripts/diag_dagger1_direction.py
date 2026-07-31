"""Diagnose the DIRECTION of dagger1's drift: mimicry pull toward pillar3k?

On a rehearsal sample (labels in-tensor = pillar3k argmax) and the on-policy
holdout (pillar3k forwarded live), measure match-to-pillar3k for vh1 vs trained
checkpoints. If the trained models' match is clearly HIGHER, the run pulled the
policy back toward the raw distillate — undoing the gate-3 (non-mimicry) gains.

    python -m alphatrain.scripts.diag_dagger1_direction \
        --models alphatrain/data/small128_dagger1_e3_s400.pt \
                 alphatrain/data/small128_dagger1_e1_s100.pt
"""
import argparse
import os

import numpy as np
import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.mcts import _legal_priors_jit
from alphatrain.evaluate import load_model
from alphatrain.scripts.diag_dagger1_regression import legal_argmax, build_holdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--base', default='alphatrain/data/small128_vh1.pt')
    p.add_argument('--prior-base',
                   default='alphatrain/data/pillar3k_small128_hardce_epoch_87.pt')
    p.add_argument('--teacher',
                   default='alphatrain/data/pillar3k_r3_dw3_T0.7_epoch_22.pt')
    p.add_argument('--corpus', default='alphatrain/data/dagger_v1.pt')
    p.add_argument('--games-dir', default='data/dagger_games_v1')
    p.add_argument('--rehearsal', default='alphatrain/data/distill_pillar3k.pt')
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    rng = np.random.default_rng(0)

    hold_path = a.corpus + '.holdout2.tmp'
    n_hold = build_holdout(a.games_dir, a.corpus, 200, 100, hold_path, rng)
    ds_hold = TensorDatasetGPU(hold_path, augment=False, color_augment=False,
                               augment_factor=1, device=a.device)
    ds_reh = TensorDatasetGPU(a.rehearsal, augment=False, color_augment=False,
                              augment_factor=1, device=a.device)
    reh_idx_np = np.sort(rng.choice(ds_reh.boards.shape[0], 20000, replace=False))
    reh_idx = torch.from_numpy(reh_idx_np).to(dev)
    hold_idx = torch.arange(n_hold).to(dev)
    print(f'holdout {n_hold:,} on-policy states, rehearsal sample 20,000',
          flush=True)

    # pillar3k reference argmax: stored labels on rehearsal, live fwd on holdout
    reh_full = torch.load(a.rehearsal, map_location='cpu', weights_only=False)
    pi = reh_full['pol_indices'][reh_idx_np].numpy()
    pv = reh_full['pol_values'][reh_idx_np].numpy()
    reh_teacher = pi[np.arange(len(pi)), pv.argmax(1)]
    del reh_full
    t_net, _ = load_model(a.teacher, dev, fp16=False)
    hold_teacher = legal_argmax(t_net, ds_hold, hold_idx, dev)
    del t_net

    print(f'\n{"model":30s} {"reh->3k%":>9s} {"hold->3k%":>10s}')
    for path in [a.prior_base, a.base] + a.models:
        net, _ = load_model(path, dev, fp16=False)
        name = os.path.basename(path).replace('.pt', '')
        m_reh = legal_argmax(net, ds_reh, reh_idx, dev)
        m_hold = legal_argmax(net, ds_hold, hold_idx, dev)
        del net
        print(f'{name:30s} {100 * (m_reh == reh_teacher).mean():9.2f} '
              f'{100 * (m_hold == hold_teacher).mean():10.2f}', flush=True)
    os.remove(hold_path)


if __name__ == '__main__':
    main()
