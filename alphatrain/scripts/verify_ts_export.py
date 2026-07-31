"""Verify a TorchScript export matches its source checkpoint (logit-diff).

Provenance rule (HISTORY 176 erratum): never generate data with a TS export
that hasn't been diffed against the intended checkpoint.

    python -m alphatrain.scripts.verify_ts_export \
        --ts alphatrain/inference_cpp/data/vh1_policy_ts.pt \
        --checkpoint alphatrain/data/small128_vh1.pt --n 512
"""
import argparse

import torch

from alphatrain.dataset import TensorDatasetGPU
from alphatrain.inference_cpp.export_ts import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ts', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--state-tensor', default='alphatrain/data/distill_states.pt')
    p.add_argument('--n', type=int, default=512)
    a = p.parse_args()

    m, nb, ch = load_model(a.checkpoint)
    ts = torch.jit.load(a.ts, map_location='cpu')
    ts.train(False)

    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device='cpu')
    obs = ds._build_obs_core(ds.boards[:a.n], next_pos=ds.next_pos[:a.n],
                             next_col=ds.next_col[:a.n],
                             n_next=ds.n_next[:a.n]).float()
    with torch.no_grad():
        eager = m(obs)
        traced = ts(obs)
        if isinstance(traced, tuple):
            traced = traced[0]
    diff = (eager - traced).abs().max().item()
    agree = (eager.argmax(1) == traced.argmax(1)).float().mean().item()
    print(f'{a.ts} vs {a.checkpoint} ({nb}b x {ch}ch, {a.n} states): '
          f'max|logit diff| = {diff:.2e}, argmax agree = {100 * agree:.2f}%')
    if diff > 1e-4 or agree < 1.0:
        raise SystemExit('FAIL: export does not match checkpoint')
    print('OK')


if __name__ == '__main__':
    main()
