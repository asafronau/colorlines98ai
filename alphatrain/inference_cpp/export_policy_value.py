"""Export a FUSED policy+value TorchScript module for the C++ engine.

forward(obs[B,18,9,9]) -> (policy_logits[B,6561], value[B])
where value = survival_to_scalar(head(backbone_features(obs))) — one backbone
pass feeds both heads (same as Python MCTS's forward_with_features path).

Writes data/policy_value_ts.pt + golden (example obs/logits/values) and
verifies traced == eager.

    python -m alphatrain.inference_cpp.export_policy_value \
        --model alphatrain/data/pillar3k_small128_hardce_epoch_87.pt \
        --head alphatrain/data/value_head_small128.pt
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, '.')
from alphatrain.evaluate import load_model
from alphatrain.dataset import TensorDatasetGPU
from alphatrain import value_head as vh


class PolicyValue(nn.Module):
    def __init__(self, net, head, horizon_weights):
        super().__init__()
        self.net = net
        self.head = head
        self.register_buffer('hw', horizon_weights)

    def forward(self, obs):
        pol, feats = self.net.forward_with_features(obs)
        out = self.head(feats)  # runs in module dtype (fp16-safe on MPS)
        v = (torch.sigmoid(out) * self.hw).sum(dim=-1)  # survival_to_scalar
        return pol, v


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3k_small128_hardce_epoch_87.pt')
    p.add_argument('--head', default='alphatrain/data/value_head_small128.pt')
    p.add_argument('--state-tensor', default='alphatrain/data/distill_states.pt')
    p.add_argument('--outdir', default='alphatrain/inference_cpp/data')
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    net, _ = load_model(a.model, torch.device('cpu'), fp16=False)
    head, ckpt, head_type = vh.load_any(a.head, torch.device('cpu'))
    assert head_type == 'value_head' and ckpt.get('target_type') == 'survival', \
        f'expected survival ValueHead, got {head_type}/{ckpt.get("target_type")}'
    head.train(False)
    net.train(False)
    hw = torch.tensor(vh.SURVIVAL_WEIGHTS
                      if hasattr(vh, 'SURVIVAL_WEIGHTS') else [1.0, 0.8, 0.5, 0.25],
                      dtype=torch.float32)
    module = PolicyValue(net, head, hw)
    module.train(False)

    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device='cpu')
    obs = ds._build_obs_core(ds.boards[0:4], next_pos=ds.next_pos[0:4],
                             next_col=ds.next_col[0:4], n_next=ds.n_next[0:4]).float()

    with torch.no_grad():
        pol_e, v_e = module(obs)
    ts = torch.jit.trace(module, obs)
    with torch.no_grad():
        pol_t, v_t = ts(obs)
    dp = (pol_e - pol_t).abs().max().item()
    dv = (v_e - v_t).abs().max().item()
    print(f'traced vs eager: pol {dp:.2e}  value {dv:.2e}')
    assert dp < 1e-4 and dv < 1e-5

    ts.save(f'{a.outdir}/policy_value_ts.pt')
    obs.numpy().astype('<f4').tofile(f'{a.outdir}/pv_example_obs.f32')
    pol_e.numpy().astype('<f4').tofile(f'{a.outdir}/pv_example_logits.f32')
    v_e.numpy().astype('<f4').tofile(f'{a.outdir}/pv_example_values.f32')
    print(f'wrote {a.outdir}/policy_value_ts.pt (+ pv_example obs/logits/values, B=4)')
    print(f'values sample: {[round(float(x), 4) for x in v_e]}')


if __name__ == '__main__':
    main()
