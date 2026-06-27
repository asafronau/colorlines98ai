"""Export PolicyNet as a TorchScript module for the C++ (LibTorch) engine.

Writes to --outdir:
  policy_ts.pt        TorchScript module: obs(B,18,9,9) -> logits(B,6561).
                      Weights + BatchNorm + everything are baked in.
  example_obs.f32     one real obs (18*9*9 float32, row-major) to test on.
  example_logits.f32  PyTorch's logits for that obs (6561 float32) = the oracle.

The C++ side just does torch::jit::load("policy_ts.pt") and forward() — that is
the whole engine. The example_*.f32 files let main.cc prove C++ == PyTorch.

    python -m alphatrain.inference_cpp.export_ts \
        --model alphatrain/data/pillar3k_small128_epoch_15.pt
"""
import argparse, os
import torch

from alphatrain.model import PolicyNet
from alphatrain.dataset import TensorDatasetGPU


def load_model(path):
    """Rebuild PolicyNet from a checkpoint, inferring arch from the weights."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    if any(k.startswith('_orig_mod.') for k in st):
        st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    ch = st['stem.0.weight'].shape[0]
    nblocks = sum(1 for k in st if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    m = PolicyNet(num_blocks=nblocks, channels=ch)
    m.load_state_dict(st)
    m.train(False)  # eval mode: BatchNorm uses running stats (important!)
    return m, nblocks, ch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3k_small128_epoch_15.pt')
    p.add_argument('--state-tensor', default='alphatrain/data/distill_states.pt')
    p.add_argument('--outdir', default='alphatrain/inference_cpp/data')
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    m, nblocks, ch = load_model(a.model)

    # one real board -> obs, to use as the test input
    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device='cpu')
    obs = ds._build_obs_core(ds.boards[0:1], next_pos=ds.next_pos[0:1],
                             next_col=ds.next_col[0:1], n_next=ds.n_next[0:1]).float()

    with torch.no_grad():
        eager = m(obs)                       # what PyTorch produces

    # Trace the eval-mode model into TorchScript. The net is a plain CNN (no
    # data-dependent control flow), so tracing records an exact, frozen graph.
    ts = torch.jit.trace(m, obs)
    with torch.no_grad():
        traced = ts(obs)
    max_diff = (eager - traced).abs().max().item()

    ts.save(f'{a.outdir}/policy_ts.pt')
    eager[0].numpy().astype('<f4').tofile(f'{a.outdir}/example_logits.f32')
    obs[0].numpy().astype('<f4').tofile(f'{a.outdir}/example_obs.f32')

    print(f'arch: {nblocks}b x {ch}ch')
    print(f'traced vs eager max|diff| = {max_diff:.2e}  '
          f'({"OK" if max_diff < 1e-4 else "WARN"})')
    print(f'wrote {a.outdir}/: policy_ts.pt, example_obs.f32 (18x9x9), '
          f'example_logits.f32 (6561)')


if __name__ == '__main__':
    main()
