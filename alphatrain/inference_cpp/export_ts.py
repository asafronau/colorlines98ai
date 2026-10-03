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
import numpy as np
import torch

from alphatrain.model import PolicyNet
from alphatrain.dataset import TensorDatasetGPU


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


def repo_path(*parts):
    """Return a stable default path independent of the caller's cwd."""
    return os.path.join(_REPO_ROOT, *parts)


def load_model(path, device, dtype, fold_bn=False):
    """Rebuild PolicyNet from a checkpoint, inferring arch from the weights."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    st = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
    from alphatrain.model_variants import net_from_state
    m, desc, flags = net_from_state(st)   # eval mode; a p4m trunk comes back frozen (plain convs, BN-foldable)
    n_fix = fp16_safe_batchnorm(m)
    if n_fix:
        print(f'fp16-safe BN: rescaled {n_fix} channel(s) with running_var > {FP16_SAFE_VAR:g}')
    if fold_bn:
        print(f'folded {fold_batchnorm(m)} BatchNorm layers into the preceding convolutions')
    m.to(device=device, dtype=dtype)
    return m, desc, flags



def fold_batchnorm(m):
    """Fold every BatchNorm that directly follows a convolution into that convolution (eval mode, exact
    up to rounding): the stem, each block's conv1 -> bn2, and the policy head's conv1 -> bn. A block's
    bn1 (pre-activation, before a ReLU) and backbone_bn (after the residual sum) cannot fold.
    Removes ~half the per-position elementwise passes (HISTORY 257)."""
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    if hasattr(m, 'slot_stem'):         # c7 slot trunk: folds into its sums of convolutions itself
        return m.fold_batchnorm()
    if hasattr(m, 'gblocks'):           # frozen p4m trunk (PolicyNetP4M.freeze): stem, blocks, 1x1 expand
        m.stem_lift = fuse_conv_bn_eval(m.stem_lift, m.stem_bn)
        m.stem_bn = torch.nn.Identity()
        blocks = m.gblocks
        m.expand_conv = fuse_conv_bn_eval(m.expand_conv, m.expand_bn)
        m.expand_bn = torch.nn.Identity()
        n = 1
    else:
        m.stem[0] = fuse_conv_bn_eval(m.stem[0], m.stem[1])
        m.stem[1] = torch.nn.Identity()
        blocks = m.blocks
        n = 0
    for blk in blocks:
        blk.conv1 = fuse_conv_bn_eval(blk.conv1, blk.bn2)
        blk.bn2 = torch.nn.Identity()
    m.policy_conv1 = fuse_conv_bn_eval(m.policy_conv1, m.policy_bn)
    m.policy_bn = torch.nn.Identity()
    return len(blocks) + 2 + n

FP16_SAFE_VAR = 1e3


def fp16_safe_batchnorm(m):
    """Exact reparameterization so BN buffers survive the C++ engine's cast to fp16.

    Trained nets can have running_var > 65504 (fp16 max): the cast makes it +inf and
    silently zeroes that channel (18b96e40 backbone_bn ch 9/48 = 100,447/70,628).
    For c = var/FP16_SAFE_VAR:  (x-mean)/sqrt(var/c+eps) * (gamma/sqrt(c))
    == (x-mean)*gamma/sqrt(var + c*eps), i.e. identical up to eps*c (~1e-8*var)."""
    n = 0
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d) and mod.running_var is not None:
                v = mod.running_var
                c = torch.where(v > FP16_SAFE_VAR, v / FP16_SAFE_VAR, torch.ones_like(v))
                if bool((c > 1).any()):
                    mod.running_var.div_(c)
                    if mod.weight is not None:
                        mod.weight.div_(c.sqrt())
                    n += int((c > 1).sum())
    return n


def load_or_build_fixture(a, device, dtype):
    """Load the stable golden probe, avoiding a 1.4GB tensor load on reruns."""
    obs_path = os.path.join(a.outdir, 'example_obs.f32')
    legal_path = os.path.join(a.outdir, 'example_legal.f32')
    if os.path.exists(obs_path) and os.path.exists(legal_path):
        obs_np = np.fromfile(obs_path, dtype='<f4')
        legal_np = np.fromfile(legal_path, dtype='<f4')
        if obs_np.size != 18 * 9 * 9 or legal_np.size != 81 * 81:
            raise ValueError('existing golden fixture has the wrong shape')
        obs = torch.from_numpy(obs_np.copy()).reshape(1, 18, 9, 9)
        legal = torch.from_numpy(legal_np.copy()).bool()
        return obs.to(device=device, dtype=dtype), legal.to(device=device)

    # First-time bootstrap only. TensorDatasetGPU constructs the observation
    # on the requested accelerator; later exports reuse the emitted fixture.
    ds = TensorDatasetGPU(a.state_tensor, augment=False, color_augment=False,
                          augment_factor=1, device=str(device))
    obs = ds._build_obs_core(ds.boards[0:1], next_pos=ds.next_pos[0:1],
                             next_col=ds.next_col[0:1],
                             n_next=ds.n_next[0:1]).to(dtype=dtype)
    from alphatrain.batched_engine_gpu import legal_priors_t
    dummy_logits = torch.zeros((1, 81 * 81), device=device, dtype=dtype)
    _, idx, _ = legal_priors_t(ds.boards[0:1], dummy_logits, top_k=6561)
    legal = torch.zeros(6561, device=device, dtype=torch.bool)
    legal[idx[0][idx[0] >= 0]] = True
    return obs, legal


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=repo_path(
        'alphatrain', 'data', 'pillar3k_small128_epoch_15.pt'))
    p.add_argument('--state-tensor', default=repo_path(
        'alphatrain', 'data', 'distill_states.pt'))
    p.add_argument('--outdir', default=repo_path(
        'alphatrain', 'inference_cpp', 'data'))
    p.add_argument('--output', default=None,
                   help='TorchScript output path. Defaults to '
                        '<outdir>/policy_ts.pt for backward compatibility.')
    p.add_argument('--device', choices=('auto', 'mps', 'cuda', 'cpu'),
                   default='auto',
                   help='Verification device. auto prefers MPS, then CUDA, '
                        'and never silently falls back to CPU.')
    p.add_argument('--precision', choices=('fp16', 'fp32'), default='fp16')
    p.add_argument('--fold-bn', action='store_true',
                   help='fold conv -> BatchNorm pairs into the convolutions (faster inference)')
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    output = a.output or f'{a.outdir}/policy_ts.pt'
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    requested_device = a.device
    if requested_device == 'auto':
        if torch.backends.mps.is_available():
            requested_device = 'mps'
        elif torch.cuda.is_available():
            requested_device = 'cuda'
        else:
            raise RuntimeError('no accelerator available; pass --device cpu '
                               '--precision fp32 only for a structural '
                               'diagnostic')
    if requested_device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('--device mps requested but MPS is unavailable')
    if requested_device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda requested but CUDA is unavailable')
    device = torch.device(requested_device)
    dtype = torch.float16 if a.precision == 'fp16' else torch.float32
    if device.type == 'cpu' and dtype == torch.float16:
        raise ValueError('fp16 export verification requires mps or cuda')

    m, desc, _ = load_model(a.model, device, dtype, fold_bn=a.fold_bn)
    obs, legal = load_or_build_fixture(a, device, dtype)

    with torch.no_grad():
        eager = m(obs)                       # what PyTorch produces

    # Trace the eval-mode model into TorchScript. The net is a plain CNN (no
    # data-dependent control flow), so tracing records an exact, frozen graph.
    ts = torch.jit.trace(m, obs)
    with torch.no_grad():
        traced = ts(obs)
    max_diff = (eager - traced).abs().max().item()

    # The fixture's legal mask is model-independent. The C++ side uses the
    # same mask to verify the deployment legal argmax for every exported net.
    legal_move = int(eager[0].float().masked_fill(~legal, float('-inf')).argmax())

    ts.save(output)
    eager[0].float().cpu().numpy().astype('<f4').tofile(
        f'{a.outdir}/example_logits.f32')
    obs[0].float().cpu().numpy().astype('<f4').tofile(
        f'{a.outdir}/example_obs.f32')
    legal.float().cpu().numpy().astype('<f4').tofile(
        f'{a.outdir}/example_legal.f32')

    print(f'arch: {desc}; verify={device} {a.precision}')
    print(f'traced vs eager max|diff| = {max_diff:.2e}  '
          f'({"OK" if max_diff < 1e-4 else "WARN"})')
    print(f'raw argmax move = {int(eager[0].argmax())}  |  '
          f'legal argmax move = {legal_move}  ({int(legal.sum())} legal moves)')
    print(f'wrote {output}; golden files in {a.outdir}/: example_obs.f32, '
          f'example_logits.f32, example_legal.f32')


if __name__ == '__main__':
    main()
