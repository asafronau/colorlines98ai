"""Uniform weight averaging (SWA-style) of checkpoints of ONE run; BN running stats averaged, optionally
re-estimated on training states afterwards (--recalib-tensor, cumulative average over --recalib-batches)."""
import argparse, sys
import torch
sys.path.insert(0, '.')
p = argparse.ArgumentParser(); p.add_argument('--ckpts', nargs='+', required=True); p.add_argument('--out', required=True)
p.add_argument('--recalib-tensor', default=None); p.add_argument('--recalib-batches', type=int, default=200); a = p.parse_args()
sds = []
for c in a.ckpts:
    ck = torch.load(c, map_location='cpu', weights_only=False); sd = ck['model']
    sds.append({k.replace('_orig_mod.', ''): v for k, v in sd.items()})
avg = {}
for k, v in sds[0].items():
    avg[k] = torch.stack([s[k].float() for s in sds]).mean(0).to(v.dtype) if v.is_floating_point() else sds[-1][k]
if a.recalib_tensor:
    from alphatrain.model import PolicyNet, head_kwargs_from_state
    from alphatrain.dataset import TensorDatasetGPU
    ch = avg['stem.0.weight'].shape[0]; nb = sum(1 for k in avg if k.startswith('blocks.') and k.endswith('.conv1.weight'))
    net = PolicyNet(num_blocks=nb, channels=ch, **head_kwargs_from_state(avg)); net.load_state_dict(avg); net.to('mps')
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm): m.reset_running_stats(); m.momentum = None
    ds = TensorDatasetGPU(a.recalib_tensor, augment=True, color_augment=True, augment_factor=1, device='mps')
    net.train(True); done = 0; g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for _ in range(a.recalib_batches):
            idx = torch.randint(0, len(ds), (4096,), generator=g).tolist()
            obs = ds.collate(idx)[0]
            net(obs.to('mps')); done += 1
    avg = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    print(f'BN re-estimated over {done} batches')
torch.save({'model': avg, 'epoch': 'avg', 'note': f'uniform average of {len(a.ckpts)} checkpoints'}, a.out)
print(f'wrote {a.out} = mean of {len(a.ckpts)} checkpoints')
