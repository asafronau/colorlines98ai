"""Cross-torch-version numeric gate: run the policy net forward on a FIXED obs batch and dump the
logits. Run under .venv (2.10) and .venv-t212 (2.12); if the outputs match (fp16 forward identical
or within tiny tol), the upgrade preserves the policy -> MCTS / mining / eval are unchanged -> SAFE.

    .venv/bin/python       scripts/check_torch_fwd.py --out /tmp/fwd_210.npz
    .venv-t212/bin/python  scripts/check_torch_fwd.py --out /tmp/fwd_212.npz
    .venv/bin/python       scripts/check_torch_fwd.py --compare /tmp/fwd_210.npz /tmp/fwd_212.npz
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

MODEL = 'alphatrain/data/pillar3b_epoch_20.pt'


def _load(dev, fp16):
    from alphatrain.model import PolicyNet
    ckpt = torch.load(MODEL, map_location='cpu', weights_only=False)
    state = ckpt['model']
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    in_ch = state['stem.0.weight'].shape[1]
    nb = sum(1 for k in state if k.endswith('.conv1.weight') and k.startswith('blocks.'))
    ch = state['stem.0.weight'].shape[0]
    state = {k: v for k, v in state.items() if not k.startswith('value_')}
    net = PolicyNet(in_channels=in_ch, num_blocks=nb, channels=ch)
    net.load_state_dict(state); net.train(False); net = net.to(dev)
    if fp16 and dev.type in ('mps', 'cuda'):
        net = net.half()
    return net


def _fixed_obs(n=64):
    """Deterministic obs batch built from fixed pseudo-random boards (same on every run/version)."""
    from alphatrain.observation import build_observation
    rng = np.random.default_rng(12345)
    obs = []
    for _ in range(n):
        b = np.where(rng.random((9, 9)) < 0.55, rng.integers(1, 8, (9, 9)), 0).astype(np.int8)
        e = np.argwhere(b == 0)
        nr = np.zeros(3, np.intp); nc = np.zeros(3, np.intp); ncol = np.zeros(3, np.intp)
        m = min(3, len(e))
        for i in range(m):
            nr[i], nc[i] = e[rng.integers(len(e))]; ncol[i] = rng.integers(1, 8)
        obs.append(build_observation(b, nr, nc, ncol, m))
    return np.stack(obs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=None)
    p.add_argument('--compare', nargs=2, default=None)
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    if a.compare:
        x = np.load(a.compare[0]); y = np.load(a.compare[1])
        lx, ly = x['logits'], y['logits']
        d = np.abs(lx - ly)
        am_x, am_y = lx.argmax(1), ly.argmax(1)
        print(f"torch {x['ver']} vs {y['ver']}  (fp16 mps forward, {lx.shape[0]} boards)")
        print(f"  logits max abs diff: {d.max():.4e}   mean abs diff: {d.mean():.4e}")
        print(f"  argmax agreement: {int((am_x == am_y).sum())}/{len(am_x)}")
        # softmax TV (policy-level drift)
        def sm(z):
            z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
        tv = 0.5 * np.abs(sm(lx.astype(np.float64)) - sm(ly.astype(np.float64))).sum(1)
        print(f"  policy softmax TV: mean {tv.mean():.4e}  max {tv.max():.4e}")
        ok = (am_x == am_y).all() and d.max() < 1e-2
        print(f"  => {'SAFE (forward preserved)' if ok else 'DRIFT — investigate before adopting'}")
        return
    dev = torch.device(a.device)
    net = _load(dev, fp16=(dev.type != 'cpu'))
    obs = _fixed_obs()
    with torch.no_grad():
        logits = net(torch.from_numpy(obs).to(dev, next(net.parameters()).dtype)).float().cpu().numpy()
    out = a.out or f'/tmp/fwd_{torch.__version__}.npz'
    np.savez(out, logits=logits, ver=torch.__version__)
    print(f"torch {torch.__version__}: wrote {out}  logits sum={logits.sum():.4f} "
          f"argmax[0:4]={logits.argmax(1)[:4].tolist()}", flush=True)


if __name__ == '__main__':
    main()
