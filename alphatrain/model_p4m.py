"""p4m (D4) group-equivariant trunk for the PAIR2 policy (HISTORY 278).

The game is exactly symmetric under the 8 rotations/reflections of the board (the dihedral group D4); a plain CNN
is not, and its 8-view average (TTA-8) dies ~35% less than its single pass (HISTORY 270). This trunk is
equivariant BY CONSTRUCTION (group convolutions, Cohen & Welling 2016), so one forward pass gives the same move
in all 8 orientations.

Feature maps live on the group: a tensor (B, C*8, 9, 9) holds C "group channels", each with one 9x9 slice per
group element g (layout c*8 + g). Every filter is learned once and applied in all 8 orientations:
  lifting (input -> group):  out[c, g] = sum_i x_i * T_g(K[c, pi_g(i)])
  group conv (group -> group): out[c, g] = sum_{c', h} f[c', h] * T_g(K[c, c', g^-1 h])
where T_g rotates/reflects a kernel and pi_g permutes the four line-direction input channels (13-16: H, V, D1,
D2), derived numerically from the observation builder. The expanded weights are an ordinary Conv2d weight, so
freeze() turns the trunk into a standard CNN for TorchScript export and BN folding (same compute as a plain CNN
with C*8 channels; 8x fewer free parameters). BatchNorm shares statistics across the 8 orientations. After the
blocks, a 1x1 group conv expands to E group channels, which are pooled over the orientations (mean and max) into
2E invariant per-cell features feeding the unchanged PAIR2 head (exactly D4-equivariant given invariant features).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphatrain.model import PolicyNet

G = 8
ELEMENTS = [(k, m) for m in (0, 1) for k in range(4)]     # index m*4 + k: rotate k quarter-turns, then mirror if m


def apply_g(x, g):
    """T_g on the last two (spatial) dims."""
    k, m = ELEMENTS[g]
    y = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if m else y


def _group_tables():
    p = torch.arange(9.).reshape(3, 3)                       # all 8 images distinct
    imgs = [apply_g(p, g) for g in range(G)]

    def find(y):
        return next(i for i, im in enumerate(imgs) if torch.equal(im, y))
    mult = [[find(apply_g(apply_g(p, b), a)) for b in range(G)] for a in range(G)]   # T_a T_b = T_mult[a][b]
    inv = [next(b for b in range(G) if mult[a][b] == 0) for a in range(G)]
    return mult, inv


MULT, INV = _group_tables()


def input_permutation():
    """pi[g][i] = j such that obs(T_g board)[i] == T_g(obs(board)[j]): the line-direction channels swap under
    rotations/reflections; every other channel maps to itself. Derived from the real observation builder."""
    from alphatrain.observation import build_observation, NUM_CHANNELS
    rng = np.random.default_rng(12345)
    board = rng.integers(0, 8, (9, 9)).astype(np.int8)
    board[rng.random((9, 9)) < 0.35] = 0
    nr, nc, col = np.array([0, 3, 7]), np.array([1, 8, 4]), np.array([2, 5, 6])
    board[nr, nc] = 0
    obs = torch.from_numpy(build_observation(board, nr, nc, col, 3))
    perms = []
    for g in range(G):
        bt = apply_g(torch.from_numpy(board), g).numpy()
        pos = []
        for r, c in zip(nr, nc):
            one = torch.zeros(9, 9); one[r, c] = 1
            rr, cc = torch.nonzero(apply_g(one, g))[0].tolist()
            pos.append((rr, cc))
        ot = torch.from_numpy(build_observation(bt, np.array([p[0] for p in pos]), np.array([p[1] for p in pos]), col, 3))
        perm = []
        for i in range(NUM_CHANNELS):
            js = [j for j in range(NUM_CHANNELS) if torch.allclose(ot[i], apply_g(obs[j], g))]
            if i in js:
                perm.append(i)
            elif len(js) == 1:
                perm.append(js[0])
            else:
                raise RuntimeError(f'input channel {i} under g={g}: ambiguous/no source channel {js}')
        if sorted(perm) != list(range(NUM_CHANNELS)) or any(perm[i] != i for i in range(NUM_CHANNELS) if not 13 <= i <= 16):
            raise RuntimeError(f'unexpected input permutation for g={g}: {perm}')
        perms.append(perm)
    return torch.tensor(perms, dtype=torch.long)


class LiftConv(nn.Module):
    """Input -> group features: out[c, g] = sum_i x_i * T_g(K[c, pi_g(i)]). The expanded (C*8, cin, 3, 3) weight is
    a fixed gather of the free (C, cin, 3, 3) weight, precomputed once as an index (one op per forward)."""
    def __init__(self, cin, cout, perm):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin, 3, 3))
        nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        src = torch.arange(self.weight.numel()).reshape(self.weight.shape)
        ws = [apply_g(src[:, perm[g]], g) for g in range(G)]                       # (cout, cin, 3, 3) each
        self.register_buffer('index', torch.stack(ws, 1).reshape(cout * G, cin, 3, 3), persistent=False)

    def expanded_weight(self):
        return self.weight.reshape(-1)[self.index]

    def forward(self, x):
        return F.conv2d(x, self.expanded_weight(), padding=1)


class GConv(nn.Module):
    """Group features -> group features: out[c, g] = sum_{c', h} f[c', h] * T_g(K[c, c', g^-1 h]); the expanded
    (cout*8, cin*8, k, k) weight is a precomputed gather of the free (cout, cin, 8, k, k) weight."""
    def __init__(self, cin, cout, k):
        super().__init__()
        self.k = k
        self.weight = nn.Parameter(torch.empty(cout, cin, G, k, k))
        nn.init.kaiming_normal_(self.weight.view(cout, cin * G, k, k), nonlinearity='relu')
        src = torch.arange(self.weight.numel()).reshape(self.weight.shape)
        ws = [apply_g(src[:, :, [MULT[INV[g]][h] for h in range(G)]], g) for g in range(G)]
        self.register_buffer('index', torch.stack(ws, 1).reshape(cout * G, cin * G, k, k), persistent=False)

    def expanded_weight(self):
        return self.weight.reshape(-1)[self.index]

    def forward(self, x):
        return F.conv2d(x, self.expanded_weight(), padding=self.k // 2)


class GBN(nn.Module):
    """BatchNorm per group channel, statistics shared across the 8 orientations (keeps equivariance)."""
    def __init__(self, c):
        super().__init__()
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        b, cg, h, w = x.shape
        return self.bn(x.reshape(b, cg // G, G * h, w)).reshape(b, cg, h, w)


class GResBlock(nn.Module):
    """Pre-activation residual block on group features (as alphatrain.model.ResBlock)."""
    def __init__(self, c):
        super().__init__()
        self.bn1 = GBN(c); self.conv1 = GConv(c, c, 3)
        self.bn2 = GBN(c); self.conv2 = GConv(c, c, 3)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out + x


class PolicyNetP4M(PolicyNet):
    """PolicyNet with a D4-equivariant trunk; same forward / forward_with_features / PAIR2 head."""

    def __init__(self, in_channels=18, num_blocks=18, group_channels=16, expand=64,
                 policy_channels=128, head='pair2', pair_dim=64):
        super().__init__(in_channels=in_channels, num_blocks=0, channels=2 * expand,
                         policy_channels=policy_channels, head=head, pair_dim=pair_dim)
        del self.stem, self.blocks, self.backbone_bn                 # replaced by the group trunk
        self.group_channels, self.expand, self.num_blocks = group_channels, expand, num_blocks
        self.stem_lift = LiftConv(in_channels, group_channels, input_permutation())
        self.stem_bn = GBN(group_channels)
        self.gblocks = nn.Sequential(*[GResBlock(group_channels) for _ in range(num_blocks)])
        self.trunk_bn = GBN(group_channels)
        self.expand_conv = GConv(group_channels, expand, 1)
        self.expand_bn = GBN(expand)

    def backbone_features(self, x):
        out = F.relu(self.stem_bn(self.stem_lift(x)))
        out = self.gblocks(out)
        out = self.expand_bn(self.expand_conv(F.relu(self.trunk_bn(out))))
        out = F.relu(out)
        b, _, h, w = out.shape
        out = out.reshape(b, self.expand, G, h * w)                  # 4-D: MPS reduces a 5-D middle dim ~20x slower
        return torch.cat([out.mean(2), out.amax(2)], dim=1).reshape(b, 2 * self.expand, h, w)   # invariant per-cell features

    def freeze(self):
        """Inference/export form (call in eval mode): every group conv becomes a plain Conv2d holding its expanded
        weight and every group BN a BatchNorm2d over the C*8 actual channels (statistics repeated per orientation).
        The result is an ordinary CNN computing the same function, so TorchScript tracing and BN folding
        (export_ts.fold_batchnorm) work as for PolicyNet. No longer trainable as a group CNN."""
        def conv(m):
            w = m.expanded_weight().detach()
            c = nn.Conv2d(w.shape[1], w.shape[0], w.shape[-1], padding=w.shape[-1] // 2, bias=False)
            c.weight.data.copy_(w)
            return c.to(device=w.device, dtype=w.dtype).train(False)

        def bn(m):
            b = m.bn
            out = nn.BatchNorm2d(b.num_features * G, eps=b.eps, momentum=b.momentum)
            for name in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(out, name).data.copy_(getattr(b, name).detach().repeat_interleave(G))
            out.num_batches_tracked.copy_(b.num_batches_tracked)
            return out.to(device=b.weight.device, dtype=b.weight.dtype).train(False)

        if self.training:
            raise RuntimeError('freeze() needs eval mode (BatchNorm running statistics)')
        self.stem_lift, self.stem_bn = conv(self.stem_lift), bn(self.stem_bn)
        for blk in self.gblocks:
            blk.bn1, blk.conv1, blk.bn2, blk.conv2 = bn(blk.bn1), conv(blk.conv1), bn(blk.bn2), conv(blk.conv2)
        self.trunk_bn = bn(self.trunk_bn)
        self.expand_conv, self.expand_bn = conv(self.expand_conv), bn(self.expand_bn)
        return self


def is_p4m_state(state):
    return any(k.endswith('stem_lift.weight') for k in state)


def p4m_kwargs_from_state(state):
    from alphatrain.model import head_kwargs_from_state
    w = next(v for k, v in state.items() if k.endswith('stem_lift.weight'))
    e = next(v for k, v in state.items() if k.endswith('expand_conv.weight'))
    nb = sum(1 for k in state if '.gblocks.' in '.' + k and k.endswith('.conv1.weight'))
    return dict(in_channels=int(w.shape[1]), num_blocks=nb, group_channels=int(w.shape[0]),
                expand=int(e.shape[0]), **head_kwargs_from_state(state))
