"""AlphaTrain ResNet model — policy-only.

10x256 ResNet for Color Lines 98:
- Input: (batch, 18, 9, 9) — tactical observation channels
- Backbone: Conv stem + N residual blocks
- Policy head: (batch, 6561) flat joint logits

The original dual-head (policy + value) architecture was retired after
HISTORY lessons 112-118. The NN value head never learned meaningful
survival signal (R² ≈ 0.03 on remaining-turns); training conflict on
the shared backbone destroyed policy quality. V10+ models are policy-
only and use a separate feature-value evaluator at MCTS leaves.

Future work (HISTORY lessons 130+): a small value head trained on a
*frozen* backbone, with multi-horizon survival classification targets,
is the planned escape from the linear evaluator's r ≈ 0.5 ceiling.
That head should be a separate `nn.Module` (not built into this class)
so the policy training loop stays clean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4  # 6561


class ResBlock(nn.Module):
    """Pre-activation residual block."""

    def __init__(self, channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + x


class PolicyNet(nn.Module):
    """Policy-only ResNet for AlphaTrain.

    Single-output forward returning policy logits of shape
    (batch, 6561) — flat joint encoding of (source_idx * 81 + target_idx).

    Args:
        in_channels: observation channels (default 18)
        num_blocks: residual blocks (default 10)
        channels: hidden width (default 256)
        policy_channels: intermediate policy conv channels (default 128)
    """

    def __init__(self, in_channels=18, num_blocks=10, channels=256,
                 policy_channels=128, head='abs', pair_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        # head='abs' (historical): logits for all 81 ABSOLUTE source cells are
        #   read out of the destination cell's features by one 1x1 conv, so the
        #   "which ball" axis shares nothing across positions or rotations.
        # head='pair': every cell gets a source embedding u and a destination
        #   embedding w (shared 1x1 convs); logit(s, d) = u(s).w(d)/sqrt(k)
        #   + a(s) + b(d). The head is exactly equivariant to any permutation
        #   of cells, so symmetry only has to be learned by the backbone.
        # head='pair2': 'pair' + two geometry-gated bilinear terms, one active only
        #   for (s, d) on a common row/column/diagonal within 4 cells (the line
        #   geometry of the game), one only for orthogonal neighbours. The gates
        #   are D4-invariant, so the head is exactly D4-equivariant but, unlike
        #   'pair', can tell where s sits relative to d.
        if head not in ('abs', 'pair', 'pair2'):
            raise ValueError(f'unknown policy head {head!r}')
        self.head = head

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Backbone
        self.blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_blocks)])
        self.backbone_bn = nn.BatchNorm2d(channels)

        # Policy head: conv → (batch, 81, 9, 9) → (batch, 6561)
        self.policy_conv1 = nn.Conv2d(
            channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        if head == 'abs':
            self.policy_conv2 = nn.Conv2d(policy_channels, 81, 1)
        else:
            if head == 'pair2':
                self.line_src = nn.Conv2d(policy_channels, pair_dim, 1)
                self.line_dst = nn.Conv2d(policy_channels, pair_dim, 1)
                self.adj_src = nn.Conv2d(policy_channels, pair_dim, 1)
                self.adj_dst = nn.Conv2d(policy_channels, pair_dim, 1)
                line = torch.zeros(81, 81); adj = torch.zeros(81, 81)
                for s_ in range(81):
                    for d_ in range(81):
                        dr, dc = s_ // 9 - d_ // 9, s_ % 9 - d_ % 9
                        dist = max(abs(dr), abs(dc))
                        if 1 <= dist <= 4 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
                            line[s_, d_] = 1.0
                        if abs(dr) + abs(dc) == 1:
                            adj[s_, d_] = 1.0
                self.register_buffer('line_gate', line, persistent=False)
                self.register_buffer('adj_gate', adj, persistent=False)
            self.pair_dim = pair_dim
            self.pair_src = nn.Conv2d(policy_channels, pair_dim, 1)
            self.pair_dst = nn.Conv2d(policy_channels, pair_dim, 1)
            self.pair_sbias = nn.Conv2d(policy_channels, 1, 1)
            self.pair_dbias = nn.Conv2d(policy_channels, 1, 1)

    def forward(self, x):
        """Returns policy_logits of shape (batch, 6561)."""
        feats = self.backbone_features(x)
        return self._policy_from_features(feats)

    def backbone_features(self, x):
        """Run stem + blocks + backbone_bn + ReLU; return (B, channels, 9, 9).

        Used by the frozen-backbone value head — the head trains on
        these features without backprop into the policy net.
        """
        out = self.stem(x)
        out = self.blocks(out)
        return F.relu(self.backbone_bn(out))

    def _policy_from_features(self, feats):
        """Apply the policy head to backbone features. Public via
        forward_with_features() for callers that want both."""
        p = F.relu(self.policy_bn(self.policy_conv1(feats)))
        if self.head == 'abs':
            p = self.policy_conv2(p)
            return p.reshape(p.size(0), -1)
        b = p.size(0)
        u = self.pair_src(p).reshape(b, self.pair_dim, 81)          # (B, k, source cells)
        w = self.pair_dst(p).reshape(b, self.pair_dim, 81)          # (B, k, destination cells)
        logits = torch.bmm(u.transpose(1, 2), w) * (self.pair_dim ** -0.5)   # (B, s, d)
        logits = logits + self.pair_sbias(p).reshape(b, 81, 1) + self.pair_dbias(p).reshape(b, 1, 81)
        if self.head == 'pair2':
            sc = self.pair_dim ** -0.5
            for src, dst, gate in ((self.line_src, self.line_dst, self.line_gate), (self.adj_src, self.adj_dst, self.adj_gate)):
                g = torch.bmm(src(p).reshape(b, self.pair_dim, 81).transpose(1, 2), dst(p).reshape(b, self.pair_dim, 81))
                logits = logits + g * sc * gate
        return logits.reshape(b, 81 * 81)                           # index = s*81 + d

    def forward_with_features(self, x):
        """Returns (policy_logits, backbone_features).

        Useful for ValueHead inference where we want the policy AND
        the features in one forward pass — avoids redundant backbone
        compute. Policy head + ValueHead can share `feats`.
        """
        feats = self.backbone_features(x)
        pol_logits = self._policy_from_features(feats)
        return pol_logits, feats


# Back-compat alias: old checkpoints reference AlphaTrainNet by name.
# `load_model` filters value_* keys so old dual-head checkpoints load
# into PolicyNet without erroring. Once V10+ pillar2x/2y/etc. are the
# only checkpoints we read, this alias can be deleted.
AlphaTrainNet = PolicyNet


def head_kwargs_from_state(state):
    """Infer policy-head constructor kwargs from a state_dict."""
    if any(k.endswith('pair_src.weight') for k in state):
        w = next(v for k, v in state.items() if k.endswith('pair_src.weight'))
        head = 'pair2' if any(k.endswith('line_src.weight') for k in state) else 'pair'
        return {'head': head, 'pair_dim': int(w.shape[0]), 'policy_channels': int(w.shape[1])}
    w = next((v for k, v in state.items() if k.endswith('policy_conv1.weight')), None)
    return {'head': 'abs', 'policy_channels': int(w.shape[0])} if w is not None else {}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
