"""AlphaTrain ResNet model.

Dual-head ResNet for Color Lines 98:
- Input: (batch, 18, 9, 9) — tactical observation channels
- Backbone: Conv stem + N residual blocks
- Policy head: (batch, 6561) flat joint logits
- Value head: (batch, num_bins) categorical score distribution

Design choices:
- Pre-activation ResBlocks (BN→ReLU→Conv) for better gradient flow
- Policy via conv (256→128→81 channels) reshpaed to 6561 — no huge linear layer
- Categorical value with two-hot targets — handles high-variance scores
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


class AlphaTrainNet(nn.Module):
    """Policy ResNet for AlphaTrain.

    Args:
        in_channels: observation channels (default 18)
        num_blocks: residual blocks (default 10)
        channels: hidden width (default 256)
        policy_channels: intermediate policy conv channels (default 128)
        num_value_bins: kept for checkpoint compatibility (ignored)
    """

    def __init__(self, in_channels=18, num_blocks=10, channels=256,
                 policy_channels=128, num_value_bins=64,
                 value_channels=32, value_hidden=512, value_dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_value_bins = num_value_bins

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        # Backbone
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.backbone_bn = nn.BatchNorm2d(channels)

        # Policy head: conv → (batch, 81, 9, 9) → (batch, 6561)
        self.policy_conv1 = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_conv2 = nn.Conv2d(policy_channels, 81, 1)

    def forward(self, x):
        """Returns (policy_logits, None)."""
        out = self.stem(x)
        out = self.blocks(out)
        out = F.relu(self.backbone_bn(out))

        p = F.relu(self.policy_bn(self.policy_conv1(out)))
        p = self.policy_conv2(p)
        policy_logits = p.reshape(p.size(0), -1)

        return policy_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
