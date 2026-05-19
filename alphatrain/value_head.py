"""Multi-horizon survival value head for MCTS leaf evaluation.

Replaces the linear feature evaluator (`feature_value_weights_*.npz`,
Pearson r ≈ 0.5 with truth) for the next iteration of NN-driven MCTS.

Design (per ChatGPT review 2026-05-06):
- Small head attached to a *frozen* PolicyNet backbone — avoids the
  policy/value gradient conflict that broke the dual-head era
  (HISTORY 90-118).
- Multi-horizon survival classification: 4 binary outputs predicting
  `P(game survives ≥ H more turns from this position)` for
  H ∈ {25, 50, 100, 200}. Short horizons (vs single
  log(remaining_turns) regression) properly handle cap censoring on
  capped games and align with what MCTS Q-norm cares about: tactical
  risk in the next few moves, not asymptotic survival.
- Risk-focused MCTS scalar — front-loaded combination favors
  tactical certainty over long-horizon noise:
    V = 1.0·p25 + 0.8·p50 + 0.5·p100 + 0.25·p200

Training pipeline lives in `alphatrain/scripts/train_value_head.py`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# Survival horizons. Order is fixed and load-bearing — the head's 4
# output logits correspond to these horizons in order. Changing this
# requires retraining + a checkpoint format bump.
SURVIVAL_HORIZONS = (25, 50, 100, 200)
NUM_HORIZONS = len(SURVIVAL_HORIZONS)

# Default risk-focused weights for combining horizon probabilities into
# a scalar leaf value for MCTS. Earlier horizons weight more — they
# reflect tactical certainty; longer horizons get noisier as the game
# transitions through regimes.
DEFAULT_HORIZON_WEIGHTS = (1.0, 0.8, 0.5, 0.25)
assert len(DEFAULT_HORIZON_WEIGHTS) == NUM_HORIZONS


class ValueHead(nn.Module):
    """Tiny head over a (frozen) policy backbone.

    Input: backbone features (B, channels, 9, 9) — what
    `PolicyNet.backbone_features()` returns.
    Output: (B, num_outputs) logits.
      - num_outputs=NUM_HORIZONS (default, 4): per-horizon survival logits.
        Apply sigmoid for probabilities, combine via survival_to_scalar().
      - num_outputs=1: scalar V directly. Used for pairwise-ranking heads
        (BPR loss during training). Inference reads the raw scalar.

    Architecture:
        conv(channels → hidden, 1×1) → BN → ReLU → GAP → linear(hidden → num_outputs)

    Parameter count: ~8K at hidden=32 (negligible vs 12M-param frozen backbone).
    """

    def __init__(self, in_channels=256, hidden=32, num_outputs=NUM_HORIZONS):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.num_outputs = num_outputs
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.fc = nn.Linear(hidden, num_outputs)

    def forward(self, backbone_features):
        """backbone_features: (B, channels, 9, 9). Returns (B, num_outputs)."""
        x = F.relu(self.bn(self.conv(backbone_features)))
        x = x.mean(dim=(2, 3))         # GAP over spatial dims → (B, hidden)
        return self.fc(x)              # (B, num_outputs) logits or scalars


def survival_to_scalar(probs, horizon_weights=DEFAULT_HORIZON_WEIGHTS):
    """Combine per-horizon survival probabilities into one scalar V.

    Args:
        probs: (B, NUM_HORIZONS) tensor of P(survive ≥ H) per horizon
        horizon_weights: optional override of the front-loaded default

    Returns:
        (B,) tensor of scalar leaf values. Higher = better board.
    """
    w = torch.tensor(horizon_weights, dtype=probs.dtype, device=probs.device)
    return (probs * w).sum(dim=-1)


def save(value_head, path, *, backbone_path, train_args=None,
         val_metrics=None, horizons=SURVIVAL_HORIZONS, target_type='survival'):
    """Serialize a trained ValueHead.

    Stores enough metadata to fully reconstruct: backbone reference
    (so MCTS knows which PolicyNet to attach it to), horizon list,
    target_type (survival vs density), optional training/val metadata.
    """
    torch.save({
        'state_dict': value_head.state_dict(),
        'in_channels': value_head.in_channels,
        'hidden': value_head.hidden,
        'num_outputs': value_head.num_outputs,
        'horizons': list(horizons),
        'target_type': target_type,
        'backbone_path': backbone_path,
        'train_args': train_args,
        'val_metrics': val_metrics,
    }, path)


def load(path, device=None):
    """Load a saved ValueHead checkpoint.

    Returns (value_head, metadata). The caller is responsible for loading
    the matching backbone (from `metadata['backbone_path']`) and wiring
    them together for inference.
    """
    map_loc = device if device is not None else 'cpu'
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    num_outputs = ckpt.get('num_outputs', NUM_HORIZONS)  # backward compat
    head = ValueHead(in_channels=ckpt['in_channels'], hidden=ckpt['hidden'],
                     num_outputs=num_outputs)
    head.load_state_dict(ckpt['state_dict'])
    if device is not None:
        head = head.to(device)
    head.train(False)  # inference mode (avoid named-call false-positive)
    if num_outputs == 1:
        # Scalar-output head (pairwise-ranking). No horizon check.
        return head, ckpt
    target_type = ckpt.get('target_type', 'survival')
    if target_type == 'density':
        # Density heads use different horizons (e.g., {5, 15, 50}). Skip check.
        return head, ckpt
    horizons = tuple(ckpt['horizons'])
    if horizons != SURVIVAL_HORIZONS:
        raise ValueError(
            f"Checkpoint horizons {horizons} don't match the current "
            f"SURVIVAL_HORIZONS {SURVIVAL_HORIZONS}. Either retrain or "
            f"update the constant.")
    return head, ckpt


class SpatialValueHead(nn.Module):
    """Spatial-preserving value head for pairwise ranking (Pillar 3a-v2).

    ~170K params. Preserves the 9×9 board geometry through residual conv
    blocks before pooling. Uses both mean and max pool to capture average
    and extreme features. Scalar output for pairwise ranking via BPR loss.

    Architecture:
        backbone features (B, 256, 9, 9)
        → 1×1 conv 256→64, BN, ReLU
        → ResBlock 3×3 (64→64), BN, ReLU
        → ResBlock 3×3 (64→64), BN, ReLU
        → [global mean pool, global max pool] → concat (128)
        → Linear 128→64, ReLU
        → Linear 64→num_outputs
    """

    def __init__(self, in_channels=256, mid_channels=64, num_outputs=1):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_outputs = num_outputs

        self.conv0 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_channels)

        # ResBlock 1
        self.res1_conv1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.res1_bn1 = nn.BatchNorm2d(mid_channels)
        self.res1_conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.res1_bn2 = nn.BatchNorm2d(mid_channels)

        # ResBlock 2
        self.res2_conv1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.res2_bn1 = nn.BatchNorm2d(mid_channels)
        self.res2_conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.res2_bn2 = nn.BatchNorm2d(mid_channels)

        self.fc1 = nn.Linear(mid_channels * 2, 64)
        self.fc2 = nn.Linear(64, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        h = F.relu(self.res1_bn1(self.res1_conv1(x)))
        h = self.res1_bn2(self.res1_conv2(h))
        x = F.relu(x + h)
        h = F.relu(self.res2_bn1(self.res2_conv1(x)))
        h = self.res2_bn2(self.res2_conv2(h))
        x = F.relu(x + h)
        mean_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        h = F.relu(self.fc1(pooled))
        return self.fc2(h)


def save_spatial(head, path, *, backbone_path, train_args=None,
                 val_metrics=None):
    """Serialize a SpatialValueHead checkpoint."""
    torch.save({
        'head_type': 'spatial',
        'state_dict': head.state_dict(),
        'in_channels': head.in_channels,
        'mid_channels': head.mid_channels,
        'num_outputs': head.num_outputs,
        'target_type': 'pairwise_ranking',
        'backbone_path': backbone_path,
        'train_args': train_args,
        'val_metrics': val_metrics,
    }, path)


def load_spatial(path, device=None):
    """Load a SpatialValueHead checkpoint. Raises if not a spatial head."""
    map_loc = device if device is not None else 'cpu'
    ckpt = torch.load(path, map_location=map_loc, weights_only=False)
    head_type = ckpt.get('head_type', 'value_head')
    if head_type != 'spatial':
        raise ValueError(
            f"Expected head_type='spatial', got {head_type!r}. "
            f"Use load() for survival/density heads.")
    head = SpatialValueHead(
        in_channels=ckpt['in_channels'],
        mid_channels=ckpt['mid_channels'],
        num_outputs=ckpt['num_outputs'],
    )
    head.load_state_dict(ckpt['state_dict'])
    if device is not None:
        head = head.to(device)
    head.train(False)
    return head, ckpt


def load_any(path, device=None):
    """Load any value-head checkpoint (ValueHead or SpatialValueHead).

    Returns (head, ckpt, head_type) where head_type is 'value_head' or 'spatial'.
    """
    map_loc = device if device is not None else 'cpu'
    peek = torch.load(path, map_location=map_loc, weights_only=False)
    head_type = peek.get('head_type', 'value_head')
    if head_type == 'spatial':
        head, ckpt = load_spatial(path, device=device)
    else:
        head, ckpt = load(path, device=device)
    return head, ckpt, head_type
