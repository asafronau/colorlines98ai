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
    Output: (B, NUM_HORIZONS) logits (apply sigmoid for probabilities).

    Architecture:
        conv(channels → hidden, 1×1) → BN → ReLU → GAP → linear(hidden → 4)

    Parameter count: ~8K (negligible vs the 12M-param frozen backbone).
    """

    def __init__(self, in_channels=256, hidden=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden)
        self.fc = nn.Linear(hidden, NUM_HORIZONS)

    def forward(self, backbone_features):
        """backbone_features: (B, channels, 9, 9). Returns (B, NUM_HORIZONS) logits."""
        x = F.relu(self.bn(self.conv(backbone_features)))
        x = x.mean(dim=(2, 3))         # GAP over spatial dims → (B, hidden)
        return self.fc(x)              # (B, NUM_HORIZONS) logits


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
         val_metrics=None, horizons=SURVIVAL_HORIZONS):
    """Serialize a trained ValueHead.

    Stores enough metadata to fully reconstruct: backbone reference
    (so MCTS knows which PolicyNet to attach it to), horizon list,
    optional training/val metadata.
    """
    torch.save({
        'state_dict': value_head.state_dict(),
        'in_channels': value_head.in_channels,
        'hidden': value_head.hidden,
        'horizons': list(horizons),
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
    head = ValueHead(in_channels=ckpt['in_channels'], hidden=ckpt['hidden'])
    head.load_state_dict(ckpt['state_dict'])
    if device is not None:
        head = head.to(device)
    head.train(False)  # inference mode (avoid named-call false-positive)
    horizons = tuple(ckpt['horizons'])
    if horizons != SURVIVAL_HORIZONS:
        raise ValueError(
            f"Checkpoint horizons {horizons} don't match the current "
            f"SURVIVAL_HORIZONS {SURVIVAL_HORIZONS}. Either retrain or "
            f"update the constant.")
    return head, ckpt
