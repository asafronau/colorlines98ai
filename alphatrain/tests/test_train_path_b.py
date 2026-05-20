"""Unit tests for V-corpus distillation trainer.

Path B oracle code was removed (HISTORY 145, 151-152). This file kept its
name for git history continuity but now tests only the surviving loss
shape: cross_entropy_soft + distillation_loss with target_temperature.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from alphatrain.train_path_b import (
    cross_entropy_soft,
    distillation_loss,
)


@pytest.fixture(autouse=True)
def _torch_seed():
    torch.manual_seed(0)


# ── cross_entropy_soft ────────────────────────────────────────────────

def test_cross_entropy_soft_matches_F_kl_when_target_normalizes():
    """When targets sum to 1, our soft CE matches negative log-likelihood
    against those targets."""
    B, V = 4, 10
    logits = torch.randn(B, V)
    targets = torch.rand(B, V)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    loss = cross_entropy_soft(logits, targets)
    expected = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
    assert math.isclose(loss.item(), expected.item(), abs_tol=1e-6)


# ── distillation_loss target_temperature ──────────────────────────────

def test_distillation_loss_T1_matches_cross_entropy_soft():
    """target_temperature=1.0 means no change — should match base CE."""
    B, V = 4, 10
    logits = torch.randn(B, V)
    targets = torch.rand(B, V)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    loss_T1 = distillation_loss(logits, targets, target_temperature=1.0)
    base = cross_entropy_soft(logits, targets)
    assert math.isclose(loss_T1.item(), base.item(), abs_tol=1e-6)


def test_distillation_loss_T_sharpens_targets():
    """T<1.0 should sharpen targets via target**(1/T) renormalized.

    Verifies the FORMULA directly — that the sharpening applied inside
    distillation_loss matches a manual `target**(1/T) / sum`.
    """
    B, V = 1, 6
    target = torch.tensor([[0.4, 0.2, 0.15, 0.15, 0.05, 0.05]])
    logits = torch.randn(B, V)  # any model output

    # Manual sharpen at T=0.5: target^2 normalized
    sharp_T05 = target ** (1.0 / 0.5)
    sharp_T05 = sharp_T05 / sharp_T05.sum(dim=-1, keepdim=True)
    # Sanity: top1 of sharp should be much higher than top1 of target
    assert sharp_T05[0, 0].item() > target[0, 0].item()
    assert math.isclose(sharp_T05[0, 0].item(), 0.64, abs_tol=0.01)

    # distillation_loss with target_temperature=0.5 should equal
    # cross_entropy_soft against the manually-sharpened target.
    loss_distill = distillation_loss(logits, target, target_temperature=0.5)
    loss_manual = cross_entropy_soft(logits, sharp_T05)
    assert math.isclose(loss_distill.item(), loss_manual.item(), abs_tol=1e-5)


def test_distillation_loss_T_collapses_to_argmax_at_low_T():
    """As T → 0, sharpened target should approach one-hot at argmax.
    Use T=0.1 (not T=0.01) to avoid float32 underflow in target**100."""
    B, V = 1, 6
    target = torch.tensor([[0.4, 0.2, 0.15, 0.15, 0.05, 0.05]])
    logits = torch.zeros_like(target)
    loss = distillation_loss(logits, target, target_temperature=0.1)
    # Hard CE on argmax (index 0) with uniform logits = log(6) ≈ 1.792
    hard_ce_uniform = -float(torch.log(torch.tensor(1.0/6)))
    # T=0.1 is enough to make sharp ~= one-hot for this target
    assert math.isclose(loss.item(), hard_ce_uniform, abs_tol=0.05)


# ── distillation_loss blend_alpha ─────────────────────────────────────

def test_distillation_loss_blend_alpha_zero_equals_hard_ce():
    """blend_alpha=0 → pure hard CE on argmax."""
    B, V = 4, 10
    logits = torch.randn(B, V)
    targets = torch.rand(B, V)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    loss_blended = distillation_loss(logits, targets, blend_alpha=0.0)
    argmax_idx = targets.argmax(dim=-1)
    expected_hard = F.cross_entropy(logits, argmax_idx)
    assert math.isclose(loss_blended.item(), expected_hard.item(),
                          abs_tol=1e-6)


def test_distillation_loss_blend_alpha_one_equals_soft_ce():
    """blend_alpha=1 → pure soft CE."""
    B, V = 4, 10
    logits = torch.randn(B, V)
    targets = torch.rand(B, V)
    targets = targets / targets.sum(dim=-1, keepdim=True)
    loss_blended = distillation_loss(logits, targets, blend_alpha=1.0)
    soft = cross_entropy_soft(logits, targets)
    assert math.isclose(loss_blended.item(), soft.item(), abs_tol=1e-6)


# ── End-to-end smoke ──────────────────────────────────────────────────

def test_smoke_one_training_step():
    """End-to-end: forward + distillation_loss + backward + step on a tiny
    network. Must not crash, must produce finite loss + gradients."""
    from alphatrain.model import AlphaTrainNet
    torch.manual_seed(0)
    net = AlphaTrainNet(num_blocks=1, channels=8)
    net.train(True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    B = 4
    V = 6561
    obs = torch.randn(B, 18, 9, 9)
    pol = torch.zeros(B, V)
    pol[:, 17] = 1.0  # one-hot at action 17

    out = net(obs)
    logits = out[0] if isinstance(out, tuple) else out
    loss = distillation_loss(logits, pol, target_temperature=0.5)
    assert torch.isfinite(loss).item()
    opt.zero_grad()
    loss.backward()
    # Check gradients flow
    any_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in net.parameters() if p.requires_grad)
    assert any_grad, "no gradient flowed during backward"
    opt.step()
