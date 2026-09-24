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


# ── distillation_loss set-valued rows ─────────────────────────────────

def test_set_loss_tie_row_costs_nothing_when_student_in_set():
    """Two acceptable moves (visits 0.5/0.5): a student fully on either pays ~0;
    a student on a third move pays -log P(set)."""
    tgt = torch.zeros(3, 10); tgt[:, 1] = 0.5; tgt[:, 2] = 0.5
    logits = torch.full((3, 10), -20.0)
    logits[0, 1] = 20.0            # student on move 1 (in set)
    logits[1, 2] = 20.0            # student on move 2 (in set)
    logits[2, 5] = 20.0            # student on move 5 (outside set)
    rows = torch.tensor([True, True, True])
    per = torch.stack([distillation_loss(logits[i:i+1], tgt[i:i+1], blend_alpha=0.0,
                                         set_rows=rows[i:i+1], set_tau=0.5) for i in range(3)])
    assert per[0] < 1e-3 and per[1] < 1e-3
    assert per[2] > 10.0
    # hard CE on the forced argmax would have punished row 1 (student on move 2)
    assert distillation_loss(logits[1:2], tgt[1:2], blend_alpha=0.0) > 10.0


def test_set_loss_singleton_equals_hard_ce_and_unmasked_rows_unchanged():
    torch.manual_seed(0)
    logits = torch.randn(4, 10)
    tgt = torch.zeros(4, 10); tgt[torch.arange(4), torch.tensor([3, 1, 7, 0])] = 1.0
    rows = torch.tensor([True, False, True, False])
    a = distillation_loss(logits, tgt, blend_alpha=0.0, set_rows=rows, set_tau=0.5)
    b = distillation_loss(logits, tgt, blend_alpha=0.0)
    assert torch.allclose(a, b, atol=1e-6)


# ── LR schedule: warmup 0 must train at the requested LR ──────────────

def test_warmup_zero_schedule_starts_at_requested_lr():
    import subprocess, sys, re
    import os
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'train_path_b.py')).read()
    assert 'if args.warmup_epochs > 0:' in src
    # emulate the scheduler block for warmup 0 and 1
    import torch
    for warm, first in ((0, 1e-4), (1, 1e-5)):
        p = torch.nn.Parameter(torch.zeros(1)); opt = torch.optim.AdamW([p], lr=1e-4)
        scheds, ms = [], []
        if warm > 0:
            scheds.append(torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm)); ms.append(warm)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, 6 - warm), eta_min=1e-6); scheds.append(cos)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, scheds, milestones=ms) if len(scheds) > 1 else cos
        assert abs(opt.param_groups[0]['lr'] - first) < 1e-9, (warm, opt.param_groups[0]['lr'])


def test_legal_mask_from_obs_matches_reference_legality():
    import numpy as np, torch
    from alphatrain.train_path_b import legal_mask_from_obs
    from alphatrain.observation import build_observation
    from alphatrain.mcts import _legal_priors_jit
    rng = np.random.default_rng(0)
    for _ in range(30):
        b = np.zeros(81, np.int8); occ = rng.choice(81, rng.integers(10, 70), replace=False); b[occ] = rng.integers(1, 8, len(occ))
        b = b.reshape(9, 9); empty = np.flatnonzero(b.reshape(81) == 0)[:3]
        obs = build_observation(b, empty // 9, empty % 9, np.ones(len(empty), np.int64), len(empty))
        got = legal_mask_from_obs(torch.from_numpy(obs[None]))[0].numpy()
        c, idx, _ = _legal_priors_jit(b, np.zeros(6561, np.float32), 6561)
        ref = np.zeros(6561, bool); ref[idx[:c]] = True
        assert np.array_equal(got, ref)
