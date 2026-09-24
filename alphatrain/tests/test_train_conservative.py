import torch
from torch import nn

from alphatrain.train_conservative import (
    conservative_policy_loss,
    make_frozen_base_and_student,
)


def test_conservative_loss_has_analytic_mixture_optimum():
    base_logits = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float64)
    base_p = torch.softmax(base_logits, dim=-1)
    teacher = torch.tensor([1])
    eta = 0.3
    target = base_p.clone()
    target[0, 1] += eta
    target /= 1.0 + eta

    student_logits = target.log().clone().detach().requires_grad_(True)
    loss, _, _ = conservative_policy_loss(
        base_logits, student_logits, teacher, torch.tensor([eta]))
    loss.backward()
    torch.testing.assert_close(
        student_logits.grad, torch.zeros_like(student_logits.grad),
        atol=1e-6, rtol=0)


def test_zero_edit_weight_preserves_base_distribution():
    base_logits = torch.tensor([[1.0, 0.0, -2.0]])
    student_logits = base_logits.clone().detach().requires_grad_(True)
    loss, kl, _ = conservative_policy_loss(
        base_logits, student_logits, torch.tensor([2]), torch.tensor([0.0]))
    loss.backward()
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-7, rtol=0)
    torch.testing.assert_close(
        student_logits.grad, torch.zeros_like(student_logits.grad),
        atol=1e-7, rtol=0)


def test_student_clone_is_trainable_after_base_is_frozen():
    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    base, student = make_frozen_base_and_student(model)

    assert not base.training
    assert all(not p.requires_grad for p in base.parameters())
    assert student.training
    assert all(p.requires_grad for p in student.parameters())
    for key, value in base.state_dict().items():
        torch.testing.assert_close(student.state_dict()[key], value)
