import torch
import numpy as np

from alphatrain.train_flywheel import (
    PooledBatchLoader, _capture_rng_state, _completed_epoch,
    _deployment_numerics, _functional_gate,
    _quantize_frozen_bn_for_deployment_,
    _restore_rng_state, _validate_resume_args, audit, cap_dataset,
    flywheel_loss,
)


def test_bounded_soft_target_has_exact_mixture_optimum():
    base_logits = torch.tensor([[0.7, -0.1, 0.2]], dtype=torch.float64)
    base_p = torch.softmax(base_logits, -1)
    search = torch.tensor([[0.1, 0.7, 0.2]], dtype=torch.float64)
    eta = 0.2
    target = (base_p + eta * search) / (1 + eta)
    student = target.log().detach().clone().requires_grad_(True)
    loss, _, _ = flywheel_loss(
        base_logits, student, search, torch.tensor([1]), torch.tensor([1.0]),
        objective='bounded', eta=eta, soft_alpha=1.0, is_target=True,
        row_weight=torch.tensor([16.0]))
    loss.backward()
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad),
                               atol=1e-6, rtol=0)


def test_anchor_is_zero_gradient_at_base_for_bounded_objective():
    base = torch.tensor([[0.3, -0.2, 0.8]])
    student = base.detach().clone().requires_grad_(True)
    loss, kl, _ = flywheel_loss(
        base, student, torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([0]),
        torch.tensor([0.0]), objective='bounded', eta=0.1,
        soft_alpha=0.5, is_target=False)
    loss.backward()
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-7, rtol=0)
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad),
                               atol=1e-7, rtol=0)


def test_legacy_anchor_is_hard_behavior_ce():
    logits = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    loss, _, _ = flywheel_loss(
        logits.detach(), logits, torch.tensor([[0.0, 1.0, 0.0]]),
        torch.tensor([2]), torch.tensor([0.0]), objective='legacy', eta=0.0,
        soft_alpha=1.0, is_target=False)
    torch.testing.assert_close(loss, torch.log(torch.tensor(3.0)))


def test_loss_accepts_mixed_target_and_anchor_rows():
    logits = torch.zeros(2, 3, requires_grad=True)
    loss, _, _ = flywheel_loss(
        logits.detach(), logits,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([1, 2]), torch.tensor([1.0, 0.0]),
        objective='legacy', eta=0.0, soft_alpha=1.0,
        is_target=torch.tensor([True, False]))
    # Target row uses action 1; anchor row uses its hard behavior action 2.
    torch.testing.assert_close(loss, torch.log(torch.tensor(3.0)))


def test_legacy_anchor_weight_changes_only_anchor_row():
    logits = torch.zeros(2, 3, requires_grad=True)
    loss, _, _ = flywheel_loss(
        logits.detach(), logits,
        torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.tensor([1, 2]), torch.tensor([1.0, 0.0]),
        objective='legacy', eta=0.0, soft_alpha=0.0,
        is_target=torch.tensor([True, False]), anchor_weight=0.25)
    expected = (1.0 + 0.25) / 2 * torch.log(torch.tensor(3.0))
    torch.testing.assert_close(loss, expected)


def test_aggregate_uses_teacher_ce_on_targets_and_base_kl_on_anchors():
    base = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
    student = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    policy = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss, kl, _ = flywheel_loss(
        base, student, policy, torch.tensor([1, 0]),
        torch.tensor([1.0, 0.0]), objective='aggregate', eta=0.0,
        soft_alpha=0.0, is_target=torch.tensor([True, False]),
        anchor_weight=0.25)
    # Target row is exactly hard teacher CE; unchanged anchor has zero KL.
    expected = -torch.log_softmax(student[0], -1)[1] / 2
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(kl[1], torch.tensor(0.0), atol=1e-7, rtol=0)


def test_legal_support_loss_ignores_illegal_logits():
    legal = torch.tensor([[True, True, False]])
    base = torch.tensor([[0.0, 0.0, 100.0]])
    student = torch.tensor([[0.0, 0.0, -100.0]], requires_grad=True)
    common = dict(
        policy_target=torch.tensor([[1.0, 0.0, 0.0]]),
        hard_target_move=torch.tensor([0]), target_weight=torch.tensor([0.0]),
        objective='bounded', eta=0.1, soft_alpha=0.0, is_target=False)
    legal_loss, legal_kl, _ = flywheel_loss(
        base, student, legal_mask=legal, **common)
    full_loss, full_kl, _ = flywheel_loss(base, student, **common)
    torch.testing.assert_close(legal_loss, torch.tensor(0.0))
    torch.testing.assert_close(legal_kl, torch.tensor([0.0]))
    assert float(full_loss.detach()) > 100.0
    assert float(full_kl[0]) > 100.0


def test_legal_support_hard_ce_normalizes_over_legal_moves():
    logits = torch.tensor([[0.0, 0.0, 100.0]], requires_grad=True)
    loss, _, _ = flywheel_loss(
        logits.detach(), logits, torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([0]), torch.tensor([1.0]), objective='legacy', eta=0.0,
        soft_alpha=0.0, is_target=True,
        legal_mask=torch.tensor([[True, True, False]]))
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


def test_pooled_loader_mixes_and_consumes_every_row_once():
    class Stub:
        def __init__(self, offset, n):
            self.offset = offset
            self.n = n

        def __len__(self):
            return self.n

        def collate(self, indices):
            x = torch.as_tensor(indices) + self.offset
            return (x,) * 6

    loader = PooledBatchLoader(
        Stub(0, 3), Stub(100, 5), 4, shuffle=False)
    batches = list(loader)
    values = torch.cat([batch[0] for batch in batches])
    masks = torch.cat([batch[-1] for batch in batches])
    assert sorted(values.tolist()) == [0, 1, 2, 100, 101, 102, 103, 104]
    assert int(masks.sum()) == 3
    assert all(len(batch[0]) <= 4 for batch in batches)


def test_pooled_loader_mid_epoch_resume_reconstructs_exact_tail():
    class Stub:
        def __init__(self, offset, n):
            self.offset = offset
            self.n = n

        def __len__(self):
            return self.n

        def collate(self, indices):
            x = torch.as_tensor(indices) + self.offset
            return (x,) * 6

    loader = PooledBatchLoader(
        Stub(0, 11), Stub(100, 13), 5, shuffle=True)
    whole = list(loader.iter_from(epoch_seed=1234))
    tail = list(loader.iter_from(start_batch=2, epoch_seed=1234))
    assert len(tail) == len(whole) - 2
    for expected, actual in zip(whole[2:], tail):
        for expected_field, actual_field in zip(expected, actual):
            torch.testing.assert_close(expected_field, actual_field)


def test_audit_restores_training_mode():
    class Fixed(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(len(obs), 6561)

    model = Fixed().train(True)
    obs = torch.zeros(1, 18, 9, 9)
    obs[:, 0, 0, 0] = 1.0
    obs[:, 7] = 1.0
    obs[:, 7, 0, 0] = 0.0
    batch = (obs, torch.zeros(1, 6561), torch.ones(1),
             torch.tensor([1]), torch.tensor([1]), torch.tensor([0]))
    audit(model, Fixed(), {'target': [batch]}, torch.device('cpu'))
    assert model.training


def test_smoke_cap_is_random_and_reproducible():
    class Stub:
        device = torch.device('cpu')
        base_indices = torch.arange(100)

    a, b = Stub(), Stub()
    cap_dataset(a, 12, np.random.default_rng(7))
    cap_dataset(b, 12, np.random.default_rng(7))
    assert torch.equal(a.base_indices, b.base_indices)
    assert len(a.base_indices) == 12
    assert not torch.equal(a.base_indices, torch.arange(12))


def test_smoke_cap_can_reserve_all_declared_edits():
    class Stub:
        device = torch.device('cpu')
        base_indices = torch.arange(100)
        flywheel_disagree = torch.zeros(100, dtype=torch.bool)

    ds = Stub()
    ds.flywheel_disagree[[3, 17, 44, 81]] = True
    cap_dataset(ds, 12, np.random.default_rng(9), min_edits=100)
    assert len(ds.base_indices) == 12
    assert set((3, 17, 44, 81)).issubset(set(ds.base_indices.tolist()))


def test_audit_action_metrics_use_exact_legal_argmax():
    class Fixed(torch.nn.Module):
        def __init__(self, illegal_action):
            super().__init__()
            logits = torch.zeros(6561)
            logits[illegal_action] = 10.0
            self.register_buffer('logits', logits)

        def forward(self, obs):
            return self.logits.expand(len(obs), -1)

    # One ball at cell 0; source cell 1 is empty, so action 82 is illegal.
    obs = torch.zeros(1, 18, 9, 9)
    obs[0, 0, 0, 0] = 1.0
    obs[0, 7] = 1.0
    obs[0, 7, 0, 0] = 0.0
    batch = (obs, torch.zeros(1, 6561), torch.ones(1),
             torch.tensor([1]), torch.tensor([1]), torch.tensor([0]))
    metrics = audit(
        Fixed(0), Fixed(82), {'target': [batch]}, torch.device('cpu'))
    target = metrics['target']
    assert target['retain'] == 1.0
    assert target['teacher_before'] == 1.0
    assert target['teacher_after'] == 1.0
    assert abs(target['mean_legal_kl']) < 1e-7
    assert target['mean_kl'] > 0.0


def test_audit_excludes_zero_weight_rows_from_teacher_metrics():
    class Fixed(torch.nn.Module):
        def forward(self, obs):
            return torch.zeros(len(obs), 6561)

    obs = torch.zeros(2, 18, 9, 9)
    obs[:, 0, 0, 0] = 1.0
    obs[:, 7] = 1.0
    obs[:, 7, 0, 0] = 0.0
    # Legal argmax is action 1.  The irrelevant zero-weight row says action 2.
    batch = (obs, torch.zeros(2, 6561), torch.tensor([1.0, 0.0]),
             torch.tensor([1, 2]), torch.tensor([1, 2]), torch.tensor([0, 0]))
    target = audit(
        Fixed(), Fixed(), {'target': [batch]}, torch.device('cpu'))['target']
    assert target['teacher_n'] == 1
    assert target['teacher_before'] == 1.0
    assert target['teacher_after'] == 1.0


def test_audit_honors_declared_edit_subset():
    class Fixed(torch.nn.Module):
        def __init__(self, action):
            super().__init__()
            logits = torch.full((6561,), -100.0)
            logits[action] = 0.0
            self.register_buffer('logits', logits)

        def forward(self, obs):
            return self.logits.expand(len(obs), -1)

    obs = torch.zeros(2, 18, 9, 9)
    obs[:, 0, 0, 0] = 1.0
    obs[:, 7] = 1.0
    obs[:, 7, 0, 0] = 0.0
    policy = torch.zeros(2, 6561)
    policy[:, 2] = 1.0
    # Both rows are online base/teacher disagreements, but only the first is
    # part of the immutable mined edit class.
    batch = (obs, policy, torch.ones(2), torch.tensor([1, 1]),
             torch.tensor([2, 2]), torch.tensor([0, 0]),
             torch.tensor([True, False]))
    target = audit(
        Fixed(2), Fixed(1), {'target': [batch]}, torch.device('cpu'),
        bounded_eta=0.5, bounded_agree_weight=0.0,
        bounded_disagree_weight=1.0)['target']
    assert target['edit_n'] == 1
    assert target['edit_adopt'] == 1.0
    assert target['bounded_optimum_edit_n'] == 1
    assert target['bounded_optimum_all_n'] == 1


def test_audit_reports_bounded_argmax_optimum_not_capacity_limit():
    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            logits = torch.full((6561,), -100.0)
            # On the board below, actions 1 and 2 are legal. These logits give
            # them probabilities 0.7 and 0.3, so the teacher needs eta > 0.4
            # to become the exact mixture argmax.
            logits[1] = 0.0
            logits[2] = -torch.log(torch.tensor(7.0 / 3.0))
            self.register_buffer('logits', logits)

        def forward(self, obs):
            return self.logits.expand(len(obs), -1)

    obs = torch.zeros(1, 18, 9, 9)
    obs[0, 0, 0, 0] = 1.0
    obs[0, 7] = 1.0
    obs[0, 7, 0, 0] = 0.0
    batch = (obs, torch.zeros(1, 6561), torch.ones(1),
             torch.tensor([1]), torch.tensor([2]), torch.tensor([0]))
    low = audit(
        Fixed(), Fixed(), {'target': [batch]}, torch.device('cpu'),
        bounded_eta=0.2)['target']
    high = audit(
        Fixed(), Fixed(), {'target': [batch]}, torch.device('cpu'),
        bounded_eta=0.5)['target']
    assert low['bounded_optimum_edit_adopt'] == 0.0
    assert high['bounded_optimum_edit_adopt'] == 1.0
    assert abs(low['edit_probability_gap_p50'] - 0.4) < 1e-5


def test_audit_dense_bounded_target_reports_exact_all_row_residual():
    class Fixed(torch.nn.Module):
        def __init__(self, p1, p2):
            super().__init__()
            logits = torch.full((6561,), -100.0)
            logits[1] = torch.log(torch.tensor(p1))
            logits[2] = torch.log(torch.tensor(p2))
            self.register_buffer('logits', logits)

        def forward(self, obs):
            return self.logits.expand(len(obs), -1)

    # base=(.7,.3), visits=(.2,.8), eta=.5 gives q=(.8,.7)/1.5.
    q1, q2 = 0.8 / 1.5, 0.7 / 1.5
    obs = torch.zeros(1, 18, 9, 9)
    obs[0, 0, 0, 0] = 1.0
    obs[0, 7] = 1.0
    obs[0, 7, 0, 0] = 0.0
    policy = torch.zeros(1, 6561)
    policy[0, 1] = 0.2
    policy[0, 2] = 0.8
    batch = (obs, policy, torch.ones(1), torch.tensor([1]),
             torch.tensor([2]), torch.tensor([0]))
    target = audit(
        Fixed(q1, q2), Fixed(0.7, 0.3), {'target': [batch]},
        torch.device('cpu'), bounded_eta=0.5,
        bounded_soft_alpha=1.0)['target']
    assert target['bounded_soft_alpha'] == 1.0
    assert target['bounded_optimum_all_n'] == 1
    assert target['bounded_optimum_all_action_change'] == 0.0
    assert target['bounded_optimum_all_teacher_match'] == 0.0
    assert target['bounded_optimum_all_residual_kl'] < 1e-6
    assert target['bounded_optimum_all_base_kl'] > 0.0


def test_resume_rng_state_replays_all_sampling_streams():
    torch.manual_seed(13)
    loader_rng = torch.Generator().manual_seed(17)
    numpy_rng = np.random.default_rng(19)
    state = _capture_rng_state(
        loader_rng, numpy_rng, torch.device('cpu'))
    expected = (
        torch.rand(4),
        torch.randperm(20, generator=loader_rng),
        numpy_rng.integers(0, 1000, size=5),
    )
    assert _restore_rng_state(
        state, loader_rng, numpy_rng, torch.device('cpu'))
    actual = (
        torch.rand(4),
        torch.randperm(20, generator=loader_rng),
        numpy_rng.integers(0, 1000, size=5),
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2], expected[2])


def test_resume_arg_validation_rejects_objective_change():
    saved = {'objective': 'legacy', 'batch_size': 4096}
    current = {'objective': 'bounded', 'batch_size': 4096}
    try:
        _validate_resume_args(saved, current)
    except ValueError as exc:
        assert 'objective' in str(exc)
    else:
        raise AssertionError('resume mismatch was accepted')


def test_completed_epoch_has_legacy_filename_fallback():
    assert _completed_epoch({'completed_epoch': 7}, 'anything.pt') == 7
    assert _completed_epoch({}, 'checkpoints/run/epoch_12.pt') == 12


def test_deployment_numerics_catches_fp16_batchnorm_overflow():
    model = torch.nn.BatchNorm2d(2).eval()
    model.running_var.copy_(torch.tensor([70_000.0, 2.0]))
    report = _deployment_numerics(model, torch.float16)
    assert report['source_nonfinite'] == 0
    assert report['bn_running_var_cast_nonfinite'] == 1
    _quantize_frozen_bn_for_deployment_(model, torch.float16)
    assert torch.isinf(model.running_var[0])
    assert model.running_var[1] == 2


def test_functional_gate_checks_retention_kl_and_new_bn_overflow():
    metrics = {
        'target': {'retain': 0.89, 'mean_legal_kl': 0.02},
        'anchor': {'retain': 0.93, 'mean_legal_kl': 0.06},
    }
    student = {'bn_running_var_cast_nonfinite': 4}
    base = {'bn_running_var_cast_nonfinite': 1}
    gate = _functional_gate(
        metrics, student, base, min_retain=0.90, max_legal_kl=0.05,
        max_new_bn_nonfinite=0)
    assert not gate['passed']
    assert gate['new_bn_running_var_cast_nonfinite'] == 3
    assert len(gate['failures']) == 3


def test_functional_gate_can_be_disabled():
    metrics = {'target': {'retain': 0.0, 'mean_legal_kl': 100.0}}
    numerics = {'bn_running_var_cast_nonfinite': 5}
    gate = _functional_gate(metrics, numerics, numerics)
    assert gate['passed']
