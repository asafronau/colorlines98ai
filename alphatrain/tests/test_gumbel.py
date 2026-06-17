"""Unit tests for the completed-Q improvement target (alphatrain/gumbel.py).

These pin the diagnostic-driven behavior: padded slots get zero mass; saturated states
fall back to the prior (no correction); a low-visit high-Q candidate does NOT trigger a
correction (the raw-Q noise trap); a well-visited higher-Q alternative DOES, shifting
mass toward it with weight 1+gamma.
"""
import torch
import pytest
from alphatrain.gumbel import completed_q_target


def _row(visits, priors, qs, nnz, K=10):
    """Pad one state's candidate lists to length K (zeros = padding)."""
    pad = lambda x, v: torch.tensor(x + [v] * (K - len(x)), dtype=torch.float32)
    return (pad(visits, 0.0), pad(priors, 0.0), pad(qs, 0.0), nnz)


def _batch(rows, root_values, K=10):
    vis, pri, q, nnz = zip(*rows)
    return (torch.stack(vis), torch.stack(pri), torch.stack(q),
            torch.tensor(nnz, dtype=torch.long),
            torch.tensor(root_values, dtype=torch.float32))


def test_padding_gets_zero_mass():
    # 3 real candidates out of 10; padded slots must receive exactly zero probability.
    r = _row([50, 40, 30], [-0.5, -1.0, -2.0], [2.55, 2.55, 2.55], nnz=3)
    cv, cp, cq, nnz, rv = _batch([r], [2.55])
    target_p, prior_p, weight, corr, _ = completed_q_target(cv, cp, cq, nnz, rv)
    assert torch.allclose(target_p[0, 3:], torch.zeros(7), atol=1e-7)
    assert torch.allclose(prior_p[0, 3:], torch.zeros(7), atol=1e-7)
    assert torch.allclose(target_p.sum(1), torch.ones(1), atol=1e-5)


def test_saturated_state_falls_back_to_prior():
    # All candidate Qs equal (value head saturated) -> no correction, target == prior, w=1.
    r = _row([50, 50, 50], [-0.3, -1.2, -2.5], [2.55, 2.55, 2.55], nnz=3)
    cv, cp, cq, nnz, rv = _batch([r], [2.55])
    target_p, prior_p, weight, corr, _ = completed_q_target(cv, cp, cq, nnz, rv)
    assert not corr[0]
    assert weight[0].item() == pytest.approx(1.0)
    assert torch.allclose(target_p[0], prior_p[0], atol=1e-6)


def test_low_visit_high_q_is_NOT_a_correction():
    # The noise trap: candidate 2 has a high Q but only 1 visit. Must be ignored
    # (completed_Q falls back to root_value) and must NOT flip the correction flag.
    r = _row([120, 90, 1], [-0.2, -0.9, -3.0], [2.55, 2.55, 2.95], nnz=3)
    cv, cp, cq, nnz, rv = _batch([r], [2.55])
    target_p, prior_p, weight, corr, _ = completed_q_target(
        cv, cp, cq, nnz, rv, visit_floor=15.0)
    assert not corr[0], "low-visit high-Q must not be trusted"
    assert weight[0].item() == pytest.approx(1.0)
    # target essentially the prior (the spike candidate gets no boost)
    assert torch.allclose(target_p[0], prior_p[0], atol=1e-6)


def test_well_visited_better_q_is_a_correction():
    # Candidate 1 is the prior's pick but candidate 2 is well-visited with clearly
    # higher Q (spread 0.1 > gate). This IS a trustworthy correction.
    r = _row([100, 80, 5], [-0.2, -0.6, -3.0], [2.50, 2.60, 2.55], nnz=3)
    cv, cp, cq, nnz, rv = _batch([r], [2.55])
    target_p, prior_p, weight, corr, _ = completed_q_target(
        cv, cp, cq, nnz, rv, visit_floor=15.0, tau=0.05, gamma=10.0, spread_gate=0.05)
    assert corr[0], "well-visited higher-Q alternative should be a correction"
    assert weight[0].item() == pytest.approx(11.0)
    # mass should move toward candidate 1 (the higher-Q well-visited move) vs the prior
    assert target_p[0, 1] > prior_p[0, 1]
    assert target_p[0, 0] < prior_p[0, 0]


def test_spread_below_gate_is_not_a_correction():
    # Well-visited disagreement but tiny spread (0.01 < gate 0.05) -> not a correction.
    r = _row([100, 80], [-0.2, -0.6], [2.550, 2.560], nnz=2)
    cv, cp, cq, nnz, rv = _batch([r], [2.555])
    _, _, weight, corr, _ = completed_q_target(
        cv, cp, cq, nnz, rv, spread_gate=0.05)
    assert not corr[0]
    assert weight[0].item() == pytest.approx(1.0)


def test_all_zero_q_state_is_safe():
    # The 0.005% legacy no-Q states (all cand_q == 0, root_value 0): must not crash,
    # must not be a correction, target == prior.
    r = _row([50, 40], [-0.5, -1.0], [0.0, 0.0], nnz=2)
    cv, cp, cq, nnz, rv = _batch([r], [0.0])
    target_p, prior_p, weight, corr, _ = completed_q_target(cv, cp, cq, nnz, rv)
    assert not corr[0]
    assert torch.allclose(target_p[0], prior_p[0], atol=1e-6)


def test_batch_mixed():
    # Saturated + correction + noise-trap in one batch; flags must be independent.
    rows = [
        _row([50, 50, 50], [-0.3, -1.2, -2.5], [2.55, 2.55, 2.55], 3),   # saturated
        _row([100, 80, 5], [-0.2, -0.6, -3.0], [2.50, 2.60, 2.55], 3),   # correction
        _row([120, 90, 1], [-0.2, -0.9, -3.0], [2.55, 2.55, 2.95], 3),   # noise trap
    ]
    cv, cp, cq, nnz, rv = _batch(rows, [2.55, 2.55, 2.55])
    _, _, weight, corr, _ = completed_q_target(cv, cp, cq, nnz, rv)
    assert corr.tolist() == [False, True, False]
    assert weight.tolist() == pytest.approx([1.0, 11.0, 1.0])
