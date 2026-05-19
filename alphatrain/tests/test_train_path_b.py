"""Unit tests for Path B trainer's oracle loss and reliability weighting."""

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from alphatrain.train_path_b import (
    NEG_INF,
    OracleDataset,
    oracle_loss,
    reliability_weight,
)


@pytest.fixture(autouse=True)
def _torch_default_dtype():
    torch.manual_seed(0)


# ── reliability_weight: ramp boundaries ────────────────────────────────

def test_reliability_weight_boundaries():
    deltas = torch.tensor([0.00, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50,
                            1.00], dtype=torch.float32)
    w = reliability_weight(deltas, noise_floor=0.05, scale=0.20)
    # Δ <= 0.05  → 0
    assert w[0].item() == 0.0
    assert w[1].item() == 0.0
    assert w[2].item() == 0.0
    # Δ = 0.10 → ((0.10 - 0.05)/0.20)**2 = 0.25**2 = 0.0625
    assert math.isclose(w[3].item(), 0.0625, abs_tol=1e-6)
    # Δ = 0.15 → (0.10/0.20)**2 = 0.5**2 = 0.25
    assert math.isclose(w[4].item(), 0.25, abs_tol=1e-6)
    # Δ = 0.20 → (0.15/0.20)**2 = 0.75**2 = 0.5625
    assert math.isclose(w[5].item(), 0.5625, abs_tol=1e-6)
    # Δ = 0.25 → exactly 1.0 (clamped)
    assert math.isclose(w[6].item(), 1.0, abs_tol=1e-6)
    # Δ >= 0.25 saturates at 1
    assert w[7].item() == 1.0
    assert w[8].item() == 1.0


def test_reliability_weight_monotone():
    """Weight is non-decreasing in delta_cap."""
    deltas = torch.linspace(0.0, 1.0, 101)
    w = reliability_weight(deltas)
    diffs = w[1:] - w[:-1]
    assert (diffs >= -1e-7).all()


# ── oracle_loss: gather correctness ────────────────────────────────────

def test_oracle_loss_gathers_intended_logits():
    """If we put a huge logit ONLY at the action that oracle says is best,
    KL ≈ 0; if we put it at the wrong action, KL is high."""
    torch.manual_seed(0)
    B, V = 4, 6561
    K = 6
    logits_good = torch.full((B, V), -10.0)
    logits_bad = torch.full((B, V), -10.0)
    # Build 4 anchors. cap_rates put all weight on slot 0; actions[b, 0] = 100*b
    actions = torch.full((B, K), -1, dtype=torch.long)
    cap_rates = torch.zeros(B, K, dtype=torch.float32)
    for b in range(B):
        for k in range(K):
            actions[b, k] = 10 + 100 * b + k
            cap_rates[b, k] = 1.0 if k == 0 else 0.1
        # In good: put high logit on action at slot 0 (matching oracle pref)
        logits_good[b, int(actions[b, 0])] = 5.0
        # In bad: put high logit on slot K-1 (oracle's last preference)
        logits_bad[b, int(actions[b, K - 1])] = 5.0
    n_moves = torch.full((B,), K, dtype=torch.long)
    delta_cap = torch.full((B,), 1.0)  # all w=1

    loss_good, _ = oracle_loss(
        logits_good, actions, cap_rates, n_moves, delta_cap,
        beta=10.0, noise_floor=0.05, scale=0.20)
    loss_bad, _ = oracle_loss(
        logits_bad, actions, cap_rates, n_moves, delta_cap,
        beta=10.0, noise_floor=0.05, scale=0.20)
    assert loss_good.item() < loss_bad.item()
    assert loss_good.item() < 0.5


def test_oracle_loss_zero_when_aligned():
    """If model logits exactly match β·cap_rates at the candidate actions
    (and uniform elsewhere), KL = 0."""
    B, V, K = 2, 6561, 6
    actions = torch.arange(B * K).reshape(B, K).long()  # disjoint indices
    cap_rates = torch.tensor([[0.9, 0.7, 0.5, 0.3, 0.2, 0.1],
                                [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]],
                                dtype=torch.float32)
    n_moves = torch.full((B,), K, dtype=torch.long)
    delta_cap = torch.full((B,), 1.0)
    beta = 10.0
    # Place β·cap_rates at the action positions; anything else is masked
    # away in the gather + softmax so its value doesn't matter for KL.
    logits = torch.full((B, V), -1000.0)
    for b in range(B):
        for k in range(K):
            logits[b, int(actions[b, k])] = beta * cap_rates[b, k]

    loss, _ = oracle_loss(logits, actions, cap_rates, n_moves, delta_cap,
                            beta=beta, noise_floor=0.05, scale=0.20)
    assert loss.item() < 1e-5, f"expected 0, got {loss.item()}"


# ── Padding / mask ─────────────────────────────────────────────────────

def test_oracle_loss_mask_ignores_pad_actions():
    """An anchor with n_moves=3 should give the same loss whether pad slots
    carry valid or garbage data."""
    B, V, K = 1, 6561, 6
    actions_valid = torch.tensor([[5, 17, 42, -1, -1, -1]], dtype=torch.long)
    actions_garbage = torch.tensor([[5, 17, 42, 99, 100, 101]],
                                     dtype=torch.long)
    cap_rates_valid = torch.tensor([[0.9, 0.6, 0.3, 0.0, 0.0, 0.0]])
    cap_rates_garbage = torch.tensor([[0.9, 0.6, 0.3, 999., 999., 999.]])
    n_moves = torch.tensor([3], dtype=torch.long)
    delta_cap = torch.tensor([0.6])  # w = 1
    logits = torch.randn(B, V)

    l_a, _ = oracle_loss(logits, actions_valid, cap_rates_valid,
                          n_moves, delta_cap)
    l_b, _ = oracle_loss(logits, actions_garbage, cap_rates_garbage,
                          n_moves, delta_cap)
    assert math.isclose(l_a.item(), l_b.item(), abs_tol=1e-5), \
        f"mask leaked: {l_a.item()} vs {l_b.item()}"


def test_oracle_loss_no_nan_with_pads():
    """Padded indices = -1 must not cause NaN even with random logits."""
    torch.manual_seed(7)
    B, V, K = 16, 6561, 6
    actions = torch.randint(0, V, (B, K), dtype=torch.long)
    n_moves = torch.randint(2, K + 1, (B,), dtype=torch.long)
    # Set unused slots in actions to -1
    for b in range(B):
        actions[b, int(n_moves[b]):] = -1
    cap_rates = torch.rand(B, K)
    cap_rates[actions == -1] = 0.0
    delta_cap = torch.rand(B)
    logits = torch.randn(B, V) * 5

    loss, w = oracle_loss(logits, actions, cap_rates, n_moves, delta_cap)
    assert torch.isfinite(loss).item()
    assert torch.isfinite(w).all().item()


# ── λ=0 contract ──────────────────────────────────────────────────────

def test_zero_lambda_oracle_gradient_skipped():
    """When λ=0, oracle_loss is multiplied by 0 → no gradient contribution.
    Test the combined loss expression directly."""
    torch.manual_seed(11)
    B, V, K = 4, 6561, 6
    logits = torch.randn(B, V, requires_grad=True)
    actions = torch.randint(0, V, (B, K), dtype=torch.long)
    cap_rates = torch.rand(B, K)
    n_moves = torch.full((B,), K, dtype=torch.long)
    delta_cap = torch.rand(B)

    L_oracle, _ = oracle_loss(logits, actions, cap_rates, n_moves, delta_cap)
    combined = 0.0 * L_oracle  # λ=0
    combined.backward()
    assert (logits.grad.abs().sum().item() == 0.0)


def test_zero_weight_anchor_does_not_contribute():
    """If Δcap < noise_floor for every anchor, every w=0 → loss is 0
    (denominator clamp lets the sum be 0/1)."""
    B, V, K = 3, 6561, 6
    actions = torch.arange(B * K).reshape(B, K).long()
    cap_rates = torch.rand(B, K)
    n_moves = torch.full((B,), K, dtype=torch.long)
    # All anchors below noise floor
    delta_cap = torch.tensor([0.01, 0.02, 0.04], dtype=torch.float32)
    logits = torch.randn(B, V)

    loss, w = oracle_loss(logits, actions, cap_rates, n_moves, delta_cap,
                            noise_floor=0.05, scale=0.20)
    assert (w == 0).all().item()
    assert loss.item() == 0.0


# ── Trainer integration smoke ─────────────────────────────────────────

def test_train_path_b_smoke_one_step():
    """End-to-end: V12-style batch + oracle batch, one optimizer step
    on a tiny network. Must not crash, must produce finite loss."""
    from alphatrain.model import AlphaTrainNet
    from alphatrain.train_path_b import (
        distillation_loss, oracle_loss as _oracle_loss)

    torch.manual_seed(0)
    net = AlphaTrainNet(num_blocks=1, channels=8)
    net.train(True)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

    B_v, B_o = 4, 4
    V = 6561
    obs_v = torch.randn(B_v, 18, 9, 9)
    pol_v = torch.zeros(B_v, V)
    pol_v[:, 17] = 1.0
    obs_o = torch.randn(B_o, 18, 9, 9)
    actions = torch.randint(0, V, (B_o, 6), dtype=torch.long)
    cap_rates = torch.rand(B_o, 6)
    n_moves = torch.full((B_o,), 6, dtype=torch.long)
    delta_cap = torch.full((B_o,), 0.30)  # w=1

    obs = torch.cat([obs_v, obs_o], dim=0)
    out = net(obs)
    logits = out[0] if isinstance(out, tuple) else out
    L_v = distillation_loss(logits[:B_v], pol_v)
    L_o, _ = _oracle_loss(logits[B_v:].float(), actions, cap_rates,
                            n_moves, delta_cap)
    total = L_v + 0.05 * L_o
    assert torch.isfinite(total).item()
    opt.zero_grad()
    total.backward()
    opt.step()


# ── OracleDataset round-trip ──────────────────────────────────────────

def test_oracle_dataset_loads(tmp_path):
    """Round-trip a tiny synthetic Path B tensor through OracleDataset."""
    N = 4
    # Build a plausible empty-ish board with a couple of balls
    boards = np.zeros((N, 9, 9), dtype=np.int8)
    boards[:, 4, 4] = 1
    boards[:, 0, 0] = 2
    next_pos = np.array(
        [[(1, 1), (2, 2), (3, 3)]] * N, dtype=np.int8)
    next_col = np.array([[3, 4, 5]] * N, dtype=np.int8)
    n_next = np.full(N, 3, dtype=np.int8)
    actions = np.zeros((N, 6), dtype=np.int64)
    actions[:] = np.array([100, 200, 300, 400, 500, 600])
    cap_rates = np.random.rand(N, 6).astype(np.float32)
    mean_turns = np.zeros((N, 6), dtype=np.float32)
    n_moves = np.full(N, 6, dtype=np.int8)
    delta_cap = np.array([0.30, 0.20, 0.10, 0.04], dtype=np.float32)

    payload = {
        'boards': torch.from_numpy(boards),
        'next_pos': torch.from_numpy(next_pos),
        'next_col': torch.from_numpy(next_col),
        'n_next': torch.from_numpy(n_next),
        'actions': torch.from_numpy(actions),
        'cap_rates': torch.from_numpy(cap_rates),
        'mean_turns': torch.from_numpy(mean_turns),
        'n_moves': torch.from_numpy(n_moves),
        'delta_cap': torch.from_numpy(delta_cap),
        'turn_origin': torch.zeros(N, dtype=torch.int32),
        'is_crisis': torch.ones(N, dtype=torch.bool),
        'meta': {'top_k': 6, 'n_anchors': N},
    }
    path = tmp_path / 'tiny.pt'
    torch.save(payload, str(path))

    ods = OracleDataset(str(path), device='cpu')
    assert ods.N == N
    assert ods.obs.shape == (N, 18, 9, 9)
    assert ods.actions.shape == (N, 6)
    # Channel 7 should be the "empty" plane
    assert torch.isfinite(ods.obs).all().item()
    train_idx, val_idx = ods.split(val_frac=0.25, seed=1)
    assert train_idx.numel() + val_idx.numel() == N

    gen = torch.Generator(device='cpu').manual_seed(0)
    o_obs, o_act, o_cap, o_nm, o_dc = ods.sample(train_idx, 2, gen)
    assert o_obs.shape == (2, 18, 9, 9)
