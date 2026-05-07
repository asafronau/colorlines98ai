"""Tests for ValueHead module + scalar combination."""

import os
import tempfile

import torch
import pytest

from alphatrain.model import PolicyNet
from alphatrain.value_head import (
    ValueHead, SURVIVAL_HORIZONS, NUM_HORIZONS, DEFAULT_HORIZON_WEIGHTS,
    survival_to_scalar, save, load,
)


class TestValueHead:
    def test_shapes(self):
        head = ValueHead(in_channels=256, hidden=32)
        feats = torch.randn(4, 256, 9, 9)
        out = head(feats)
        assert out.shape == (4, NUM_HORIZONS)

    def test_param_count_small(self):
        head = ValueHead(in_channels=256, hidden=32)
        n = sum(p.numel() for p in head.parameters())
        # 256*32 conv (8192) + 32*2 BN (64) + 32*4 + 4 fc (132) ≈ 8400
        assert 5_000 < n < 15_000, f"unexpected param count: {n}"

    def test_works_with_policy_backbone(self):
        """End-to-end: frozen PolicyNet backbone → ValueHead → 4 logits."""
        net = PolicyNet(num_blocks=2, channels=64)
        head = ValueHead(in_channels=64, hidden=16)
        net.train(False)
        for p in net.parameters():
            p.requires_grad_(False)
        x = torch.randn(2, 18, 9, 9)
        with torch.no_grad():
            feats = net.backbone_features(x)
        logits = head(feats)
        assert logits.shape == (2, NUM_HORIZONS)

    def test_gradient_flows_only_through_head(self):
        """Confirm frozen-backbone setup: only head params get gradients."""
        net = PolicyNet(num_blocks=2, channels=64)
        head = ValueHead(in_channels=64, hidden=16)
        for p in net.parameters():
            p.requires_grad_(False)
        x = torch.randn(2, 18, 9, 9)
        feats = net.backbone_features(x)
        logits = head(feats)
        loss = logits.sum()
        loss.backward()
        # Backbone gets no grad
        for name, p in net.named_parameters():
            assert p.grad is None, f"backbone param got grad: {name}"
        # Head gets grad
        for name, p in head.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"head param missing grad: {name}"


class TestSurvivalToScalar:
    def test_default_weights_sum(self):
        # Sanity check: front-loaded weights, sum=2.55
        assert sum(DEFAULT_HORIZON_WEIGHTS) == pytest.approx(2.55)
        # Each horizon weighted less than the previous
        for i in range(len(DEFAULT_HORIZON_WEIGHTS) - 1):
            assert DEFAULT_HORIZON_WEIGHTS[i] > DEFAULT_HORIZON_WEIGHTS[i + 1]

    def test_all_zero(self):
        probs = torch.zeros(3, NUM_HORIZONS)
        v = survival_to_scalar(probs)
        assert torch.equal(v, torch.zeros(3))

    def test_all_one(self):
        probs = torch.ones(3, NUM_HORIZONS)
        v = survival_to_scalar(probs)
        assert torch.allclose(v, torch.full((3,), 2.55))

    def test_dim(self):
        probs = torch.rand(5, NUM_HORIZONS)
        v = survival_to_scalar(probs)
        assert v.shape == (5,)

    def test_custom_weights(self):
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        v = survival_to_scalar(probs, horizon_weights=(1, 1, 1, 1))
        assert torch.allclose(v, torch.tensor([1.0]))


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        head = ValueHead(in_channels=128, hidden=16)
        # Set known weights so we can verify byte-perfect round-trip
        with torch.no_grad():
            head.fc.weight.fill_(0.5)
            head.fc.bias.fill_(0.1)
        path = tmp_path / "head.pt"
        save(head, str(path), backbone_path='/path/to/backbone.pt',
             train_args={'epochs': 5}, val_metrics={'auc': [0.7]*4})

        loaded, meta = load(str(path))
        assert loaded.in_channels == 128
        assert loaded.hidden == 16
        assert torch.equal(loaded.fc.weight, head.fc.weight)
        assert torch.equal(loaded.fc.bias, head.fc.bias)
        assert meta['backbone_path'] == '/path/to/backbone.pt'
        assert meta['train_args'] == {'epochs': 5}
        assert meta['val_metrics'] == {'auc': [0.7]*4}
        assert tuple(meta['horizons']) == SURVIVAL_HORIZONS

    def test_horizons_mismatch_rejected(self, tmp_path):
        """If a saved head's horizons don't match the current
        SURVIVAL_HORIZONS, load() raises rather than silently producing
        wrong scalars in MCTS."""
        head = ValueHead(in_channels=64, hidden=8)
        path = tmp_path / "head.pt"
        # Hand-craft a checkpoint with wrong horizons
        torch.save({
            'state_dict': head.state_dict(),
            'in_channels': 64,
            'hidden': 8,
            'horizons': [10, 20, 30, 40],
            'backbone_path': 'x',
            'train_args': None,
            'val_metrics': None,
        }, path)
        with pytest.raises(ValueError, match="horizons"):
            load(str(path))
