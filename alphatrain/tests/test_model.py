"""Unit tests for the policy-only AlphaTrain model."""

import torch
import pytest
from alphatrain.model import (
    PolicyNet, AlphaTrainNet, count_parameters, NUM_MOVES, BOARD_SIZE,
)


class TestModelArchitecture:
    def test_default_construction(self):
        model = PolicyNet()
        assert model.in_channels == 18
        assert model.channels == 256

    def test_custom_construction(self):
        model = PolicyNet(in_channels=12, num_blocks=5, channels=128)
        assert model.in_channels == 12
        assert model.channels == 128

    def test_parameter_count(self):
        model = PolicyNet(num_blocks=10, channels=256)
        params = count_parameters(model)
        # Policy-only model is ~12M params (down from 13M in dual-head era).
        assert 10_000_000 < params < 13_000_000

    def test_small_model(self):
        model = PolicyNet(num_blocks=2, channels=64)
        params = count_parameters(model)
        assert params < 500_000

    def test_alphatrainnet_alias(self):
        """Back-compat: AlphaTrainNet is an alias for PolicyNet."""
        assert AlphaTrainNet is PolicyNet


class TestModelForward:
    @pytest.fixture
    def model(self):
        return PolicyNet(num_blocks=2, channels=64)

    def test_output_shape(self, model):
        x = torch.randn(4, 18, 9, 9)
        policy = model(x)
        assert policy.shape == (4, NUM_MOVES)

    def test_single_sample(self, model):
        x = torch.randn(1, 18, 9, 9)
        policy = model(x)
        assert policy.shape == (1, NUM_MOVES)

    def test_returns_single_tensor(self, model):
        """forward() returns a tensor, not a tuple."""
        x = torch.randn(2, 18, 9, 9)
        out = model(x)
        assert isinstance(out, torch.Tensor)

    def test_batch_independence(self, model):
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        p_batch = model(x)
        p1 = model(x[0:1])
        p2 = model(x[1:2])
        assert torch.allclose(p_batch[0], p1[0], atol=1e-5)
        assert torch.allclose(p_batch[1], p2[0], atol=1e-5)

    def test_deterministic(self, model):
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        p1 = model(x)
        p2 = model(x)
        assert torch.equal(p1, p2)

    def test_gradient_flow(self, model):
        x = torch.randn(2, 18, 9, 9)
        policy = model(x)
        policy.sum().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestBackboneFeatures:
    """Tests for the planned frozen-backbone value head extension point."""

    def test_backbone_shape(self):
        model = PolicyNet(num_blocks=2, channels=64)
        x = torch.randn(4, 18, 9, 9)
        feats = model.backbone_features(x)
        # (B, channels, 9, 9) — the spatial feature map a value head
        # would consume after a pooling step.
        assert feats.shape == (4, 64, 9, 9)

    def test_backbone_matches_internal(self):
        """backbone_features matches the start of forward() up through
        backbone_bn + ReLU."""
        import torch.nn.functional as F
        model = PolicyNet(num_blocks=2, channels=64)
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        with torch.no_grad():
            feats = model.backbone_features(x)
            # Reproduce manually
            out = model.stem(x)
            out = model.blocks(out)
            out = F.relu(model.backbone_bn(out))
            assert torch.equal(feats, out)


class TestFP16:
    def test_half_precision(self):
        model = PolicyNet(num_blocks=2, channels=64).half()
        x = torch.randn(2, 18, 9, 9).half()
        out = model(x)
        assert out.dtype == torch.float16

    def test_channels_last(self):
        model = PolicyNet(num_blocks=2, channels=64)
        model = model.to(memory_format=torch.channels_last)
        x = torch.randn(2, 18, 9, 9).to(memory_format=torch.channels_last)
        out = model(x)
        assert out.shape == (2, NUM_MOVES)
