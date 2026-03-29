"""Unit tests for AlphaTrain model."""

import torch
import numpy as np
import pytest
from alphatrain.model import AlphaTrainNet, count_parameters, NUM_MOVES, BOARD_SIZE


class TestModelArchitecture:
    def test_default_construction(self):
        model = AlphaTrainNet()
        assert model.in_channels == 18
        assert model.channels == 256
        assert model.num_value_bins == 64

    def test_custom_construction(self):
        model = AlphaTrainNet(in_channels=12, num_blocks=5, channels=128,
                               num_value_bins=32)
        assert model.in_channels == 12
        assert model.channels == 128
        assert model.num_value_bins == 32

    def test_parameter_count(self):
        model = AlphaTrainNet(num_blocks=10, channels=256)
        params = count_parameters(model)
        assert 10_000_000 < params < 15_000_000

    def test_small_model(self):
        model = AlphaTrainNet(num_blocks=2, channels=64)
        params = count_parameters(model)
        assert params < 500_000


class TestModelForward:
    @pytest.fixture
    def model(self):
        return AlphaTrainNet(num_blocks=2, channels=64)

    def test_output_shapes(self, model):
        x = torch.randn(4, 18, 9, 9)
        policy, value = model(x)
        assert policy.shape == (4, NUM_MOVES)
        assert value.shape == (4, 64)

    def test_single_sample(self, model):
        x = torch.randn(1, 18, 9, 9)
        policy, value = model(x)
        assert policy.shape == (1, NUM_MOVES)
        assert value.shape == (1, 64)

    def test_batch_independence(self, model):
        """Each sample in batch should be independently processed."""
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        p_batch, v_batch = model(x)
        p1, v1 = model(x[0:1])
        p2, v2 = model(x[1:2])
        assert torch.allclose(p_batch[0], p1[0], atol=1e-5)
        assert torch.allclose(p_batch[1], p2[0], atol=1e-5)

    def test_deterministic(self, model):
        """Same input produces same output in eval mode."""
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        p1, v1 = model(x)
        p2, v2 = model(x)
        assert torch.equal(p1, p2)
        assert torch.equal(v1, v2)

    def test_gradient_flow(self, model):
        """Gradients flow through all parameters."""
        x = torch.randn(2, 18, 9, 9)
        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestPredictValue:
    def test_output_shape(self):
        model = AlphaTrainNet(num_blocks=2, channels=64)
        x = torch.randn(4, 18, 9, 9)
        _, value_logits = model(x)
        scores = model.predict_value(value_logits)
        assert scores.shape == (4,)

    def test_value_range(self):
        model = AlphaTrainNet(num_blocks=2, channels=64)
        model.eval()
        x = torch.randn(100, 18, 9, 9)
        with torch.no_grad():
            _, value_logits = model(x)
            scores = model.predict_value(value_logits, min_val=0, max_val=30000)
        assert scores.min() >= 0
        assert scores.max() <= 30000

    def test_custom_range(self):
        model = AlphaTrainNet(num_blocks=2, channels=64)
        model.eval()
        x = torch.randn(4, 18, 9, 9)
        with torch.no_grad():
            _, value_logits = model(x)
            scores = model.predict_value(value_logits, min_val=0, max_val=6)
        assert scores.min() >= 0
        assert scores.max() <= 6


class TestFP16:
    def test_half_precision(self):
        model = AlphaTrainNet(num_blocks=2, channels=64).half()
        x = torch.randn(2, 18, 9, 9).half()
        policy, value = model(x)
        assert policy.dtype == torch.float16

    def test_channels_last(self):
        model = AlphaTrainNet(num_blocks=2, channels=64)
        model = model.to(memory_format=torch.channels_last)
        x = torch.randn(2, 18, 9, 9).to(memory_format=torch.channels_last)
        policy, value = model(x)
        assert policy.shape == (2, NUM_MOVES)
