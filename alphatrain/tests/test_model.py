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

    def test_custom_construction(self):
        model = AlphaTrainNet(in_channels=12, num_blocks=5, channels=128)
        assert model.in_channels == 12
        assert model.channels == 128

    def test_parameter_count(self):
        model = AlphaTrainNet(num_blocks=10, channels=256)
        params = count_parameters(model)
        # Without value head, should be smaller than before
        assert 5_000_000 < params < 15_000_000

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
        policy = model(x)
        assert policy.shape == (4, NUM_MOVES)

    def test_single_sample(self, model):
        x = torch.randn(1, 18, 9, 9)
        policy = model(x)
        assert policy.shape == (1, NUM_MOVES)

    def test_batch_independence(self, model):
        """Each sample in batch should be independently processed."""
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        p_batch = model(x)
        p1 = model(x[0:1])
        p2 = model(x[1:2])
        assert torch.allclose(p_batch[0], p1[0], atol=1e-5)
        assert torch.allclose(p_batch[1], p2[0], atol=1e-5)

    def test_deterministic(self, model):
        """Same input produces same output in eval mode."""
        model.eval()
        x = torch.randn(2, 18, 9, 9)
        p1 = model(x)
        p2 = model(x)
        assert torch.equal(p1, p2)

    def test_gradient_flow(self, model):
        """Gradients flow through all parameters."""
        x = torch.randn(2, 18, 9, 9)
        policy = model(x)
        loss = policy.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestFP16:
    def test_half_precision(self):
        model = AlphaTrainNet(num_blocks=2, channels=64).half()
        x = torch.randn(2, 18, 9, 9).half()
        policy = model(x)
        assert policy.dtype == torch.float16

    def test_channels_last(self):
        model = AlphaTrainNet(num_blocks=2, channels=64)
        model = model.to(memory_format=torch.channels_last)
        x = torch.randn(2, 18, 9, 9).to(memory_format=torch.channels_last)
        policy = model(x)
        assert policy.shape == (2, NUM_MOVES)


class TestWarmStart:
    def test_load_old_checkpoint_skips_value_keys(self):
        """Loading a checkpoint with value head into new model skips value keys."""
        # Simulate old checkpoint with value head keys
        model = AlphaTrainNet(num_blocks=2, channels=64)
        state = model.state_dict()

        # Add fake value head keys (as if from old checkpoint)
        state['value_conv.weight'] = torch.randn(32, 64, 1, 1)
        state['value_bn.weight'] = torch.randn(32)
        state['value_fc1.weight'] = torch.randn(512, 32 * 81)
        state['value_fc2.weight'] = torch.randn(64, 512)

        # Load with strict=False (like warm start does)
        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        # Value keys should not be in missing (they don't exist in new model)
        # Backbone keys should load fine
        assert 'stem.0.weight' not in missing
