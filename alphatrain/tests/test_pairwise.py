"""Tests for pairwise training pipeline: afterstate pairs, collate, loss."""

import numpy as np
import torch
import torch.nn.functional as F
import pytest
from alphatrain.model import AlphaTrainNet


class TestPairwiseCollate:
    """Test collate_pairwise returns correct shapes and types."""

    @pytest.fixture
    def dataset(self):
        """Load pairwise dataset if available."""
        import os
        path = 'alphatrain/data/alphatrain_pairwise.pt'
        if not os.path.exists(path):
            pytest.skip("Pairwise tensor file not available")
        from alphatrain.dataset import TensorDatasetGPU
        return TensorDatasetGPU(path, augment=True, device='cpu')

    def test_has_pairs(self, dataset):
        assert dataset.has_pairs
        assert dataset.n_pairs > 0
        assert dataset.good_boards.shape[0] == dataset.n_pairs
        assert dataset.bad_boards.shape[0] == dataset.n_pairs
        assert dataset.margins.shape[0] == dataset.n_pairs

    def test_collate_pairwise_shapes(self, dataset):
        indices = list(range(32))
        obs, policy, val, good_obs, bad_obs, margin = dataset.collate_pairwise(indices)
        B = len(indices)
        assert obs.shape == (B, 18, 9, 9)
        assert policy.shape == (B, 6561)
        assert val.shape == (B, 64)
        assert good_obs.shape == (B, 18, 9, 9)
        assert bad_obs.shape == (B, 18, 9, 9)
        assert margin.shape == (B,)

    def test_margins_positive(self, dataset):
        """Margins should be non-negative (good >= bad by definition)."""
        assert (dataset.margins >= 0).all()

    def test_good_bad_boards_differ(self, dataset):
        """Good and bad afterstate boards should differ (different moves)."""
        n_diff = 0
        for i in range(min(100, dataset.n_pairs)):
            if not torch.equal(dataset.good_boards[i], dataset.bad_boards[i]):
                n_diff += 1
        assert n_diff > 50, "Most pairs should have different boards"

    def test_collate_standard_still_works(self, dataset):
        """Standard collate should still work on pairwise dataset."""
        indices = list(range(16))
        obs, policy, val = dataset.collate(indices)
        assert obs.shape == (16, 18, 9, 9)

    def test_max_score_loaded(self, dataset):
        assert dataset.max_score == 500.0


class TestMarginRankingLoss:
    """Test the ranking loss computation matches expectations."""

    def test_correct_ranking_zero_loss(self):
        """When V(good) > V(bad), loss should be zero."""
        loss_fn = torch.nn.MarginRankingLoss(margin=0.0)
        v_good = torch.tensor([10.0, 20.0])
        v_bad = torch.tensor([5.0, 8.0])
        target = torch.ones(2)
        loss = loss_fn(v_good, v_bad, target)
        assert loss.item() == 0.0

    def test_wrong_ranking_positive_loss(self):
        """When V(good) < V(bad), loss should be positive."""
        loss_fn = torch.nn.MarginRankingLoss(margin=0.0)
        v_good = torch.tensor([5.0])
        v_bad = torch.tensor([10.0])
        target = torch.ones(1)
        loss = loss_fn(v_good, v_bad, target)
        assert loss.item() > 0.0

    def test_gradient_flows_through_model(self):
        """Verify ranking loss produces gradients for the model."""
        net = AlphaTrainNet(num_blocks=2, channels=32)
        obs1 = torch.randn(4, 18, 9, 9)
        obs2 = torch.randn(4, 18, 9, 9)
        _, val1 = net(obs1)
        _, val2 = net(obs2)
        v_good = net.predict_value(val1, max_val=500.0)
        v_bad = net.predict_value(val2, max_val=500.0)
        # Match actual training loss: F.relu(margin - (v_good - v_bad))
        margin = torch.tensor([5.0, 10.0, 3.0, 8.0])
        loss = F.relu(margin - (v_good - v_bad)).mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in net.parameters())
        assert has_grad, "Ranking loss should produce gradients"


class TestBuildObsBoardsOnly:
    """Test _build_obs_boards_only produces valid observations."""

    @pytest.fixture
    def dataset(self):
        import os
        path = 'alphatrain/data/alphatrain_pairwise.pt'
        if not os.path.exists(path):
            pytest.skip("Pairwise tensor file not available")
        from alphatrain.dataset import TensorDatasetGPU
        return TensorDatasetGPU(path, augment=True, device='cpu')

    def test_output_shape(self, dataset):
        boards = dataset.good_boards[:8]
        obs = dataset._build_obs_boards_only(boards)
        assert obs.shape == (8, 18, 9, 9)

    def test_color_channels_correct(self, dataset):
        """Color one-hot channels should match board values."""
        boards = dataset.good_boards[:4]
        obs = dataset._build_obs_boards_only(boards)
        for b in range(4):
            for r in range(9):
                for c in range(9):
                    v = boards[b, r, c].item()
                    if v == 0:
                        assert obs[b, 7, r, c] == 1.0
                    else:
                        assert obs[b, v - 1, r, c] == 1.0

    def test_boards_only_has_no_next_balls(self, dataset):
        """_build_obs_boards_only should have zero next_ball channels."""
        boards = dataset.good_boards[:4]
        obs = dataset._build_obs_boards_only(boards)
        assert obs[:, 8:12].abs().sum() == 0.0


class TestAfterStateNextBalls:
    """Verify collate_pairwise includes next_balls in afterstate obs."""

    @pytest.fixture
    def dataset(self):
        import os
        path = 'alphatrain/data/alphatrain_pairwise.pt'
        if not os.path.exists(path):
            pytest.skip("Pairwise tensor file not available")
        from alphatrain.dataset import TensorDatasetGPU
        return TensorDatasetGPU(path, augment=True, device='cpu')

    def test_afterstate_has_next_balls(self, dataset):
        """Afterstate obs from collate_pairwise must have next_balls (ch 8-11)."""
        indices = list(range(32))
        _, _, _, good_obs, bad_obs, _ = dataset.collate_pairwise(indices)
        assert good_obs[:, 8:12].abs().sum() > 0, \
            "Afterstate good_obs missing next_balls"
        assert bad_obs[:, 8:12].abs().sum() > 0, \
            "Afterstate bad_obs missing next_balls"

    def test_good_bad_share_next_balls(self, dataset):
        """Good and bad afterstates from same parent must have same next_balls."""
        indices = list(range(32))
        _, _, _, good_obs, bad_obs, _ = dataset.collate_pairwise(indices)
        assert torch.allclose(good_obs[:, 8:12], bad_obs[:, 8:12]), \
            "Good/bad afterstates should share next_balls from same parent"
