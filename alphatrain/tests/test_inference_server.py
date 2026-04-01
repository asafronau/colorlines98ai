"""Tests for inference server: shared memory, GPU batching, client/server."""

import pytest
import numpy as np
import torch
import time

from alphatrain.inference_server import (
    InferenceServer, InferenceClient, OBS_SHAPE, POL_SIZE, MAX_BATCH
)


def _get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        pytest.skip("No GPU available")


MODEL_PATH = 'alphatrain/data/alphatrain_td_best.pt'


@pytest.fixture
def server():
    """Create and start a 2-worker inference server."""
    import os
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model not found")
    device = _get_device()
    srv = InferenceServer(MODEL_PATH, num_workers=2, device=device,
                          max_batch_per_worker=8)
    srv.start()
    time.sleep(1)  # let GPU process initialize
    yield srv
    srv.shutdown()


class TestInferenceServer:
    """Test InferenceServer lifecycle."""

    def test_server_starts_and_is_alive(self, server):
        assert server.is_alive()

    def test_shared_memory_shapes(self, server):
        assert server.obs_buf.shape == (2, 8) + OBS_SHAPE
        assert server.pol_buf.shape == (2, 8, POL_SIZE)
        assert server.val_buf.shape == (2, 8)

    def test_make_client(self, server):
        client = server.make_client(0)
        assert client.slot_id == 0
        assert client.obs_buf is server.obs_buf

    def test_shutdown(self):
        import os
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Model not found")
        device = _get_device()
        srv = InferenceServer(MODEL_PATH, num_workers=1, device=device)
        srv.start()
        time.sleep(1)
        assert srv.is_alive()
        srv.shutdown()
        time.sleep(0.5)
        assert not srv.is_alive()


class TestInferenceClient:
    """Test InferenceClient single and batch evaluation."""

    def test_single_evaluate(self, server):
        client = server.make_client(0)
        obs = np.random.randn(*OBS_SHAPE).astype(np.float32)
        pol, val = client.evaluate(obs)

        assert pol.shape == (POL_SIZE,)
        assert pol.dtype == np.float32
        assert isinstance(val, float)
        assert 0 <= val <= 1000  # sigmoid * max_score

    def test_batch_evaluate(self, server):
        client = server.make_client(0)
        batch = np.random.randn(4, *OBS_SHAPE).astype(np.float32)
        pol, val = client.evaluate_batch(batch, 4)

        assert pol.shape[0] == 4
        assert pol.shape[1] == POL_SIZE
        assert val.shape == (4,)

    def test_two_clients_independent(self, server):
        """Two clients get independent results for different inputs."""
        c0 = server.make_client(0)
        c1 = server.make_client(1)

        obs0 = np.zeros(OBS_SHAPE, dtype=np.float32)
        obs1 = np.ones(OBS_SHAPE, dtype=np.float32)

        pol0, val0 = c0.evaluate(obs0)
        pol1, val1 = c1.evaluate(obs1)

        # Different inputs should give different outputs
        assert not np.allclose(pol0, pol1, atol=0.01)

    def test_deterministic_same_input(self, server):
        """Same input gives same output (deterministic inference)."""
        client = server.make_client(0)
        obs = np.random.randn(*OBS_SHAPE).astype(np.float32)

        pol1, val1 = client.evaluate(obs)
        pol2, val2 = client.evaluate(obs)

        np.testing.assert_allclose(pol1, pol2, atol=1e-4)
        assert abs(val1 - val2) < 0.1

    def test_matches_direct_inference(self, server):
        """Server results match direct model inference."""
        from alphatrain.evaluate import load_model
        from alphatrain.mcts import _build_obs_for_game
        from game.board import ColorLinesGame

        device = torch.device(_get_device())
        net, max_score = load_model(MODEL_PATH, device,
                                    fp16=True, jit_trace=True)

        game = ColorLinesGame(seed=42)
        game.reset()
        obs_np = _build_obs_for_game(game)

        # Direct inference
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device).half()
        with torch.inference_mode():
            pol_logits, val_logits = net(obs_t)
            direct_val = net.predict_value(
                val_logits, max_val=max_score).item()
        direct_pol = pol_logits[0].float().cpu().numpy()

        # Server inference
        client = server.make_client(0)
        server_pol, server_val = client.evaluate(obs_np)

        # Should be close (FP16 rounding may differ slightly)
        np.testing.assert_allclose(
            direct_pol, server_pol, atol=0.5,
            err_msg="Policy logits diverge between direct and server")
        assert abs(direct_val - server_val) < 5.0, \
            f"Value diverges: direct={direct_val:.1f} server={server_val:.1f}"


class TestServerWithMCTS:
    """Test MCTS using inference server produces valid results."""

    def test_mcts_search_with_server(self, server):
        from alphatrain.mcts import MCTS
        from game.board import ColorLinesGame

        client = server.make_client(0)
        mcts = MCTS(inference_client=client, max_score=500,
                     num_simulations=50, batch_size=8,
                     top_k=30, c_puct=2.5)

        game = ColorLinesGame(seed=42)
        game.reset()
        action = mcts.search(game)

        assert action is not None
        (sr, sc), (tr, tc) = action
        assert 0 <= sr < 9 and 0 <= sc < 9
        assert 0 <= tr < 9 and 0 <= tc < 9

    def test_mcts_selfplay_with_server(self, server):
        """Full self-play game via server produces valid training data."""
        from alphatrain.mcts import MCTS
        from alphatrain.scripts.selfplay import play_selfplay_game

        client = server.make_client(0)
        mcts = MCTS(inference_client=client, max_score=500,
                     num_simulations=50, batch_size=8,
                     top_k=30, c_puct=2.5)

        result = play_selfplay_game(mcts, seed=42,
                                    temperature_moves=5,
                                    dirichlet_alpha=0.3,
                                    dirichlet_weight=0.25)

        assert result['score'] >= 0
        assert result['turns'] > 0
        assert result['observations'].shape[0] == result['turns']
        assert result['policy_targets'].shape == (result['turns'], 6561)
        assert result['value_targets'].shape == (result['turns'],)
