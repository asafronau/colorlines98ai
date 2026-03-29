"""Unit tests for AlphaTrain dataset."""

import os
import json
import tempfile
import numpy as np
import torch
import pytest
from alphatrain.dataset import (
    make_flat_policy_target, score_to_twohot,
    precompute_tensors, TensorDatasetGPU,
    NUM_MOVES, BOARD_SIZE
)
from alphatrain.observation import NUM_CHANNELS


class TestPolicyTarget:
    def test_empty_input(self):
        t = make_flat_policy_target([], [])
        assert t.shape == (NUM_MOVES,)
        assert t.sum() == 0

    def test_single_move(self):
        moves = [{'sr': 0, 'sc': 0, 'tr': 8, 'tc': 8}]
        scores = [10.0]
        t = make_flat_policy_target(moves, scores)
        assert t.sum() == pytest.approx(1.0)
        idx = 0 * 81 + 8 * 9 + 8
        assert t[idx] == pytest.approx(1.0)

    def test_two_moves_softmax(self):
        moves = [{'sr': 0, 'sc': 0, 'tr': 0, 'tc': 1},
                 {'sr': 0, 'sc': 0, 'tr': 0, 'tc': 2}]
        scores = [10.0, 10.0]  # equal scores
        t = make_flat_policy_target(moves, scores)
        assert t.sum() == pytest.approx(1.0)
        idx1 = 0 * 81 + 0 * 9 + 1
        idx2 = 0 * 81 + 0 * 9 + 2
        assert t[idx1] == pytest.approx(0.5, abs=0.01)
        assert t[idx2] == pytest.approx(0.5, abs=0.01)

    def test_temperature(self):
        moves = [{'sr': 0, 'sc': 0, 'tr': 0, 'tc': 1},
                 {'sr': 0, 'sc': 0, 'tr': 0, 'tc': 2}]
        scores = [20.0, 10.0]
        # High temperature → more uniform
        t_high = make_flat_policy_target(moves, scores, temperature=10.0)
        # Low temperature → more peaked
        t_low = make_flat_policy_target(moves, scores, temperature=0.1)
        idx1 = 0 * 81 + 0 * 9 + 1
        assert t_low[idx1] > t_high[idx1]


class TestScoreToTwohot:
    def test_output_shape(self):
        t = score_to_twohot(5000, num_bins=64)
        assert t.shape == (64,)

    def test_sums_to_one(self):
        t = score_to_twohot(5000, num_bins=64)
        assert t.sum() == pytest.approx(1.0)

    def test_min_value(self):
        t = score_to_twohot(0, num_bins=64)
        assert t[0] == pytest.approx(1.0)

    def test_max_value(self):
        t = score_to_twohot(30000, num_bins=64)
        assert t[63] == pytest.approx(1.0)

    def test_clamping(self):
        t = score_to_twohot(-100, num_bins=64)
        assert t[0] == pytest.approx(1.0)
        t = score_to_twohot(99999, num_bins=64)
        assert t[63] == pytest.approx(1.0)


class TestPrecompute:
    def test_precompute_creates_file(self, tmp_path):
        """Create a minimal game file and precompute."""
        game_dir = tmp_path / "games"
        game_dir.mkdir()

        # Minimal game with 2 moves
        board = [[0]*9 for _ in range(9)]
        board[0][0] = 1
        board[4][4] = 2
        game = {
            'seed': 1, 'score': 100, 'turns': 2, 'num_moves': 2,
            'moves': [
                {
                    'board': board,
                    'next_balls': [{'row': 1, 'col': 1, 'color': 3}],
                    'num_next': 1,
                    'chosen_move': {'sr': 0, 'sc': 0, 'tr': 0, 'tc': 1},
                    'top_moves': [{'sr': 0, 'sc': 0, 'tr': 0, 'tc': 1}],
                    'top_scores': [15.0],
                    'num_top': 1,
                    'game_score': 100,
                },
                {
                    'board': board,
                    'next_balls': [{'row': 2, 'col': 2, 'color': 5}],
                    'num_next': 1,
                    'chosen_move': {'sr': 4, 'sc': 4, 'tr': 4, 'tc': 5},
                    'top_moves': [{'sr': 4, 'sc': 4, 'tr': 4, 'tc': 5}],
                    'top_scores': [12.0],
                    'num_top': 1,
                    'game_score': 100,
                },
            ]
        }
        with open(game_dir / "game_seed1_score100.json", 'w') as f:
            json.dump(game, f)

        out_path = str(tmp_path / "tensors.pt")
        precompute_tensors(str(game_dir), out_path)
        assert os.path.exists(out_path)

        data = torch.load(out_path, weights_only=True)
        assert data['boards'].shape == (2, 9, 9)
        assert data['pol_indices'].shape == (2, 10)
        assert data['val_targets'].shape == (2, 64)


class TestDihedralAugmentation:
    def test_identity_transform(self):
        """Transform 0 should not change anything."""
        from alphatrain.dataset import _OBS_LUTS, _POL_LUTS
        obs_lut = _OBS_LUTS[0]
        pol_lut = _POL_LUTS[0]
        assert np.all(obs_lut == np.arange(81))
        assert np.all(pol_lut == np.arange(NUM_MOVES))

    def test_rotation_is_valid_permutation(self):
        from alphatrain.dataset import _OBS_LUTS
        for t in range(8):
            lut = _OBS_LUTS[t]
            assert len(set(lut)) == 81, f"Transform {t} has duplicates"

    def test_policy_lut_is_valid_permutation(self):
        from alphatrain.dataset import _POL_LUTS
        for t in range(8):
            lut = _POL_LUTS[t]
            assert len(set(lut)) == NUM_MOVES, f"Transform {t} has duplicates"


class TestTDValues:
    """Tests using synthetic game data (works everywhere, including Colab)."""

    def _make_synthetic_game(self, tmp_path):
        """Create a real game via engine and save as JSON with chosen_moves."""
        from game.board import ColorLinesGame
        game = ColorLinesGame(seed=999)
        game.reset()
        moves_data = []
        for _ in range(80):  # play enough turns to score some points
            if game.game_over:
                break
            legal = game.get_legal_moves()
            if not legal:
                break
            move = legal[0]  # always pick first legal move
            board_snap = game.board.tolist()
            nb = [{'row': r, 'col': c, 'color': int(col)}
                  for (r, c), col in game.next_balls]
            moves_data.append({
                'board': board_snap,
                'next_balls': nb,
                'num_next': len(nb),
                'chosen_move': {'sr': move[0][0], 'sc': move[0][1],
                                'tr': move[1][0], 'tc': move[1][1]},
                'top_moves': [{'sr': move[0][0], 'sc': move[0][1],
                               'tr': move[1][0], 'tc': move[1][1]}],
                'top_scores': [10.0],
                'num_top': 1,
                'game_score': game.score,  # will be overwritten
            })
            game.move(move[0], move[1])
        final_score = game.score
        game_data = {
            'seed': 999, 'score': final_score,
            'turns': len(moves_data), 'num_moves': len(moves_data),
            'moves': moves_data,
        }
        # Update game_score to final (matches original format)
        for m in game_data['moves']:
            m['game_score'] = final_score
        game_dir = tmp_path / "games"
        game_dir.mkdir(exist_ok=True)
        path = game_dir / f"game_seed999_score{final_score}.json"
        with open(path, 'w') as f:
            json.dump(game_data, f)
        return game_data, str(game_dir)

    def test_replay_game_scores(self, tmp_path):
        """Verify _replay_game_scores returns correct running scores."""
        from alphatrain.dataset import _replay_game_scores
        game_data, _ = self._make_synthetic_game(tmp_path)
        scores = _replay_game_scores(game_data)
        assert len(scores) == len(game_data['moves'])
        assert scores[0] == 0  # starts at 0
        assert all(scores[i] <= scores[i+1] for i in range(len(scores)-1))
        assert scores[-1] <= game_data['score']

    def test_td_targets_differ_within_game(self, tmp_path):
        """TD targets should vary within a game (unlike game_score)."""
        from alphatrain.dataset import _replay_game_scores
        game_data, _ = self._make_synthetic_game(tmp_path)
        scores = _replay_game_scores(game_data)
        final = game_data['score']
        remaining = [final - s for s in scores]
        assert remaining[0] == final
        assert remaining[-1] <= final
        # Should have at least some distinct values
        assert len(set(remaining)) >= 2

    def test_td_remaining_score_monotonic(self, tmp_path):
        """Remaining score decreases (or stays same) along trajectory."""
        from alphatrain.dataset import _replay_game_scores
        game_data, _ = self._make_synthetic_game(tmp_path)
        scores = _replay_game_scores(game_data)
        final = game_data['score']
        remaining = [final - s for s in scores]
        for i in range(len(remaining) - 1):
            assert remaining[i] >= remaining[i+1]

    def test_precompute_td_mode(self, tmp_path):
        """Precompute with value_mode='td' produces different targets per state."""
        game_data, game_dir = self._make_synthetic_game(tmp_path)
        out_path = str(tmp_path / "td_tensors.pt")
        precompute_tensors(game_dir, out_path, value_mode='td', gamma=1.0)
        data = torch.load(out_path, weights_only=True)
        assert data['val_targets'].shape[0] == len(game_data['moves'])
        assert data.get('value_mode') == 'td'
        # Decode two-hot back to scalar
        bins = torch.linspace(0, data['max_score'], data['num_value_bins'])
        vals = (data['val_targets'] * bins).sum(dim=-1)
        # First state should have higher value than last
        assert vals[0] >= vals[-1]
