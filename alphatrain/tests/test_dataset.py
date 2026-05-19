"""Unit tests for AlphaTrain dataset."""

import os
import json
import tempfile
import numpy as np
import torch
import pytest
from alphatrain.dataset import (
    make_flat_policy_target,
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
        # High temperature -> more uniform
        t_high = make_flat_policy_target(moves, scores, temperature=10.0)
        # Low temperature -> more peaked
        t_low = make_flat_policy_target(moves, scores, temperature=0.1)
        idx1 = 0 * 81 + 0 * 9 + 1
        assert t_low[idx1] > t_high[idx1]


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
        # No value targets in policy-only mode
        assert 'val_targets' not in data


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


def _make_fake_tensor_file(tmp_path, n_states=8, device='cpu'):
    """Synthesize a small tensor file matching the precompute_tensors format."""
    rng = np.random.default_rng(0)
    boards = rng.integers(0, 8, size=(n_states, 9, 9), dtype=np.int8)
    next_pos = np.zeros((n_states, 3, 2), dtype=np.int8)
    next_col = rng.integers(1, 8, size=(n_states, 3), dtype=np.int8)
    n_next = np.full(n_states, 3, dtype=np.int8)
    for i in range(n_states):
        # Place next balls on empty cells if possible
        for j in range(3):
            r, c = rng.integers(0, 9), rng.integers(0, 9)
            next_pos[i, j, 0], next_pos[i, j, 1] = r, c
    # Sparse policy: one move per state
    pol_indices = np.zeros((n_states, 10), dtype=np.int64)
    pol_values = np.zeros((n_states, 10), dtype=np.float32)
    for i in range(n_states):
        pol_indices[i, 0] = (i * 100) % NUM_MOVES
        pol_values[i, 0] = 1.0
    pol_nnz = np.ones(n_states, dtype=np.int64)
    data = {
        'boards': torch.from_numpy(boards),
        'next_pos': torch.from_numpy(next_pos),
        'next_col': torch.from_numpy(next_col),
        'n_next': torch.from_numpy(n_next),
        'pol_indices': torch.from_numpy(pol_indices),
        'pol_values': torch.from_numpy(pol_values),
        'pol_nnz': torch.from_numpy(pol_nnz),
        'max_score': 30000.0,
        'num_channels': NUM_CHANNELS,
    }
    path = str(tmp_path / "fake_tensors.pt")
    torch.save(data, path)
    return path


class TestTrainValSplit:
    """Verify train/val do not share base states (no leakage)."""

    def test_base_indices_disjoint(self, tmp_path):
        path = _make_fake_tensor_file(tmp_path, n_states=20)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        train, val = TensorDatasetGPU.make_train_val_split(
            path, val_split=0.25, augment=False, color_augment=False,
            device='cpu', seed=42)
        train_base = set(train.base_indices.cpu().tolist())
        val_base = set(val.base_indices.cpu().tolist())
        assert train_base.isdisjoint(val_base)
        assert train_base | val_base == set(range(20))

    def test_split_deterministic(self, tmp_path):
        path = _make_fake_tensor_file(tmp_path, n_states=20)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        t1, v1 = TensorDatasetGPU.make_train_val_split(
            path, val_split=0.25, augment=False, color_augment=False,
            device='cpu', seed=42)
        t2, v2 = TensorDatasetGPU.make_train_val_split(
            path, val_split=0.25, augment=False, color_augment=False,
            device='cpu', seed=42)
        assert torch.equal(t1.base_indices, t2.base_indices)
        assert torch.equal(v1.base_indices, v2.base_indices)


class TestColorPermutationAugmentation:
    """Color permutation is a full symmetry of the game.

    Permuting all 7 ball colors (and the next-ball colors) produces an
    equivalent game state. Move actions don't depend on color labels,
    so policy targets should be unchanged.
    """

    def test_color_perm_preserves_color_counts(self, tmp_path):
        """After permutation, each color count should be reassigned 1-to-1."""
        path = _make_fake_tensor_file(tmp_path, n_states=32)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        ds = TensorDatasetGPU(path, augment=False, color_augment=True,
                               device='cpu')
        torch.manual_seed(0)
        obs, policy = ds.collate(list(range(8)))
        # Sum over channels 0-6 should equal occupancy = (board != 0).sum()
        # over the original board (which is a fixed quantity per sample)
        # but we need to compute it from the *permuted* obs, which should still
        # have the same total occupancy.
        per_sample_occupancy = obs[:, 0:7].sum(dim=(1, 2, 3))
        # Each occupied cell sets exactly one of 7 color channels to 1.0.
        # Expected total = total ball count from original boards.
        n_balls_orig = (ds.boards[:8] != 0).sum(dim=(1, 2)).float()
        assert torch.allclose(per_sample_occupancy, n_balls_orig), \
            "Color permutation broke total occupancy"

    def test_color_perm_does_not_remap_policy_target(self, tmp_path):
        """Policy target indices/values must be unchanged under color perm."""
        path = _make_fake_tensor_file(tmp_path, n_states=8)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        ds_noaug = TensorDatasetGPU(path, augment=False, color_augment=False,
                                      device='cpu')
        ds_color = TensorDatasetGPU(path, augment=False, color_augment=True,
                                      device='cpu')
        # Run multiple times with different RNG; policy targets should still
        # match across the two datasets (color perm doesn't touch policy).
        torch.manual_seed(0)
        _, p_color = ds_color.collate(list(range(8)))
        torch.manual_seed(0)
        _, p_noaug = ds_noaug.collate(list(range(8)))
        assert torch.allclose(p_color, p_noaug), \
            "Color permutation changed policy target (it shouldn't!)"

    def test_color_perm_changes_obs(self, tmp_path):
        """With high probability, color perm produces different observations."""
        path = _make_fake_tensor_file(tmp_path, n_states=8)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        ds_noaug = TensorDatasetGPU(path, augment=False, color_augment=False,
                                      device='cpu')
        ds_color = TensorDatasetGPU(path, augment=False, color_augment=True,
                                      device='cpu')
        torch.manual_seed(0)
        obs_color, _ = ds_color.collate(list(range(8)))
        obs_noaug, _ = ds_noaug.collate(list(range(8)))
        # Color channels (0-6) should differ; topology/line channels (12-17)
        # should NOT (color-invariant).
        assert not torch.allclose(obs_color[:, 0:7], obs_noaug[:, 0:7]), \
            "Color channels unchanged — perm not applied"
        # Channel 7 (empty) and channel 12 (component area) should be unchanged
        assert torch.allclose(obs_color[:, 7], obs_noaug[:, 7]), \
            "Empty channel should be color-invariant"
        assert torch.allclose(obs_color[:, 12], obs_noaug[:, 12]), \
            "Component area channel should be color-invariant"
        # Line potentials (13-17) are based on same-color runs — they depend on
        # which cells share a color, not which color it is. So they should be
        # SAME UNDER PERMUTATION as long as we permute identically (per-sample).
        assert torch.allclose(obs_color[:, 13:18], obs_noaug[:, 13:18]), \
            "Line potential channels should be color-invariant"

    def test_color_perm_next_balls_remapped(self, tmp_path):
        """Next ball channels (8-10) encode color/7; perm must remap them."""
        path = _make_fake_tensor_file(tmp_path, n_states=8)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        ds_color = TensorDatasetGPU(path, augment=False, color_augment=True,
                                      device='cpu')
        ds_noaug = TensorDatasetGPU(path, augment=False, color_augment=False,
                                      device='cpu')
        torch.manual_seed(0)
        obs_color, _ = ds_color.collate(list(range(8)))
        obs_noaug, _ = ds_noaug.collate(list(range(8)))
        # The next-ball mask channel (11) should be identical (positions same)
        assert torch.allclose(obs_color[:, 11], obs_noaug[:, 11]), \
            "Next-ball mask should not change under color perm"
        # The next-ball color channels (8-10) carry color/7.0 values.
        # Under permutation the SUM over a single channel changes (different
        # next balls map to different scalar slot), but the SET of nonzero
        # values per sample should still be a subset of {1/7, ..., 7/7}.
        for i in range(min(8, obs_color.shape[0])):
            ch_colors = obs_color[i, 8:11]
            nonzero = ch_colors[ch_colors > 0]
            for v in nonzero.cpu().tolist():
                k = round(v * 7.0)
                assert 1 <= k <= 7, f"Bad next-ball color value {v}"

    def test_color_perm_consistent_board_and_next(self, tmp_path):
        """Per-sample permutation must be applied IDENTICALLY to board colors
        and next-ball colors. We construct a controlled fake state where each
        sample has a single unique color on the board, then verify that the
        permutation inferred from the board matches the one applied to
        next_col.
        """
        # Hand-crafted state: sample i has color (i % 7 + 1) at (0, 0) on
        # the board and the SAME color as next ball #0 at (1, 1).
        n_states = 7  # one per color
        boards = np.zeros((n_states, 9, 9), dtype=np.int8)
        next_pos = np.zeros((n_states, 3, 2), dtype=np.int8)
        next_col = np.zeros((n_states, 3), dtype=np.int8)
        n_next = np.full(n_states, 1, dtype=np.int8)
        pol_indices = np.zeros((n_states, 10), dtype=np.int64)
        pol_values = np.zeros((n_states, 10), dtype=np.float32)
        pol_nnz = np.ones(n_states, dtype=np.int64)
        for i in range(n_states):
            color = i + 1
            boards[i, 0, 0] = color    # board has just this one color
            next_pos[i, 0, 0] = 1
            next_pos[i, 0, 1] = 1
            next_col[i, 0] = color     # next ball is SAME color
            pol_indices[i, 0] = 100
            pol_values[i, 0] = 1.0
        data = {
            'boards': torch.from_numpy(boards),
            'next_pos': torch.from_numpy(next_pos),
            'next_col': torch.from_numpy(next_col),
            'n_next': torch.from_numpy(n_next),
            'pol_indices': torch.from_numpy(pol_indices),
            'pol_values': torch.from_numpy(pol_values),
            'pol_nnz': torch.from_numpy(pol_nnz),
            'max_score': 30000.0,
            'num_channels': NUM_CHANNELS,
        }
        p = str(tmp_path / "consistency_check.pt")
        torch.save(data, p)
        from alphatrain.dataset import _BACKING_CACHE
        _BACKING_CACHE.clear()
        ds = TensorDatasetGPU(p, augment=False, color_augment=True,
                                device='cpu')
        torch.manual_seed(123)
        obs, _ = ds.collate(list(range(n_states)))
        # For each sample, infer the new color from channel index where (0,0)
        # has value 1.0, and check next-ball at (1,1) in channel 8 has the
        # same scaled value.
        for i in range(n_states):
            board_planes = obs[i, 0:7, 0, 0]
            assert board_planes.sum().item() == pytest.approx(1.0), \
                f"sample {i}: board cell should have exactly one color plane on"
            new_color_idx = int(board_planes.argmax().item())  # 0..6
            new_color = new_color_idx + 1  # 1..7
            # Channel 8 at (1,1) holds the first next ball's color / 7.0
            next_scalar = obs[i, 8, 1, 1].item()
            inferred = round(next_scalar * 7.0)
            assert inferred == new_color, (
                f"sample {i}: board permuted to color {new_color} "
                f"but next-ball channel encoded color {inferred}")


@pytest.mark.skip(reason="TD value targets — deleted in arch/v2-clean Stage 2")
class TestTDValues:
    """Tests using synthetic game data (works everywhere, including Colab)."""

    def _make_synthetic_game(self, tmp_path):
        """Create a real game via engine and save as JSON with chosen_moves."""
        from game.board import ColorLinesGame
        game = ColorLinesGame(seed=669)  # scores at multiple turns with first-legal-move
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
            'seed': 669, 'score': final_score,
            'turns': len(moves_data), 'num_moves': len(moves_data),
            'moves': moves_data,
        }
        # Update game_score to final (matches original format)
        for m in game_data['moves']:
            m['game_score'] = final_score
        game_dir = tmp_path / "games"
        game_dir.mkdir(exist_ok=True)
        path = game_dir / f"game_seed669_score{final_score}.json"
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

    def test_precompute_policy_only(self, tmp_path):
        """Precompute produces policy-only tensors (no value targets)."""
        game_data, game_dir = self._make_synthetic_game(tmp_path)
        out_path = str(tmp_path / "policy_tensors.pt")
        precompute_tensors(game_dir, out_path)
        data = torch.load(out_path, weights_only=True)
        assert data['boards'].shape[0] == len(game_data['moves'])
        assert 'val_targets' not in data
        assert 'num_value_bins' not in data
