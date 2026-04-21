"""AlphaTrain dataset: loads expert game data with 18-channel observations.

Each sample: (observation, policy_target)
- observation: (18, 9, 9) float32 with tactical features
- policy_target: (6561,) flat joint move probability

Supports precomputed tensors for GPU-native training (TensorDatasetGPU).
"""

import os
import json
import glob
import time
import numpy as np
import torch
from torch.utils.data import Dataset

from alphatrain.observation import build_observation, build_line_potentials_batch, NUM_CHANNELS

BOARD_SIZE = 9
NUM_MOVES = BOARD_SIZE ** 4


def make_flat_policy_target(top_moves, top_scores, temperature=1.0):
    """Convert top moves + scores to flat (6561,) probability distribution."""
    target = np.zeros(NUM_MOVES, dtype=np.float32)
    if not top_moves or not top_scores:
        return target

    scores = np.array(top_scores, dtype=np.float64)
    scores = scores / max(temperature, 1e-8)
    scores -= scores.max()
    probs = np.exp(scores)
    probs /= probs.sum()

    for i, m in enumerate(top_moves):
        src_idx = m['sr'] * BOARD_SIZE + m['sc']
        tgt_idx = m['tr'] * BOARD_SIZE + m['tc']
        target[src_idx * 81 + tgt_idx] = probs[i]

    return target


def _replay_game_scores(game_data):
    """Replay a game from seed to get running scores at each move.

    Returns list of running_score values, one per move in game_data['moves'].
    running_score[i] = game score BEFORE move i is executed.
    """
    from game.board import ColorLinesGame as CLG
    game = CLG(seed=game_data['seed'])
    game.reset()
    running = []
    for move in game_data['moves']:
        running.append(game.score)
        cm = move['chosen_move']
        game.move((cm['sr'], cm['sc']), (cm['tr'], cm['tc']))
    return running


def precompute_tensors(data_path, output_path, policy_temperature=1.0,
                        max_score=30000.0, **kwargs):
    """Convert JSON game data to precomputed tensor file.

    Stores sparse policy (indices + values) + raw boards
    for on-the-fly 18-channel observation building on GPU.

    Extra kwargs are accepted for backward compatibility but ignored.
    """
    files = sorted(glob.glob(os.path.join(data_path, 'game_*.json')))
    print(f"Loading {len(files)} game files...", flush=True)

    all_moves = []
    for fi, f in enumerate(files):
        with open(f) as fh:
            game_data = json.load(fh)
        for move in game_data['moves']:
            all_moves.append(move)

        if (fi + 1) % 100 == 0:
            print(f"  Loaded {fi+1}/{len(files)} games "
                  f"({len(all_moves):,} states)", flush=True)

    n = len(all_moves)
    print(f"Precomputing {n:,} states...", flush=True)
    t0 = time.time()

    boards = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    next_pos = np.zeros((n, 3, 2), dtype=np.int8)
    next_col = np.zeros((n, 3), dtype=np.int8)
    n_next = np.zeros(n, dtype=np.int8)
    pol_indices = np.zeros((n, 10), dtype=np.int16)
    pol_values = np.zeros((n, 10), dtype=np.float32)
    pol_nnz = np.zeros(n, dtype=np.int8)

    for i, move in enumerate(all_moves):
        board = np.array(move['board'], dtype=np.int8)
        boards[i] = board

        for j, nb in enumerate(move['next_balls']):
            if j >= 3:
                break
            next_pos[i, j, 0] = nb['row']
            next_pos[i, j, 1] = nb['col']
            next_col[i, j] = nb['color']
        n_next[i] = min(len(move['next_balls']), 3)

        flat = make_flat_policy_target(
            move['top_moves'], move['top_scores'], policy_temperature)
        nz = np.nonzero(flat)[0]
        k = min(len(nz), 10)
        pol_indices[i, :k] = nz[:k].astype(np.int16)
        pol_values[i, :k] = flat[nz[:k]]
        pol_nnz[i] = k

        if (i + 1) % 200000 == 0:
            print(f"  {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%)", flush=True)

    elapsed = time.time() - t0
    print(f"Precomputed in {elapsed:.1f}s", flush=True)

    data = {
        'boards': torch.from_numpy(boards),
        'next_pos': torch.from_numpy(next_pos),
        'next_col': torch.from_numpy(next_col),
        'n_next': torch.from_numpy(n_next),
        'pol_indices': torch.from_numpy(pol_indices.astype(np.int64)),
        'pol_values': torch.from_numpy(pol_values),
        'pol_nnz': torch.from_numpy(pol_nnz.astype(np.int64)),
        'max_score': max_score,
        'num_channels': NUM_CHANNELS,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved to {output_path} ({size_mb:.0f} MB)", flush=True)


# -- Dihedral augmentation LUTs --

def _build_dihedral_luts():
    """Precompute LUTs for 8 dihedral transforms on 9x9 board."""
    obs_luts = []
    pol_luts = []
    for t in range(8):
        k = t % 4
        flip = t >= 4
        obs_lut = np.zeros(81, dtype=np.int32)
        pol_lut = np.zeros(NUM_MOVES, dtype=np.int32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                nr, nc = r, c
                if flip:
                    nc = BOARD_SIZE - 1 - nc
                for _ in range(k):
                    nr, nc = nc, BOARD_SIZE - 1 - nr
                obs_lut[r * 9 + c] = nr * 9 + nc
                for tr in range(BOARD_SIZE):
                    for tc in range(BOARD_SIZE):
                        ntr, ntc = tr, tc
                        if flip:
                            ntc = BOARD_SIZE - 1 - ntc
                        for _ in range(k):
                            ntr, ntc = ntc, BOARD_SIZE - 1 - ntr
                        old_idx = (r * 9 + c) * 81 + tr * 9 + tc
                        new_idx = (nr * 9 + nc) * 81 + ntr * 9 + ntc
                        pol_lut[old_idx] = new_idx
        obs_luts.append(obs_lut)
        pol_luts.append(pol_lut)
    return obs_luts, pol_luts

_OBS_LUTS, _POL_LUTS = _build_dihedral_luts()


class TensorDatasetGPU(Dataset):
    """GPU-resident dataset with on-the-fly 18-channel observation building.

    Stores compact board data on GPU. The collate function builds observations,
    applies augmentation, and reconstructs sparse policy targets -- all on GPU.
    """

    def __init__(self, tensor_path, augment=True, device='cuda'):
        print(f"Loading tensors to {device}...", flush=True)
        t0 = time.time()
        data = torch.load(tensor_path, weights_only=True)

        self.device = torch.device(device)
        self.boards = data['boards'].to(self.device)
        self.next_pos = data['next_pos'].to(self.device)
        self.next_col = data['next_col'].to(self.device)
        self.n_next = data['n_next'].to(self.device)
        self.pol_indices = data['pol_indices'].to(self.device)
        self.pol_values = data['pol_values'].to(self.device)
        self.max_score = float(data['max_score'])

        self.augment = augment
        self.augment_factor = 8 if augment else 1
        n = self.boards.shape[0]

        # Dihedral LUTs on GPU
        self._obs_luts = torch.tensor(
            np.stack(_OBS_LUTS), dtype=torch.long, device=self.device)
        self._pol_luts = torch.tensor(
            np.stack(_POL_LUTS), dtype=torch.long, device=self.device)

        # Neighbor table for connected components (cached, not rebuilt per batch)
        neighbors = torch.full((81, 4), -1, dtype=torch.long, device=self.device)
        for r in range(9):
            for ci in range(9):
                cell = r * 9 + ci
                ni = 0
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = r + dr, ci + dc
                    if 0 <= nr < 9 and 0 <= nc < 9:
                        neighbors[cell, ni] = nr * 9 + nc
                        ni += 1
        self._neighbors = neighbors

        print(f"Loaded {n:,} states in {time.time()-t0:.1f}s "
              f"(x{self.augment_factor} = {n * self.augment_factor:,} effective)",
              flush=True)

    def __len__(self):
        return self.boards.shape[0] * self.augment_factor

    def __getitem__(self, idx):
        return idx

    @torch.no_grad()
    def collate(self, indices):
        """Build batch on GPU: observations + policy targets."""
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        base_indices = indices // self.augment_factor
        transforms = indices % self.augment_factor

        B = len(indices)
        boards = self.boards[base_indices]  # (B, 9, 9)

        # Build 18-channel observations on GPU
        obs = self._build_obs_gpu(boards, base_indices)

        # Sparse -> dense policy
        policy = torch.zeros(B, NUM_MOVES, device=self.device)
        pol_idx = self.pol_indices[base_indices]
        pol_val = self.pol_values[base_indices]
        policy.scatter_(1, pol_idx, pol_val)

        # Apply dihedral augmentation
        for t in range(1, 8 if self.augment else 1):
            mask = transforms == t
            if not mask.any():
                continue
            obs[mask] = obs[mask].reshape(-1, NUM_CHANNELS, 81
                )[:, :, self._obs_luts[t]].reshape(-1, NUM_CHANNELS, 9, 9)
            policy[mask] = policy[mask][:, self._pol_luts[t]]

        return obs, policy

    def _build_obs_core(self, boards, next_pos=None, next_col=None, n_next=None):
        """Build (B, 18, 9, 9) observations on GPU.

        If next_pos/next_col/n_next are None, channels 8-11 are left as zeros
        (used for afterstate observations).
        """
        B = boards.shape[0]
        obs = torch.zeros(B, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE,
                           device=self.device)

        # Channels 0-6: one-hot colors
        for c in range(1, 8):
            obs[:, c - 1] = (boards == c).float()
        # Channel 7: empty
        obs[:, 7] = (boards == 0).float()

        # Channels 8-10: next ball color, Channel 11: mask
        if next_pos is not None:
            batch_idx = torch.arange(B, device=self.device)
            for i in range(3):
                mask = (i < n_next.long())
                if not mask.any():
                    continue
                idx = batch_idx[mask]
                r = next_pos[mask, i, 0].long()
                c = next_pos[mask, i, 1].long()
                col = next_col[mask, i].float() / 7.0
                obs[idx, 8 + i, r, c] = col
                obs[idx, 11, r, c] = 1.0

        # Channel 12: component area heatmap
        # Min-label propagation via 4-directional shifts (no gather/scatter)
        empty = (boards == 0)  # (B, 9, 9) bool
        # Labels: 1-81 for empty cells, 0 for occupied
        labels = torch.arange(1, 82, device=self.device, dtype=torch.long
                               ).reshape(1, 9, 9).expand(B, 9, 9).clone()
        labels = labels * empty.long()

        for _ in range(20):
            old = labels
            # Propagate min label from 4 neighbors via shifts
            for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                nb = self._shift(labels, dr, dc)
                both = (labels > 0) & (nb > 0)
                labels = torch.where(both & (nb < labels), nb, labels)
            if (labels == old).all():
                break

        # Component sizes via scatter_add
        offsets = torch.arange(B, device=self.device).reshape(B, 1, 1) * 82
        flat_labels = (labels + offsets) * (labels > 0).long()
        flat = flat_labels.reshape(-1)
        max_label = int(flat.max().item()) + 1
        counts = torch.zeros(max_label, device=self.device)
        mask = flat > 0
        counts.scatter_add_(0, flat[mask],
                             torch.ones(mask.sum(), device=self.device))
        heatmap_full = torch.zeros_like(flat, dtype=torch.float32)
        heatmap_full[mask] = counts[flat[mask]] / 81.0
        obs[:, 12] = heatmap_full.reshape(B, 9, 9)

        # Channels 13-17: line potentials + max line (pure GPU, no CPU round-trip)
        self._build_line_potentials_gpu(boards, obs, B)

        return obs

    @staticmethod
    def _shift(tensor, dr, dc):
        """Shift tensor: result[r,c] = tensor[r+dr, c+dc], zero-padded."""
        S = BOARD_SIZE
        result = torch.zeros_like(tensor)
        src_r = slice(max(dr, 0), S + min(dr, 0) or S)
        dst_r = slice(max(-dr, 0), S + min(-dr, 0) or S)
        src_c = slice(max(dc, 0), S + min(dc, 0) or S)
        dst_c = slice(max(-dc, 0), S + min(-dc, 0) or S)
        result[:, dst_r, dst_c] = tensor[:, src_r, src_c]
        return result

    @staticmethod
    def _scan_direction(boards, dr, dc):
        """Count consecutive same-color cells going in direction (dr,dc).

        Returns (B, 9, 9): for each cell, how many same-color cells extend
        from it in the given direction (including self). 8 sequential
        propagation steps -- each step extends chains by 1 cell.
        """
        S = BOARD_SIZE
        occ = (boards != 0)
        run = occ.float()

        for _ in range(S - 1):
            # Get forward neighbor's board value and current run count
            nb_board = TensorDatasetGPU._shift(boards, dr, dc)
            nb_run = TensorDatasetGPU._shift(run, dr, dc)
            # Extend: if same color and neighbor has a chain, take max
            same = (boards == nb_board) & occ & (nb_board != 0)
            run = torch.where(same, torch.maximum(run, nb_run + 1), run)

        return run

    def _build_line_potentials_gpu(self, boards, obs, B):
        """Compute line potentials entirely on GPU -- no CPU round-trip.

        For each occupied cell, count same-color line length through it
        in 4 directions (H, V, D1, D2). Line length = forward + backward - 1.
        """
        max_line = torch.zeros(B, BOARD_SIZE, BOARD_SIZE, device=self.device)
        occ_f = (boards != 0).float()

        for di, (dr, dc) in enumerate([(0, 1), (1, 0), (1, 1), (1, -1)]):
            fwd = self._scan_direction(boards, dr, dc)
            bwd = self._scan_direction(boards, -dr, -dc)
            total = (fwd + bwd - 1.0) * occ_f
            obs[:, 13 + di] = total / 9.0
            max_line = torch.maximum(max_line, total)

        obs[:, 17] = max_line / 9.0

    def _build_obs_gpu(self, boards, base_indices):
        """Build observations with next_balls from stored data."""
        return self._build_obs_core(
            boards,
            next_pos=self.next_pos[base_indices],
            next_col=self.next_col[base_indices],
            n_next=self.n_next[base_indices],
        )

    def _build_obs_boards_only(self, boards):
        """Build observations from boards only (no next_balls). For afterstates."""
        return self._build_obs_core(boards)
