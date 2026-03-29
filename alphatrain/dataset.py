"""AlphaTrain dataset: loads expert game data with 18-channel observations.

Each sample: (observation, policy_target, value_target)
- observation: (18, 9, 9) float32 with tactical features
- policy_target: (6561,) flat joint move probability
- value_target: (num_bins,) two-hot categorical of game_score

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


def score_to_twohot(score, num_bins=64, min_val=0.0, max_val=30000.0):
    """Encode scalar score as two-hot categorical."""
    target = np.zeros(num_bins, dtype=np.float32)
    score = max(min_val, min(score, max_val))
    pos = (score - min_val) / (max_val - min_val) * (num_bins - 1)
    low = int(pos)
    high = min(low + 1, num_bins - 1)
    frac = pos - low
    target[low] = 1.0 - frac
    target[high] += frac
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


def precompute_tensors(data_path, output_path, num_value_bins=64,
                        max_score=30000.0, policy_temperature=1.0,
                        value_mode='td', gamma=1.0):
    """Convert JSON game data to precomputed tensor file.

    Stores sparse policy (indices + values) + value targets + raw boards
    for on-the-fly 18-channel observation building on GPU.

    value_mode:
        'game_score': original mode — every state gets final game_score
        'td': remaining future score — V(t) = final_score - score_at_t
              with optional discounting (gamma < 1.0)
    """
    files = sorted(glob.glob(os.path.join(data_path, 'game_*.json')))
    print(f"Loading {len(files)} game files...", flush=True)

    # Load all games and compute value targets
    all_moves = []
    all_value_scalars = []

    for fi, f in enumerate(files):
        with open(f) as fh:
            game_data = json.load(fh)
        final_score = game_data['score']
        moves = game_data['moves']

        if value_mode == 'td':
            running = _replay_game_scores(game_data)
            n_moves = len(moves)
            if gamma >= 1.0:
                for i, move in enumerate(moves):
                    all_moves.append(move)
                    all_value_scalars.append(final_score - running[i])
            else:
                # Discounted returns: compute once per game, O(N)
                deltas = [0.0] * n_moves
                for j in range(n_moves - 1):
                    deltas[j] = running[j + 1] - running[j]
                deltas[-1] = final_score - running[-1]
                returns = [0.0] * n_moves
                returns[-1] = deltas[-1]
                for j in range(n_moves - 2, -1, -1):
                    returns[j] = deltas[j] + gamma * returns[j + 1]
                for i, move in enumerate(moves):
                    all_moves.append(move)
                    all_value_scalars.append(returns[i])
        else:
            for move in moves:
                all_moves.append(move)
                all_value_scalars.append(final_score)

        if (fi + 1) % 100 == 0:
            print(f"  Loaded {fi+1}/{len(files)} games "
                  f"({len(all_moves):,} states)", flush=True)

    n = len(all_moves)
    print(f"Precomputing {n:,} states (value_mode={value_mode}, "
          f"gamma={gamma})...", flush=True)
    t0 = time.time()

    # Store boards as int8 + next_balls + policy (sparse) + value
    boards = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    next_pos = np.zeros((n, 3, 2), dtype=np.int8)
    next_col = np.zeros((n, 3), dtype=np.int8)
    n_next = np.zeros(n, dtype=np.int8)
    pol_indices = np.zeros((n, 10), dtype=np.int16)
    pol_values = np.zeros((n, 10), dtype=np.float32)
    pol_nnz = np.zeros(n, dtype=np.int8)
    val_targets = np.zeros((n, num_value_bins), dtype=np.float32)

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

        val_targets[i] = score_to_twohot(
            all_value_scalars[i], num_value_bins, 0.0, max_score)

        if (i + 1) % 200000 == 0:
            print(f"  {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%)", flush=True)

    elapsed = time.time() - t0
    print(f"Precomputed in {elapsed:.1f}s", flush=True)

    # Stats on value targets
    vals = np.array(all_value_scalars)
    print(f"Value target stats: mean={vals.mean():.0f}, std={vals.std():.0f}, "
          f"min={vals.min():.0f}, max={vals.max():.0f}", flush=True)

    data = {
        'boards': torch.from_numpy(boards),
        'next_pos': torch.from_numpy(next_pos),
        'next_col': torch.from_numpy(next_col),
        'n_next': torch.from_numpy(n_next),
        'pol_indices': torch.from_numpy(pol_indices.astype(np.int64)),
        'pol_values': torch.from_numpy(pol_values),
        'pol_nnz': torch.from_numpy(pol_nnz.astype(np.int64)),
        'val_targets': torch.from_numpy(val_targets),
        'num_value_bins': num_value_bins,
        'max_score': max_score,
        'num_channels': NUM_CHANNELS,
        'value_mode': value_mode,
        'gamma': gamma,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(data, output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Saved to {output_path} ({size_mb:.0f} MB)", flush=True)


def precompute_pairwise_tensors(data_path, output_path, num_value_bins=64,
                                 max_score=500.0, policy_temperature=1.0,
                                 gamma=0.99):
    """Build tensor file with afterstate pairs for pairwise value training.

    For each state with 2+ top_moves, generates a pair:
    - good_board: afterstate of the best move
    - bad_board: afterstate of a worse move
    - margin: score difference (from tournament evaluation)

    Also stores the original pre-move data for policy training.
    """
    from alphatrain.afterstate import compute_afterstate

    files = sorted(glob.glob(os.path.join(data_path, 'game_*.json')))
    print(f"Loading {len(files)} game files...", flush=True)

    # First pass: collect all moves with TD values
    all_moves = []
    all_td_values = []

    for fi, f in enumerate(files):
        with open(f) as fh:
            game_data = json.load(fh)
        final_score = game_data['score']
        moves = game_data['moves']
        running = _replay_game_scores(game_data)
        n_moves = len(moves)

        # Compute discounted returns
        if gamma >= 1.0:
            returns = [final_score - running[i] for i in range(n_moves)]
        else:
            deltas = [0.0] * n_moves
            for j in range(n_moves - 1):
                deltas[j] = running[j + 1] - running[j]
            deltas[-1] = final_score - running[-1]
            returns = [0.0] * n_moves
            returns[-1] = deltas[-1]
            for j in range(n_moves - 2, -1, -1):
                returns[j] = deltas[j] + gamma * returns[j + 1]

        for i, move in enumerate(moves):
            all_moves.append(move)
            all_td_values.append(returns[i])

        if (fi + 1) % 100 == 0:
            print(f"  Loaded {fi+1}/{len(files)} games "
                  f"({len(all_moves):,} states)", flush=True)

    # Second pass: build arrays
    n = len(all_moves)
    # Filter to states with 2+ top_moves for pair generation
    pair_indices = [i for i in range(n)
                    if len(all_moves[i].get('top_moves', [])) >= 2]
    n_pairs = len(pair_indices)
    print(f"Building {n:,} states + {n_pairs:,} afterstate pairs "
          f"(gamma={gamma})...", flush=True)
    t0 = time.time()

    # Pre-move data (same as before)
    boards = np.zeros((n, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    next_pos = np.zeros((n, 3, 2), dtype=np.int8)
    next_col = np.zeros((n, 3), dtype=np.int8)
    n_next = np.zeros(n, dtype=np.int8)
    pol_indices = np.zeros((n, 10), dtype=np.int16)
    pol_values = np.zeros((n, 10), dtype=np.float32)
    pol_nnz = np.zeros(n, dtype=np.int8)
    val_targets = np.zeros((n, num_value_bins), dtype=np.float32)

    # Afterstate pair data
    good_boards = np.zeros((n_pairs, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    bad_boards = np.zeros((n_pairs, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    margins = np.zeros(n_pairs, dtype=np.float32)
    pair_base_idx = np.zeros(n_pairs, dtype=np.int64)  # index into pre-move arrays

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

        val_targets[i] = score_to_twohot(
            all_td_values[i], num_value_bins, 0.0, max_score)

        if (i + 1) % 200000 == 0:
            print(f"  Pre-move: {i+1:,}/{n:,} ({(i+1)/n*100:.0f}%)",
                  flush=True)

    # Build afterstate pairs
    for pi, idx in enumerate(pair_indices):
        move = all_moves[idx]
        board = boards[idx].astype(np.int64)  # numba needs int64
        top_moves = move['top_moves']
        top_scores = move['top_scores']

        # Best move
        best = top_moves[0]
        good_after, _ = compute_afterstate(
            board, best['sr'], best['sc'], best['tr'], best['tc'])
        good_boards[pi] = good_after.astype(np.int8)

        # Pick a worse move (last in top list, or second if only 2)
        worse_idx = min(len(top_moves) - 1, max(1, len(top_moves) // 2))
        worse = top_moves[worse_idx]
        bad_after, _ = compute_afterstate(
            board, worse['sr'], worse['sc'], worse['tr'], worse['tc'])
        bad_boards[pi] = bad_after.astype(np.int8)

        # Margin from tournament scores (normalized)
        margins[pi] = top_scores[0] - top_scores[worse_idx]
        pair_base_idx[pi] = idx

        if (pi + 1) % 200000 == 0:
            print(f"  Pairs: {pi+1:,}/{n_pairs:,} ({(pi+1)/n_pairs*100:.0f}%)",
                  flush=True)

    elapsed = time.time() - t0
    print(f"Built in {elapsed:.1f}s", flush=True)

    # Margin stats
    print(f"Margin stats: mean={margins.mean():.2f}, std={margins.std():.2f}, "
          f"max={margins.max():.2f}", flush=True)

    data = {
        # Pre-move data (for policy training)
        'boards': torch.from_numpy(boards),
        'next_pos': torch.from_numpy(next_pos),
        'next_col': torch.from_numpy(next_col),
        'n_next': torch.from_numpy(n_next),
        'pol_indices': torch.from_numpy(pol_indices.astype(np.int64)),
        'pol_values': torch.from_numpy(pol_values),
        'pol_nnz': torch.from_numpy(pol_nnz.astype(np.int64)),
        'val_targets': torch.from_numpy(val_targets),
        # Afterstate pairs (for value ranking)
        'good_boards': torch.from_numpy(good_boards),
        'bad_boards': torch.from_numpy(bad_boards),
        'margins': torch.from_numpy(margins),
        'pair_base_idx': torch.from_numpy(pair_base_idx),
        # Metadata
        'num_value_bins': num_value_bins,
        'max_score': max_score,
        'num_channels': NUM_CHANNELS,
        'value_mode': 'pairwise',
        'gamma': gamma,
        'n_pairs': n_pairs,
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
    applies augmentation, and reconstructs sparse policy targets — all on GPU.
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
        self.val_targets = data['val_targets'].to(self.device)
        self.num_value_bins = int(data['num_value_bins'])
        self.max_score = float(data['max_score'])

        # Pairwise afterstate data (optional)
        self.has_pairs = 'good_boards' in data
        if self.has_pairs:
            self.good_boards = data['good_boards'].to(self.device)
            self.bad_boards = data['bad_boards'].to(self.device)
            self.margins = data['margins'].to(self.device)
            self.pair_base_idx = data['pair_base_idx'].to(self.device)
            self.n_pairs = int(data['n_pairs'])

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

        pairs_str = f", {self.n_pairs:,} pairs" if self.has_pairs else ""
        print(f"Loaded {n:,} states{pairs_str} in {time.time()-t0:.1f}s "
              f"(x{self.augment_factor} = {n * self.augment_factor:,} effective)",
              flush=True)

    def __len__(self):
        return self.boards.shape[0] * self.augment_factor

    def __getitem__(self, idx):
        return idx

    def collate(self, indices):
        """Build batch on GPU: observations + policy + value targets."""
        indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        base_indices = indices // self.augment_factor
        transforms = indices % self.augment_factor

        B = len(indices)
        boards = self.boards[base_indices]  # (B, 9, 9)
        val = self.val_targets[base_indices]

        # Build 18-channel observations on GPU
        obs = self._build_obs_gpu(boards, base_indices)

        # Sparse → dense policy
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

        return obs, policy, val

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
        heatmap = torch.zeros(B * 81, device=self.device)
        flat_mask = flat[mask]
        heatmap_full = torch.zeros_like(flat, dtype=torch.float32)
        heatmap_full[mask] = counts[flat_mask] / 81.0
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
        propagation steps — each step extends chains by 1 cell.
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
        """Compute line potentials entirely on GPU — no CPU round-trip.

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

    def collate_pairwise(self, indices):
        """Build batch with afterstate pairs for pairwise value training.

        Returns (obs, policy, val, good_obs, bad_obs, margin).
        """
        # Standard collate for policy training
        obs, policy, val = self.collate(indices)

        # Sample afterstate pairs (randomly from the pair pool)
        B = len(indices)
        pair_idx = torch.randint(0, self.n_pairs, (B,), device=self.device)
        base = self.pair_base_idx[pair_idx]

        # Fuse good+bad into single obs build with parent's next_balls
        # (next_balls are known information — visible in UI before the move)
        both_boards = torch.cat([self.good_boards[pair_idx],
                                  self.bad_boards[pair_idx]], dim=0)
        both_next_pos = self.next_pos[base].repeat(2, 1, 1)
        both_next_col = self.next_col[base].repeat(2, 1)
        both_n_next = self.n_next[base].repeat(2)
        both_obs = self._build_obs_core(both_boards,
                                         next_pos=both_next_pos,
                                         next_col=both_next_col,
                                         n_next=both_n_next)
        good_obs, bad_obs = both_obs.chunk(2, dim=0)
        margin = self.margins[pair_idx]

        return obs, policy, val, good_obs, bad_obs, margin

    @property
    def num_bins(self):
        return self.num_value_bins
