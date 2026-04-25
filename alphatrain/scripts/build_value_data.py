"""Build training data for a separate value network.

Extracts (board, next_balls, survival_target) from game JSONs.
Target: min(turns_remaining / horizon, 1.0) — continuous [0, 1].

For capped games, turns_remaining is measured from position to cap
(conservative lower bound — the game was still healthy at the cap).

Usage:
    python -m alphatrain.scripts.build_value_data \
        --dirs data/selfplay_v7_s1600 data/selfplay_v8_s1600 data/crisis_v2 \
        --horizon 50 \
        --output alphatrain/data/value_train.pt
"""

import os
import json
import argparse
import time
import numpy as np
import torch

BOARD_SIZE = 9


def extract_game(path, horizon, max_distance):
    """Extract (board, next_balls, target) pairs from one game JSON.

    Only keeps positions within max_distance turns of game end.
    For capped games, skips entirely if max_distance < turns in game
    (all positions are too far from any meaningful end).
    """
    data = json.load(open(path))
    moves = data['moves']
    n_moves = len(moves)
    capped = data.get('capped', False)

    boards = []
    next_balls_list = []
    targets = []

    # For capped games, the last max_distance turns all have target≈1.0.
    # Include a small sample (every 10th) as healthy-board examples.
    if capped and max_distance < n_moves:
        sample_step = 10
        for turn_idx in range(0, n_moves, sample_step):
            move_data = moves[turn_idx]
            board = np.array(move_data['board'], dtype=np.int8)
            nb = move_data['next_balls']
            n_next = move_data.get('num_next', len(nb))
            nb_packed = np.zeros((3, 3), dtype=np.int8)
            for i in range(min(n_next, 3)):
                nb_packed[i, 0] = nb[i]['row']
                nb_packed[i, 1] = nb[i]['col']
                nb_packed[i, 2] = nb[i]['color']
            boards.append(board)
            next_balls_list.append(nb_packed)
            targets.append(1.0)
        return boards, next_balls_list, targets

    boards = []
    next_balls_list = []
    targets = []

    start_idx = max(0, n_moves - max_distance)
    for turn_idx in range(start_idx, n_moves):
        move_data = moves[turn_idx]
        board = np.array(move_data['board'], dtype=np.int8)

        nb = move_data['next_balls']
        n_next = move_data.get('num_next', len(nb))
        nb_packed = np.zeros((3, 3), dtype=np.int8)
        for i in range(min(n_next, 3)):
            nb_packed[i, 0] = nb[i]['row']
            nb_packed[i, 1] = nb[i]['col']
            nb_packed[i, 2] = nb[i]['color']

        turns_remaining = n_moves - turn_idx
        target = min(turns_remaining / horizon, 1.0)

        boards.append(board)
        next_balls_list.append(nb_packed)
        targets.append(target)

    return boards, next_balls_list, targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True,
                   help='Directories with game JSON files')
    p.add_argument('--horizon', type=int, default=50,
                   help='Survival horizon in turns')
    p.add_argument('--output', default='alphatrain/data/value_train.pt')
    p.add_argument('--max-games', type=int, default=0,
                   help='Max games per directory (0=all)')
    p.add_argument('--max-distance', type=int, default=200,
                   help='Only keep positions within N turns of game end')
    args = p.parse_args()

    all_files = []
    for d in args.dirs:
        files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
        if args.max_games > 0:
            files = files[:args.max_games]
        all_files.extend(os.path.join(d, f) for f in files)

    print(f"Processing {len(all_files)} games, horizon={args.horizon}",
          flush=True)

    all_boards = []
    all_next_balls = []
    all_targets = []
    t0 = time.time()

    for i, path in enumerate(all_files):
        boards, nbs, targets = extract_game(path, args.horizon, args.max_distance)
        all_boards.extend(boards)
        all_next_balls.extend(nbs)
        all_targets.extend(targets)

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(all_files)}] {len(all_targets):,} positions, "
                  f"{elapsed:.0f}s", flush=True)

    n = len(all_targets)
    elapsed = time.time() - t0
    print(f"\nExtracted {n:,} positions from {len(all_files)} games "
          f"({elapsed:.0f}s)", flush=True)

    # Convert to tensors
    boards_t = torch.tensor(np.array(all_boards), dtype=torch.int8)
    next_balls_t = torch.tensor(np.array(all_next_balls), dtype=torch.int8)
    targets_t = torch.tensor(all_targets, dtype=torch.float32)

    # Stats
    print(f"Boards: {boards_t.shape}", flush=True)
    print(f"Targets: mean={targets_t.mean():.3f} std={targets_t.std():.3f} "
          f"<0.5: {(targets_t < 0.5).sum():,} ({100*(targets_t < 0.5).float().mean():.1f}%) "
          f"=1.0: {(targets_t == 1.0).sum():,} ({100*(targets_t == 1.0).float().mean():.1f}%)",
          flush=True)

    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save({
        'boards': boards_t,
        'next_balls': next_balls_t,
        'targets': targets_t,
        'horizon': args.horizon,
        'n_games': len(all_files),
    }, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output} ({size_mb:.0f} MB)", flush=True)


if __name__ == '__main__':
    main()
