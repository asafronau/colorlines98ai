"""Build exact post-move afterstate pairs for ranking head training.

For each sampled position, reconstructs the game state from the saved
board + next_balls, then applies each candidate move through the real
game engine (move + clear + spawn) to get the exact post-move board.

No replay from seed needed — uses saved board states directly.

Usage:
    python -m alphatrain.scripts.build_ranking_data_exact \
        --dirs data/selfplay_v7_s1600 data/selfplay_v8_s1600 data/crisis_v2 \
        --output alphatrain/data/ranking_pairs_exact.pt \
        --max-games 5000
"""

import os
import json
import argparse
import time
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.observation import build_observation


def build_obs_from_game(game):
    """Build 18-channel observation from a live game state."""
    nb = game.next_balls
    n_next = len(nb)
    next_r = np.zeros(3, dtype=np.intp)
    next_c = np.zeros(3, dtype=np.intp)
    next_color = np.zeros(3, dtype=np.intp)
    for i in range(min(n_next, 3)):
        next_r[i] = nb[i][0][0]
        next_c[i] = nb[i][0][1]
        next_color[i] = nb[i][1]
    return build_observation(game.board, next_r, next_c, next_color, n_next)


def game_from_snapshot(board, next_balls_json, seed=0):
    """Reconstruct a game state from saved board + next_balls."""
    game = ColorLinesGame.__new__(ColorLinesGame)
    from game.rng import SimpleRng
    game.rng = SimpleRng(seed)
    game.num_colors = 7
    game.board = np.array(board, dtype=np.int8)
    game.next_balls = [
        ((nb['row'], nb['col']), nb['color'])
        for nb in next_balls_json
    ]
    game.score = 0
    game.game_over = False
    game.turns = 0
    game._cc_labels = None
    return game


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True)
    p.add_argument('--output', default='alphatrain/data/ranking_pairs_exact.pt')
    p.add_argument('--max-games', type=int, default=5000)
    p.add_argument('--max-pairs', type=int, default=800000)
    args = p.parse_args()

    all_files = []
    for d in args.dirs:
        files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
        all_files.extend(os.path.join(d, f) for f in files)
    all_files = all_files[:args.max_games]
    print(f"Processing {len(all_files)} games", flush=True)

    obs_better_list = []
    obs_worse_list = []
    margin_list = []
    game_id_list = []  # track which game each pair came from
    t0 = time.time()

    for gi, path in enumerate(all_files):
        if len(margin_list) >= args.max_pairs:
            break

        data = json.load(open(path))
        moves = data['moves']
        step = max(1, len(moves) // 100)

        for turn_idx in range(0, len(moves), step):
            if len(margin_list) >= args.max_pairs:
                break

            move_data = moves[turn_idx]
            top_moves = move_data.get('top_moves', [])
            top_scores = move_data.get('top_scores', [])
            if len(top_moves) < 2:
                continue

            board = move_data['board']
            nb = move_data['next_balls']

            # Reconstruct game state from snapshot
            # Apply MCTS-preferred move
            m0 = top_moves[0]
            game_0 = game_from_snapshot(board, nb, seed=turn_idx * 37)
            result_0 = game_0.move(
                (m0['sr'], m0['sc']), (m0['tr'], m0['tc']))

            if not result_0['valid']:
                continue

            obs_0 = build_obs_from_game(game_0)
            # Zero out next_balls channels (8-11) — they contain
            # fake values from the invented RNG, not the real game
            obs_0[8:12] = 0

            for i in range(1, len(top_moves)):
                mi = top_moves[i]
                margin = top_scores[0] - top_scores[i]

                game_i = game_from_snapshot(board, nb, seed=turn_idx * 37)
                result_i = game_i.move(
                    (mi['sr'], mi['sc']), (mi['tr'], mi['tc']))

                if not result_i['valid']:
                    continue

                obs_i = build_obs_from_game(game_i)
                obs_i[8:12] = 0  # zero fake next_balls
                obs_better_list.append(obs_0)
                obs_worse_list.append(obs_i)
                margin_list.append(margin)
                game_id_list.append(gi)

        if (gi + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"  [{gi+1}/{len(all_files)}] {len(margin_list):,} pairs "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    n = len(margin_list)
    print(f"\nExtracted {n:,} exact pairs from {gi+1} games ({elapsed:.0f}s)",
          flush=True)

    obs_better = torch.from_numpy(np.array(obs_better_list, dtype=np.float32))
    obs_worse = torch.from_numpy(np.array(obs_worse_list, dtype=np.float32))
    margins = torch.tensor(margin_list, dtype=torch.float32)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    game_ids = torch.tensor(game_id_list, dtype=torch.int32)
    torch.save({
        'obs_better': obs_better,
        'obs_worse': obs_worse,
        'margins': margins,
        'game_ids': game_ids,
        'n_games': gi + 1,
    }, args.output)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved: {args.output} ({size_mb:.0f} MB)", flush=True)
    print(f"Margin stats: mean={margins.mean():.3f} std={margins.std():.3f}",
          flush=True)


if __name__ == '__main__':
    main()
