"""Mine board features from death trajectories.

Analyzes games that died naturally. For each game, extracts board features
at multiple offsets before death AND at healthy mid-game positions.
Produces a clear picture of how board health degrades toward death.

Usage:
    python -m alphatrain.scripts.mine_death_features \
        --dirs data/selfplay_v7_s1600 data/selfplay_v8_s1600 \
               data/crisis_v1 data/crisis_v2 \
        --max-games 5000
"""

import os
import json
import argparse
import numpy as np
from numba import njit
from game.config import BOARD_SIZE


@njit(cache=True)
def board_features(board):
    """Compute comprehensive board health features."""
    # === Empty count + connected components ===
    empty = 0
    labels = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    queue_r = np.empty(81, dtype=np.int32)
    queue_c = np.empty(81, dtype=np.int32)
    current = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                empty += 1
                if labels[r, c] == 0:
                    current += 1
                    labels[r, c] = current
                    queue_r[0] = r
                    queue_c[0] = c
                    head, tail = 0, 1
                    while head < tail:
                        cr, cc = queue_r[head], queue_c[head]
                        head += 1
                        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                                if board[nr, nc] == 0 and labels[nr, nc] == 0:
                                    labels[nr, nc] = current
                                    queue_r[tail] = nr
                                    queue_c[tail] = nc
                                    tail += 1

    n_components = current
    comp_sizes = np.zeros(current + 1, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if labels[r, c] > 0:
                comp_sizes[labels[r, c]] += 1

    largest = 0
    tiny_count = 0  # components with <= 3 cells
    for i in range(1, current + 1):
        if comp_sizes[i] > largest:
            largest = comp_sizes[i]
        if comp_sizes[i] <= 3:
            tiny_count += 1

    # === Mobility: legal moves proxy ===
    # Count (ball, reachable-empty) pairs using component labels
    # A ball can reach any empty cell in an adjacent component
    mobility = 0
    reachable_per_ball = np.zeros(81, dtype=np.int32)
    ball_idx = 0
    low_mobility_balls = 0  # balls with < 5 reachable targets

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] > 0:
                # Find reachable empty cells via adjacent components
                reachable = 0
                seen_comps = np.zeros(current + 1, dtype=np.int8)
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        lbl = labels[nr, nc]
                        if lbl > 0 and seen_comps[lbl] == 0:
                            seen_comps[lbl] = 1
                            reachable += comp_sizes[lbl]
                reachable_per_ball[ball_idx] = reachable
                mobility += reachable
                if reachable < 5:
                    low_mobility_balls += 1
                ball_idx += 1

    n_balls = ball_idx
    avg_reach = mobility / max(n_balls, 1)
    min_reach = 999
    for i in range(n_balls):
        if reachable_per_ball[i] < min_reach:
            min_reach = reachable_per_ball[i]
    if n_balls == 0:
        min_reach = 0

    # === Color analysis ===
    color_counts = np.zeros(7, dtype=np.int32)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            v = board[r, c]
            if v > 0:
                color_counts[v - 1] += 1

    n_colors_present = 0
    for i in range(7):
        if color_counts[i] > 0:
            n_colors_present += 1

    # === Same-color adjacency / line potential ===
    same_color_adj = 0
    diff_color_adj = 0
    line3_count = 0  # 3-in-a-row (potential lines)
    line4_count = 0  # 4-in-a-row (one away from clearing)

    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            color = board[r, c]
            if color == 0:
                continue
            # Adjacency
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    nc_val = board[nr, nc]
                    if nc_val > 0:
                        if nc_val == color:
                            same_color_adj += 1
                        else:
                            diff_color_adj += 1
            # Line potential (only count from start of line)
            for dr, dc in dirs:
                # Check we're at the start (no same-color behind)
                pr, pc = r - dr, c - dc
                if 0 <= pr < BOARD_SIZE and 0 <= pc < BOARD_SIZE:
                    if board[pr, pc] == color:
                        continue  # not the start
                length = 1
                cr, cc = r + dr, c + dc
                while 0 <= cr < BOARD_SIZE and 0 <= cc < BOARD_SIZE \
                        and board[cr, cc] == color:
                    length += 1
                    cr += dr
                    cc += dc
                if length == 3:
                    line3_count += 1
                elif length == 4:
                    line4_count += 1

    # === Central density (3x3 center) ===
    center_balls = 0
    center_colors = np.zeros(7, dtype=np.int8)
    for r in range(3, 6):
        for c in range(3, 6):
            v = board[r, c]
            if v > 0:
                center_balls += 1
                center_colors[v - 1] = 1
    center_color_count = 0
    for i in range(7):
        if center_colors[i] > 0:
            center_color_count += 1

    return (empty, n_components, largest, tiny_count,
            mobility, avg_reach, min_reach, low_mobility_balls,
            n_balls, n_colors_present,
            same_color_adj, diff_color_adj,
            line3_count, line4_count,
            center_balls, center_color_count)


FEATURE_NAMES = [
    'empty', 'components', 'largest', 'tiny_comp',
    'mobility', 'avg_reach', 'min_reach', 'low_mob_balls',
    'balls', 'colors_present',
    'same_adj', 'diff_adj',
    'line3', 'line4',
    'center_balls', 'center_colors',
]

# Derived features added in Python
DERIVED_NAMES = ['ratio', 'frag_score']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dirs', nargs='+', required=True)
    p.add_argument('--max-games', type=int, default=5000)
    args = p.parse_args()

    all_files = []
    for d in args.dirs:
        files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
        all_files.extend(os.path.join(d, f) for f in files)

    # Only naturally-died games with enough turns
    died_games = []
    for path in all_files:
        data = json.load(open(path))
        if not data.get('capped', False) and len(data['moves']) > 10:
            died_games.append(path)
        if len(died_games) >= args.max_games:
            break

    print(f"Analyzing {len(died_games)} naturally-died games", flush=True)

    # Offsets to sample (turns before death)
    offsets = [10, 15, 20, 25, 30, 40, 50, 75]

    feat_by_offset = {o: [] for o in offsets}
    healthy_feats = []  # turn 500 positions from long games
    game_lengths = []

    for gi, path in enumerate(died_games):
        data = json.load(open(path))
        moves = data['moves']
        n = len(moves)
        game_lengths.append(n)

        for offset in offsets:
            idx = n - 1 - offset
            if idx < 0:
                continue
            board = np.array(moves[idx]['board'], dtype=np.int8)
            raw = board_features(board)
            feat = dict(zip(FEATURE_NAMES, raw))
            feat['ratio'] = feat['largest'] / max(feat['empty'], 1)
            feat['frag_score'] = (feat['empty'] - feat['largest']) * \
                feat['components']
            feat_by_offset[offset].append(feat)

        # Healthy: sample from the first third of the game
        healthy_idx = min(n // 3, 500)
        if healthy_idx > 50:
            board = np.array(moves[healthy_idx]['board'], dtype=np.int8)
            raw = board_features(board)
            feat = dict(zip(FEATURE_NAMES, raw))
            feat['ratio'] = feat['largest'] / max(feat['empty'], 1)
            feat['frag_score'] = (feat['empty'] - feat['largest']) * \
                feat['components']
            healthy_feats.append(feat)

        if (gi + 1) % 500 == 0:
            print(f"  [{gi+1}/{len(died_games)}]", flush=True)

    # Print results
    all_names = FEATURE_NAMES + DERIVED_NAMES

    print(f"\nGame length: mean={np.mean(game_lengths):.0f} "
          f"median={np.median(game_lengths):.0f} "
          f"P10={np.percentile(game_lengths, 10):.0f} "
          f"P90={np.percentile(game_lengths, 90):.0f}", flush=True)

    # Key features trajectory
    key_feats = ['empty', 'components', 'largest', 'ratio', 'tiny_comp',
                 'avg_reach', 'low_mob_balls', 'line3', 'line4',
                 'center_balls', 'center_colors', 'frag_score']

    print(f"\n{'='*90}", flush=True)
    print(f"DEATH TRAJECTORY ({len(died_games)} games)", flush=True)
    print(f"{'='*90}", flush=True)

    header = f"{'T-offset':>8}"
    for f in key_feats:
        header += f" {f:>10}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # Print healthy baseline first
    if healthy_feats:
        row = f"{'HEALTHY':>8}"
        for f in key_feats:
            vals = [feat[f] for feat in healthy_feats]
            row += f" {np.mean(vals):>10.1f}"
        print(row, flush=True)
        print("-" * len(header), flush=True)

    for offset in sorted(feat_by_offset.keys(), reverse=True):
        feats = feat_by_offset[offset]
        if not feats:
            continue
        row = f"{offset:>8}"
        for f in key_feats:
            vals = [feat[f] for feat in feats]
            row += f" {np.mean(vals):>10.1f}"
        print(row, flush=True)

    # Distributions at key offsets
    for offset in [75, 50, 25, 10]:
        feats = feat_by_offset.get(offset, [])
        if not feats:
            continue
        print(f"\n  T-{offset} distributions (n={len(feats)}):", flush=True)
        for f in ['empty', 'components', 'ratio', 'avg_reach',
                   'low_mob_balls', 'frag_score']:
            vals = [feat[f] for feat in feats]
            print(f"    {f:>15}: P10={np.percentile(vals,10):>6.1f} "
                  f"P25={np.percentile(vals,25):>6.1f} "
                  f"P50={np.percentile(vals,50):>6.1f} "
                  f"P75={np.percentile(vals,75):>6.1f} "
                  f"P90={np.percentile(vals,90):>6.1f}", flush=True)

    # Healthy distributions for comparison
    if healthy_feats:
        print(f"\n  HEALTHY distributions (n={len(healthy_feats)}):",
              flush=True)
        for f in ['empty', 'components', 'ratio', 'avg_reach',
                   'low_mob_balls', 'frag_score']:
            vals = [feat[f] for feat in healthy_feats]
            print(f"    {f:>15}: P10={np.percentile(vals,10):>6.1f} "
                  f"P25={np.percentile(vals,25):>6.1f} "
                  f"P50={np.percentile(vals,50):>6.1f} "
                  f"P75={np.percentile(vals,75):>6.1f} "
                  f"P90={np.percentile(vals,90):>6.1f}", flush=True)


if __name__ == '__main__':
    main()
