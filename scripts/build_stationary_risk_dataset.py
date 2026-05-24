"""Build a stationary-risk training dataset.

Phase 1 step of the floor-lift experiment (per ChatGPT 2026-05-23 plan).

For each game in each corpus, sample windows at stride S from turn t >=
min_start_turn (or 0 for crisis). At each window start:
  - Save state representation (board, next_balls)
  - Compute forward H-turn labels:
      min_empty_H, min_lec_H, empty_delta_H, lec_delta_H,
      score_rate_H, clear_rate_H
  - Save as multi-output regression labels for an auxiliary head

Phase 1 goal: prove the network can predict these labels from a frozen
backbone. Decision gate at training: AUC > 0.75 on derived binary signals
(min_lec_H < 10, min_empty_H < 25).

Output: a single .pt file with N windows ready for training.

Usage:
    python scripts/build_stationary_risk_dataset.py \\
        --selfplay-dirs data/selfplay_v13 data/selfplay_v14 \\
        --crisis-dirs data/crisis_v13 \\
        --H 100 --stride 50 \\
        --output alphatrain/data/stationary_risk_v1.pt
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def largest_empty_component(board_2d):
    """BFS over empty cells. Returns (lec_size, n_components)."""
    visited = [[False] * 9 for _ in range(9)]
    best = 0
    n_comp = 0
    for r0 in range(9):
        for c0 in range(9):
            if board_2d[r0][c0] != 0 or visited[r0][c0]:
                continue
            n_comp += 1
            sz = 0
            stack = [(r0, c0)]
            visited[r0][c0] = True
            while stack:
                r, c = stack.pop()
                sz += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < 9 and 0 <= nc < 9
                            and not visited[nr][nc]
                            and board_2d[nr][nc] == 0):
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            if sz > best:
                best = sz
    return best, n_comp


def score_from_clear(n):
    return n * (n - 4) if n >= 5 else 0


def derive_timeline(moves):
    """Per-move empties, lec, n_components, cleared_balls, score_step."""
    n = len(moves)
    empties = np.empty(n, dtype=np.int32)
    lec = np.empty(n, dtype=np.int32)
    n_comp = np.empty(n, dtype=np.int32)
    occ = np.empty(n, dtype=np.int32)
    for t in range(n):
        board = moves[t]['board']
        c = 0
        for row in board:
            for v in row:
                if v == 0:
                    c += 1
        empties[t] = c
        occ[t] = 81 - c
        l, nc = largest_empty_component(board)
        lec[t] = l
        n_comp[t] = nc
    cleared_balls = np.zeros(n, dtype=np.int32)
    score_step = np.zeros(n, dtype=np.int32)
    for t in range(n - 1):
        delta = occ[t + 1] - occ[t]
        if delta < 0:
            cleared = -delta
            cleared_balls[t] = cleared
            score_step[t] = score_from_clear(cleared)
    return empties, lec, n_comp, cleared_balls, score_step


def extract_state(move):
    """Returns (board_arr_9x9_int8, next_pos_3x2, next_col_3, n_next)."""
    board = np.asarray(move['board'], dtype=np.int8)
    nb = move['next_balls']
    n_next = min(len(nb), 3)
    next_pos = np.zeros((3, 2), dtype=np.int8)
    next_col = np.zeros(3, dtype=np.int8)
    for i in range(n_next):
        e = nb[i]
        if isinstance(e, dict):
            next_pos[i, 0] = e['row']
            next_pos[i, 1] = e['col']
            next_col[i] = e['color']
        else:
            # tuple format ([r, c], color) from some pipelines
            (r, c), col = e
            next_pos[i, 0] = r
            next_pos[i, 1] = c
            next_col[i] = col
    return board, next_pos, next_col, np.int8(n_next)


def process_game(moves, H, stride, min_start_turn, corpus_idx, seed):
    """Yield (state_tuple, label_vec, meta) for each window."""
    n = len(moves)
    if n <= min_start_turn + H + 1:
        return
    empties, lec, n_comp, cleared, score_step = derive_timeline(moves)
    for t in range(min_start_turn, n - H, stride):
        board, np_, nc_, n_n = extract_state(moves[t])
        e_window = empties[t:t + H + 1]
        l_window = lec[t:t + H + 1]
        clear_window = cleared[t:t + H]
        score_window = score_step[t:t + H]
        # Per-turn LEC sustained-fragmentation features (ChatGPT spec):
        #   lec_under_10_frac  = mean(lec_t < 10) in window
        #   lec_under_15_frac  = mean(lec_t < 15) in window
        #   lec_shortfall_15   = mean(max(0, 15 - lec_t)) in window
        # These distinguish transient drops from sustained low-LEC regimes.
        l_window_inner = lec[t:t + H]  # exclude end so length = H
        lec_under_10 = float((l_window_inner < 10).mean())
        lec_under_15 = float((l_window_inner < 15).mean())
        lec_shortfall = float(np.clip(15 - l_window_inner, 0, None).mean())
        labels = np.array([
            float(e_window.min()),                  # 0 min_empty_H
            float(l_window.min()),                  # 1 min_lec_H
            float(e_window[-1] - e_window[0]),      # 2 empty_delta_H
            float(l_window[-1] - l_window[0]),      # 3 lec_delta_H
            float(score_window.sum()) / H,          # 4 score_rate_H
            float((clear_window > 0).sum()) / H,    # 5 clear_rate_H
            lec_under_10,                           # 6 lec_under_10_frac_H
            lec_under_15,                           # 7 lec_under_15_frac_H
            lec_shortfall,                          # 8 lec_shortfall_15_H
        ], dtype=np.float32)
        meta = np.array([
            corpus_idx, seed, t,
            int(empties[t]), int(lec[t]), int(n_comp[t]),
        ], dtype=np.int32)
        yield (board, np_, nc_, n_n), labels, meta


def collect_corpus(directory, corpus_idx, H, stride, min_start_turn,
                    progress_every=200):
    """Walk a directory of game JSONs and yield (state, labels, meta)."""
    files = sorted(glob.glob(os.path.join(directory, 'game_seed*.json')))
    t0 = time.time()
    for i, path in enumerate(files):
        try:
            with open(path) as f:
                g = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        moves = g.get('moves', [])
        if len(moves) < min_start_turn + H + 2:
            continue
        seed = g.get('seed', 0)
        # Crisis JSONs have original_seed; selfplay uses seed
        if 'original_seed' in g:
            seed = g['original_seed']
        yield from process_game(moves, H, stride, min_start_turn,
                                  corpus_idx, seed)
        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            print(f"    [{directory}] {i+1}/{len(files)}  {elapsed:.0f}s",
                  flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--selfplay-dirs', nargs='*',
                   default=['data/selfplay_v13', 'data/selfplay_v14'])
    p.add_argument('--crisis-dirs', nargs='*',
                   default=['data/crisis_v13'])
    p.add_argument('--H', type=int, default=100)
    p.add_argument('--stride', type=int, default=50)
    p.add_argument('--selfplay-min-start-turn', type=int, default=100,
                   help='Skip pre-stationary fill-up phase for selfplay.')
    p.add_argument('--crisis-min-start-turn', type=int, default=0,
                   help='Crisis starts mid-game; no fill-up to skip.')
    p.add_argument('--output',
                   default='alphatrain/data/stationary_risk_v1.pt')
    args = p.parse_args()

    # Corpus index → name mapping (saved in tensor for reference)
    corpus_names = []
    sources = []
    for d in args.selfplay_dirs:
        corpus_names.append(f'sp:{os.path.basename(d)}')
        sources.append((d, args.selfplay_min_start_turn))
    for d in args.crisis_dirs:
        corpus_names.append(f'cr:{os.path.basename(d)}')
        sources.append((d, args.crisis_min_start_turn))
    print(f"Building stationary-risk dataset H={args.H}, stride={args.stride}",
          flush=True)
    for idx, name in enumerate(corpus_names):
        print(f"  [{idx}] {name}", flush=True)

    # Collect all windows
    boards_list = []
    nextpos_list = []
    nextcol_list = []
    nnext_list = []
    labels_list = []
    meta_list = []
    t0 = time.time()
    for corpus_idx, (directory, min_start) in enumerate(sources):
        print(f"\n--- Processing corpus {corpus_idx} ({directory}) ---", flush=True)
        for state, labels, meta in collect_corpus(
                directory, corpus_idx, args.H, args.stride, min_start):
            board, np_, nc_, n_n = state
            boards_list.append(board)
            nextpos_list.append(np_)
            nextcol_list.append(nc_)
            nnext_list.append(n_n)
            labels_list.append(labels)
            meta_list.append(meta)

    n = len(boards_list)
    print(f"\nTotal windows: {n:,}  ({time.time() - t0:.0f}s)", flush=True)

    if n == 0:
        print("No data — aborting.")
        return

    boards = torch.from_numpy(np.stack(boards_list))
    next_pos = torch.from_numpy(np.stack(nextpos_list))
    next_col = torch.from_numpy(np.stack(nextcol_list))
    n_next = torch.from_numpy(np.array(nnext_list, dtype=np.int8))
    labels = torch.from_numpy(np.stack(labels_list))
    meta = torch.from_numpy(np.stack(meta_list))

    label_names = ['min_empty_H', 'min_lec_H', 'empty_delta_H',
                    'lec_delta_H', 'score_rate_H', 'clear_rate_H',
                    'lec_under_10_frac_H', 'lec_under_15_frac_H',
                    'lec_shortfall_15_H']
    meta_names = ['corpus_idx', 'seed', 'turn',
                   'empties_now', 'lec_now', 'n_components_now']

    print(f"\nTensor shapes:")
    print(f"  boards: {boards.shape} {boards.dtype}")
    print(f"  next_pos: {next_pos.shape} {next_pos.dtype}")
    print(f"  labels: {labels.shape}  {label_names}")
    print(f"  meta: {meta.shape}  {meta_names}")

    # Quick label statistics
    labels_np = labels.numpy()
    print(f"\nLabel statistics (mean ± std):")
    for i, name in enumerate(label_names):
        col = labels_np[:, i]
        print(f"  {name:<18s}: mean={col.mean():>7.3f}  std={col.std():>7.3f}  "
              f"P10={np.percentile(col, 10):>7.2f}  P50={np.percentile(col, 50):>7.2f}  "
              f"P90={np.percentile(col, 90):>7.2f}")

    print(f"\nRisk event rates (derived from labels):")
    print(f"  min_empty_H < 30: {(labels_np[:, 0] < 30).mean()*100:.1f}%")
    print(f"  min_empty_H < 25: {(labels_np[:, 0] < 25).mean()*100:.1f}%")
    print(f"  min_lec_H   < 15: {(labels_np[:, 1] < 15).mean()*100:.1f}%")
    print(f"  min_lec_H   < 10: {(labels_np[:, 1] < 10).mean()*100:.1f}%")
    print(f"  lec_delta_H <=-10: {(labels_np[:, 3] <= -10).mean()*100:.1f}%")

    # Save tensor
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'boards': boards,
        'next_pos': next_pos,
        'next_col': next_col,
        'n_next': n_next,
        'labels': labels,
        'meta': meta,
        'label_names': label_names,
        'meta_names': meta_names,
        'corpus_names': corpus_names,
        'H': args.H,
        'stride': args.stride,
    }, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved {args.output} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
