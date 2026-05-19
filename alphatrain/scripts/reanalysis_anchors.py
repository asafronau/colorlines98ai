"""Re-analyze sampled V12 anchor states at stronger sim count to produce
Q labels for pairwise-ranking value-head training.

The key constraint (per ChatGPT/Reanalyse): the teacher MCTS sim count must
be MEANINGFULLY stronger than what produced the V12 corpus (400 sims), so
its Q estimates carry information the current head doesn't already encode.
Default 800 sims = 2× generation budget.

For each sampled anchor state, run one MCTS search and save root top-K
child stats: (action, visits N, Q = value_sum/N, prior). These become the
supervision for pairwise ranking in the next step.

Usage:
    python -m alphatrain.scripts.reanalysis_anchors \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --value-head-path alphatrain/data/value_head_v12.pt \\
        --games-dir data/selfplay_v12 data/crisis_v12 \\
        --num-anchors 5000 --sims 800 --top-k 5 \\
        --output alphatrain/data/reanalysis_v12_800sim.pt
"""

import os
import json
import glob
import time
import argparse
from random import Random
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS


def sample_anchor_states(games_dirs, num_anchors, seed=42):
    """Uniformly sample anchor states from V12 JSONs.

    Each anchor = (board, next_balls, turn_idx) drawn from a random move
    in a random game file. Returns list of dicts.
    """
    files = []
    for d in games_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, '*.json'))))
    print(f"Found {len(files)} game files across {len(games_dirs)} dirs",
          flush=True)
    rng = Random(seed)
    anchors = []
    attempts = 0
    while len(anchors) < num_anchors and attempts < num_anchors * 5:
        attempts += 1
        f = rng.choice(files)
        try:
            with open(f) as fp:
                game = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        moves = game.get('moves', [])
        if not moves:
            continue
        mi = rng.randint(0, len(moves) - 1)
        move = moves[mi]
        anchors.append({
            'board': np.array(move['board'], dtype=np.int8),
            'next_balls': [(int(nb['row']), int(nb['col']), int(nb['color']))
                           for nb in move['next_balls']],
            'num_next': int(move['num_next']),
            'seed_origin': int(game.get('seed', 0)),
            'turn_origin': mi,
        })
    return anchors


def run_reanalysis(anchors, model_path, head_path, sims, top_k, device, q_weight):
    """For each anchor, run MCTS and capture top-K root child stats."""
    print(f"Loading {model_path}...", flush=True)
    # jit_trace=False since we need backbone access for value head
    net, max_score = load_model(model_path, torch.device(device),
                                fp16=(device != 'cpu'), jit_trace=False)

    print(f"Building MCTS with {sims} sims, q_weight={q_weight}, head={head_path}",
          flush=True)
    mcts = MCTS(net, torch.device(device), max_score=max_score,
                num_simulations=sims, batch_size=8, top_k=30, c_puct=2.5,
                value_head_path=head_path, q_weight=q_weight,
                early_stop=False)  # need full sims for clean Q estimates

    boards_l = []
    next_pos_l = []
    next_col_l = []
    n_next_l = []
    top_k_actions_l = []
    top_k_visits_l = []
    top_k_qs_l = []
    top_k_priors_l = []

    t0 = time.time()
    skipped = 0
    for i, anchor in enumerate(anchors):
        game = ColorLinesGame(seed=anchor['seed_origin'])
        # Recreate state. The .reset() signature varies; mimic crisis_mining.py.
        next_balls = [((r, c), col) for r, c, col in anchor['next_balls']]
        game.reset(board=anchor['board'].copy(), next_balls=next_balls)
        game.turns = anchor['turn_origin']

        if game.game_over:
            skipped += 1
            continue

        action = mcts.search(game, temperature=0.0, return_policy=False)
        if action is None:
            skipped += 1
            continue

        root = mcts._last_root
        if root is None or not root.children:
            skipped += 1
            continue

        # Top-K children by visit count (most informative ranking)
        sorted_children = sorted(root.children.items(),
                                 key=lambda x: x[1].visit_count, reverse=True)
        children = sorted_children[:top_k]

        # Pad to top_k if fewer children
        actions = np.zeros(top_k, dtype=np.int32)
        visits = np.zeros(top_k, dtype=np.int32)
        qs = np.zeros(top_k, dtype=np.float32)
        priors = np.zeros(top_k, dtype=np.float32)
        for j, (act, ch) in enumerate(children):
            actions[j] = int(act)
            visits[j] = ch.visit_count
            qs[j] = (ch.value_sum / ch.visit_count) if ch.visit_count > 0 else 0.0
            priors[j] = ch.prior

        # Store anchor state
        boards_l.append(anchor['board'])
        npos = np.zeros((3, 2), dtype=np.int8)
        ncol = np.zeros(3, dtype=np.int8)
        for k, (r, c, col) in enumerate(anchor['next_balls'][:3]):
            npos[k, 0] = r
            npos[k, 1] = c
            ncol[k] = col
        next_pos_l.append(npos)
        next_col_l.append(ncol)
        n_next_l.append(anchor['num_next'])

        top_k_actions_l.append(actions)
        top_k_visits_l.append(visits)
        top_k_qs_l.append(qs)
        top_k_priors_l.append(priors)

        if (i + 1) % 50 == 0 or i == len(anchors) - 1:
            elapsed = time.time() - t0
            done = i + 1 - skipped
            rate = done / max(elapsed, 1)
            eta = (len(anchors) - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{len(anchors)}] done={done} skip={skipped} "
                  f"{rate:.1f} anchors/s ETA {eta:.0f}s", flush=True)

    return {
        'boards': np.stack(boards_l),
        'next_pos': np.stack(next_pos_l),
        'next_col': np.stack(next_col_l),
        'n_next': np.array(n_next_l, dtype=np.int8),
        'top_k_actions': np.stack(top_k_actions_l),
        'top_k_visits': np.stack(top_k_visits_l),
        'top_k_qs': np.stack(top_k_qs_l),
        'top_k_priors': np.stack(top_k_priors_l),
        'n_anchors': len(boards_l),
        'n_skipped': skipped,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--value-head-path', required=True)
    p.add_argument('--games-dir', nargs='+', default=['data/selfplay_v12'])
    p.add_argument('--num-anchors', type=int, default=5000)
    p.add_argument('--sims', type=int, default=800)
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--q-weight', type=float, default=2.0)
    p.add_argument('--device', default='mps')
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    t0 = time.time()
    print(f"Sampling {args.num_anchors} anchors from {args.games_dir}...",
          flush=True)
    anchors = sample_anchor_states(args.games_dir, args.num_anchors,
                                   args.sample_seed)
    print(f"Sampled {len(anchors)} anchors in {time.time()-t0:.0f}s",
          flush=True)

    print(f"\nRunning MCTS reanalysis at {args.sims} sims, top-{args.top_k}, "
          f"q={args.q_weight}...", flush=True)
    result = run_reanalysis(anchors, args.model, args.value_head_path,
                            args.sims, args.top_k, args.device, args.q_weight)
    print(f"\n{result['n_anchors']} anchors processed "
          f"({result['n_skipped']} skipped)",
          flush=True)

    out = {k: torch.tensor(v) if isinstance(v, np.ndarray) else v
           for k, v in result.items()}
    out['args'] = vars(args)
    out['sims'] = args.sims
    out['top_k'] = args.top_k

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    torch.save(out, args.output)
    print(f"\nSaved to {args.output} "
          f"({os.path.getsize(args.output)/1e6:.0f} MB)", flush=True)
    print(f"Total wall: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
