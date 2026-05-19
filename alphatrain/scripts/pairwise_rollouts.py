"""Build pairwise ranking labels via shared-RNG policy-only rollouts.

For each sampled anchor:
  1. Take top-M candidate moves from the raw policy (no MCTS).
  2. For each candidate: K rollouts with shared RNG seeds (variance control).
  3. Score = mean(final_score) over K rollouts per move.
  4. Build pair: (afterstate_A, afterstate_B, label) when |score_A - score_B| > margin.

The CRITICAL choice vs Q-bootstrap: rollouts produce SCORES from actual play,
which bypass the saturated value head entirely. Score margins (hundreds of
points) are far larger than Q margins (<0.01) and don't inherit head saturation.

Usage:
    python -m alphatrain.scripts.pairwise_rollouts \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --games-dir data/selfplay_v12 data/crisis_v12 \\
        --num-anchors 5000 --top-moves 4 --rollouts-per-move 8 \\
        --horizon 200 --margin 200 \\
        --output alphatrain/data/pairwise_v12.pt
"""

import os
import time
import json
import glob
import argparse
from random import Random
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat


def sample_anchor_states(games_dirs, num_anchors, seed=42):
    """Uniformly sample anchor states from V12 JSONs."""
    files = []
    for d in games_dirs:
        files.extend(sorted(glob.glob(os.path.join(d, '*.json'))))
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
            'next_balls': [((int(nb['row']), int(nb['col'])), int(nb['color']))
                           for nb in move['next_balls']],
            'num_next': int(move['num_next']),
            'seed_origin': int(game.get('seed', 0)),
            'turn_origin': mi,
        })
    return anchors


def policy_forward(net, game, device, fp16):
    """Single policy forward over one game state. Returns flat 6561 logits."""
    obs = _build_obs_for_game(game)
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    if fp16:
        obs_t = obs_t.half()
    with torch.inference_mode():
        out = net(obs_t)
        pol_logits = out[0] if isinstance(out, tuple) else out
    return pol_logits[0].float().cpu().numpy()


def get_top_M_moves(net, anchor, M, device, fp16):
    """Top-M legal moves at the anchor state."""
    game = ColorLinesGame(seed=anchor['seed_origin'])
    game.reset(board=anchor['board'].copy(),
               next_balls=list(anchor['next_balls']))
    game.turns = anchor['turn_origin']
    pol_np = policy_forward(net, game, device, fp16)
    priors = _get_legal_priors_flat(game.board, pol_np, M)
    if not priors:
        return []
    sorted_actions = sorted(priors.items(), key=lambda x: x[1], reverse=True)
    return [a for a, _ in sorted_actions[:M]]


def run_rollout(net, anchor, first_action, rollout_seed, horizon, device, fp16):
    """Policy-only rollout. Returns (final_score, final_turns, afterstate_board,
    afterstate_next_balls, afterstate_n_next) or None on illegal first move."""
    game = ColorLinesGame(seed=rollout_seed)
    game.reset(board=anchor['board'].copy(),
               next_balls=list(anchor['next_balls']))
    game.turns = anchor['turn_origin']

    sr = first_action // 81 // 9
    sc = first_action // 81 % 9
    tr = first_action % 81 // 9
    tc = first_action % 81 % 9
    result = game.move((sr, sc), (tr, tc))
    if not result['valid']:
        return None
    if game.game_over:
        return (game.score, game.turns,
                game.board.copy(), list(game.next_balls), len(game.next_balls))

    afterstate_board = game.board.copy()
    afterstate_next_balls = list(game.next_balls)
    afterstate_n_next = len(game.next_balls)

    for _ in range(horizon):
        if game.game_over:
            break
        pol_np = policy_forward(net, game, device, fp16)
        priors = _get_legal_priors_flat(game.board, pol_np, 30)
        if not priors:
            break
        best_action = max(priors.items(), key=lambda x: x[1])[0]
        src_flat = best_action // 81
        tgt_flat = best_action % 81
        r = game.move((src_flat // 9, src_flat % 9),
                      (tgt_flat // 9, tgt_flat % 9))
        if not r['valid']:
            break

    return (game.score, game.turns,
            afterstate_board, afterstate_next_balls, afterstate_n_next)


def next_balls_to_arrays(next_balls):
    """Convert next_balls list to (npos, ncol) arrays of fixed shape (3,2) and (3,)."""
    npos = np.zeros((3, 2), dtype=np.int8)
    ncol = np.zeros(3, dtype=np.int8)
    for k, (pos, col) in enumerate(next_balls[:3]):
        npos[k, 0] = pos[0]
        npos[k, 1] = pos[1]
        ncol[k] = col
    return npos, ncol


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--games-dir', nargs='+', default=['data/selfplay_v12'])
    p.add_argument('--num-anchors', type=int, default=5000)
    p.add_argument('--top-moves', type=int, default=4)
    p.add_argument('--rollouts-per-move', type=int, default=8)
    p.add_argument('--horizon', type=int, default=200)
    p.add_argument('--margin', type=float, default=200.0)
    p.add_argument('--device', default='mps')
    p.add_argument('--sample-seed', type=int, default=42)
    p.add_argument('--rollout-base-seed', type=int, default=100000)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    t0 = time.time()
    print(f"Sampling {args.num_anchors} anchors...", flush=True)
    anchors = sample_anchor_states(args.games_dir, args.num_anchors,
                                   args.sample_seed)
    print(f"Sampled {len(anchors)} in {time.time()-t0:.0f}s", flush=True)

    print(f"\nLoading {args.model}...", flush=True)
    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')
    net, _ = load_model(args.model, device, fp16=fp16, jit_trace=False)

    print(f"\nRollouts: {args.top_moves}M x {args.rollouts_per_move}K "
          f"x {args.horizon}T, margin={args.margin}", flush=True)

    out_buffers = {k: [] for k in [
        'anchor_boards', 'anchor_next_pos', 'anchor_next_col', 'anchor_n_next',
        'after_A_boards', 'after_A_next_pos', 'after_A_next_col', 'after_A_n_next',
        'after_B_boards', 'after_B_next_pos', 'after_B_next_col', 'after_B_n_next',
        'score_diffs',
    ]}

    n_anchors_used = 0
    n_pairs_accepted = 0
    n_pairs_filtered = 0
    margin_data = []

    for ai, anchor in enumerate(anchors):
        top_moves = get_top_M_moves(net, anchor, args.top_moves, device, fp16)
        if len(top_moves) < 2:
            continue

        per_move_scores = {}
        per_move_afterstate = {}
        for move_action in top_moves:
            scores = []
            afterstate = None
            for k in range(args.rollouts_per_move):
                rollout_seed = args.rollout_base_seed + k
                res = run_rollout(net, anchor, move_action, rollout_seed,
                                  args.horizon, device, fp16)
                if res is None:
                    continue
                score, _, after_board, after_nb, after_nn = res
                scores.append(score)
                if afterstate is None:
                    afterstate = (after_board, after_nb, after_nn)
            if scores:
                per_move_scores[move_action] = scores
                per_move_afterstate[move_action] = afterstate

        if len(per_move_scores) < 2:
            continue

        moves = list(per_move_scores.keys())
        means = {m: float(np.mean(per_move_scores[m])) for m in moves}

        n_pairs_this_anchor = 0
        for i in range(len(moves)):
            for j in range(i + 1, len(moves)):
                m_i, m_j = moves[i], moves[j]
                diff = means[m_i] - means[m_j]
                margin_data.append(abs(diff))
                if abs(diff) < args.margin:
                    n_pairs_filtered += 1
                    continue
                if diff > 0:
                    a_move, b_move = m_i, m_j
                else:
                    a_move, b_move = m_j, m_i

                out_buffers['anchor_boards'].append(anchor['board'])
                an_npos, an_ncol = next_balls_to_arrays(anchor['next_balls'])
                out_buffers['anchor_next_pos'].append(an_npos)
                out_buffers['anchor_next_col'].append(an_ncol)
                out_buffers['anchor_n_next'].append(anchor['num_next'])

                a_b, a_nb, a_nn = per_move_afterstate[a_move]
                out_buffers['after_A_boards'].append(a_b)
                a_np, a_nc = next_balls_to_arrays(a_nb)
                out_buffers['after_A_next_pos'].append(a_np)
                out_buffers['after_A_next_col'].append(a_nc)
                out_buffers['after_A_n_next'].append(a_nn)

                b_b, b_nb, b_nn = per_move_afterstate[b_move]
                out_buffers['after_B_boards'].append(b_b)
                b_np, b_nc = next_balls_to_arrays(b_nb)
                out_buffers['after_B_next_pos'].append(b_np)
                out_buffers['after_B_next_col'].append(b_nc)
                out_buffers['after_B_n_next'].append(b_nn)

                out_buffers['score_diffs'].append(abs(diff))
                n_pairs_accepted += 1
                n_pairs_this_anchor += 1

        if n_pairs_this_anchor > 0:
            n_anchors_used += 1

        if (ai + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (ai + 1) / max(elapsed, 1)
            eta = (len(anchors) - ai - 1) / max(rate, 0.01)
            print(f"  [{ai+1}/{len(anchors)}] used={n_anchors_used} "
                  f"pairs={n_pairs_accepted} filtered={n_pairs_filtered} "
                  f"{rate:.2f}/s ETA {eta:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)
    print(f"Anchors used: {n_anchors_used}/{len(anchors)}", flush=True)
    print(f"Pairs: accepted={n_pairs_accepted}  filtered={n_pairs_filtered}",
          flush=True)
    if margin_data:
        m = np.array(margin_data)
        print(f"\n|score_diff| stats across all pairs:", flush=True)
        print(f"  mean={m.mean():.0f}  median={np.median(m):.0f}  "
              f"P75={np.percentile(m, 75):.0f}  P90={np.percentile(m, 90):.0f}",
              flush=True)
        print(f"  At margin={args.margin}: accepted "
              f"{(m >= args.margin).sum()}/{len(m)} = "
              f"{100*(m >= args.margin).mean():.1f}%", flush=True)

    if n_pairs_accepted == 0:
        print("\nERROR: 0 pairs accepted. Lower --margin or check rollouts.",
              flush=True)
        return

    print(f"\nStacking tensors...", flush=True)
    out = {}
    for k, v in out_buffers.items():
        if k == 'score_diffs':
            out[k] = torch.tensor(v, dtype=torch.float32)
        elif 'n_next' in k:
            out[k] = torch.tensor(v, dtype=torch.int8)
        else:
            out[k] = torch.tensor(np.stack(v), dtype=torch.int8)
    out['args'] = vars(args)
    torch.save(out, args.output)
    print(f"Saved {args.output} ({os.path.getsize(args.output)/1e6:.0f} MB)",
          flush=True)


if __name__ == '__main__':
    main()
