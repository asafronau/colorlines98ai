"""Image-3 rollout judge.

Compare two candidate moves from the image-3 state by playing K policy-only
continuations from each post-state with common RNG. Reports per-branch score
gained over horizon, turns survived, and bigger-clear frequency.

Phase 1 of the protocol in docs/image3_value_head_diagnosis.md (after ChatGPT
review 2026-05-22): don't retrain anything until we know whether the model is
actually wrong on this state.

Usage:
    python scripts/image3_rollout_judge.py
    python scripts/image3_rollout_judge.py --K 512 --H 500 \\
        --model alphatrain/data/pillar3b_epoch_20.pt
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.rng import SimpleRng
from game.config import BOARD_SIZE
from alphatrain.evaluate import load_model, make_policy_player

# Image 3 state (Score 35, Turn 27). Same as scripts/debug_image3*.py.
BOARD = np.zeros((9, 9), dtype=np.int8)
for (r, c), col in {
    (0, 0): 4, (0, 1): 3, (0, 7): 7,
    (1, 0): 6, (1, 1): 7, (1, 2): 3,
    (2, 1): 5,
    (3, 1): 2, (3, 2): 2, (3, 3): 2, (3, 4): 2, (3, 7): 1,
    (4, 1): 5, (4, 2): 7, (4, 5): 6, (4, 6): 1,
    (5, 0): 3, (5, 6): 1, (5, 7): 3,
    (6, 0): 7, (6, 3): 1, (6, 6): 2,
    (7, 5): 6, (7, 6): 6, (7, 7): 6,
    (8, 0): 6, (8, 6): 3, (8, 8): 4,
}.items():
    BOARD[r, c] = col
NEXT_BALLS = [((3, 5), 7), ((2, 3), 7), ((6, 5), 2)]
START_SCORE = 35
START_TURN = 27

CANDIDATES = {
    'A_green_clear':  ((6, 6), (3, 5)),   # the +5pt clear
    'B_pink_setup':   ((4, 5), (7, 4)),   # pillar3a model's choice
    'B2_pink_setup2': ((4, 5), (7, 8)),   # pillar3b model's choice
}


def build_initial(seed_for_rng):
    g = ColorLinesGame()
    g.reset(board=BOARD.copy(), next_balls=list(NEXT_BALLS))
    g.score = START_SCORE
    g.turns = START_TURN
    g.rng = SimpleRng(seed_for_rng)
    return g


def rollout(game, player, max_turns, track_clears=True):
    """Play policy-only from `game` for up to max_turns. Returns stats."""
    start_score = game.score
    start_turn = game.turns
    biggest_clear = 0
    clears_count = 0
    turns_played = 0
    while not game.game_over and turns_played < max_turns:
        mv = player(game)
        if mv is None:
            break
        res = game.move(mv[0], mv[1])
        if not res['valid']:
            break
        if track_clears and res.get('cleared', 0) >= 5:
            clears_count += 1
            biggest_clear = max(biggest_clear, res['cleared'])
        turns_played += 1
    return {
        'score_gained': game.score - start_score,
        'turns_played': turns_played,
        'game_over': game.game_over,
        'final_score': game.score,
        'biggest_clear': biggest_clear,
        'clears_count': clears_count,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',
                   default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--K', type=int, default=256,
                   help='Replicates per candidate move.')
    p.add_argument('--H', type=int, default=500,
                   help='Horizon (max turns per rollout).')
    args = p.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available()
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    net, _ = load_model(args.model, device, fp16=False)
    player = make_policy_player(net, device)

    candidates = list(CANDIDATES.items())

    # Warmup
    print("Warmup ...", flush=True)
    g = build_initial(0)
    g.move(*CANDIDATES['A_green_clear'])
    rollout(g, player, max_turns=5)

    results = {name: [] for name, _ in candidates}
    t0 = time.time()
    last_print = t0
    total_units = args.K * len(candidates)
    done = 0
    for seed in range(args.K):
        for name, move in candidates:
            g = build_initial(seed)
            res = g.move(*move)
            assert res['valid'], f"Move {name}={move} invalid on init state"
            res_stats = rollout(g, player, max_turns=args.H)
            # Tag whether THIS move itself was a clear (move A clears, B doesn't)
            res_stats['initial_clear'] = res.get('cleared', 0)
            res_stats['initial_pts'] = res.get('score', 0)
            results[name].append(res_stats)
            done += 1
        if time.time() - last_print > 20:
            elapsed = time.time() - t0
            eta = elapsed / done * (total_units - done)
            print(f"  {done}/{total_units}  elapsed={elapsed:.0f}s "
                  f"eta={eta:.0f}s", flush=True)
            last_print = time.time()

    print(f"\nDone in {time.time() - t0:.0f}s. K={args.K}, H={args.H}")
    print(f"\n{'='*92}")
    print(f"  ROLLOUT JUDGE — model={os.path.basename(args.model)}")
    print(f"{'='*92}")
    print(f"  Metric per branch: distribution over {args.K} common-RNG seeds.")
    print(f"  Branches start from image-3 state (Score={START_SCORE}, "
          f"Turn={START_TURN}), apply the candidate move, then play "
          f"policy-only for up to {args.H} turns.\n")

    # Per-branch summary
    header = f"  {'branch':<20s} | {'init clr+pts':<13s} | " \
             f"{'mean Δscore':<11s} | {'P50':<5s} | {'P25':<5s} | " \
             f"{'mean turns':<10s} | {'die%':<5s} | {'bigclr ≥6%':<10s} | " \
             f"{'mean #clears':<12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary = {}
    for name, _ in candidates:
        r = results[name]
        scores = np.array([x['score_gained'] for x in r])
        turns = np.array([x['turns_played'] for x in r])
        died = np.array([x['game_over'] for x in r])
        big = np.array([x['biggest_clear'] >= 6 for x in r])
        clears = np.array([x['clears_count'] for x in r])
        # The "initial" move stats are the same for all seeds (deterministic move)
        ic = r[0]['initial_clear']
        ip = r[0]['initial_pts']
        summary[name] = {
            'mean_score': float(scores.mean()),
            'p50_score': float(np.percentile(scores, 50)),
            'p25_score': float(np.percentile(scores, 25)),
            'mean_turns': float(turns.mean()),
            'die_rate': float(died.mean()),
            'big_rate': float(big.mean()),
            'mean_clears': float(clears.mean()),
        }
        print(f"  {name:<20s} | {ic:>2d}clr +{ip:>3d}pt    | "
              f"{scores.mean():>8.1f}    | "
              f"{np.percentile(scores, 50):>5.0f} | "
              f"{np.percentile(scores, 25):>5.0f} | "
              f"{turns.mean():>8.1f}   | "
              f"{died.mean()*100:>4.1f}% | "
              f"{big.mean()*100:>7.1f}%   | "
              f"{clears.mean():>10.2f}")

    # Pairwise paired-seed comparison
    print(f"\n  PAIRED COMPARISON (same-seed rollouts):")
    base_name = 'A_green_clear'
    for name, _ in candidates:
        if name == base_name:
            continue
        a_scores = np.array([x['score_gained'] for x in results[base_name]])
        b_scores = np.array([x['score_gained'] for x in results[name]])
        diff = a_scores - b_scores  # positive ⇒ green-clear wins
        wins = (diff > 0).sum()
        losses = (diff < 0).sum()
        ties = (diff == 0).sum()
        a_died = np.array([x['game_over'] for x in results[base_name]])
        b_died = np.array([x['game_over'] for x in results[name]])
        print(f"\n    {base_name} vs {name}")
        print(f"      Δscore (A−B): mean={diff.mean():+.1f}  "
              f"median={np.median(diff):+.0f}  std={diff.std():.0f}")
        print(f"      A wins {wins}/{args.K} ({100*wins/args.K:.1f}%), "
              f"B wins {losses}/{args.K} ({100*losses/args.K:.1f}%), "
              f"ties {ties}")
        print(f"      Die rates: A={a_died.mean()*100:.1f}%, "
              f"B={b_died.mean()*100:.1f}%")


if __name__ == '__main__':
    main()
