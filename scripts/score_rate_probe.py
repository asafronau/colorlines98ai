"""Score-rate probe: is pillar3b's ~2 pts/turn real, or a crippled-harness artifact?

Plays pillar3b argmax from fresh seeds for a long horizon, printing cumulative
score at turn checkpoints. If the rate is ~constant ~2/turn out to thousands of
turns, then the place-anywhere table (P50~400 over 200 turns) is correct and the
17k mean is survival-driven (score = rate x turns_survived), not a bug.

Usage:
    PYTHONPATH=. python scripts/score_rate_probe.py --seeds 1 2 3 4 --max-turns 4000
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from scripts.mine_stationary_counterfactuals import policy_argmax

CHECKPOINTS = (100, 200, 500, 1000, 2000, 4000, 8000)


def play_trajectory(net, device, dtype, seed, max_turns):
    g = ColorLinesGame(seed=seed)
    g.reset()
    marks = {}
    t0 = time.time()
    while not g.game_over and g.turns < max_turns:
        mv = policy_argmax(net, device, dtype, g)
        if mv is None:
            break
        r = g.move(*mv)
        if not r['valid']:
            break
        if g.turns in CHECKPOINTS:
            marks[g.turns] = int(g.score)
            print(f"  seed {seed}: turn {g.turns:>5} | score {g.score:>6} | "
                  f"rate {g.score / max(1, g.turns):.3f}/turn | "
                  f"{time.time() - t0:.0f}s", flush=True)
    return seed, int(g.score), int(g.turns), bool(g.game_over), marks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/pillar3b_epoch_20.pt')
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--max-turns', type=int, default=4000)
    args = p.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model(args.model, device, fp16=(device.type != 'cpu'))
    dtype = next(net.parameters()).dtype
    print(f"device={device} dtype={dtype}; {len(args.seeds)} seeds, "
          f"max_turns={args.max_turns}", flush=True)

    results = []
    for s in args.seeds:
        results.append(play_trajectory(net, device, dtype, s, args.max_turns))

    print("\n=== summary ===", flush=True)
    for seed, score, turns, over, marks in results:
        tag = 'DIED' if over else 'cap'
        print(f"seed {seed:>3}: {score:>6} pts in {turns:>5} turns "
              f"({score / max(1, turns):.3f}/turn) [{tag}]", flush=True)
    rates = [score / max(1, turns) for _, score, turns, _, _ in results]
    print(f"\nmean rate: {sum(rates) / len(rates):.3f} pts/turn", flush=True)
    print("If ~2/turn and roughly constant across checkpoints -> "
          "P50~400 over 200 turns is CORRECT, no bug.", flush=True)


if __name__ == '__main__':
    main()
