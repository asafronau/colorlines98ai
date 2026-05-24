"""MCTS sweep on the image-3 state.

Question: under what configuration (sims, q_weight, c_puct, exploration) does
MCTS pick the green-clear move? The aim is to figure out WHERE the bias lives:
- More sims fixes it → policy distillation needs higher-sim teacher
- More c_puct fixes it → root exploration too narrow at default
- Lower q_weight fixes it → value head over-rewarding pink-setup
- Higher q_weight fixes it → value head under-trusted (unlikely)
- Nothing fixes it → bug deeper than search; in the value head or backbone
"""
from __future__ import annotations
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.config import BOARD_SIZE
from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS

COLOR_NAME = ['_', 'red', 'green', 'blue', 'yellow', 'cyan', 'pink', 'brown']

# Image 3: Score 35, Turn 27
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

CLEAR_MOVE = ((6, 6), (3, 5))  # the only legal move that clears a line
MODEL_PICK = ((4, 5), (7, 4))  # policy-only top1

VALUE_HEAD = 'alphatrain/data/value_head_sharp25_ep12.pt'


def fresh_game():
    g = ColorLinesGame()
    g.reset(board=BOARD.copy(), next_balls=list(NEXT_BALLS))
    g.score = 35
    g.turns = 27
    return g


def top_k_from_policy(policy, k=3):
    """Given a 6561 visit distribution, return top-k (idx, prob, src, tgt)."""
    top = np.argsort(policy)[::-1][:k]
    out = []
    for idx in top:
        if policy[idx] <= 0:
            break
        sr, sc = idx // 81 // 9, idx // 81 % 9
        tr, tc = idx % 81 // 9, idx % 81 % 9
        out.append((idx, float(policy[idx]), (sr, sc), (tr, tc)))
    return out


def move_prob(policy, src, tgt):
    idx = (src[0] * 9 + src[1]) * 81 + tgt[0] * 9 + tgt[1]
    return float(policy[idx]), idx


def fmt_move(src, tgt):
    color = COLOR_NAME[BOARD[src[0], src[1]]]
    return f"{color:6s} ({src[0]},{src[1]})→({tgt[0]},{tgt[1]})"


def run_one(net, device, value_head_path, *, sims, q_weight=2.0, c_puct=2.5,
            dirichlet_alpha=0.0, dirichlet_weight=0.0,
            top_k=30, batch_size=8):
    game = fresh_game()
    mcts = MCTS(net=net, device=device,
                num_simulations=sims, c_puct=c_puct,
                top_k=top_k, batch_size=batch_size,
                q_weight=q_weight,
                value_head_path=value_head_path)
    t0 = time.time()
    action, policy = mcts.search(game, temperature=0.0,
                                  dirichlet_alpha=dirichlet_alpha,
                                  dirichlet_weight=dirichlet_weight,
                                  return_policy=True)
    elapsed = time.time() - t0
    return action, policy, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='alphatrain/data/sharp_25_epoch_12.pt')
    parser.add_argument('--value-head', default=VALUE_HEAD,
                         help='Value head path (must match backbone).')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Value head: {args.value_head}")
    net, _ = load_model(args.model, device, fp16=False)
    vh = args.value_head

    # Warmup so MPS compile cost doesn't pollute the first row's wall time.
    print("Warmup MCTS@50 ...", flush=True)
    run_one(net, device, vh, sims=50)

    def header(title):
        print(f"\n{'=' * 92}\n  {title}\n{'=' * 92}")

    def row(label, action, policy, elapsed):
        top3 = top_k_from_policy(policy, k=3)
        clear_p, _ = move_prob(policy, *CLEAR_MOVE)
        pick_p, _ = move_prob(policy, *MODEL_PICK)
        is_clear = (action == CLEAR_MOVE)
        winner = "GREEN-CLEAR ✓" if is_clear else (
            f"{fmt_move(*action)}")
        s = f"  {label:<28s} chose: {winner}  | clear_p={clear_p*100:5.1f}%  "
        s += f"pickprev_p={pick_p*100:5.1f}%  | {elapsed:5.1f}s"
        print(s)
        for i, (_, p, src, tgt) in enumerate(top3, 1):
            tag = "✦CLEAR" if (src, tgt) == CLEAR_MOVE else ""
            print(f"    #{i}: {fmt_move(src, tgt)}  p={p*100:5.1f}%  {tag}")

    # === Sim sweep at default settings ===
    header("Sim budget sweep (q_weight=2.0, c_puct=2.5, no Dirichlet)")
    for sims in [100, 200, 400, 800, 1200, 1600, 3200]:
        action, policy, elapsed = run_one(net, device, vh, sims=sims)
        row(f"sims={sims}", action, policy, elapsed)

    # === q_weight sweep at sims=400 (matches selfplay) ===
    header("q_weight sweep (sims=400, c_puct=2.5, no Dirichlet)")
    for q in [0.0, 1.0, 2.0, 3.0]:
        action, policy, elapsed = run_one(net, device, vh, sims=400, q_weight=q)
        row(f"q_weight={q}", action, policy, elapsed)

    # === c_puct sweep at sims=400, q=2.0 ===
    header("c_puct sweep (sims=400, q=2.0, no Dirichlet)")
    for cp in [1.5, 2.5, 4.0, 6.0]:
        action, policy, elapsed = run_one(net, device, vh, sims=400, c_puct=cp)
        row(f"c_puct={cp}", action, policy, elapsed)

    # === Dirichlet noise ablation at sims=400 ===
    header("Dirichlet noise (sims=400, q=2.0, c_puct=2.5)")
    for alpha, w in [(0.0, 0.0), (0.3, 0.25), (1.0, 0.5)]:
        action, policy, elapsed = run_one(net, device, vh, sims=400,
                                            dirichlet_alpha=alpha,
                                            dirichlet_weight=w)
        row(f"α={alpha} w={w}", action, policy, elapsed)


if __name__ == '__main__':
    main()
