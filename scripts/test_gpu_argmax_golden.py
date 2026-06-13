"""Golden test 2: GPU argmax-legal player == eval_policy's CPU player.

Realistic boards (random-play snapshots, 7-color and 3-color mixes — sparse through
near-full) × random fp32 logits: choose_moves (legal_priors_t top-1 masked argmax) must
pick the SAME move as the CPU path eval_policy uses (_get_legal_priors_flat top-30 →
max-prior; equal to the global legal argmax since softmax is monotone), and agree on
death (no legal move). fp32 random logits make exact ties measure-zero.

    PYTHONPATH=. python scripts/test_gpu_argmax_golden.py --boards 400 --device mps
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.mcts import _get_legal_priors_flat
from alphatrain.gpu_eval_engine import choose_moves, BOARD


def collect_boards(n, seed0=777):
    """Snapshots across whole random games — early sparse to terminal dense."""
    boards = []
    s = seed0
    while len(boards) < n:
        g = ColorLinesGame(seed=s, num_colors=3 if s % 2 else 7)
        g.reset()
        pick = np.random.RandomState(s)
        while not g.game_over and len(boards) < n:
            boards.append(g.board.copy())
            moves = g.get_legal_moves()
            if not moves:
                break
            (sr, sc), (tr, tc) = moves[pick.randint(len(moves))]
            g.move((sr, sc), (tr, tc))
        s += 1
    return boards[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--boards', type=int, default=400)
    p.add_argument('--device', default='mps')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    dev = torch.device(a.device)

    boards = collect_boards(a.boards)
    # Death coverage: a FULL board has no legal moves (the only no-legal case —
    # any empty cell is enterable by an adjacent ball); plus a one-empty board.
    full = np.full((BOARD, BOARD), 1, dtype=np.int8)
    full[::2, ::2] = 2
    boards.append(full.copy())
    one_empty = full.copy()
    one_empty[4, 4] = 0
    boards.append(one_empty)
    rng = np.random.RandomState(a.seed)
    logits_np = rng.randn(len(boards), BOARD ** 4).astype(np.float32)

    boards_t = torch.from_numpy(np.stack(boards)).to(dev)
    logits_t = torch.from_numpy(logits_np).to(dev)
    moves_t, has_t = choose_moves(boards_t, logits_t)
    moves_g = moves_t.cpu().numpy()
    has_g = has_t.cpu().numpy()

    mism = 0
    deaths = 0
    for i, b in enumerate(boards):
        priors = _get_legal_priors_flat(b, logits_np[i], 30)
        if not priors:
            deaths += 1
            if has_g[i]:
                mism += 1
                print(f"[{i}] CPU says NO legal moves, GPU chose {moves_g[i]}")
            continue
        cpu_move = max(priors.items(), key=lambda x: x[1])[0]
        if not has_g[i] or moves_g[i] != cpu_move:
            mism += 1
            if mism <= 5:
                print(f"[{i}] CPU {cpu_move} vs GPU {int(moves_g[i])} "
                      f"(has_legal={bool(has_g[i])}) "
                      f"logit_cpu={logits_np[i, cpu_move]:.4f} "
                      f"logit_gpu={logits_np[i, int(moves_g[i])] if moves_g[i] >= 0 else float('nan'):.4f}")
    print(f"\n{len(boards)} boards ({deaths} dead positions), mismatches={mism}")
    if mism:
        raise SystemExit("GOLDEN FAIL (argmax-legal)")
    print("GOLDEN PASS: GPU argmax-legal == CPU player on every board.")


if __name__ == '__main__':
    main()
