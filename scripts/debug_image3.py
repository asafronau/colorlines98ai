"""Why does pillar3a prefer extending the pink line over completing the green line?

Hand-transcribed from image 3 (Score 35, Turn 27). Compares the model's top-K
choices against move-of-interest candidates: in particular, completing the
4-greens-in-a-row at row 3, which becomes impossible next turn when brown
spawns at (3,5).
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import ColorLinesGame
from game.config import BOARD_SIZE
from alphatrain.evaluate import load_model
from alphatrain.observation import build_observation

# Colors: 1=red, 2=green, 3=blue, 4=yellow, 5=cyan, 6=pink, 7=brown
COLOR_NAME = ['_', 'red', 'green', 'blue', 'yellow', 'cyan', 'pink', 'brown']

BOARD = np.zeros((9, 9), dtype=np.int8)
balls = {
    (0, 0): 4, (0, 1): 3, (0, 7): 7,
    (1, 0): 6, (1, 1): 7, (1, 2): 3,
    (2, 1): 5,
    (3, 1): 2, (3, 2): 2, (3, 3): 2, (3, 4): 2, (3, 7): 1,
    (4, 1): 5, (4, 2): 7, (4, 5): 6, (4, 6): 1,
    (5, 0): 3, (5, 6): 1, (5, 7): 3,
    (6, 0): 7, (6, 3): 1, (6, 6): 2,
    (7, 5): 6, (7, 6): 6, (7, 7): 6,
    (8, 0): 6, (8, 6): 3, (8, 8): 4,
}
for (r, c), col in balls.items():
    BOARD[r, c] = col

# Next balls from the right panel
NEXT_BALLS = [((3, 5), 7), ((2, 3), 7), ((6, 5), 2)]


def print_board(b):
    print("    " + " ".join(str(c) for c in range(9)))
    for r in range(9):
        row = [str(int(b[r, c])) if b[r, c] != 0 else '.'
               for c in range(9)]
        print(f"  {r} " + " ".join(row))


def main():
    game = ColorLinesGame()
    game.reset(board=BOARD.copy(), next_balls=list(NEXT_BALLS))
    game.score = 35
    game.turns = 27
    print(f"State: Score={game.score}, Turn={game.turns}")
    print(f"Next balls: {NEXT_BALLS}")
    print()
    print_board(game.board)
    print()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, _ = load_model('alphatrain/data/sharp_25_epoch_12.pt', device,
                         fp16=False)
    net_dtype = next(net.parameters()).dtype

    # Build observation
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    obs = torch.from_numpy(
        build_observation(game.board, nr, nc, ncol, nn)
    ).unsqueeze(0).to(device=device, dtype=net_dtype)

    with torch.no_grad():
        logits = net(obs)[0].float().cpu().numpy()

    # Mask legal moves
    source_mask = game.get_source_mask()
    legal_idx = []
    legal_logits = np.full(6561, -np.inf, dtype=np.float64)
    for sr in range(BOARD_SIZE):
        for sc in range(BOARD_SIZE):
            if source_mask[sr, sc] == 0:
                continue
            tmask = game.get_target_mask((sr, sc))
            for tr in range(BOARD_SIZE):
                for tc in range(BOARD_SIZE):
                    if tmask[tr, tc] > 0:
                        idx = (sr * 9 + sc) * 81 + tr * 9 + tc
                        legal_logits[idx] = logits[idx]
                        legal_idx.append(idx)

    # Softmax over legal moves
    finite = legal_logits[legal_logits > -np.inf]
    m = finite.max()
    e = np.exp(legal_logits - m)
    e[legal_logits == -np.inf] = 0
    probs = e / e.sum()
    print(f"# legal moves: {len(legal_idx)}")
    print(f"policy top1 prob: {probs.max()*100:.2f}%")
    print()

    # Simulate every legal move once to find ALL moves that clear lines
    clears = []
    for idx in legal_idx:
        sr, sc, tr, tc = idx // 81 // 9, idx // 81 % 9, idx % 81 // 9, idx % 81 % 9
        g2 = game.clone()
        r = g2.move((sr, sc), (tr, tc))
        if r.get('cleared', 0) > 0:
            clears.append((idx, r['cleared'], r['score']))

    print(f"=== LEGAL MOVES THAT CLEAR A LINE THIS TURN: {len(clears)} ===")
    # Sort clears by points then probability rank
    clears_sorted = sorted(clears, key=lambda x: (-x[2], -probs[x[0]]))
    for idx, cleared, pts in clears_sorted[:10]:
        sr, sc, tr, tc = idx // 81 // 9, idx // 81 % 9, idx % 81 // 9, idx % 81 % 9
        color = COLOR_NAME[BOARD[sr, sc]]
        p = probs[idx]
        rank = int((probs > p).sum()) + 1
        print(f"  {color} ({sr},{sc}) → ({tr},{tc})  clear={cleared} +{pts}pts  "
              f"policy p={p*100:.4f}%  rank=#{rank}/{len(legal_idx)}")
    print()

    # Top 15 policy choices
    print("=== TOP 15 POLICY CHOICES ===")
    top_idx = np.argsort(legal_logits)[::-1][:15]
    for rank, idx in enumerate(top_idx, 1):
        if legal_logits[idx] == -np.inf:
            break
        sr, sc, tr, tc = idx // 81 // 9, idx // 81 % 9, idx % 81 // 9, idx % 81 % 9
        color = COLOR_NAME[BOARD[sr, sc]]
        p = probs[idx]
        # Simulate
        g2 = game.clone()
        r = g2.move((sr, sc), (tr, tc))
        cleared = r.get('cleared', 0) if r['valid'] else 0
        pts = r.get('score', 0) if r['valid'] else 0
        clear_str = f"   ✦CLEAR {cleared} (+{pts})" if cleared else ""
        print(f"  #{rank:2d}: {color:6s} ({sr},{sc}) → ({tr},{tc})  "
              f"p={p*100:6.2f}%{clear_str}")
    print()

    # Moves of interest
    def find_idx(src, tgt):
        return (src[0] * 9 + src[1]) * 81 + tgt[0] * 9 + tgt[1]

    interest = [
        ("green(6,6)→(3,5) — complete 5-line in row 3 (blocked next turn!)",
         (6, 6), (3, 5)),
        ("green(6,6)→(3,0) — complete 5-line via other end",
         (6, 6), (3, 0)),
        ("pink(4,5)→(7,4)  — model's apparent choice: setup row-7 pink line",
         (4, 5), (7, 4)),
    ]
    print("=== MOVES OF INTEREST ===")
    for name, src, tgt in interest:
        idx = find_idx(src, tgt)
        if legal_logits[idx] == -np.inf:
            print(f"  {name}")
            print(f"    -- ILLEGAL (source can't reach target)")
            continue
        p = probs[idx]
        rank = int((probs > p).sum()) + 1
        g2 = game.clone()
        r = g2.move(src, tgt)
        cleared = r.get('cleared', 0) if r['valid'] else 0
        pts = r.get('score', 0) if r['valid'] else 0
        print(f"  {name}")
        print(f"    rank #{rank}/{len(legal_idx)}, p={p*100:.4f}%, "
              f"cleared={cleared}, +{pts}pts")


if __name__ == '__main__':
    main()
