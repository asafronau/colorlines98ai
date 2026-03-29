"""Debug value head predictions for move ranking.

Shows how policy and value heads rank moves from the same position,
to diagnose whether the value head is useful for search.

Usage:
    python -m alphatrain.scripts.debug_value_head
    python -m alphatrain.scripts.debug_value_head --model alphatrain/data/alphatrain_best.pt --seed 42
"""

import argparse
import os
import numpy as np
import torch
from scipy.stats import spearmanr

from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='alphatrain/data/alphatrain_best.pt')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net, max_score = load_model(args.model, device)
    print(f"Value head range: [0, {max_score:.0f}]")

    game = ColorLinesGame(seed=args.seed)
    game.reset()

    # Root evaluation
    obs = torch.from_numpy(_build_obs_for_game(game)).unsqueeze(0).to(device)
    with torch.no_grad():
        pol, val = net(obs)
        root_val = net.predict_value(val, max_val=max_score).item()
    print(f"Root value prediction: {root_val:.0f} (actual score: {game.score})")

    # Collect all legal moves with policy scores
    legal = game.get_legal_moves()
    pol_np = pol[0].cpu().numpy()
    move_scores = []
    for m in legal:
        idx = (m[0][0] * 9 + m[0][1]) * 81 + m[1][0] * 9 + m[1][1]
        move_scores.append((pol_np[idx], m))
    move_scores.sort(reverse=True)

    print(f"\nTop-5 policy moves and post-move value predictions:")
    for rank, (pol_score, move) in enumerate(move_scores[:5]):
        g = game.clone()
        result = g.move(move[0], move[1])
        obs2 = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, val2 = net(obs2)
            v = net.predict_value(val2, max_val=max_score).item()
        print(f"  #{rank+1} {move[0]}->{move[1]}: pol={pol_score:.2f}, "
              f"value={v:.0f}, score_delta={g.score - game.score}, "
              f"over={g.game_over}")

    print(f"\nBottom-5 policy moves:")
    for rank, (pol_score, move) in enumerate(move_scores[-5:]):
        g = game.clone()
        g.move(move[0], move[1])
        obs2 = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, val2 = net(obs2)
            v = net.predict_value(val2, max_val=max_score).item()
        print(f"  {move[0]}->{move[1]}: pol={pol_score:.2f}, value={v:.0f}, "
              f"score_delta={g.score - game.score}")

    # Value ranking: which moves does the value head prefer?
    print(f"\nValue head top-5 (all {len(legal)} legal moves evaluated):")
    val_scores = []
    for pol_score, move in move_scores:
        g = game.clone()
        g.move(move[0], move[1])
        obs2 = torch.from_numpy(_build_obs_for_game(g)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, val2 = net(obs2)
            v = net.predict_value(val2, max_val=max_score).item()
        val_scores.append((v, pol_score, move))
    val_scores.sort(reverse=True)
    for rank, (v, pol_score, move) in enumerate(val_scores[:5]):
        print(f"  #{rank+1} {move[0]}->{move[1]}: value={v:.0f}, pol={pol_score:.2f}")

    # Correlation between policy and value rankings
    pol_ranks = {m: i for i, (_, m) in enumerate(move_scores)}
    val_ranks = {m: i for i, (_, _, m) in enumerate(val_scores)}
    pol_r = [pol_ranks[m] for m in pol_ranks]
    val_r = [val_ranks[m] for m in pol_ranks]
    corr, pval = spearmanr(pol_r, val_r)
    print(f"\nPolicy vs Value rank correlation: rho={corr:.3f} (p={pval:.3f})")
    print(f"Value range: {val_scores[-1][0]:.0f} - {val_scores[0][0]:.0f} "
          f"(spread={val_scores[0][0] - val_scores[-1][0]:.0f})")


if __name__ == '__main__':
    main()
