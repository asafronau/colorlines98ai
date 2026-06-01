"""Count legal moves + policy rank of the floor-best at death-game frames.

Sizes the candidate-set question: top-K vs all-legal. Reports n_legal per frame
and where the policy ranks each legal move (to see if broadening is needed).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from game.board import ColorLinesGame
from alphatrain.evaluate import load_model
from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat

d = json.load(open('alphatrain/data/worst_game.json'))
frames = d['frames']
net, _ = load_model('alphatrain/data/pillar3b_epoch_20.pt', torch.device('cpu'), fp16=False)
dtype = next(net.parameters()).dtype

print(f"{'turn':>5} {'empt':>5} {'lec':>4} {'ncomp':>5} {'n_legal':>8} "
      f"{'top1_p':>7} {'p@rank5':>8} {'p@rank20':>9}")
for turn in (40, 55, 70, 85, 100, 110, 118, 120, 122, 130):
    fr = frames[turn]
    g = ColorLinesGame()
    g.reset(board=np.array(fr['board'], dtype=np.int8),
            next_balls=[(tuple(p), int(c)) for p, c in fr['next_balls']])
    legal = g.get_legal_moves()
    obs = _build_obs_for_game(g)
    with torch.no_grad():
        logits = net(torch.from_numpy(obs).unsqueeze(0).to(torch.device('cpu'), dtype))[0]
    pri = _get_legal_priors_flat(g.board, logits.float().numpy(), 9999)
    probs = sorted(pri.values(), reverse=True)
    empt = int((g.board == 0).sum())
    print(f"{turn:>5} {empt:>5} {fr.get('lec','?'):>4} {fr.get('n_components','?'):>5} "
          f"{len(legal):>8} {probs[0]:>7.3f} "
          f"{(probs[4] if len(probs)>4 else 0):>8.4f} "
          f"{(probs[19] if len(probs)>19 else 0):>9.5f}")
