import numpy as np
from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, Node
from game.rng import SimpleRng
import torch

class DummyNet:
    def __call__(self, obs):
        batch = obs.shape[0]
        # output uniform policy and value 3000
        pol = torch.ones(batch, 6561) / 6561
        val = torch.zeros(batch, 1) # predict_value will be mocked
        return pol, val
    def predict_value(self, val_logits, max_val):
        return torch.ones(val_logits.shape[0]) * 3000.0

game = ColorLinesGame(seed=42)
game.reset()

mcts = MCTS(net=DummyNet(), device=torch.device('cpu'), num_simulations=128, batch_size=128, top_k=30, dynamic_sims=False)

# Mock _nn_evaluate_single
mcts._fp16 = False
mcts._obs_buf = torch.empty(128, 18, 9, 9)

# We want to see if trusted_move raises an exception or corrupts the board.
# We'll patch trusted_move to assert legality.
original_trusted_move = ColorLinesGame.trusted_move
def safe_trusted_move(self, sr, sc, tr, tc):
    if self.board[sr, sc] == 0:
        print("GHOST MOVE DETECTED! Source cell is empty.")
        raise RuntimeError("Ghost Move")
    if self.board[tr, tc] != 0:
        print("GHOST MOVE DETECTED! Target cell is occupied.")
        raise RuntimeError("Ghost Move")
    original_trusted_move(self, sr, sc, tr, tc)

ColorLinesGame.trusted_move = safe_trusted_move

try:
    mcts.search(game)
    print("Search completed without corruption.")
except Exception as e:
    print(f"Search failed: {e}")
