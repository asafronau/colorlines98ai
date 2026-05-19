import torch
import numpy as np
from game.board import ColorLinesGame
from alphatrain.mcts import MCTS
from alphatrain.evaluate import load_model

def main():
    device = torch.device('cpu')
    net, max_score = load_model('alphatrain/data/pillar2w2_epoch_10.pt', device, fp16=False, jit_trace=False)
    mcts = MCTS(net, device, max_score=max_score, num_simulations=400, batch_size=8, top_k=30, dynamic_sims=False)

    game = ColorLinesGame(seed=36)
    game.reset()

    # Step 1: Get raw policy prior
    priors, root_value = mcts._nn_evaluate_single(game)
    top_policy_move = max(priors.items(), key=lambda x: x[1])[0]

    # Step 2: Run MCTS
    action, policy_target = mcts.search(game, return_policy=True)
    
    # Re-decode action to flat for comparison
    flat_action = action[0][0]*81 + action[0][1]*9 + action[1][0]*9 + action[1][1]

    print(f"Top Policy Move Prior: {priors[top_policy_move]:.4f}")
    print(f"Top Policy Move MCTS Visits: {policy_target[top_policy_move]*400:.1f}")
    
    print(f"MCTS Chosen Move Prior: {priors.get(flat_action, 0.0):.4f}")
    print(f"MCTS Chosen Move Visits: {policy_target[flat_action]*400:.1f}")

if __name__ == '__main__':
    main()
