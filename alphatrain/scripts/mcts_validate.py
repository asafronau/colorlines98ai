import argparse
import time
import torch
from game.board import ColorLinesGame
from alphatrain.mcts import MCTS
from alphatrain.evaluate import load_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=400000)
    p.add_argument('--sims', type=int, default=800)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-turns', type=int, default=2000)
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=True, jit_trace=True)
    print(f"Model loaded, max_score={max_score}, device={args.device}")

    mcts = MCTS(net, device, max_score=max_score, num_simulations=args.sims,
                batch_size=8, top_k=30, dynamic_sims=False)

    game = ColorLinesGame(seed=args.seed)
    game.reset()

    t0 = time.time()
    turn = 0
    ghost_breaks = 0
    while not game.game_over and turn < args.max_turns:
        # Patch trusted_move to detect ghosts and raise an exception we can catch
        original_trusted_move = game.__class__.trusted_move
        def safe_trusted_move(self, sr, sc, tr, tc):
            if self.board[sr, sc] == 0 or self.board[tr, tc] != 0:
                raise ValueError("Ghost Move")
            original_trusted_move(self, sr, sc, tr, tc)
        game.__class__.trusted_move = safe_trusted_move

        try:
            action = mcts.search(game)
        except ValueError:
             ghost_breaks += 1
             game.__class__.trusted_move = original_trusted_move
             continue
        finally:
             game.__class__.trusted_move = original_trusted_move

        if action is None:
            break
        game.move(action[0], action[1])
        turn += 1

    elapsed = time.time() - t0
    print(f"Seed {args.seed}: Score {game.score} in {turn} turns ({elapsed:.1f}s)")
    print(f"Ghost breaks: {ghost_breaks}")

if __name__ == '__main__':
    main()
