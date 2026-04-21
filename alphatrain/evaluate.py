"""AlphaTrain evaluation: play games with trained ResNet.

Usage:
    python -m alphatrain.evaluate --player policy --games 20 --seed 42
    python -m alphatrain.evaluate --player mcts --simulations 50 --games 5
"""

import os
import time
import argparse
import numpy as np
import torch
from typing import Callable, Optional

from game.board import ColorLinesGame
from alphatrain.model import AlphaTrainNet
from alphatrain.observation import build_observation

BOARD_SIZE = 9


class _JitWrapper:
    """Wraps a JIT-traced model."""

    def __init__(self, net, dummy_input):
        self._traced = torch.jit.trace(net, dummy_input)

    def __call__(self, x):
        return self._traced(x)

    def parameters(self):
        return self._traced.parameters()

    def train(self, mode):
        return self

Player = Callable[[ColorLinesGame], Optional[tuple[tuple[int, int], tuple[int, int]]]]


def load_model(model_path, device, fp16=False, jit_trace=False):
    """Load AlphaTrainNet from checkpoint, auto-detecting architecture.

    Args:
        fp16: convert to half precision (2x faster on MPS/CUDA)
        jit_trace: apply torch.jit.trace (10-15% faster forward pass)

    Returns net.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"  Download from Drive or specify --model path")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt['model']
    # Strip torch.compile prefix if present
    if any(k.startswith('_orig_mod.') for k in state):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    in_ch = state['stem.0.weight'].shape[1]
    nb = sum(1 for k in state if k.endswith('.conv1.weight')
             and k.startswith('blocks.'))
    ch = state['stem.0.weight'].shape[0]
    # Filter out value head keys from old checkpoints
    VALUE_PREFIXES = ('value_conv.', 'value_bn.', 'value_fc1.', 'value_fc2.')
    state = {k: v for k, v in state.items()
             if not any(k.startswith(p) for p in VALUE_PREFIXES)}
    net = AlphaTrainNet(in_channels=in_ch, num_blocks=nb, channels=ch).to(device)
    net.load_state_dict(state)
    net.train(False)
    epoch = ckpt.get('epoch', '?')

    opts = []
    if fp16 and device.type in ('mps', 'cuda'):
        net = net.half()
        opts.append('fp16')
    if jit_trace:
        dtype = torch.float16 if fp16 and device.type in ('mps', 'cuda') else torch.float32
        dummy = torch.randn(1, in_ch, 9, 9, device=device, dtype=dtype)
        net = _JitWrapper(net, dummy)
        opts.append('jit')

    opt_str = f" [{'+'.join(opts)}]" if opts else ""
    print(f"Loaded {model_path}: {nb}b x {ch}ch, epoch={epoch}"
          + opt_str, flush=True)
    return net


def make_policy_player(net, device):
    """Greedy argmax policy player from AlphaTrainNet."""
    def player(game: ColorLinesGame):
        board = game.board
        nr = np.zeros(3, dtype=np.intp)
        nc = np.zeros(3, dtype=np.intp)
        ncol = np.zeros(3, dtype=np.intp)
        nn = min(len(game.next_balls), 3)
        for i, ((r, c), col) in enumerate(game.next_balls):
            if i >= 3:
                break
            nr[i], nc[i], ncol[i] = r, c, col

        obs = torch.from_numpy(
            build_observation(board, nr, nc, ncol, nn)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = net(obs)[0].cpu().numpy()

        source_mask = game.get_source_mask()
        best_score = -1e18
        best_move = None
        for sr in range(BOARD_SIZE):
            for sc in range(BOARD_SIZE):
                if source_mask[sr, sc] == 0:
                    continue
                target_mask = game.get_target_mask((sr, sc))
                for tr in range(BOARD_SIZE):
                    for tc in range(BOARD_SIZE):
                        if target_mask[tr, tc] > 0:
                            idx = (sr * 9 + sc) * 81 + tr * 9 + tc
                            if logits[idx] > best_score:
                                best_score = logits[idx]
                                best_move = ((sr, sc), (tr, tc))
        return best_move

    return player


def play_game(player, seed=None, verbose=False):
    """Play one game, return stats dict."""
    game = ColorLinesGame(seed=seed)
    game.reset()
    total_cleared = 0
    clears = 0

    while not game.game_over:
        move = player(game)
        if move is None:
            break
        result = game.move(move[0], move[1])
        if not result['valid']:
            break
        if result['cleared'] > 0:
            total_cleared += result['cleared']
            clears += 1
        if verbose and result['cleared'] > 0:
            print(f"  Turn {game.turns}: cleared {result['cleared']}, "
                  f"score={game.score}", flush=True)

    return {
        'score': game.score,
        'turns': game.turns,
        'total_cleared': total_cleared,
        'clears': clears,
    }


def run_evaluation(player, num_games=100, seed=42):
    """Run games and print aggregate stats."""
    scores, turns_list = [], []
    t0 = time.time()

    for i in range(num_games):
        game_seed = seed + i if seed is not None else None
        result = play_game(player, seed=game_seed)
        scores.append(result['score'])
        turns_list.append(result['turns'])

        if (i + 1) % max(1, num_games // 10) == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_games}] mean={np.mean(scores):.0f} "
                  f"max={np.max(scores)} ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Results over {num_games} games ({elapsed:.1f}s):")
    print(f"  Score: {np.mean(scores):.0f} +/- {np.std(scores):.0f} "
          f"(median={np.median(scores):.0f}, "
          f"max={np.max(scores)}, min={np.min(scores)})")
    print(f"  Turns: {np.mean(turns_list):.1f}")
    print(f"{'='*50}")

    return {
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores)),
        'max_score': int(np.max(scores)),
        'min_score': int(np.min(scores)),
        'num_games': num_games,
    }


def play_game_verbose(player, seed=None, report_every=500):
    """Play one game with periodic progress (for MCTS which is slow)."""
    game = ColorLinesGame(seed=seed)
    game.reset()
    t0 = time.time()

    while not game.game_over:
        move = player(game)
        if move is None:
            break
        result = game.move(move[0], move[1])
        if not result['valid']:
            break
        if game.turns % report_every == 0:
            elapsed = time.time() - t0
            print(f"    turn {game.turns}, score={game.score}, "
                  f"{elapsed:.0f}s", flush=True)

    return {
        'score': game.score,
        'turns': game.turns,
        'time': time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--player', choices=['policy', 'mcts'], default='policy')
    p.add_argument('--model', default='alphatrain/data/alphatrain_best.pt')
    p.add_argument('--games', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    # MCTS params
    p.add_argument('--simulations', type=int, default=200)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--top-k', type=int, default=30)
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    net = load_model(args.model, device)

    if args.player == 'policy':
        player = make_policy_player(net, device)
    elif args.player == 'mcts':
        from alphatrain.mcts import make_mcts_player
        player = make_mcts_player(
            net, device,
            num_simulations=args.simulations,
            c_puct=args.c_puct, top_k=args.top_k)

    print(f"Running {args.player} on {device} "
          f"({args.games} games, seed={args.seed}"
          + (f", sims={args.simulations}" if args.player == 'mcts' else "")
          + ")", flush=True)
    run_evaluation(player, num_games=args.games, seed=args.seed)


if __name__ == '__main__':
    main()
