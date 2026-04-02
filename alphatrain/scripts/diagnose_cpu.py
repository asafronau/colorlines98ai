"""Diagnose CPU vs MPS inference differences.

Compares raw NN outputs and MCTS search results between CPU and MPS
for the same game state.
"""

import numpy as np
import torch
import time

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game
from alphatrain.evaluate import load_model


def compare_outputs(game, label=""):
    """Compare CPU and MPS forward pass outputs for the same game state."""
    obs_np = _build_obs_for_game(game)

    print(f"\n{'='*60}")
    print(f"Game state: {label}")
    print(f"Score={game.score}, Turns={game.turns}, "
          f"Balls on board={np.count_nonzero(game.board)}")
    print(f"{'='*60}")

    results = {}
    for device_str in ['cpu', 'mps']:
        device = torch.device(device_str)
        net, max_score = load_model(
            'alphatrain/data/alphatrain_td_best.pt', device,
            fp16=(device_str != 'cpu'), jit_trace=True)

        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        if device_str != 'cpu':
            obs = obs.half()

        with torch.inference_mode():
            pol_logits, val_logits = net(obs)
            value = net.predict_value(val_logits, max_val=max_score).item()

        pol_np = pol_logits[0].float().cpu().numpy()
        val_np = val_logits[0].float().cpu().numpy()

        # Policy stats
        top5_idx = np.argsort(pol_np)[-5:][::-1]
        print(f"\n  {device_str.upper()}:")
        print(f"    Value: {value:.2f}")
        print(f"    Policy logits: min={pol_np.min():.3f}, max={pol_np.max():.3f}, "
              f"mean={pol_np.mean():.3f}, std={pol_np.std():.3f}")
        print(f"    Value logits: {val_np}")
        print(f"    Top-5 policy indices: {top5_idx}")
        print(f"    Top-5 policy logits: {pol_np[top5_idx]}")

        results[device_str] = {'pol': pol_np, 'val': value}

    # Compare
    pol_diff = np.abs(results['cpu']['pol'] - results['mps']['pol'])
    val_diff = abs(results['cpu']['val'] - results['mps']['val'])
    print(f"\n  DIFF:")
    print(f"    Value diff: {val_diff:.4f}")
    print(f"    Policy max diff: {pol_diff.max():.4f}")
    print(f"    Policy mean diff: {pol_diff.mean():.6f}")
    print(f"    Policy correlation: {np.corrcoef(results['cpu']['pol'], results['mps']['pol'])[0,1]:.6f}")


def compare_mcts(game, sims=100, label=""):
    """Compare MCTS search results between CPU and MPS."""
    print(f"\n{'='*60}")
    print(f"MCTS comparison ({sims} sims): {label}")
    print(f"{'='*60}")

    for device_str in ['cpu', 'mps']:
        device = torch.device(device_str)
        net, max_score = load_model(
            'alphatrain/data/alphatrain_td_best.pt', device,
            fp16=(device_str != 'cpu'), jit_trace=True)

        mcts = MCTS(net, device, max_score=max_score,
                     num_simulations=sims, batch_size=8,
                     top_k=30, c_puct=2.5)

        # Greedy search (no exploration)
        t0 = time.time()
        action = mcts.search(game)
        t1 = time.time()
        print(f"  {device_str.upper()} greedy: {action}  ({(t1-t0)*1000:.0f}ms)")

        # Self-play search (with exploration)
        np.random.seed(42)  # reproducible noise
        t0 = time.time()
        action_sp, policy = mcts.search(
            game, temperature=1.0,
            dirichlet_alpha=0.3, dirichlet_weight=0.25,
            return_policy=True)
        t1 = time.time()
        print(f"  {device_str.upper()} selfplay: {action_sp}  ({(t1-t0)*1000:.0f}ms)")

        # Show visit distribution
        root_visits = policy[policy > 0]
        print(f"    Non-zero policy entries: {len(root_visits)}")
        print(f"    Top policy: {np.sort(root_visits)[-5:][::-1]}")


def test_selfplay_quality(device_str, sims=100, seed=42, max_turns=50):
    """Run a short self-play game and show every move."""
    device = torch.device(device_str)
    net, max_score = load_model(
        'alphatrain/data/alphatrain_td_best.pt', device,
        fp16=(device_str != 'cpu'), jit_trace=True)

    mcts = MCTS(net, device, max_score=max_score,
                 num_simulations=sims, batch_size=8,
                 top_k=30, c_puct=2.5)

    game = ColorLinesGame(seed=seed)
    game.reset()

    print(f"\n{'='*60}")
    print(f"Self-play game on {device_str.upper()} ({sims} sims, seed={seed})")
    print(f"{'='*60}")

    for turn in range(max_turns):
        if game.game_over:
            break

        temp = 1.0 if turn < 30 else 0.0
        np.random.seed(seed * 1000 + turn)

        result = mcts.search(
            game, temperature=temp,
            dirichlet_alpha=0.3, dirichlet_weight=0.25,
            return_policy=True)

        if result[0] is None:
            print(f"  Turn {turn}: No moves available")
            break

        action, policy = result
        top_policy = np.sort(policy[policy > 0])[-3:][::-1]

        move_result = game.move(action[0], action[1])
        cleared = move_result.get('cleared', 0)
        status = f"CLEAR {cleared}" if cleared > 0 else ""

        print(f"  Turn {turn}: {action[0]}->{action[1]}, "
              f"score={game.score}, balls={np.count_nonzero(game.board)}, "
              f"top_pol={top_policy} {status}", flush=True)

    print(f"\nFinal: score={game.score}, turns={game.turns}")


if __name__ == '__main__':
    # Test 1: Compare raw outputs
    game = ColorLinesGame(seed=42)
    game.reset()
    compare_outputs(game, label="seed=42, start")

    # Play a few moves to get a mid-game state
    game2 = ColorLinesGame(seed=42)
    game2.reset()
    # Make some simple moves
    for sr in range(9):
        for sc in range(9):
            if game2.board[sr, sc] != 0:
                for tr in range(9):
                    for tc in range(9):
                        if not game2.game_over and game2.board[tr, tc] == 0:
                            r = game2.move((sr, sc), (tr, tc))
                            if r['valid']:
                                break
                    if game2.turns >= 5:
                        break
            if game2.turns >= 5:
                break
        if game2.turns >= 5:
            break

    if not game2.game_over:
        compare_outputs(game2, label="seed=42, 5 turns in")

    # Test 2: Compare MCTS (low sims for speed)
    game3 = ColorLinesGame(seed=42)
    game3.reset()
    compare_mcts(game3, sims=50, label="seed=42, start")

    # Test 3: Short self-play games on both devices
    test_selfplay_quality('mps', sims=100, seed=99, max_turns=40)
    test_selfplay_quality('cpu', sims=100, seed=99, max_turns=40)
