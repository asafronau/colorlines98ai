"""Determinism golden-file test.

Captures and verifies a known-good MCTS action sequence for a few seeds
under the production config (pillar2x2 + feature_value_weights_2x). This
is the contract the architectural refactor must preserve: same model,
same seed, same code path → same actions.

Run modes:
- `pytest alphatrain/tests/test_determinism_golden.py` — verifies that
  current code reproduces the golden actions. Fails if MCTS behavior
  has drifted.
- `python -m alphatrain.tests.test_determinism_golden --capture` —
  captures fresh golden actions from current code into the .json file.
  Use AFTER intentional behavior changes (e.g. before a refactor stage
  to lock in the current truth, or after a stage with verified-correct
  behavior).
"""
import json
import os

import numpy as np
import pytest
import torch

from alphatrain.evaluate import load_model
from alphatrain.mcts import MCTS
from game.board import ColorLinesGame

GOLDEN = os.path.join(os.path.dirname(__file__), 'determinism_golden.json')

# Tight, fast config for CI: 5 seeds × 20 turns × 50 sims = ~5 sec on CPU.
SEEDS = [0, 7, 14, 23, 42]
TURNS_PER_SEED = 20
SIMS = 50


def _action_sequence(model_path, weights_path):
    device = torch.device('cpu')   # CPU forces full determinism
    net, max_score = load_model(model_path, device, fp16=False, jit_trace=False)
    mcts = MCTS(net=net, device=device, max_score=max_score,
                num_simulations=SIMS, batch_size=8, top_k=30,
                feature_weights_path=weights_path)

    out = {}
    for seed in SEEDS:
        game = ColorLinesGame(seed=seed)
        game.reset()
        actions = []
        for _ in range(TURNS_PER_SEED):
            if game.game_over:
                break
            move = mcts.search(game, temperature=0.0)
            if move is None:
                break
            (sr, sc), (tr, tc) = move
            actions.append([sr, sc, tr, tc])
            game.move(move[0], move[1])
        out[str(seed)] = {'score': int(game.score),
                          'turns': int(game.turns),
                          'actions': actions}
    return out


def _default_paths():
    """Look for production assets, fall back to a tiny test model if missing."""
    model = 'alphatrain/data/pillar2x2_epoch_10.pt'
    weights = 'alphatrain/data/feature_value_weights_2x.npz'
    if not (os.path.exists(model) and os.path.exists(weights)):
        return None, None
    return model, weights


def test_determinism_against_golden():
    """Re-run MCTS and assert actions match the captured golden file."""
    model, weights = _default_paths()
    if model is None:
        pytest.skip("production model/weights missing; capture golden first")
    if not os.path.exists(GOLDEN):
        pytest.skip(f"golden file missing: {GOLDEN}; "
                    f"run with --capture to create it")

    expected = json.load(open(GOLDEN))
    actual = _action_sequence(model, weights)

    for seed in SEEDS:
        s = str(seed)
        assert s in expected, f"seed {seed} missing from golden"
        e = expected[s]
        a = actual[s]
        assert a['actions'] == e['actions'], \
            f"seed {seed}: action sequence drifted\n" \
            f"  expected: {e['actions']}\n" \
            f"  actual:   {a['actions']}"
        assert a['score'] == e['score'], \
            f"seed {seed}: score drifted ({e['score']} → {a['score']})"


def main():
    """Capture mode: write current actions to the golden file."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--capture', action='store_true')
    args = p.parse_args()

    model, weights = _default_paths()
    if model is None:
        raise SystemExit(
            f"Production assets not found. Need:\n"
            f"  alphatrain/data/pillar2x2_epoch_10.pt\n"
            f"  alphatrain/data/feature_value_weights_2x.npz")

    print(f"Capturing actions for seeds {SEEDS}, "
          f"{TURNS_PER_SEED} turns each, {SIMS} sims, CPU...")
    actions = _action_sequence(model, weights)

    if args.capture or not os.path.exists(GOLDEN):
        json.dump(actions, open(GOLDEN, 'w'), indent=1)
        print(f"Wrote golden to {GOLDEN}")
    else:
        print(f"Golden exists at {GOLDEN}. Pass --capture to overwrite.")
        for seed in SEEDS:
            print(f"  seed {seed}: score={actions[str(seed)]['score']}, "
                  f"actions={actions[str(seed)]['actions'][:3]}...")


if __name__ == '__main__':
    main()
