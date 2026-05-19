"""Build Source C anchors: first-divergence states between two policies.

For each seed S:
  - Init two games with seed S (deterministic spawn sequence)
  - Step through, running policy_A and policy_B on the (identical) state
  - Find first turn where their argmax differs → save the anchor state plus
    both policies' chosen moves
  - Stop comparing (games diverge after that point)

Output: a list of anchor records suitable for the tournament's judging path.

Usage:
    python -m alphatrain.scripts.build_first_divergence_anchors \\
        --model-a alphatrain/data/pillar2z_epoch_19.pt \\
        --model-b alphatrain/data/policy_dagger_v9.pt \\
        --n-seeds 300 --max-turns 2000 \\
        --device mps \\
        --output alphatrain/data/source_c_first_divergence.pt
"""

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-a', required=True, help='Reference policy (e.g., 2z).')
    p.add_argument('--model-b', required=True, help='Variant policy (e.g., v9).')
    p.add_argument('--n-seeds', type=int, default=300)
    p.add_argument('--max-turns', type=int, default=2000,
                   help='Cap on how many turns to step before giving up if no divergence.')
    p.add_argument('--device', default='mps')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    from alphatrain.evaluate import load_model
    from alphatrain.observation import build_observation
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')

    print(f"Loading {args.model_a} (A)...", flush=True)
    net_a, _ = load_model(args.model_a, device, fp16=fp16, jit_trace=False)
    net_a.train(False)
    print(f"Loading {args.model_b} (B)...", flush=True)
    net_b, _ = load_model(args.model_b, device, fp16=fp16, jit_trace=False)
    net_b.train(False)

    def policy_argmax(net, game):
        obs = _build_obs_for_game(game)
        x = torch.from_numpy(obs[None]).to(
            device, dtype=torch.float16 if fp16 else torch.float32)
        with torch.inference_mode():
            logits = net(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            pol = torch.softmax(logits.float()[0], dim=-1).cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol, 10)
        if not priors:
            return None, None
        # argmax over legal moves
        return max(priors.items(), key=lambda x: x[1])[0], pol

    anchors = []
    skipped = 0
    t0 = time.time()
    for seed in range(args.n_seeds):
        # Paired identical-seed games. Must call reset() to spawn initial
        # balls + generate next_balls preview (constructor alone leaves the
        # game in pre-start state).
        gA = ColorLinesGame(seed=seed)
        gA.reset()
        gB = ColorLinesGame(seed=seed)
        gB.reset()
        diverged_at = None
        anchor_state = None
        for turn in range(args.max_turns):
            if gA.game_over or gB.game_over:
                break
            # Both games are in identical state until first divergence.
            mv_a, _ = policy_argmax(net_a, gA)
            mv_b, _ = policy_argmax(net_b, gB)
            if mv_a is None or mv_b is None:
                break
            if mv_a != mv_b:
                # Save anchor (state where they diverge), plus both moves.
                # Use gA's state since both are identical at this point.
                diverged_at = turn
                anchor_state = {
                    'board': gA.board.copy(),
                    'next_balls': list(gA.next_balls),
                    'num_next': len(gA.next_balls),
                    'turn_origin': gA.turns,
                    'seed_origin': seed,
                    'source_label': 'first_divergence',
                    'move_a': int(mv_a),
                    'move_b': int(mv_b),
                }
                break
            # Apply the (shared) move to both games (RNG is per-game-instance,
            # but seeded identically → same spawn sequence).
            sr = mv_a // 81 // 9; sc = mv_a // 81 % 9
            tr = mv_a % 81 // 9; tc = mv_a % 81 % 9
            rA = gA.move((sr, sc), (tr, tc))
            rB = gB.move((sr, sc), (tr, tc))
            if not rA['valid'] or not rB['valid']:
                break

        if anchor_state is not None:
            anchors.append(anchor_state)
        else:
            skipped += 1

        if (seed + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (seed + 1) * (args.n_seeds - seed - 1)
            print(f"  [{seed+1}/{args.n_seeds}] anchors={len(anchors)} "
                  f"skipped={skipped} {elapsed:.0f}s ETA {eta:.0f}s",
                  flush=True)

    print(f"\nDone: {len(anchors)} first-divergence anchors "
          f"(skipped {skipped}, no divergence within {args.max_turns} turns "
          f"or early game-over)", flush=True)
    if anchors:
        turns_at_diverge = [a['turn_origin'] for a in anchors]
        print(f"  divergence turn stats: "
              f"mean={np.mean(turns_at_diverge):.0f} "
              f"median={int(np.median(turns_at_diverge))} "
              f"min={min(turns_at_diverge)} max={max(turns_at_diverge)}",
              flush=True)
        # How often do the two policies' moves come from same top-K?
        # (Just a sanity check.)
    for i, a in enumerate(anchors):
        a['id'] = i
    torch.save({'args': vars(args), 'anchors': anchors}, args.output)
    print(f"Saved {args.output}", flush=True)


if __name__ == '__main__':
    main()
