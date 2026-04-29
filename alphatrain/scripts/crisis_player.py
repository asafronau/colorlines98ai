"""Crisis-mode player: policy default, MCTS on crisis boards.

Plays with instant greedy policy on healthy boards. When board features
indicate crisis, switches to deep MCTS search.

Crisis detection uses avg_reach (average reachable targets per ball)
as the primary signal, combining emptiness and connectivity.

Usage:
    python -m alphatrain.scripts.crisis_player \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --seeds 0 1 2 3 14 --device mps
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import argparse
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model
from alphatrain.scripts.mine_death_features import board_features


def policy_move(net, device, game, fp16=False):
    obs_np = _build_obs_for_game(game)
    obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
    if fp16:
        obs = obs.half()
    with torch.inference_mode():
        pol_logits, _ = net(obs)
    pol_np = pol_logits[0].float().cpu().numpy()
    priors = _get_legal_priors_flat(game.board, pol_np, 30)
    if not priors:
        return None
    best = max(priors.items(), key=lambda x: x[1])[0]
    sr, sc = best // 81 // 9, best // 81 % 9
    tr, tc = best % 81 // 9, best % 81 % 9
    return (sr, sc), (tr, tc)


def detect_crisis(board):
    """Detect crisis level from board features.

    Returns: 'emergency', 'prevention', or None (healthy).
    """
    feats = board_features(board)
    avg_reach = feats[5]
    n_components = feats[1]
    empty = feats[0]

    if avg_reach < 5:
        return 'emergency'
    if avg_reach < 10 and n_components > 4:
        return 'prevention'
    if empty < 20 and n_components > 5:
        return 'emergency'
    return None


def play_game(net, device, mcts_prevention, mcts_emergency,
              seed, max_turns=5000, fp16=True, verbose=False,
              override_threshold=0.2):
    game = ColorLinesGame(seed=seed)
    game.reset()

    turn = 0
    policy_turns = 0
    prevention_turns = 0
    emergency_turns = 0
    overrides = 0
    t0 = time.time()

    while not game.game_over and turn < max_turns:
        crisis = detect_crisis(game.board)

        if crisis is not None:
            # Get policy move first
            pol_action = policy_move(net, device, game, fp16)
            if pol_action is None:
                break

            # Run MCTS
            mcts_obj = mcts_emergency if crisis == 'emergency' \
                else mcts_prevention
            mcts_result = mcts_obj.search(
                game, return_policy=True)
            if mcts_result[0] is None:
                action = pol_action
            else:
                mcts_action, visit_dist = mcts_result

                # Check if MCTS strongly prefers a different move
                pol_flat = (pol_action[0][0] * 9 + pol_action[0][1]) * 81 + \
                           (pol_action[1][0] * 9 + pol_action[1][1])
                mcts_flat = (mcts_action[0][0] * 9 + mcts_action[0][1]) * 81 + \
                            (mcts_action[1][0] * 9 + mcts_action[1][1])

                pol_visits = visit_dist[pol_flat]
                mcts_visits = visit_dist[mcts_flat]

                if mcts_flat != pol_flat and \
                        mcts_visits > pol_visits * (1 + override_threshold):
                    action = mcts_action
                    overrides += 1
                else:
                    action = pol_action

            if crisis == 'emergency':
                emergency_turns += 1
            else:
                prevention_turns += 1
        else:
            action = policy_move(net, device, game, fp16)
            policy_turns += 1

        if action is None:
            break
        result = game.move(action[0], action[1])
        if not result['valid']:
            break
        turn += 1

        if verbose and turn % 500 == 0:
            elapsed = time.time() - t0
            empty = int(np.sum(game.board == 0))
            print(f"    turn={turn} score={game.score} empty={empty} "
                  f"pol={policy_turns} prev={prevention_turns} "
                  f"emerg={emergency_turns} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'score': game.score,
        'turns': turn,
        'policy_turns': policy_turns,
        'prevention_turns': prevention_turns,
        'emergency_turns': emergency_turns,
        'overrides': overrides,
        'time': elapsed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--device', default='mps')
    p.add_argument('--prevention-sims', type=int, default=1600)
    p.add_argument('--emergency-sims', type=int, default=2400)
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--override-threshold', type=float, default=0.2,
                   help='Only override policy if MCTS top move has >X '
                        'more visits than policy move (0.2 = 20%%)')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(
        args.model, device, fp16=True, jit_trace=False)
    net.eval()

    mcts_prev = MCTS(net, device, max_score=max_score,
                     num_simulations=args.prevention_sims,
                     batch_size=args.batch_size, top_k=30, c_puct=2.5)
    mcts_emerg = MCTS(net, device, max_score=max_score,
                      num_simulations=args.emergency_sims,
                      batch_size=args.batch_size, top_k=30, c_puct=2.5)

    print(f"Crisis player: prevention={args.prevention_sims} sims, "
          f"emergency={args.emergency_sims} sims", flush=True)
    print(f"{'='*70}", flush=True)

    # First run policy-only baseline for all seeds
    print(f"\nPolicy-only baseline:", flush=True)
    pol_scores = {}
    for seed in args.seeds:
        game = ColorLinesGame(seed=seed)
        game.reset()
        while not game.game_over and game.turns < args.max_turns:
            action = policy_move(net, device, game, fp16=True)
            if action is None:
                break
            result = game.move(action[0], action[1])
            if not result['valid']:
                break
        pol_scores[seed] = game.score
        print(f"  seed={seed}: {game.score}", flush=True)

    print(f"\nCrisis player:", flush=True)
    results = []
    for seed in args.seeds:
        print(f"\n  Seed {seed} (policy baseline: {pol_scores[seed]}):",
              flush=True)
        r = play_game(net, device, mcts_prev, mcts_emerg,
                      seed, args.max_turns, fp16=True, verbose=True,
                      override_threshold=args.override_threshold)
        search_turns = r['prevention_turns'] + r['emergency_turns']
        total = r['turns']
        print(f"    score={r['score']} turns={total} "
              f"({r['time']:.0f}s, {r['time']/max(total,1)*1000:.0f}ms/turn)"
              f" | pol={r['policy_turns']} prev={r['prevention_turns']} "
              f"emerg={r['emergency_turns']} "
              f"overrides={r['overrides']} "
              f"({100*search_turns/max(total,1):.0f}% searched)",
              flush=True)
        results.append(r)

    print(f"\n{'='*70}", flush=True)
    print(f"\n{'Seed':>6} {'Pol':>7} {'Crisis':>7} {'Chg':>6} "
          f"{'Turns':>6} {'Time':>6} "
          f"{'Prev':>5} {'Emerg':>5} {'Over':>5} {'%Srch':>6}",
          flush=True)
    print(f"{'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*6}", flush=True)
    for r in results:
        total = r['turns']
        search = r['prevention_turns'] + r['emergency_turns']
        ps = pol_scores[r['seed']]
        cs = r['score']
        chg = (cs - ps) / max(ps, 1) * 100
        print(f"{r['seed']:>6} {ps:>7} {cs:>7} {chg:>+5.0f}% "
              f"{total:>6} {r['time']:>5.0f}s "
              f"{r['prevention_turns']:>5} {r['emergency_turns']:>5} "
              f"{r['overrides']:>5} "
              f"{100*search/max(total,1):>5.0f}%", flush=True)

    pol_list = [pol_scores[s] for s in args.seeds]
    crisis_list = [r['score'] for r in results]
    print(f"\nPolicy:  mean={np.mean(pol_list):.0f} "
          f"median={np.median(pol_list):.0f} "
          f"min={min(pol_list)} max={max(pol_list)}", flush=True)
    print(f"Crisis:  mean={np.mean(crisis_list):.0f} "
          f"median={np.median(crisis_list):.0f} "
          f"min={min(crisis_list)} max={max(crisis_list)}", flush=True)


if __name__ == '__main__':
    main()
