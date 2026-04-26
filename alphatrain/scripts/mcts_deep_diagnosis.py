"""Deep MCTS diagnosis: inspect PUCT components at every disagreement.

For each turn where MCTS overrides the policy, logs the full decision:
- Top-5 root children: prior P, visits N, raw Q, q_norm, U, PUCT score
- Value spread across all children (min/max/std of Q)
- Whether this is a crisis or healthy board
- q_range at time of decision

Usage:
    python -m alphatrain.scripts.mcts_deep_diagnosis \
        --model alphatrain/data/pillar2w2_epoch_10.pt \
        --value-net alphatrain/data/value_net.pt \
        --seed 1 --sims 400 --device mps
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'

import time
import argparse
import math
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors_flat
from alphatrain.evaluate import load_model


def fmt_action(flat):
    s, t = flat // 81, flat % 81
    return f"({s//9},{s%9})->({t//9},{t%9})"


def inspect_root(root, c_puct):
    """Extract PUCT components for all root children."""
    children = []
    sqrt_parent = math.sqrt(root.visit_count)

    # Compute q_range from children
    q_values = []
    for child in root.children.values():
        if child.visit_count > 0:
            q_values.append(child.value_sum / child.visit_count)
    if len(q_values) >= 2:
        min_q = min(q_values)
        max_q = max(q_values)
    elif len(q_values) == 1:
        min_q = max_q = q_values[0]
    else:
        min_q = max_q = 0.5
    q_range = max_q - min_q

    for action, child in root.children.items():
        vc = child.visit_count
        if vc > 0:
            q_raw = child.value_sum / vc
            q_norm = (q_raw - min_q) / q_range if q_range > 0 else 0.5
        else:
            q_raw = 0.0
            q_norm = 0.5
        u = c_puct * child.prior * sqrt_parent / (1 + vc)
        score = q_norm + u
        children.append({
            'action': action,
            'prior': child.prior,
            'visits': vc,
            'q_raw': q_raw,
            'q_norm': q_norm,
            'u': u,
            'score': score,
        })

    children.sort(key=lambda x: -x['visits'])
    return children, q_range, min_q, max_q


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--sims', type=int, default=400)
    p.add_argument('--device', default='mps')
    p.add_argument('--max-turns', type=int, default=5000)
    p.add_argument('--c-puct', type=float, default=2.5)
    p.add_argument('--value-net', default=None)
    p.add_argument('--max-disagree-log', type=int, default=20,
                   help='Max disagreements to log in detail')
    args = p.parse_args()

    device = torch.device(args.device)
    net, max_score = load_model(args.model, device, fp16=True, jit_trace=True)

    vnet = None
    if args.value_net:
        from alphatrain.model import ValueNet
        ckpt = torch.load(args.value_net, map_location='cpu', weights_only=False)
        vnet = ValueNet(in_channels=18,
                        num_blocks=ckpt['num_blocks'],
                        channels=ckpt['channels'],
                        num_value_bins=1)
        vnet.load_state_dict(ckpt['model'])
        vnet = vnet.to(device)
        vnet.requires_grad_(False)
        print(f"ValueNet: {ckpt['num_blocks']}b x {ckpt['channels']}ch, "
              f"acc={ckpt.get('accuracy', 0):.1f}%", flush=True)

    print(f"Model: max_score={max_score}, sims={args.sims}, "
          f"c_puct={args.c_puct}", flush=True)

    mcts = MCTS(net, device, max_score=max_score, num_simulations=args.sims,
                batch_size=8, top_k=30, c_puct=args.c_puct,
                value_net=vnet)

    game = ColorLinesGame(seed=args.seed)
    game.reset()

    t0 = time.time()
    turn = 0
    disagreements = 0
    logged = 0

    # Track stats by board regime
    healthy_agree = 0
    healthy_disagree = 0
    crisis_agree = 0
    crisis_disagree = 0
    q_ranges = []
    value_spreads = []  # std of Q across children

    while not game.game_over and turn < args.max_turns:
        # Get raw policy top move
        obs_np = _build_obs_for_game(game)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        if next(net.parameters()).dtype == torch.float16:
            obs = obs.half()
        with torch.inference_mode():
            pol_logits, _ = net(obs)
        pol_np = pol_logits[0].float().cpu().numpy()
        priors = _get_legal_priors_flat(game.board, pol_np, 30)
        if not priors:
            break

        policy_action = max(priors.items(), key=lambda x: x[1])[0]
        p_max = priors[policy_action]
        empty = int(np.sum(game.board == 0))

        # Run MCTS
        action = mcts.search(game, temperature=0.0,
                             dirichlet_alpha=0.0, dirichlet_weight=0.0)
        if action is None:
            break

        mcts_flat = (action[0][0] * 9 + action[0][1]) * 81 + \
                    (action[1][0] * 9 + action[1][1])

        # Inspect root
        root = mcts._last_root
        children_info, q_range, min_q, max_q = inspect_root(root, args.c_puct)
        q_ranges.append(q_range)

        # Value spread: std of Q across visited children
        visited_qs = [c['q_raw'] for c in children_info if c['visits'] > 0]
        v_spread = np.std(visited_qs) if len(visited_qs) > 1 else 0.0
        value_spreads.append(v_spread)

        is_disagree = (mcts_flat != policy_action)
        is_crisis = empty < 35

        if is_disagree:
            disagreements += 1
            if is_crisis:
                crisis_disagree += 1
            else:
                healthy_disagree += 1

            if logged < args.max_disagree_log:
                logged += 1
                print(f"\n  turn={turn} DISAGREE | empty={empty} "
                      f"{'CRISIS' if is_crisis else 'HEALTHY'} | "
                      f"q_range={q_range:.4f} min_q={min_q:.4f} "
                      f"max_q={max_q:.4f}", flush=True)
                print(f"  Policy top: {fmt_action(policy_action)} "
                      f"P={p_max:.3f}", flush=True)
                print(f"  {'Action':<20} {'P':>6} {'N':>6} {'Q_raw':>8} "
                      f"{'Q_norm':>7} {'U':>7} {'Score':>7}", flush=True)
                for c in children_info[:7]:
                    marker = ""
                    if c['action'] == policy_action:
                        marker = " <-- POLICY"
                    if c['action'] == mcts_flat:
                        marker = " <-- MCTS"
                    if c['action'] == policy_action == mcts_flat:
                        marker = " <-- BOTH"
                    print(f"  {fmt_action(c['action']):<20} "
                          f"{c['prior']:>6.3f} {c['visits']:>6} "
                          f"{c['q_raw']:>8.4f} {c['q_norm']:>7.4f} "
                          f"{c['u']:>7.4f} {c['score']:>7.4f}{marker}",
                          flush=True)
        else:
            if is_crisis:
                crisis_agree += 1
            else:
                healthy_agree += 1

        game.move(action[0], action[1])
        turn += 1

        if turn % 500 == 0:
            elapsed = time.time() - t0
            qr = np.array(q_ranges[-500:])
            vs = np.array(value_spreads[-500:])
            print(f"\n  turn={turn} score={game.score} | "
                  f"disagree={disagreements}/{turn} ({100*disagreements/turn:.0f}%) | "
                  f"q_range: mean={qr.mean():.4f} med={np.median(qr):.4f} | "
                  f"val_spread: mean={vs.mean():.4f} | "
                  f"{elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    qr = np.array(q_ranges)
    vs = np.array(value_spreads)

    print(f"\n{'='*60}", flush=True)
    print(f"DIAGNOSIS: seed={args.seed} score={game.score} "
          f"turns={turn} ({elapsed:.0f}s)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Disagreements: {disagreements}/{turn} "
          f"({100*disagreements/turn:.0f}%)", flush=True)
    print(f"  Healthy boards: {healthy_disagree} disagree / "
          f"{healthy_agree + healthy_disagree} total "
          f"({100*healthy_disagree/max(healthy_agree+healthy_disagree,1):.0f}%)",
          flush=True)
    print(f"  Crisis boards:  {crisis_disagree} disagree / "
          f"{crisis_agree + crisis_disagree} total "
          f"({100*crisis_disagree/max(crisis_agree+crisis_disagree,1):.0f}%)",
          flush=True)
    print(f"Q-range: mean={qr.mean():.4f} median={np.median(qr):.4f} "
          f"max={qr.max():.4f}", flush=True)
    print(f"Value spread (Q std): mean={vs.mean():.4f} "
          f"median={np.median(vs):.4f}", flush=True)


if __name__ == '__main__':
    main()
