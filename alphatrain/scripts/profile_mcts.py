"""Profile MCTS to find bottlenecks.

Usage:
    python -m alphatrain.scripts.profile_mcts
"""

import time
import torch
import numpy as np
from alphatrain.evaluate import load_model, play_game_verbose
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors
from game.board import ColorLinesGame


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    net, max_score = load_model('alphatrain/data/alphatrain_td_best.pt', device)

    game = ColorLinesGame(seed=42)
    game.reset()

    mcts = MCTS(net, device, max_score=max_score, num_simulations=400,
                c_puct=2.5, top_k=30)

    # Profile one search call (one move)
    print("\n=== Profiling one MCTS search (400 sims) ===", flush=True)

    # Warm up
    _ = mcts.search(game)
    game2 = ColorLinesGame(seed=42)
    game2.reset()

    # Detailed timing
    t_clone = 0
    t_nn = 0
    t_legal = 0
    t_obs = 0
    t_select = 0
    t_move = 0
    t_backup = 0
    n_nn = 0
    n_sims = 400

    from alphatrain.mcts import Node
    import math

    # Manual profiled search
    root = Node()
    t0 = time.perf_counter()

    # Root expansion
    ts = time.perf_counter()
    obs = torch.from_numpy(_build_obs_for_game(game2)).unsqueeze(0).to(device)
    t_obs += time.perf_counter() - ts

    ts = time.perf_counter()
    with torch.no_grad():
        pol_logits, val_logits = net(obs)
        root_value = net.predict_value(val_logits, max_val=max_score).item()
    t_nn += time.perf_counter() - ts
    n_nn += 1

    ts = time.perf_counter()
    priors = _get_legal_priors(game2, pol_logits[0].cpu().numpy(), 30)
    t_legal += time.perf_counter() - ts

    for action, prior in priors.items():
        root.children[action] = Node(prior=prior)
    root.visit_count = 1
    root.value_sum = root_value
    min_q = root_value
    max_q = root_value

    for sim in range(n_sims):
        node = root

        ts = time.perf_counter()
        sim_game = game2.clone()
        t_clone += time.perf_counter() - ts

        path = [node]

        # Selection
        while node.expanded() and not sim_game.game_over:
            ts = time.perf_counter()
            # Inline PUCT
            best_score = float('-inf')
            best_action = None
            best_child = None
            sqrt_parent = math.sqrt(node.visit_count)
            q_range = max_q - min_q
            for action, child in node.children.items():
                if child.visit_count > 0:
                    q = child.value_sum / child.visit_count
                    q_norm = (q - min_q) / q_range if q_range > 0 else 0.5
                else:
                    q_norm = 0.5
                u = 2.5 * child.prior * sqrt_parent / (1 + child.visit_count)
                score = q_norm + u
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child
            t_select += time.perf_counter() - ts

            ts = time.perf_counter()
            sim_game.move(best_action[0], best_action[1])
            t_move += time.perf_counter() - ts

            path.append(best_child)
            node = best_child

        # Leaf evaluation
        if sim_game.game_over:
            value = float(sim_game.score)
        else:
            ts = time.perf_counter()
            obs = torch.from_numpy(_build_obs_for_game(sim_game)).unsqueeze(0).to(device)
            t_obs += time.perf_counter() - ts

            ts = time.perf_counter()
            with torch.no_grad():
                pol_logits, val_logits = net(obs)
                value = net.predict_value(val_logits, max_val=max_score).item()
            t_nn += time.perf_counter() - ts
            n_nn += 1

            ts = time.perf_counter()
            priors = _get_legal_priors(sim_game, pol_logits[0].cpu().numpy(), 30)
            t_legal += time.perf_counter() - ts

            for act, prior in priors.items():
                node.children[act] = Node(prior=prior)

        min_q = min(min_q, value)
        max_q = max(max_q, value)

        ts = time.perf_counter()
        for n in path:
            n.visit_count += 1
            n.value_sum += value
        t_backup += time.perf_counter() - ts

    total = time.perf_counter() - t0
    print(f"\nTotal: {total*1000:.0f}ms for {n_sims} sims ({n_nn} NN evals)", flush=True)
    print(f"  NN forward:   {t_nn*1000:7.0f}ms ({t_nn/total*100:5.1f}%) "
          f"[{t_nn/n_nn*1000:.1f}ms/eval]", flush=True)
    print(f"  Obs build:    {t_obs*1000:7.0f}ms ({t_obs/total*100:5.1f}%)", flush=True)
    print(f"  Legal priors: {t_legal*1000:7.0f}ms ({t_legal/total*100:5.1f}%)", flush=True)
    print(f"  Game clone:   {t_clone*1000:7.0f}ms ({t_clone/total*100:5.1f}%)", flush=True)
    print(f"  Game move:    {t_move*1000:7.0f}ms ({t_move/total*100:5.1f}%)", flush=True)
    print(f"  PUCT select:  {t_select*1000:7.0f}ms ({t_select/total*100:5.1f}%)", flush=True)
    print(f"  Backup:       {t_backup*1000:7.0f}ms ({t_backup/total*100:5.1f}%)", flush=True)
    other = total - t_nn - t_obs - t_legal - t_clone - t_move - t_select - t_backup
    print(f"  Other:        {other*1000:7.0f}ms ({other/total*100:5.1f}%)", flush=True)

    # Estimate per-game time
    avg_turns = 300
    per_move = total
    print(f"\nEstimated per-game: {avg_turns * per_move / 60:.1f} min "
          f"({per_move:.1f}s/move × {avg_turns} turns)", flush=True)


if __name__ == '__main__':
    main()
