"""Profile MCTS worker with inference server to find remaining bottlenecks.

Usage:
    python -m alphatrain.scripts.profile_server_mcts
"""

import time
import numpy as np
import torch
from alphatrain.inference_server import InferenceServer
from alphatrain.mcts import MCTS, _build_obs_for_game, _get_legal_priors, Node, VIRTUAL_LOSS
from game.board import ColorLinesGame
import math


def main():
    model_path = 'alphatrain/data/alphatrain_td_best.pt'
    server = InferenceServer(model_path, num_workers=1, max_batch_per_worker=16)
    server.start()
    time.sleep(1)  # wait for GPU process to load model

    client = server.make_client(0)

    game = ColorLinesGame(seed=43)
    game.reset()

    # Profile one search (400 sims, batch=16)
    mcts = MCTS(inference_client=client, max_score=500.0,
                num_simulations=400, top_k=30, batch_size=16)

    # Warmup
    mcts.search(game)

    # Detailed profile of 5 searches
    n_searches = 5
    t_clone = t_move = t_select = t_obs = t_server = t_legal = t_backup = 0
    t_total = 0

    for _ in range(n_searches):
        root = Node()
        t0 = time.perf_counter()

        # Root expansion via server
        ts = time.perf_counter()
        obs_np = _build_obs_for_game(game)
        t_obs += time.perf_counter() - ts

        ts = time.perf_counter()
        pol_np, value = client.evaluate(obs_np)
        t_server += time.perf_counter() - ts

        ts = time.perf_counter()
        priors = _get_legal_priors(game, pol_np, 30)
        t_legal += time.perf_counter() - ts

        for a, p in priors.items():
            root.children[a] = Node(prior=p)
        root.visit_count = 1
        root.value_sum = value
        min_q = value
        max_q = value

        obs_np_buf = np.empty((16, 18, 9, 9), dtype=np.float32)
        sims_done = 0
        while sims_done < 400:
            bs = min(16, 400 - sims_done)
            batch_paths = []
            batch_games = []
            batch_leaf_nodes = []
            batch_game_over = []
            obs_count = 0

            for _ in range(bs):
                node = root

                ts = time.perf_counter()
                sim_game = game.clone()
                t_clone += time.perf_counter() - ts

                path = [node]
                while node.expanded() and not sim_game.game_over:
                    ts = time.perf_counter()
                    # PUCT
                    best_score = -1e18
                    best_action = best_child = None
                    sp = math.sqrt(node.visit_count)
                    qr = max_q - min_q
                    for a, c in node.children.items():
                        qn = ((c.value_sum/c.visit_count - min_q)/qr
                              if c.visit_count > 0 and qr > 0 else 0.5)
                        u = 2.5 * c.prior * sp / (1 + c.visit_count)
                        s = qn + u
                        if s > best_score:
                            best_score = s; best_action = a; best_child = c
                    t_select += time.perf_counter() - ts

                    ts = time.perf_counter()
                    sim_game.move(best_action[0], best_action[1])
                    t_move += time.perf_counter() - ts
                    path.append(best_child)
                    node = best_child

                for n in path:
                    n.visit_count += 1
                    n.value_sum -= VIRTUAL_LOSS

                batch_paths.append(path)
                batch_leaf_nodes.append(node)
                batch_games.append(sim_game)
                if sim_game.game_over:
                    batch_game_over.append(True)
                else:
                    batch_game_over.append(False)
                    ts = time.perf_counter()
                    obs_np_buf[obs_count] = _build_obs_for_game(sim_game)
                    t_obs += time.perf_counter() - ts
                    obs_count += 1

            # Server eval
            if obs_count > 0:
                ts = time.perf_counter()
                pol_np, val_np = client.evaluate_batch(obs_np_buf, obs_count)
                pol_np = pol_np.copy()
                val_np = val_np.copy()
                t_server += time.perf_counter() - ts

            nn_idx = 0
            for b in range(bs):
                path = batch_paths[b]
                if batch_game_over[b]:
                    value = float(batch_games[b].score)
                else:
                    value = float(val_np[nn_idx])
                    ts = time.perf_counter()
                    pr = _get_legal_priors(batch_games[b], pol_np[nn_idx], 30)
                    t_legal += time.perf_counter() - ts
                    for a, p in pr.items():
                        batch_leaf_nodes[b].children[a] = Node(prior=p)
                    nn_idx += 1
                min_q = min(min_q, value)
                max_q = max(max_q, value)
                ts = time.perf_counter()
                for n in path:
                    n.value_sum += VIRTUAL_LOSS + value
                t_backup += time.perf_counter() - ts

            sims_done += bs

        t_total += time.perf_counter() - t0

    # Report
    t_avg = t_total / n_searches
    print(f"\nProfile: {n_searches} searches, 400 sims each\n", flush=True)
    print(f"  Total:        {t_avg*1000:7.0f}ms/search", flush=True)
    print(f"  Server eval:  {t_server/n_searches*1000:7.0f}ms "
          f"({t_server/t_total*100:5.1f}%)", flush=True)
    print(f"  Obs build:    {t_obs/n_searches*1000:7.0f}ms "
          f"({t_obs/t_total*100:5.1f}%)", flush=True)
    print(f"  Legal priors: {t_legal/n_searches*1000:7.0f}ms "
          f"({t_legal/t_total*100:5.1f}%)", flush=True)
    print(f"  Game clone:   {t_clone/n_searches*1000:7.0f}ms "
          f"({t_clone/t_total*100:5.1f}%)", flush=True)
    print(f"  Game move:    {t_move/n_searches*1000:7.0f}ms "
          f"({t_move/t_total*100:5.1f}%)", flush=True)
    print(f"  PUCT select:  {t_select/n_searches*1000:7.0f}ms "
          f"({t_select/t_total*100:5.1f}%)", flush=True)
    print(f"  Backup:       {t_backup/n_searches*1000:7.0f}ms "
          f"({t_backup/t_total*100:5.1f}%)", flush=True)
    other = t_total - t_server - t_obs - t_legal - t_clone - t_move - t_select - t_backup
    print(f"  Other:        {other/n_searches*1000:7.0f}ms "
          f"({other/t_total*100:5.1f}%)", flush=True)

    server.shutdown()


if __name__ == '__main__':
    main()
