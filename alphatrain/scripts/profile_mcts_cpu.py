"""Profile MCTS CPU-side bottlenecks (clone, move, obs, select, legal priors).

Uses DummyNet to isolate CPU overhead from GPU inference latency.
This profiles the code that runs on worker processes in self-play.

Usage:
    python -m alphatrain.scripts.profile_mcts_cpu
"""

import time
import math
import numpy as np
import torch
from game.board import ColorLinesGame
from alphatrain.mcts import (
    MCTS, _build_obs_for_game, _get_legal_priors, _legal_priors_jit,
    Node, VIRTUAL_LOSS, NUM_MOVES
)
from alphatrain.observation import build_observation


class DummyNet:
    """Fast mock net for CPU-side profiling."""
    def __init__(self, value=500.0, num_value_bins=64):
        self.num_value_bins = num_value_bins
        self._value = value
        self._pol = torch.zeros(1, NUM_MOVES)
        self._val = torch.full((1, num_value_bins), 0.0)

    def __call__(self, obs):
        B = obs.shape[0]
        return self._pol.expand(B, -1), self._val.expand(B, -1)

    def predict_value(self, val_logits, max_val=30000.0):
        B = val_logits.shape[0]
        return torch.full((B,), self._value)

    def parameters(self):
        return iter([self._pol])

    def train(self, mode):
        return self


def profile_individual_ops(game, n_iters=1000):
    """Micro-benchmark each operation in the hot loop."""
    print("=== Micro-benchmarks (individual operations) ===\n", flush=True)

    # 1. game.clone()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        g = game.clone()
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  game.clone():           {elapsed:8.1f} us/call", flush=True)

    # 2. board.copy() alone
    t0 = time.perf_counter()
    for _ in range(n_iters):
        b = game.board.copy()
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  board.copy() alone:     {elapsed:8.1f} us/call", flush=True)

    # 3. np.random.default_rng()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        r = np.random.default_rng()
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  default_rng():          {elapsed:8.1f} us/call", flush=True)

    # 4. list(game.next_balls)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        nb = list(game.next_balls)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  list(next_balls):       {elapsed:8.1f} us/call", flush=True)

    # 5. _build_obs_for_game()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        obs = _build_obs_for_game(game)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  _build_obs_for_game():  {elapsed:8.1f} us/call", flush=True)

    # 6. build_observation() JIT alone
    nr = np.zeros(3, dtype=np.intp)
    nc = np.zeros(3, dtype=np.intp)
    ncol = np.zeros(3, dtype=np.intp)
    nn = min(len(game.next_balls), 3)
    for i, ((r, c), col) in enumerate(game.next_balls):
        if i >= 3:
            break
        nr[i], nc[i], ncol[i] = r, c, col
    t0 = time.perf_counter()
    for _ in range(n_iters):
        obs = build_observation(game.board, nr, nc, ncol, nn)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  build_observation():    {elapsed:8.1f} us/call", flush=True)

    # 7. _legal_priors_jit()
    pol = np.random.randn(6561).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        k, idx, pri = _legal_priors_jit(game.board, pol, 30)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  _legal_priors_jit():    {elapsed:8.1f} us/call", flush=True)

    # 8. _get_legal_priors() (JIT + Python dict conversion)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        priors = _get_legal_priors(game, pol, 30)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  _get_legal_priors():    {elapsed:8.1f} us/call", flush=True)

    # 9. game.move() on a clone
    moves = game.get_legal_moves()
    if moves:
        move = moves[0]
        t0 = time.perf_counter()
        for _ in range(n_iters):
            g = game.clone()
            g.move(move[0], move[1])
        elapsed_both = (time.perf_counter() - t0) / n_iters * 1e6
        # Subtract clone time
        t0 = time.perf_counter()
        for _ in range(n_iters):
            g = game.clone()
        clone_t = (time.perf_counter() - t0) / n_iters * 1e6
        print(f"  game.move() alone:      {elapsed_both - clone_t:8.1f} us/call", flush=True)

    # 10. PUCT selection (simulated)
    # Create mock node with 30 children
    node = Node()
    for i in range(30):
        child = Node(prior=1.0 / 30)
        child.visit_count = max(1, i)
        child.value_sum = float(i * 100)
        node.children[((i // 9, i % 9), (8, 8))] = child
    node.visit_count = 100
    min_q, max_q = 0.0, 3000.0

    t0 = time.perf_counter()
    for _ in range(n_iters):
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
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  PUCT select (30 ch):    {elapsed:8.1f} us/call", flush=True)

    # 11. Node creation
    t0 = time.perf_counter()
    for _ in range(n_iters):
        n = Node(prior=0.5)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  Node() creation:        {elapsed:8.1f} us/call", flush=True)

    # 12. Dict insertion (30 entries)
    priors_dict = {((i // 9, i % 9), (8, 8)): 1.0/30 for i in range(30)}
    t0 = time.perf_counter()
    for _ in range(n_iters):
        n = Node()
        for act, pri in priors_dict.items():
            n.children[act] = Node(prior=pri)
    elapsed = (time.perf_counter() - t0) / n_iters * 1e6
    print(f"  Expand node (30 ch):    {elapsed:8.1f} us/call", flush=True)

    print(flush=True)


def profile_full_search(game, n_searches=20, num_sims=400, batch_size=8):
    """Profile a full MCTS search with DummyNet, measuring each phase."""
    print(f"=== Full MCTS search profile ({num_sims} sims, bs={batch_size}) ===\n",
          flush=True)

    net = DummyNet()
    device = torch.device('cpu')
    mcts = MCTS(net, device, max_score=30000.0,
                num_simulations=num_sims, batch_size=batch_size,
                top_k=30, c_puct=2.5)

    # Warmup
    mcts.search(game)

    t_clone = t_move = t_select = t_obs = t_nn = t_legal = t_backup = 0
    t_expand = t_vl_add = t_vl_remove = 0
    t_total = 0
    n_obs_total = 0
    n_moves_total = 0
    n_selects_total = 0

    for s in range(n_searches):
        root = Node()
        t0 = time.perf_counter()

        # Root expansion
        ts = time.perf_counter()
        obs_np = _build_obs_for_game(game)
        t_obs += time.perf_counter() - ts

        ts = time.perf_counter()
        obs = torch.from_numpy(obs_np).unsqueeze(0)
        with torch.inference_mode():
            pol, val = net(obs)
            root_value = net.predict_value(val, max_val=30000.0).item()
        t_nn += time.perf_counter() - ts

        ts = time.perf_counter()
        priors = _get_legal_priors(game, pol[0].numpy(), 30)
        t_legal += time.perf_counter() - ts

        ts = time.perf_counter()
        for act, pri in priors.items():
            root.children[act] = Node(prior=pri)
        t_expand += time.perf_counter() - ts

        root.visit_count = 1
        root.value_sum = root_value
        min_q = root_value
        max_q = root_value

        sims_done = 0
        while sims_done < num_sims:
            bs = min(batch_size, num_sims - sims_done)
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
                    best_score = float('-inf')
                    best_action = best_child = None
                    sp = math.sqrt(node.visit_count)
                    qr = max_q - min_q
                    for a, c in node.children.items():
                        qn = ((c.value_sum / c.visit_count - min_q) / qr
                              if c.visit_count > 0 and qr > 0 else 0.5)
                        u = 2.5 * c.prior * sp / (1 + c.visit_count)
                        sc = qn + u
                        if sc > best_score:
                            best_score = sc
                            best_action = a
                            best_child = c
                    t_select += time.perf_counter() - ts
                    n_selects_total += 1

                    ts = time.perf_counter()
                    sim_game.move(best_action[0], best_action[1])
                    t_move += time.perf_counter() - ts
                    n_moves_total += 1

                    path.append(best_child)
                    node = best_child

                ts = time.perf_counter()
                for n in path:
                    n.visit_count += 1
                    n.value_sum -= VIRTUAL_LOSS
                t_vl_add += time.perf_counter() - ts

                batch_paths.append(path)
                batch_leaf_nodes.append(node)
                batch_games.append(sim_game)

                if sim_game.game_over:
                    batch_game_over.append(True)
                else:
                    batch_game_over.append(False)
                    ts = time.perf_counter()
                    obs = _build_obs_for_game(sim_game)
                    mcts._obs_buf[obs_count] = torch.from_numpy(obs)
                    t_obs += time.perf_counter() - ts
                    obs_count += 1
                    n_obs_total += 1

            # NN eval
            if obs_count > 0:
                ts = time.perf_counter()
                with torch.inference_mode():
                    pol_logits, val_logits = net(mcts._obs_buf[:obs_count])
                    values_t = net.predict_value(val_logits, max_val=30000.0)
                pol_np = pol_logits.numpy()
                val_np = values_t.numpy()
                t_nn += time.perf_counter() - ts

            nn_idx = 0
            for b_idx in range(bs):
                path = batch_paths[b_idx]
                if batch_game_over[b_idx]:
                    value = float(batch_games[b_idx].score)
                else:
                    value = float(val_np[nn_idx])
                    ts = time.perf_counter()
                    pr = _get_legal_priors(batch_games[b_idx], pol_np[nn_idx], 30)
                    t_legal += time.perf_counter() - ts

                    ts = time.perf_counter()
                    node = batch_leaf_nodes[b_idx]
                    for act, p in pr.items():
                        node.children[act] = Node(prior=p)
                    t_expand += time.perf_counter() - ts
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
    ns = n_searches
    t_avg = t_total / ns * 1000
    print(f"  Total:        {t_avg:7.1f} ms/search ({n_searches} searches avg)", flush=True)
    print(f"  Game clone:   {t_clone/ns*1000:7.1f} ms ({t_clone/t_total*100:5.1f}%) "
          f"[{t_clone/ns/num_sims*1e6:.1f} us/call x {num_sims}]", flush=True)
    print(f"  Game move:    {t_move/ns*1000:7.1f} ms ({t_move/t_total*100:5.1f}%) "
          f"[{t_move/max(n_moves_total,1)*1e6:.1f} us/call x {n_moves_total/ns:.0f}]", flush=True)
    print(f"  PUCT select:  {t_select/ns*1000:7.1f} ms ({t_select/t_total*100:5.1f}%) "
          f"[{t_select/max(n_selects_total,1)*1e6:.1f} us/call x {n_selects_total/ns:.0f}]", flush=True)
    print(f"  Obs build:    {t_obs/ns*1000:7.1f} ms ({t_obs/t_total*100:5.1f}%) "
          f"[{t_obs/max(n_obs_total+ns,1)*1e6:.1f} us/call x {(n_obs_total+ns)/ns:.0f}]", flush=True)
    print(f"  NN forward:   {t_nn/ns*1000:7.1f} ms ({t_nn/t_total*100:5.1f}%) [DummyNet]", flush=True)
    print(f"  Legal priors: {t_legal/ns*1000:7.1f} ms ({t_legal/t_total*100:5.1f}%)", flush=True)
    print(f"  Node expand:  {t_expand/ns*1000:7.1f} ms ({t_expand/t_total*100:5.1f}%)", flush=True)
    print(f"  VL add:       {t_vl_add/ns*1000:7.1f} ms ({t_vl_add/t_total*100:5.1f}%)", flush=True)
    print(f"  Backup:       {t_backup/ns*1000:7.1f} ms ({t_backup/t_total*100:5.1f}%)", flush=True)
    other = t_total - t_clone - t_move - t_select - t_obs - t_nn - t_legal - t_expand - t_vl_add - t_backup
    print(f"  Other:        {other/ns*1000:7.1f} ms ({other/t_total*100:5.1f}%)", flush=True)
    print(flush=True)


def main():
    print("Warming up Numba JIT...", flush=True)
    game = ColorLinesGame(seed=42)
    game.reset()
    # Warmup JIT
    _ = _build_obs_for_game(game)
    _ = _legal_priors_jit(game.board, np.zeros(6561, dtype=np.float32), 30)
    _ = _get_legal_priors(game, np.zeros(6561, dtype=np.float32), 30)

    print("JIT warm, starting profile...\n", flush=True)

    profile_individual_ops(game, n_iters=5000)
    profile_full_search(game, n_searches=20, num_sims=400, batch_size=8)
    profile_full_search(game, n_searches=20, num_sims=400, batch_size=16)


if __name__ == '__main__':
    main()
