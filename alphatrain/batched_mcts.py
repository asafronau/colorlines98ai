"""Array-based batched-tree MCTS: K independent open-loop searches at once.

One process advances K trees with a VECTORIZED descent and one batched GPU forward per
simulation (bs=1 per tree, no virtual loss — cleaner than scalar's bs=8). Edge-based tree
(mctx-style): child stats live on the parent edge; a child node id is allocated only when
that child is itself expanded. Engine primitives are vectorized (alphatrain.batched_engine);
leaf obs/legal-priors/feature-value reuse the scalar numba per-leaf (v1; vectorize later).

NOT bit-identical to scalar MCTS (no virtual loss + fp16 batch + independent spawn draws) —
validated by argmax-agreement + low TV of the root visit distribution (scripts/validate_batched_mcts.py).

See docs/batched_mcts_plan.md.
"""
import numpy as np
import torch

from alphatrain.observation import build_observation
from alphatrain.mcts import _legal_priors_jit, _evaluate_features_linear

from alphatrain import batched_engine as be

BOARD = 9
NACT = 6561


def _evaluate(net, dev, dtype, boards, npos, ncol, nn, coefs, means, stds, bias):
    """K leaf boards -> (pol_logits[K,6561], feature_value[K]). One batched forward."""
    K = boards.shape[0]
    obs = np.stack([
        build_observation(boards[k],
                          npos[k, :, 0].astype(np.intp), npos[k, :, 1].astype(np.intp),
                          ncol[k].astype(np.intp), int(nn[k]))
        for k in range(K)])
    with torch.no_grad():
        logits = net(torch.from_numpy(obs).to(dev, dtype)).float().cpu().numpy()
    vals = np.array([
        _evaluate_features_linear(boards[k],
                                  npos[k, :, 0].astype(np.int8), npos[k, :, 1].astype(np.int8),
                                  ncol[k].astype(np.int8), int(nn[k]),
                                  coefs, means, stds, bias)
        for k in range(K)], dtype=np.float64)
    return logits, vals


def batched_search(net, dev, dtype, boards0, next_pos0, next_col0, next_n0,
                   fv, rng, sims=4800, top_k=300, c_puct=2.5, q_weight=2.0,
                   dirichlet_alpha=0.3, dirichlet_weight=0.25, max_depth=96):
    """Return root child visit distribution [K, 6561] (normalized) for K trees."""
    coefs, means, stds, bias = fv
    K = boards0.shape[0]
    W = top_k
    N = sims + 2
    kar = np.arange(K)

    node_visits = np.zeros((K, N), dtype=np.float64)
    node_vsum = np.zeros((K, N), dtype=np.float64)
    ch_action = np.full((K, N, W), -1, dtype=np.int64)
    ch_prior = np.zeros((K, N, W), dtype=np.float64)
    ch_visits = np.zeros((K, N, W), dtype=np.float64)
    ch_vsum = np.zeros((K, N, W), dtype=np.float64)
    ch_nodeid = np.full((K, N, W), -1, dtype=np.int64)
    n_children = np.zeros((K, N), dtype=np.int64)
    n_nodes = np.ones(K, dtype=np.int64)        # node 0 = root

    # ---- root expansion ----
    logits, val0 = _evaluate(net, dev, dtype, boards0, next_pos0, next_col0, next_n0,
                             coefs, means, stds, bias)
    for k in range(K):
        cnt, idx, pri = _legal_priors_jit(boards0[k], logits[k], top_k)
        cnt = int(cnt)
        if cnt == 0:
            continue
        pri = pri.astype(np.float64)
        if dirichlet_weight > 0 and dirichlet_alpha > 0:
            noise = rng.dirichlet([dirichlet_alpha] * cnt)
            pri = (1.0 - dirichlet_weight) * pri + dirichlet_weight * noise
        ch_action[k, 0, :cnt] = idx[:cnt]
        ch_prior[k, 0, :cnt] = pri[:cnt]
        n_children[k, 0] = cnt
    node_visits[:, 0] = 1.0
    node_vsum[:, 0] = val0
    min_q = val0.copy()
    max_q = val0.copy()
    slot_idx = np.arange(W)[None, :]

    for _ in range(sims):
        cur_board = boards0.copy()
        cur_pos = next_pos0.copy(); cur_col = next_col0.copy(); cur_n = next_n0.copy()
        cur_node = np.zeros(K, dtype=np.int64)
        active = np.ones(K, dtype=bool)
        path_nodes = np.full((K, max_depth), -1, dtype=np.int64)
        path_slots = np.full((K, max_depth), -1, dtype=np.int64)
        plen = np.zeros(K, dtype=np.int64)
        leaf_node = np.full(K, -1, dtype=np.int64)
        expand_parent = np.full(K, -1, dtype=np.int64)
        expand_slot = np.full(K, -1, dtype=np.int64)
        terminal = np.zeros(K, dtype=bool)

        for _d in range(max_depth):
            if not active.any():
                break
            cn = cur_node
            nch = n_children[kar, cn]                      # [K]
            slot_valid = slot_idx < nch[:, None]           # [K,W]
            ch_act = ch_action[kar, cn]                    # [K,W]
            ch_pri = ch_prior[kar, cn]
            ch_vis = ch_visits[kar, cn]
            ch_vs = ch_vsum[kar, cn]
            ch_nid = ch_nodeid[kar, cn]
            sf = ch_act // 81; tf = ch_act % 81
            sr = np.clip(sf // 9, 0, 8); sc = np.clip(sf % 9, 0, 8)
            tr = np.clip(tf // 9, 0, 8); tc = np.clip(tf % 9, 0, 8)
            bsrc = cur_board[kar[:, None], sr, sc]
            btgt = cur_board[kar[:, None], tr, tc]
            occ = (bsrc != 0) & (btgt == 0)
            labels = be.label_components(cur_board)
            reach = be.reachable_many(labels, np.stack([sr, sc], -1), np.stack([tr, tc], -1))
            legal = slot_valid & occ & reach & active[:, None]

            nv = node_visits[kar, cn]
            sqrt_n = np.sqrt(nv)[:, None]
            qr = (max_q - min_q)
            qpos = ch_vis > 0
            q = np.where(qpos, ch_vs / np.where(qpos, ch_vis, 1.0), 0.0)
            q_norm = (q - min_q[:, None]) / np.where(qr > 0, qr, 1.0)[:, None]
            q_norm = np.where(qpos & (qr[:, None] > 0), q_norm, 0.5)
            u = c_puct * ch_pri * sqrt_n / (1.0 + ch_vis)
            score = np.where(legal, q_weight * q_norm + u, -1e30)
            best = np.argmax(score, axis=1)                # [K]
            has_legal = legal.any(axis=1)

            move_mask = active & has_legal
            no_legal = active & ~has_legal
            leaf_node[no_legal] = cur_node[no_legal]

            # record the chosen edge for moving trees
            mm = move_mask
            if mm.any():
                rows = np.where(mm)[0]
                path_nodes[rows, plen[rows]] = cn[rows]
                path_slots[rows, plen[rows]] = best[rows]
                plen[rows] += 1

            best_act = ch_act[kar, best]
            bsf = best_act // 81; btf = best_act % 81
            mv_src = np.stack([np.clip(bsf // 9, 0, 8), np.clip(bsf % 9, 0, 8)], -1)
            mv_tgt = np.stack([np.clip(btf // 9, 0, 8), np.clip(btf % 9, 0, 8)], -1)
            chosen_nid = ch_nid[kar, best]

            go, cur_pos, cur_col, cur_n = be.apply_move(
                cur_board, cur_pos, cur_col, cur_n, mv_src, mv_tgt, rng, active=mm)

            unexp = move_mask & (chosen_nid < 0)
            exp = move_mask & (chosen_nid >= 0)
            exp_term = exp & go
            exp_cont = exp & ~go
            expand_parent[unexp] = cur_node[unexp]
            expand_slot[unexp] = best[unexp]
            terminal[unexp & go] = True
            leaf_node[exp_term] = chosen_nid[exp_term]
            terminal[exp_term] = True
            cur_node = np.where(exp_cont, chosen_nid, cur_node)
            active = exp_cont
        # trees that hit max_depth still active: leaf = current node
        leaf_node[active] = cur_node[active]

        # ---- evaluate all leaves (cur_board is each tree's leaf board) ----
        logits, value = _evaluate(net, dev, dtype, cur_board, cur_pos, cur_col, cur_n,
                                  coefs, means, stds, bias)

        # ---- expand new-node (unexpanded, non-terminal) leaves ----
        for k in range(K):
            if expand_parent[k] >= 0 and not terminal[k]:
                new = n_nodes[k]; n_nodes[k] += 1
                ch_nodeid[k, expand_parent[k], expand_slot[k]] = new
                cnt, idx, pri = _legal_priors_jit(cur_board[k], logits[k], top_k)
                cnt = int(cnt)
                if cnt > 0:
                    ch_action[k, new, :cnt] = idx[:cnt]
                    ch_prior[k, new, :cnt] = pri[:cnt].astype(np.float64)
                    n_children[k, new] = cnt
                leaf_node[k] = new
            # terminal unexpanded leaf: value rides the leaf edge (already in path); no node

        # ---- backup ----
        for k in range(K):
            v = value[k]
            L = int(plen[k])
            for i in range(L):
                nd = path_nodes[k, i]; sl = path_slots[k, i]
                node_visits[k, nd] += 1.0; node_vsum[k, nd] += v
                ch_visits[k, nd, sl] += 1.0; ch_vsum[k, nd, sl] += v
            ln = leaf_node[k]
            if ln >= 0:
                node_visits[k, ln] += 1.0; node_vsum[k, ln] += v
            if v < min_q[k]:
                min_q[k] = v
            if v > max_q[k]:
                max_q[k] = v

    # ---- root visit distribution ----
    dist = np.zeros((K, NACT), dtype=np.float64)
    for k in range(K):
        cnt = int(n_children[k, 0])
        acts = ch_action[k, 0, :cnt]
        vis = ch_visits[k, 0, :cnt]
        tot = vis.sum()
        if tot > 0:
            dist[k, acts] = vis / tot
    return dist
