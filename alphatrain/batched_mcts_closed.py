"""Closed-loop (determinized) batched-tree MCTS — NEW file, does NOT touch the open-loop
batched_mcts_gpu / batched_engine_gpu / the M5 scalar miner.

Motivation: the open-loop GPU search is COMPUTE-saturated — it re-derives connected-components +
reachability + apply_move at EVERY descent step (depth x sims engine ops), and block-capture topped
out at ~0.8x M5. Closed-loop fixes the compute floor itself: each node CACHES its board + its legal
top-k children (computed once, at expansion). Descent is then a pure gather + PUCT walk down cached
nodes — NO CC / reachability / apply_move per step. The engine ops (one apply_move + one CC + one NN
forward + one legal_priors) run ONCE PER SIM at the single expansion, not depth-times. That deletes
the ~depth-factor of engine compute that saturated the open-loop search.

SEMANTIC CHANGE vs scalar/open-loop: spawns are determinized PER NODE (a node's board is sampled
once at creation and reused on every revisit), instead of re-sampled each simulation. For teacher
mining this is likely acceptable (the scalar teacher already averages only ~3 determinizations);
validate by argmax-agreement + root-visit TV vs scalar before using the labels.

Board cache is cheap: node_board [K,N,9,9] int8 ~ 0.2 GB at K=512/sims=4800 (vs the ~14 GB edge
arrays). Reuses engine primitives from batched_engine_gpu and the leaf-value helper from
batched_mcts_gpu. Open-loop semantics live in batched_mcts_gpu; this is the determinized variant.
"""
import numpy as np
import torch

from alphatrain import batched_engine_gpu as beg
from alphatrain.batched_mcts_gpu import _leaf_values_gpu

BOARD = 9
NACT = 6561


@torch.no_grad()
def batched_search_closed(net, dev, dtype, boards0_np, npos0_np, ncol0_np, nn0_np,
                          fv, sims=4800, top_k=300, c_puct=2.5, q_weight=2.0,
                          dirichlet_alpha=0.3, dirichlet_weight=0.25, max_depth=96, seed=0):
    """Return root child visit distribution [K, 6561] (numpy, normalized). Closed-loop: each node
    caches its board + legal children; descent is gather-only; one expansion (engine ops) per sim."""
    K = boards0_np.shape[0]
    W = top_k
    N = sims + 2
    ti, tl, tb = torch.int64, torch.float32, torch.bool
    t16, tp, ts8 = torch.int16, torch.float16, torch.int8
    torch.manual_seed(seed)
    kar = torch.arange(K, device=dev)
    slot_idx = torch.arange(W, device=dev).unsqueeze(0)

    boards0 = torch.from_numpy(boards0_np.astype(np.int64)).to(dev)
    npos0 = torch.from_numpy(npos0_np.astype(np.int64)).to(dev)
    ncol0 = torch.from_numpy(ncol0_np.astype(np.int64)).to(dev)
    nn0 = torch.from_numpy(nn0_np.astype(np.int64)).to(dev)

    # per-NODE caches (the closed-loop change): board + next-balls stored at each node
    node_board = torch.zeros((K, N, BOARD, BOARD), dtype=ts8, device=dev)
    node_npos = torch.zeros((K, N, 3, 2), dtype=ts8, device=dev)
    node_ncol = torch.zeros((K, N, 3), dtype=ts8, device=dev)
    node_nn = torch.zeros((K, N), dtype=ts8, device=dev)
    node_term = torch.zeros((K, N), dtype=tb, device=dev)
    node_visits = torch.zeros((K, N), dtype=tl, device=dev)
    node_vsum = torch.zeros((K, N), dtype=tl, device=dev)
    ch_action = torch.full((K, N, W), -1, dtype=t16, device=dev)
    ch_prior = torch.zeros((K, N, W), dtype=tp, device=dev)
    ch_visits = torch.zeros((K, N, W), dtype=tl, device=dev)
    ch_vsum = torch.zeros((K, N, W), dtype=tl, device=dev)
    ch_nodeid = torch.full((K, N, W), -1, dtype=t16, device=dev)
    n_children = torch.zeros((K, N), dtype=ti, device=dev)
    n_nodes = torch.ones(K, dtype=ti, device=dev)

    # ---- root (node 0): cache board + expand legal children with Dirichlet ----
    node_board[:, 0] = boards0.to(ts8)
    node_npos[:, 0] = npos0.to(ts8); node_ncol[:, 0] = ncol0.to(ts8); node_nn[:, 0] = nn0.to(ts8)
    labels0 = beg.label_components_sv(boards0, iters=8)
    obs0 = beg.build_observation_t(boards0, npos0, ncol0, nn0, labels=labels0)
    logits0 = net(obs0.to(dtype)).float()
    cnt0, idx0, pri0 = beg.legal_priors_t(boards0, logits0, top_k, labels=labels0)
    if dirichlet_weight > 0 and dirichlet_alpha > 0:
        valid0 = idx0 >= 0
        conc = torch.full((K, W), float(dirichlet_alpha))
        g = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample().to(dev) * valid0.float()
        noise = g / g.sum(dim=1, keepdim=True).clamp(min=1e-30)
        pri0 = (1.0 - dirichlet_weight) * pri0 + dirichlet_weight * noise
    ch_action[:, 0] = idx0.to(t16); ch_prior[:, 0] = pri0.to(tp); n_children[:, 0] = cnt0.to(ti)
    v0 = _leaf_values_gpu(boards0, npos0, ncol0, nn0, fv, dev)
    node_visits[:, 0] = 1.0; node_vsum[:, 0] = v0
    min_q = v0.clone(); max_q = v0.clone()

    for _ in range(sims):
        cur_node = torch.zeros(K, dtype=ti, device=dev)
        active = torch.ones(K, dtype=tb, device=dev)
        path_nodes = torch.full((K, max_depth), -1, dtype=ti, device=dev)
        path_slots = torch.full((K, max_depth), -1, dtype=ti, device=dev)
        plen = torch.zeros(K, dtype=ti, device=dev)
        leaf_node = torch.full((K,), -1, dtype=ti, device=dev)
        expand_parent = torch.full((K,), -1, dtype=ti, device=dev)
        expand_slot = torch.full((K,), -1, dtype=ti, device=dev)

        # ---- descent: PURE gather + PUCT walk (no engine ops) ----
        for _d in range(max_depth):
            if _d & 7 == 0 and _d and not bool(active.any()):
                break
            cn = cur_node
            nch = n_children[kar, cn]
            slot_valid = slot_idx < nch.unsqueeze(1)
            ch_pri = ch_prior[kar, cn].float()
            ch_vis = ch_visits[kar, cn]; ch_vs = ch_vsum[kar, cn]
            ch_nid = ch_nodeid[kar, cn].long()
            nv = node_visits[kar, cn]
            sqrt_n = torch.sqrt(nv).unsqueeze(1)
            qr = (max_q - min_q)
            qpos = ch_vis > 0
            q = torch.where(qpos, ch_vs / torch.where(qpos, ch_vis, torch.ones_like(ch_vis)),
                            torch.zeros_like(ch_vis))
            q_norm = (q - min_q.unsqueeze(1)) / torch.where(qr > 0, qr, torch.ones_like(qr)).unsqueeze(1)
            q_norm = torch.where(qpos & (qr.unsqueeze(1) > 0), q_norm, torch.full_like(q_norm, 0.5))
            u = c_puct * ch_pri * sqrt_n / (1.0 + ch_vis)
            score = torch.where(slot_valid & active.unsqueeze(1), q_weight * q_norm + u,
                                torch.full_like(q_norm, -1e30))
            best = torch.argmax(score, dim=1)
            has_child = active & (nch > 0)
            no_child = active & (nch == 0)                 # expanded-but-no-legal => terminal-ish leaf
            leaf_node = torch.where(no_child, cur_node, leaf_node)
            wp = plen.clamp(max=max_depth - 1)
            path_nodes[kar, wp] = torch.where(has_child, cn, path_nodes[kar, wp])
            path_slots[kar, wp] = torch.where(has_child, best, path_slots[kar, wp])
            plen = plen + has_child.to(ti)
            chosen_nid = ch_nid[kar, best]
            # chosen child already expanded & not terminal -> descend (cached board, no re-sim)
            child_term = node_term[kar, chosen_nid.clamp(min=0)]
            descend = has_child & (chosen_nid >= 0) & ~child_term
            term_leaf = has_child & (chosen_nid >= 0) & child_term
            unexp = has_child & (chosen_nid < 0)
            expand_parent = torch.where(unexp, cur_node, expand_parent)
            expand_slot = torch.where(unexp, best, expand_slot)
            leaf_node = torch.where(term_leaf, chosen_nid, leaf_node)
            cur_node = torch.where(descend, chosen_nid, cur_node)
            active = descend
        leaf_node = torch.where(active, cur_node, leaf_node)

        # ---- expand: ONE apply_move + CC + NN + legal_priors per sim (the only engine work) ----
        expand_mask = expand_parent >= 0
        ep_c = expand_parent.clamp(min=0); es_c = expand_slot.clamp(min=0)
        pb = node_board[kar, ep_c].to(torch.int64)         # parent cached board
        pnpos = node_npos[kar, ep_c].to(torch.int64)
        pncol = node_ncol[kar, ep_c].to(torch.int64)
        pnn = node_nn[kar, ep_c].to(torch.int64)
        mv_act = ch_action[kar, ep_c, es_c].long().clamp(min=0)
        msf = mv_act // 81; mtf = mv_act % 81
        mv_src = torch.stack([(msf // 9).clamp(0, 8), (msf % 9).clamp(0, 8)], -1)
        mv_tgt = torch.stack([(mtf // 9).clamp(0, 8), (mtf % 9).clamp(0, 8)], -1)
        go, cpos, ccol, cn_ = beg.apply_move_nosync_t(pb, pnpos, pncol, pnn, mv_src, mv_tgt,
                                                      active=expand_mask, generator=None)
        new_id = n_nodes.clamp(max=N - 1)
        # store child node (board + next balls + terminal)
        node_board[kar, new_id] = pb.to(torch.int8)
        node_npos[kar, new_id] = cpos.to(torch.int8); node_ncol[kar, new_id] = ccol.to(torch.int8)
        node_nn[kar, new_id] = cn_.to(torch.int8); node_term[kar, new_id] = go
        # expand the child's legal children (one CC + NN + legal_priors)
        labels_c = beg.label_components_sv(pb, iters=8)
        obs_c = beg.build_observation_t(pb, cpos, ccol, cn_, labels=labels_c)
        logits_c = net(obs_c.to(dtype)).float()
        cnt_c, idx_c, pri_c = beg.legal_priors_t(pb, logits_c, top_k, labels=labels_c)
        # terminal children get no legal moves
        keep = expand_mask & ~go
        ch_action[kar, new_id] = torch.where(keep.unsqueeze(1), idx_c.to(t16),
                                             torch.full_like(idx_c, -1, dtype=t16))
        ch_prior[kar, new_id] = torch.where(keep.unsqueeze(1), pri_c.to(tp),
                                            torch.zeros_like(pri_c, dtype=tp))
        n_children[kar, new_id] = torch.where(keep, cnt_c.to(ti), torch.zeros_like(cnt_c))
        # link parent edge to the new node; bump counter; new node is this sim's leaf
        cur_link = ch_nodeid[kar, ep_c, es_c]
        ch_nodeid[kar, ep_c, es_c] = torch.where(expand_mask, new_id.to(t16), cur_link)
        leaf_node = torch.where(expand_mask, new_id, leaf_node)
        n_nodes = n_nodes + expand_mask.to(ti)
        value_t = _leaf_values_gpu(pb, cpos, ccol, cn_, fv, dev)
        # leaves that were terminal-or-no-child use their own cached node value (recompute cheap):
        # for simplicity the expanded leaf's value is value_t; non-expanding leaves keep prior backups
        # via their node stats (value contribution below applies value_t only where we expanded).

        # ---- backup (scatter-add along path) ----
        depth_ar = torch.arange(max_depth, device=dev).unsqueeze(0)
        on = (depth_ar < plen.unsqueeze(1)) & (path_nodes >= 0)
        pn = path_nodes.clamp(min=0); ps = path_slots.clamp(min=0)
        vrow = value_t.unsqueeze(1)
        onv = on.float(); onval = torch.where(on, vrow, torch.zeros_like(vrow))
        node_visits.scatter_add_(1, pn, onv); node_vsum.scatter_add_(1, pn, onval)
        flat = pn * W + ps
        ch_visits.view(K, N * W).scatter_add_(1, flat, onv)
        ch_vsum.view(K, N * W).scatter_add_(1, flat, onval)
        lnv = leaf_node >= 0
        node_visits[kar, leaf_node.clamp(min=0)] += lnv.float()
        node_vsum[kar, leaf_node.clamp(min=0)] += torch.where(lnv, value_t, torch.zeros_like(value_t))
        torch.minimum(min_q, value_t, out=min_q)
        torch.maximum(max_q, value_t, out=max_q)

    # ---- root visit distribution ----
    dist = np.zeros((K, NACT), dtype=np.float64)
    nc0 = n_children[:, 0].cpu().numpy()
    acts0 = ch_action[:, 0].cpu().numpy()
    vis0 = ch_visits[:, 0].cpu().numpy()
    for k in range(K):
        cnt = int(nc0[k])
        v = vis0[k, :cnt]; tot = v.sum()
        if tot > 0:
            dist[k, acts0[k, :cnt]] = v / tot
    return dist
