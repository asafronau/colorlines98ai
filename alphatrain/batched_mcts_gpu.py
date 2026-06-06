"""GPU batched-tree MCTS (torch). Tree arrays + vectorized PUCT descent on GPU using the
validated batched_engine_gpu primitives; leaf eval (obs / legal_priors / feature_value)
v1 = CPU per-leaf via transfer (called 1x/sim, not 1x/step) to validate the descent port
before vectorizing it in torch (Phase 2). Faithful torch translation of the numpy, 6/6-
validated alphatrain.batched_mcts.batched_search.

Validate: scripts/validate_batched_mcts.py --gpu (argmax + TV vs scalar).
"""
import numpy as np
import torch

from alphatrain.observation import build_observation
from alphatrain.mcts import _legal_priors_jit, _evaluate_features_linear
from alphatrain import batched_engine_gpu as beg

BOARD = 9
NACT = 6561


def _evaluate(net, dev, dtype, boards_t, npos_t, ncol_t, nn_t, fv, eval_net=True):
    """K leaf boards (torch) -> (pol_logits[K,6561] np, feature_value[K] np). Leaf eval on CPU
    (numba) via transfer; the NN forward runs on GPU. eval_net=False skips the forward (zeros
    logits) to isolate its cost in the --ablate no_net benchmark."""
    coefs, means, stds, bias = fv
    boards = boards_t.to(torch.int8).cpu().numpy()
    npos = npos_t.cpu().numpy(); ncol = ncol_t.cpu().numpy(); nn = nn_t.cpu().numpy()
    K = boards.shape[0]
    obs = np.stack([
        build_observation(boards[k], npos[k, :, 0].astype(np.intp), npos[k, :, 1].astype(np.intp),
                          ncol[k].astype(np.intp), int(nn[k])) for k in range(K)])
    with torch.no_grad():
        if eval_net:
            logits = net(torch.from_numpy(obs).to(dev, dtype)).float().cpu().numpy()
        else:
            logits = np.zeros((K, NACT), dtype=np.float32)
    vals = np.array([
        _evaluate_features_linear(boards[k], npos[k, :, 0].astype(np.int8),
                                  npos[k, :, 1].astype(np.int8), ncol[k].astype(np.int8),
                                  int(nn[k]), coefs, means, stds, bias)
        for k in range(K)], dtype=np.float64)
    return logits, vals, boards


from numba import njit as _njit


@_njit(cache=True)
def _feature_values_batch(boards, npos, ncol, nn, coefs, means, stds, bias):
    """Batched feature-value (one numba call over K leaves, no Python per-tree overhead).
    boards[K,9,9] int8, npos[K,3,2] int8, ncol[K,3] int8, nn[K] int. -> values[K] float64."""
    K = boards.shape[0]
    out = np.empty(K, dtype=np.float64)
    for k in range(K):
        out[k] = _evaluate_features_linear(boards[k], npos[k, :, 0], npos[k, :, 1],
                                           ncol[k], nn[k], coefs, means, stds, bias)
    return out


def _leaf_values_gpu(cur_board, cur_pos, cur_col, cur_n, fv, dev):
    """Leaf values on GPU [K] via one batched CPU feature eval (Stage-1 intermediate: the NN path
    is fully on-device, only the linear feature value still round-trips; port to GPU if it
    dominates the profile)."""
    coefs, means, stds, bias = fv
    boards = cur_board.to(torch.int8).cpu().numpy()
    npos = cur_pos.to(torch.int8).cpu().numpy()
    ncol = cur_col.to(torch.int8).cpu().numpy()
    nn = cur_n.to(torch.int32).cpu().numpy()
    vals = _feature_values_batch(boards, npos, ncol, nn,
                                 coefs.astype(np.float64), means.astype(np.float64),
                                 stds.astype(np.float64), float(bias))
    return torch.from_numpy(vals.astype(np.float32)).to(dev)


@torch.no_grad()
def batched_search_gpu(net, dev, dtype, boards0_np, npos0_np, ncol0_np, nn0_np,
                       fv, sims=4800, top_k=300, c_puct=2.5, q_weight=2.0,
                       dirichlet_alpha=0.3, dirichlet_weight=0.25, max_depth=96, seed=0,
                       ablate=None):
    """Return root child visit distribution [K, 6561] (numpy, normalized).

    ablate (perf isolation only — result is NOT a valid search):
      'descent_only' = prefill the tree with root children on every node and run sims x descent
        with NO leaf eval / expand / backup (hence zero per-sim CPU syncs). Isolates the pure GPU
        descent launch cost — the decisive test of the launch-bound hypothesis.
      'no_net' = full search but skip the NN forward at leaves (isolates the forward's cost)."""
    K = boards0_np.shape[0]
    W = top_k
    N = sims + 2
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    ti, tl, tb = torch.int64, torch.float32, torch.bool  # float32: MPS-safe + standard for MCTS
    t16, tp = torch.int16, torch.float16  # compact tree storage (cast up on use); halves [K,N,W] mem
    assert N < 32000, "ch_nodeid is int16; raise dtype if sims is huge"

    boards0 = torch.from_numpy(boards0_np.astype(np.int64)).to(dev)
    npos0 = torch.from_numpy(npos0_np.astype(np.int64)).to(dev)
    ncol0 = torch.from_numpy(ncol0_np.astype(np.int64)).to(dev)
    nn0 = torch.from_numpy(nn0_np.astype(np.int64)).to(dev)
    kar = torch.arange(K, device=dev)

    node_visits = torch.zeros((K, N), dtype=tl, device=dev)
    node_vsum = torch.zeros((K, N), dtype=tl, device=dev)
    ch_action = torch.full((K, N, W), -1, dtype=t16, device=dev)   # flat action <= 6560
    ch_prior = torch.zeros((K, N, W), dtype=tp, device=dev)         # prior in [0,1]
    ch_visits = torch.zeros((K, N, W), dtype=tl, device=dev)        # SUMS stay fp32 (precision)
    ch_vsum = torch.zeros((K, N, W), dtype=tl, device=dev)
    ch_nodeid = torch.full((K, N, W), -1, dtype=t16, device=dev)    # node id < N < 32000
    n_children = torch.zeros((K, N), dtype=ti, device=dev)
    n_nodes = torch.ones(K, dtype=ti, device=dev)

    # ---- root expansion (GPU obs+forward+legal_priors; vectorized Dirichlet into node 0) ----
    torch.manual_seed(seed)
    labels0 = beg.label_components_sv(boards0, iters=8)
    obs0 = beg.build_observation_t(boards0, npos0, ncol0, nn0, labels=labels0)
    logits0 = net(obs0.to(dtype)).float()
    cnt0, idx0, pri0 = beg.legal_priors_t(boards0, logits0, top_k, labels=labels0)
    if dirichlet_weight > 0 and dirichlet_alpha > 0:
        valid0 = idx0 >= 0                                          # [K,W] legal slots
        conc = torch.full((K, W), float(dirichlet_alpha))           # sample on CPU (mps lacks
        g = torch.distributions.Gamma(conc, torch.ones_like(conc)).sample().to(dev) * valid0.float()  # _standard_gamma); root-only
        noise = g / g.sum(dim=1, keepdim=True).clamp(min=1e-30)     # Dirichlet over legal slots
        pri0 = (1.0 - dirichlet_weight) * pri0 + dirichlet_weight * noise
    ch_action[:, 0] = idx0.to(t16)                                  # -1 pads preserved
    ch_prior[:, 0] = pri0.to(tp)
    n_children[:, 0] = cnt0.to(ti)
    val0_t = _leaf_values_gpu(boards0, npos0, ncol0, nn0, fv, dev)
    node_visits[:, 0] = 1.0
    node_vsum[:, 0] = val0_t
    min_q = val0_t.clone()
    max_q = val0_t.clone()
    slot_idx = torch.arange(W, device=dev).unsqueeze(0)

    if ablate == 'descent_only':
        # Replicate node-0's children onto every node and chain node ids so descent always has
        # somewhere to go; then run pure descent (no eval/expand/backup → no CPU sync) to isolate
        # the descent launch cost. The resulting visit dist is meaningless (backup never runs).
        ch_action[:] = ch_action[:, 0:1, :]
        ch_prior[:] = ch_prior[:, 0:1, :]
        n_children[:] = n_children[:, 0:1]
        nxt = torch.arange(N, device=dev, dtype=t16).clamp(max=N - 1)
        nxt[:-1] += 1                                    # node i's children point to node i+1
        ch_nodeid[:] = nxt.view(1, N, 1)
        node_visits[:] = 1.0                            # nonzero so PUCT sqrt term is live

    for _ in range(sims):
        cur_board = boards0.clone()
        cur_pos = npos0.clone(); cur_col = ncol0.clone(); cur_n = nn0.clone()
        cur_node = torch.zeros(K, dtype=ti, device=dev)
        active = torch.ones(K, dtype=tb, device=dev)
        path_nodes = torch.full((K, max_depth), -1, dtype=ti, device=dev)
        path_slots = torch.full((K, max_depth), -1, dtype=ti, device=dev)
        plen = torch.zeros(K, dtype=ti, device=dev)
        leaf_node = torch.full((K,), -1, dtype=ti, device=dev)
        expand_parent = torch.full((K,), -1, dtype=ti, device=dev)
        expand_slot = torch.full((K,), -1, dtype=ti, device=dev)
        terminal = torch.zeros(K, dtype=tb, device=dev)

        for _d in range(max_depth):
            if _d & 7 == 0 and _d and not bool(active.any()):  # coarse early-exit (1 sync / 8 steps)
                break
            cn = cur_node
            nch = n_children[kar, cn]
            slot_valid = slot_idx < nch.unsqueeze(1)
            ch_act = ch_action[kar, cn].long()        # int16 storage -> long for index/arith
            ch_pri = ch_prior[kar, cn].float()        # fp16 storage -> fp32 for PUCT math
            ch_vis = ch_visits[kar, cn]
            ch_vs = ch_vsum[kar, cn]
            ch_nid = ch_nodeid[kar, cn].long()
            sf = ch_act // 81; tf = ch_act % 81
            sr = (sf // 9).clamp(0, 8); sc = (sf % 9).clamp(0, 8)
            tr = (tf // 9).clamp(0, 8); tc = (tf % 9).clamp(0, 8)
            kk = kar.unsqueeze(1)
            occ = (cur_board[kk, sr, sc] != 0) & (cur_board[kk, tr, tc] == 0)
            labels = beg.label_components_sv(cur_board, iters=8)  # O(log) CC (converges ~6)
            reach = beg.reachable_many_t(labels, torch.stack([sr, sc], -1), torch.stack([tr, tc], -1))
            legal = slot_valid & occ & reach & active.unsqueeze(1)

            nv = node_visits[kar, cn]
            sqrt_n = torch.sqrt(nv).unsqueeze(1)
            qr = (max_q - min_q)
            qpos = ch_vis > 0
            q = torch.where(qpos, ch_vs / torch.where(qpos, ch_vis, torch.ones_like(ch_vis)),
                            torch.zeros_like(ch_vis))
            q_norm = (q - min_q.unsqueeze(1)) / torch.where(qr > 0, qr, torch.ones_like(qr)).unsqueeze(1)
            q_norm = torch.where(qpos & (qr.unsqueeze(1) > 0), q_norm,
                                 torch.full_like(q_norm, 0.5))
            u = c_puct * ch_pri * sqrt_n / (1.0 + ch_vis)
            score = torch.where(legal, q_weight * q_norm + u, torch.full_like(q_norm, -1e30))
            best = torch.argmax(score, dim=1)
            has_legal = legal.any(dim=1)

            move_mask = active & has_legal
            no_legal = active & ~has_legal
            leaf_node = torch.where(no_legal, cur_node, leaf_node)

            # record (node, slot) at depth plen for moving trees — full-K masked write (no nonzero)
            wp = plen.clamp(max=max_depth - 1)
            path_nodes[kar, wp] = torch.where(move_mask, cn, path_nodes[kar, wp])
            path_slots[kar, wp] = torch.where(move_mask, best, path_slots[kar, wp])
            plen = plen + move_mask.to(ti)

            best_act = ch_act[kar, best]
            bsf = best_act // 81; btf = best_act % 81
            mv_src = torch.stack([(bsf // 9).clamp(0, 8), (bsf % 9).clamp(0, 8)], -1)
            mv_tgt = torch.stack([(btf // 9).clamp(0, 8), (btf % 9).clamp(0, 8)], -1)
            chosen_nid = ch_nid[kar, best]

            go, cur_pos, cur_col, cur_n = beg.apply_move_t(
                cur_board, cur_pos, cur_col, cur_n, mv_src, mv_tgt, active=move_mask, generator=gen)

            unexp = move_mask & (chosen_nid < 0)
            exp = move_mask & (chosen_nid >= 0)
            exp_term = exp & go
            exp_cont = exp & ~go
            expand_parent = torch.where(unexp, cur_node, expand_parent)
            expand_slot = torch.where(unexp, best, expand_slot)
            terminal = terminal | (unexp & go) | exp_term
            leaf_node = torch.where(exp_term, chosen_nid, leaf_node)
            cur_node = torch.where(exp_cont, chosen_nid, cur_node)
            active = exp_cont
        leaf_node = torch.where(active, cur_node, leaf_node)

        if ablate == 'descent_only':
            continue                                    # skip eval/expand/backup (no CPU sync)

        # ---- leaf eval (obs+logits+priors on GPU; value via batched CPU feature eval),
        #      VECTORIZED expand, backup ----
        labels_leaf = beg.label_components_sv(cur_board, iters=8)
        if ablate != 'no_net':
            obs = beg.build_observation_t(cur_board, cur_pos, cur_col, cur_n, labels=labels_leaf)
            logits = net(obs.to(dtype)).float()
        else:
            logits = torch.zeros((K, NACT), device=dev)
        cnt_l, idx_l, pri_l = beg.legal_priors_t(cur_board, logits, top_k, labels=labels_leaf)
        value_t = _leaf_values_gpu(cur_board, cur_pos, cur_col, cur_n, fv, dev)

        # expand: every tree writes its children into its next free slot new_id for ALL K (harmless
        # scratch for non-expanders, overwritten next sim); only expanders bump n_nodes, link the
        # parent edge, and adopt the new node as leaf. No boolean indexing / no host sync.
        expand_mask = (expand_parent >= 0) & ~terminal
        new_id = n_nodes                                              # [K] next free node per tree
        ch_action[kar, new_id] = idx_l.to(t16)                        # -1 pads preserved
        ch_prior[kar, new_id] = pri_l.to(tp)
        n_children[kar, new_id] = cnt_l.to(ti)
        ep_c = expand_parent.clamp(min=0); es_c = expand_slot.clamp(min=0)
        cur_link = ch_nodeid[kar, ep_c, es_c]                         # non-expanders rewrite unchanged
        ch_nodeid[kar, ep_c, es_c] = torch.where(expand_mask, new_id.to(t16), cur_link)
        leaf_node = torch.where(expand_mask, new_id, leaf_node)
        n_nodes = n_nodes + expand_mask.to(ti)

        # backup: scatter-add the leaf value to every (node, edge) on each path in ONE shot (no
        # Python loop, no .item()). A path visits each node once, so no within-row scatter collision.
        depth_ar = torch.arange(max_depth, device=dev).unsqueeze(0)
        on = (depth_ar < plen.unsqueeze(1)) & (path_nodes >= 0)       # [K,max_depth] valid steps
        pn = path_nodes.clamp(min=0); ps = path_slots.clamp(min=0)
        vrow = value_t.unsqueeze(1)
        onv = on.float(); onval = torch.where(on, vrow, torch.zeros_like(vrow))
        node_visits.scatter_add_(1, pn, onv)
        node_vsum.scatter_add_(1, pn, onval)
        flat = pn * W + ps
        ch_visits.view(K, N * W).scatter_add_(1, flat, onv)
        ch_vsum.view(K, N * W).scatter_add_(1, flat, onval)
        lnv = leaf_node >= 0
        node_visits[kar, leaf_node.clamp(min=0)] += lnv.float()
        node_vsum[kar, leaf_node.clamp(min=0)] += torch.where(lnv, value_t, torch.zeros_like(value_t))
        min_q = torch.minimum(min_q, value_t)
        max_q = torch.maximum(max_q, value_t)

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
