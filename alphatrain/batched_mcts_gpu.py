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

    # ---- root expansion (leaf eval on CPU; priors+dirichlet written into node 0) ----
    logits, val0, _ = _evaluate(net, dev, dtype, boards0, npos0, ncol0, nn0, fv)
    rng = np.random.default_rng(seed)
    for k in range(K):
        cnt, idx, pri = _legal_priors_jit(boards0_np[k].astype(np.int8), logits[k], top_k)
        cnt = int(cnt)
        if cnt == 0:
            continue
        pri = pri.astype(np.float64)
        if dirichlet_weight > 0 and dirichlet_alpha > 0:
            noise = rng.dirichlet([dirichlet_alpha] * cnt)
            pri = (1.0 - dirichlet_weight) * pri + dirichlet_weight * noise
        ch_action[k, 0, :cnt] = torch.from_numpy(idx[:cnt].astype(np.int16)).to(dev)
        ch_prior[k, 0, :cnt] = torch.from_numpy(pri[:cnt].astype(np.float16)).to(dev)
        n_children[k, 0] = cnt
    node_visits[:, 0] = 1.0
    node_vsum[:, 0] = torch.from_numpy(val0.astype(np.float32)).to(dev)
    min_q = torch.from_numpy(val0.astype(np.float32)).to(dev).clone()
    max_q = torch.from_numpy(val0.astype(np.float32)).to(dev).clone()
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
            if not bool(active.any()):
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

            if bool(move_mask.any()):
                rows = torch.nonzero(move_mask, as_tuple=True)[0]
                path_nodes[rows, plen[rows]] = cn[rows]
                path_slots[rows, plen[rows]] = best[rows]
                plen[rows] += 1

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

        # ---- evaluate leaves (CPU leaf eval), expand new-node leaves, backup ----
        logits, value, _ = _evaluate(net, dev, dtype, cur_board, cur_pos, cur_col, cur_n, fv,
                                     eval_net=(ablate != 'no_net'))
        cur_board_np = cur_board.to(torch.int8).cpu().numpy()
        ep = expand_parent.cpu().numpy(); es = expand_slot.cpu().numpy()
        term = terminal.cpu().numpy(); ln = leaf_node.cpu().numpy()
        nn_host = n_nodes.cpu().numpy()
        for k in range(K):
            if ep[k] >= 0 and not term[k]:
                new = int(nn_host[k]); nn_host[k] += 1
                ch_nodeid[k, int(ep[k]), int(es[k])] = new
                cnt, idx, pri = _legal_priors_jit(cur_board_np[k], logits[k], top_k)
                cnt = int(cnt)
                if cnt > 0:
                    ch_action[k, new, :cnt] = torch.from_numpy(idx[:cnt].astype(np.int16)).to(dev)
                    ch_prior[k, new, :cnt] = torch.from_numpy(pri[:cnt].astype(np.float16)).to(dev)
                    n_children[k, new] = cnt
                ln[k] = new
        n_nodes = torch.from_numpy(nn_host).to(dev)
        leaf_node = torch.from_numpy(ln).to(dev)

        # backup (vectorized over the path via scatter; per-step torch)
        value_t = torch.from_numpy(value.astype(np.float32)).to(dev)
        max_l = int(plen.max().item())
        for i in range(max_l):
            on = (i < plen)
            nd = path_nodes[:, i]; sl = path_slots[:, i]
            valid = on & (nd >= 0)
            vi = torch.where(valid, value_t, torch.zeros_like(value_t))
            node_visits[kar, nd.clamp(min=0)] += valid.float()
            node_vsum[kar, nd.clamp(min=0)] += vi
            ch_visits[kar, nd.clamp(min=0), sl.clamp(min=0)] += valid.float()
            ch_vsum[kar, nd.clamp(min=0), sl.clamp(min=0)] += vi
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
