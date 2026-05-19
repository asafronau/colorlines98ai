"""Phase 0 — pre-screening: does the oracle ever beat policy top-1?

Uses the existing pairwise dataset (each stable pair = oracle prefers move X
over move Y at some anchor). Reconstructs the policy ranking of each pair's
winner and loser by re-running the policy on the anchor and matching
afterstate boards.

Reports:
  - distribution of winner ranks (1=policy top-1, ..., 4=top-4)
  - P(loser_rank < winner_rank)  i.e., policy ranked the WORSE move higher
  - P(loser = policy_top_1)       i.e., policy top-1 is the oracle loser
  - by metric (cap_rate vs turns) and source (crisis vs selfplay)

If P(loser = policy_top_1) is small (< 10%), policy is mostly at its ceiling.
If it's substantial (> 20%), oracle has new info -> Phase 1 is justified.

Usage:
    python -m alphatrain.scripts.phase0_oracle_screen \\
        --model alphatrain/data/pillar2z_epoch_19.pt \\
        --pairwise alphatrain/data/pairwise_v3_20260514_1754.pt \\
        --device mps
"""

import argparse
import hashlib
import numpy as np
import torch
import time


def board_hash(board_int8):
    return hashlib.md5(np.ascontiguousarray(board_int8, dtype=np.int8).tobytes()
                       ).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--pairwise', required=True)
    p.add_argument('--top-k', type=int, default=4,
                   help='Same top-K the miner used; pairs are guaranteed to be '
                        'within these K moves.')
    p.add_argument('--device', default='mps')
    args = p.parse_args()

    from alphatrain.evaluate import load_model
    from alphatrain.observation import build_observation
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from game.board import ColorLinesGame

    device = torch.device(args.device)
    fp16 = (args.device != 'cpu')

    print(f"Loading model {args.model}...", flush=True)
    net, _ = load_model(args.model, device, fp16=fp16, jit_trace=False)
    for pp in net.parameters():
        pp.requires_grad = False
    net.train(False)

    print(f"Loading pairwise {args.pairwise}...", flush=True)
    data = torch.load(args.pairwise, weights_only=False)
    N = data['anchor_boards'].shape[0]
    print(f"  pairs: {N:,}", flush=True)

    anchor_bs = data['anchor_boards'].numpy()
    anchor_np = data['anchor_next_pos'].numpy()
    anchor_nc = data['anchor_next_col'].numpy()
    anchor_nn = data['anchor_n_next'].numpy()

    win_bs = data['win_boards'].numpy()
    lose_bs = data['lose_boards'].numpy()
    metrics = data['metric_names']
    src_label_map = data.get('source_label_map', {})
    inv_src = {v: k for k, v in src_label_map.items()}
    sources = [inv_src.get(int(s), str(int(s)))
               for s in data['source_labels'].tolist()]

    # Dedupe anchors by board hash.
    print(f"Deduping anchors...", flush=True)
    anchor_key_to_idx = {}
    anchor_pair_indices = {}
    for i in range(N):
        k = board_hash(anchor_bs[i])
        if k not in anchor_key_to_idx:
            anchor_key_to_idx[k] = i
            anchor_pair_indices[k] = []
        anchor_pair_indices[k].append(i)
    n_unique = len(anchor_key_to_idx)
    print(f"  unique anchors: {n_unique:,} "
          f"(avg {N/n_unique:.1f} pairs/anchor)", flush=True)

    def policy_at(board, next_pos, next_col, n_next):
        obs = build_observation(
            np.asarray(board, dtype=np.int8),
            np.asarray(next_pos[:, 0], dtype=np.intp),
            np.asarray(next_pos[:, 1], dtype=np.intp),
            np.asarray(next_col, dtype=np.intp),
            int(n_next))
        x = torch.from_numpy(obs[None]).to(
            device, dtype=torch.float16 if fp16 else torch.float32)
        with torch.inference_mode():
            logits = net(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            pol = torch.softmax(logits.float()[0], dim=-1).cpu().numpy()
        return pol

    def apply_move(board, next_balls_list, move_action):
        game = ColorLinesGame(seed=0)
        game.reset(board=np.asarray(board, dtype=np.int8).copy(),
                   next_balls=list(next_balls_list))
        sr = move_action // 81 // 9
        sc = move_action // 81 % 9
        tr = move_action % 81 // 9
        tc = move_action % 81 % 9
        r = game.move((sr, sc), (tr, tc))
        if not r['valid']:
            return None
        return game.board.copy()

    print(f"\nResolving move ranks for {n_unique:,} anchors...", flush=True)
    t0 = time.time()
    anchor_move_map = {}
    for ai, (anchor_key, pair_i) in enumerate(anchor_pair_indices.items()):
        first = pair_i[0]
        a_np = anchor_np[first]
        a_nc = anchor_nc[first]
        a_nn = int(anchor_nn[first])
        a_board = anchor_bs[first]
        next_balls = [((int(a_np[k, 0]), int(a_np[k, 1])), int(a_nc[k]))
                       for k in range(a_nn)]
        pol = policy_at(a_board, a_np, a_nc, a_nn)
        priors = _get_legal_priors_flat(a_board.astype(np.int8), pol, args.top_k)
        if not priors:
            continue
        top = sorted(priors.items(), key=lambda x: x[1], reverse=True)
        move_map = {}
        for rank, (move, prior) in enumerate(top, start=1):
            after = apply_move(a_board, next_balls, move)
            if after is None:
                continue
            h = board_hash(after)
            move_map[h] = (move, rank, prior)
        anchor_move_map[anchor_key] = move_map
        if (ai + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (ai + 1) * (n_unique - ai - 1)
            print(f"  [{ai+1}/{n_unique}] {elapsed:.0f}s ETA {eta:.0f}s",
                  flush=True)
    print(f"  resolved in {time.time()-t0:.0f}s", flush=True)

    print(f"\nMatching pair afterstates to policy ranks...", flush=True)
    win_rank = np.full(N, -1, dtype=np.int8)
    lose_rank = np.full(N, -1, dtype=np.int8)
    matched = 0
    unmatched = 0
    for i in range(N):
        k = board_hash(anchor_bs[i])
        mmap = anchor_move_map.get(k)
        if mmap is None:
            unmatched += 1
            continue
        wh = board_hash(win_bs[i])
        lh = board_hash(lose_bs[i])
        w = mmap.get(wh)
        l = mmap.get(lh)
        if w is None or l is None:
            unmatched += 1
            continue
        win_rank[i] = w[1]
        lose_rank[i] = l[1]
        matched += 1
    print(f"  matched {matched}/{N}  (unmatched: {unmatched})", flush=True)

    if matched == 0:
        print("ERROR: no pairs matched. Check seeds / board hashing.",
              flush=True)
        return

    ok = win_rank > 0
    wr = win_rank[ok]
    lr = lose_rank[ok]

    print(f"\n=== PHASE 0 RESULTS ===", flush=True)
    print(f"Matched pairs: {matched:,}", flush=True)

    print(f"\nWinner rank distribution (policy rank of oracle winner):",
          flush=True)
    for r in range(1, args.top_k + 1):
        c = int((wr == r).sum())
        pct = 100 * c / matched
        print(f"  rank {r}: {c:5d} ({pct:5.1f}%)", flush=True)

    print(f"\nLoser rank distribution (policy rank of oracle loser):",
          flush=True)
    for r in range(1, args.top_k + 1):
        c = int((lr == r).sum())
        pct = 100 * c / matched
        print(f"  rank {r}: {c:5d} ({pct:5.1f}%)", flush=True)

    p_policy_top1_loses = int((lr == 1).sum())
    p_policy_top1_wins = int((wr == 1).sum())
    p_disagree = int((wr > lr).sum())
    p_agree = int((wr < lr).sum())

    print(f"\n=== KEY METRICS ===", flush=True)
    print(f"P(loser = policy_top_1):  "
          f"{p_policy_top1_loses}/{matched} = "
          f"{100*p_policy_top1_loses/matched:.1f}%   "
          f"<- policy's #1 is oracle-suboptimal", flush=True)
    print(f"P(winner = policy_top_1): "
          f"{p_policy_top1_wins}/{matched} = "
          f"{100*p_policy_top1_wins/matched:.1f}%   "
          f"<- policy and oracle agree on #1", flush=True)
    print(f"P(winner_rank > loser_rank, i.e., policy mis-ranks): "
          f"{p_disagree}/{matched} = "
          f"{100*p_disagree/matched:.1f}%", flush=True)
    print(f"P(winner_rank < loser_rank, i.e., policy agrees with oracle): "
          f"{p_agree}/{matched} = {100*p_agree/matched:.1f}%", flush=True)

    print(f"\n=== Per metric ===", flush=True)
    metric_arr = np.array([metrics[i] for i in range(N)])
    src_arr = np.array(sources)
    for m in sorted(set(metric_arr[ok])):
        mmask = (metric_arr[ok] == m)
        n_m = int(mmask.sum())
        if n_m == 0:
            continue
        wr_m = wr[mmask]; lr_m = lr[mmask]
        top1_lose = int((lr_m == 1).sum())
        top1_win = int((wr_m == 1).sum())
        disagree = int((wr_m > lr_m).sum())
        print(f"  {m}: n={n_m}  "
              f"P(loser=top1)={100*top1_lose/n_m:.1f}%  "
              f"P(winner=top1)={100*top1_win/n_m:.1f}%  "
              f"P(mis-rank)={100*disagree/n_m:.1f}%", flush=True)

    print(f"\n=== Per source ===", flush=True)
    for s in sorted(set(src_arr[ok])):
        smask = (src_arr[ok] == s)
        n_s = int(smask.sum())
        if n_s == 0:
            continue
        wr_s = wr[smask]; lr_s = lr[smask]
        top1_lose = int((lr_s == 1).sum())
        top1_win = int((wr_s == 1).sum())
        disagree = int((wr_s > lr_s).sum())
        print(f"  {s}: n={n_s}  "
              f"P(loser=top1)={100*top1_lose/n_s:.1f}%  "
              f"P(winner=top1)={100*top1_win/n_s:.1f}%  "
              f"P(mis-rank)={100*disagree/n_s:.1f}%", flush=True)

    print(f"\n=== INTERPRETATION ===", flush=True)
    top1_lose_pct = 100 * p_policy_top1_loses / matched
    if top1_lose_pct < 10:
        print(f"P(loser=top1) = {top1_lose_pct:.1f}% < 10%: "
              f"policy is mostly at ceiling -- oracle rarely disagrees on #1. "
              f"Root-oracle Phase 1 unlikely to find a learnable signal.",
              flush=True)
    elif top1_lose_pct < 20:
        print(f"P(loser=top1) = {top1_lose_pct:.1f}%: marginal. "
              f"Some room but inconclusive. Phase 1 worth running but "
              f"don't expect a huge win.", flush=True)
    else:
        print(f"P(loser=top1) = {top1_lose_pct:.1f}% >= 20%: "
              f"policy is suboptimal on its own top-1 in a meaningful "
              f"fraction of states. Run Phase 1 root-oracle to characterize "
              f"and commit to DAgger-style direct policy training.",
              flush=True)


if __name__ == '__main__':
    main()
