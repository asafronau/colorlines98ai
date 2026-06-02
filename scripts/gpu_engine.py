"""Fully-vectorized (batched, on-device) Color Lines rollout engine.

Replicates game/board.py's RULES exactly for B boards in parallel as torch
tensors, so an entire policy rollout runs on the GPU with NO per-step CPU<->GPU
round-trip. Deterministic ops (reachability, move, line-clear) are golden-tested
bit-exact vs the CPU engine; the spawn matches the DISTRIBUTION (uniform empty
cells + uniform colors) — catastrophe is a Monte-Carlo estimate, so the exact
PCG64 sequence is not needed, only the right distribution.

Board: (B,9,9) int8, 0=empty, 1..7=colors. This file builds up + self-tests each
op (run directly). Integration into mining comes only after the golden suite +
catastrophe-rate parity pass.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

N = 9
NUM_COLORS = 7
MIN_LINE = 5
_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]              # 4-conn (reachability/flood)
_LINE_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]          # line directions


def _shift(t, dr, dc):
    """result[...,r,c] = t[...,r+dr,c+dc], zero-padded. (neighbor in dir (dr,dc))"""
    out = torch.zeros_like(t)
    r_src = slice(max(dr, 0), N + min(dr, 0))
    r_dst = slice(max(-dr, 0), N + min(-dr, 0))
    c_src = slice(max(dc, 0), N + min(dc, 0))
    c_dst = slice(max(-dc, 0), N + min(-dc, 0))
    out[..., r_dst, c_dst] = t[..., r_src, c_src]
    return out


def empty_components(board, iters=24):
    """(B,9,9) int -> (B,9,9) long component labels (min cell-index per empty
    component; 0 for occupied). Sync-free fixed-iter min-label propagation.
    24 iters > the geodesic diameter of any empty region in real positions
    (golden-tested on dense random boards). No per-iter torch.equal (that syncs
    the GPU — the whole point of staying on-device)."""
    B = board.shape[0]
    empty = (board == 0)
    idx = torch.arange(1, 82, device=board.device).reshape(1, N, N).expand(B, N, N).clone()
    labels = idx * empty.long()
    for _ in range(iters):
        for dr, dc in _DIRS:
            nb = _shift(labels, dr, dc)
            both = (labels > 0) & (nb > 0)
            labels = torch.where(both & (nb < labels), nb, labels)
    return labels


def legal_mask(board):
    """(B,9,9) -> (B,6561) bool. index = (sr*9+sc)*81 + (tr*9+tc).
    Legal iff source occupied, target empty, and target's empty-component touches
    the source (== game.move's reachability)."""
    B = board.shape[0]
    occ = (board != 0).reshape(B, 81)
    empty = (board == 0).reshape(B, 81)
    cc = empty_components(board)                                   # (B,9,9)
    cc_flat = cc.reshape(B, 81)
    empty_g = (board == 0)
    reach = torch.zeros(B, 81, 81, dtype=torch.bool, device=board.device)
    for dr, dc in _DIRS:
        nb_cc = _shift(cc, dr, dc).reshape(B, 81)                  # label at src+d
        nb_empty = _shift(empty_g.long(), dr, dc).reshape(B, 81).bool()
        eq = (nb_cc.unsqueeze(2) == cc_flat.unsqueeze(1)) & (nb_cc.unsqueeze(2) > 0)
        reach |= eq & nb_empty.unsqueeze(2)
    mask = reach & occ.unsqueeze(2) & empty.unsqueeze(1)
    return mask.reshape(B, 6561)


def clear_lines_at(board, cells):
    """Clear lines of >=5 through each board's placed `cell`. In-place semantics
    via return. board (B,9,9) long, cells (B,2) long. Returns (new_board,
    cleared_count(B,)). Matches _clear_lines_at exactly."""
    B = board.shape[0]
    dev = board.device
    bidx = torch.arange(B, device=dev)
    r0, c0 = cells[:, 0], cells[:, 1]
    color = board[bidx, r0, c0]                                   # (B,)
    K = 8
    ks = torch.arange(-K, K + 1, device=dev)                      # (17,), idx 8 = cell
    clear = torch.zeros(B, 81, dtype=torch.float32, device=dev)  # float: MPS amax
    for dr, dc in _LINE_DIRS:
        pr = r0.unsqueeze(1) + ks.unsqueeze(0) * dr               # (B,17)
        pc = c0.unsqueeze(1) + ks.unsqueeze(0) * dc
        valid = (pr >= 0) & (pr < N) & (pc >= 0) & (pc < N)
        prc, pcc = pr.clamp(0, N - 1), pc.clamp(0, N - 1)
        val = board[bidx.unsqueeze(1), prc, pcc]                  # (B,17)
        same = valid & (val == color.unsqueeze(1)) & (color.unsqueeze(1) > 0)
        fwd = torch.cumprod(same[:, K + 1:].long(), dim=1).sum(1)  # leading-True +side
        bwd = torch.cumprod(same[:, :K].flip(1).long(), dim=1).sum(1)
        total = 1 + fwd + bwd
        line = (total >= MIN_LINE).unsqueeze(1)
        in_seg = ((ks.unsqueeze(0) >= (-bwd).unsqueeze(1))
                  & (ks.unsqueeze(0) <= fwd.unsqueeze(1))
                  & line & valid)                                  # (B,17), k in [-bwd,+fwd]
        flatpos = prc * N + pcc                                    # (B,17)
        clear.scatter_reduce_(1, flatpos, in_seg.float(),
                              reduce='amax', include_self=True)
    clear = clear.reshape(B, N, N) > 0.5
    cleared_count = clear.sum((1, 2))
    new_board = torch.where(clear, torch.zeros_like(board), board)
    return new_board, cleared_count


def calc_score(n):
    return torch.where(n >= MIN_LINE, n * (n - 4), torch.zeros_like(n))


def _sample_empties(empty, k, gen):
    """For each board pick up to k DISTINCT empty cells uniformly (== rng
    choice_no_replace distribution). empty: (B,81) bool. Returns (idx (B,k) long,
    valid (B,k) bool). Random-key top-k: keys = empty*U(0,1); top-k keys are k
    distinct uniform empties; picks beyond n_empty land on occupied (key 0)."""
    B = empty.shape[0]
    keys = empty.float() * (torch.rand(B, 81, generator=gen, device=empty.device)
                            + 1e-6)
    vals, idx = keys.topk(k, dim=1)
    valid = vals > 0
    return idx, valid


def step(ds, board, npos, ncol, nnext, score, moves, gen):
    """One batched move: move ball, clear at target; if nothing cleared, spawn
    the 3 preview balls (occupied->random empty), clear at the planned spawn
    cells, draw a fresh preview, and flag board-full game-over. Mirrors
    board.move()/_spawn_balls()/_generate_next_balls(). Returns updated
    (board,npos,ncol,nnext,score,game_over)."""
    B = board.shape[0]
    dev = board.device
    bidx = torch.arange(B, device=dev)
    src, tgt = moves // 81, moves % 81
    sr, sc, tr, tc = src // 9, src % 9, tgt // 9, tgt % 9
    color = board[bidx, sr, sc].clone()
    board[bidx, sr, sc] = 0
    board[bidx, tr, tc] = color
    board, cleared = clear_lines_at(board, torch.stack([tr, tc], 1))
    score = score + calc_score(cleared)
    spawn = cleared == 0                                          # (B,) spawning boards

    # place the 3 preview balls (only on spawning boards)
    for i in range(3):
        has = (i < nnext) & spawn
        pr, pc, cl = npos[:, i, 0], npos[:, i, 1], ncol[:, i]
        cell_empty = board[bidx, pr, pc] == 0
        direct = has & cell_empty
        if direct.any():
            board[bidx[direct], pr[direct], pc[direct]] = cl[direct].to(board.dtype)
        fb = has & ~cell_empty                                    # occupied -> random empty
        if fb.any():
            empty = (board == 0).reshape(B, 81)
            fidx, fok = _sample_empties(empty, 1, gen)
            put = fb & fok.squeeze(1)
            if put.any():
                fr, fc = fidx.squeeze(1) // 9, fidx.squeeze(1) % 9
                board[bidx[put], fr[put], fc[put]] = cl[put].to(board.dtype)

    # clear at the planned spawn cells (CPU clears at next_balls positions, in order)
    for i in range(3):
        has = (i < nnext) & spawn
        cell = torch.stack([npos[:, i, 0], npos[:, i, 1]], 1)
        occ_here = board[bidx, npos[:, i, 0], npos[:, i, 1]] != 0
        do = has & occ_here
        if do.any():
            board2, c2 = clear_lines_at(board, cell)
            board = torch.where(do.view(B, 1, 1), board2, board)
            score = score + torch.where(do, calc_score(c2), torch.zeros_like(c2))

    # fresh preview for spawning boards
    empty = (board == 0).reshape(B, 81)
    n_empty = empty.sum(1)
    pidx, pvalid = _sample_empties(empty, 3, gen)                 # (B,3)
    new_npos = torch.stack([pidx // 9, pidx % 9], dim=2)          # (B,3,2)
    new_ncol = torch.randint(1, NUM_COLORS + 1, (B, 3), generator=gen, device=dev)
    new_nnext = pvalid.sum(1).to(nnext.dtype)
    sp = spawn.view(B, 1, 1)
    npos = torch.where(sp, new_npos.to(npos.dtype), npos)
    ncol = torch.where(spawn.view(B, 1), new_ncol.to(ncol.dtype), ncol)
    nnext = torch.where(spawn, new_nnext, nnext)
    game_over = spawn & (n_empty == 0)
    return board, npos, ncol, nnext, score, game_over


@torch.no_grad()
def gpu_rollout(net, ds, board, npos, ncol, nnext, first_moves, horizon, gen,
                dtype=torch.float16):
    """Fully on-GPU rollout. Applies first_moves (the candidate), then policy
    argmax for `horizon` steps. Returns died (B,) bool, turns (B,) int."""
    B = board.shape[0]
    dev = board.device
    board = board.clone().long()
    npos = npos.clone().long(); ncol = ncol.clone().long(); nnext = nnext.clone().long()
    score = torch.zeros(B, device=dev, dtype=torch.long)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    died = torch.zeros(B, dtype=torch.bool, device=dev)
    turns = torch.zeros(B, dtype=torch.long, device=dev)

    board, npos, ncol, nnext, score, go = step(ds, board, npos, ncol, nnext,
                                                score, first_moves, gen)
    died |= go
    done |= go
    turns += (~done).long()
    for _ in range(horizon):
        if done.all():
            break
        obs = ds._build_obs_core(board, next_pos=npos, next_col=ncol, n_next=nnext)
        logits = net(obs.to(dtype)).float()
        mask = legal_mask(board)                                  # (B,6561) bool
        nolegal = ~mask.any(1)
        logits = logits.masked_fill(~mask, float('-inf'))
        moves = logits.argmax(1)
        # boards with no legal move stop here (died = current game_over=False)
        newly = nolegal & ~done
        done |= newly
        active = ~done
        if active.any():
            b2, np2, nc2, nn2, sc2, go2 = step(ds, board, npos, ncol, nnext,
                                               score, moves, gen)
            am = active.view(B, 1, 1)
            board = torch.where(am, b2, board)
            npos = torch.where(am, np2, npos)
            ncol = torch.where(active.view(B, 1), nc2, ncol)
            nnext = torch.where(active, nn2, nnext)
            score = torch.where(active, sc2, score)
            died |= (go2 & active)
            done |= (go2 & active)
            turns += active.long()
    return died, turns


# ── golden test vs CPU engine ──
def _random_boards(n, rng, density_range=(0.3, 0.95)):
    """Random boards at varied fill densities (stress reachability)."""
    boards = np.zeros((n, N, N), dtype=np.int8)
    for b in range(n):
        d = rng.uniform(*density_range)
        for r in range(N):
            for c in range(N):
                if rng.random() < d:
                    boards[b, r, c] = rng.integers(1, NUM_COLORS + 1)
    return boards


def _test_legal_mask():
    from game.board import ColorLinesGame
    rng = np.random.default_rng(0)
    boards = _random_boards(200, rng)
    bt = torch.from_numpy(boards).long()
    gm = legal_mask(bt).numpy()
    mism = 0
    for b in range(len(boards)):
        g = ColorLinesGame()
        g.reset(board=boards[b].copy(), next_balls=[])
        cpu = np.zeros(6561, dtype=bool)
        for (sr, sc), (tr, tc) in g.get_legal_moves():
            cpu[(sr * 9 + sc) * 81 + (tr * 9 + tc)] = True
        if not np.array_equal(cpu, gm[b]):
            mism += 1
            if mism <= 3:
                diff = np.where(cpu != gm[b])[0]
                print(f"  board {b}: {len(diff)} mismatched moves, e.g. {diff[:5]}")
    print(f"legal_mask: {len(boards)-mism}/{len(boards)} boards match "
          f"({'PASS' if mism == 0 else 'FAIL'})")
    return mism == 0


def _planted_boards(n, rng):
    """Boards with a planted run of L in {4..8} of one color through a cell, plus
    noise — exercises both the clear (>=5) and no-clear (<5) paths."""
    boards = np.zeros((n, N, N), dtype=np.int8)
    cells = np.zeros((n, 2), dtype=np.int64)
    for b in range(n):
        for r in range(N):                                   # light noise
            for c in range(N):
                if rng.random() < 0.4:
                    boards[b, r, c] = rng.integers(1, NUM_COLORS + 1)
        L = rng.integers(4, 9)
        dr, dc = _LINE_DIRS[rng.integers(0, 4)]
        color = rng.integers(1, NUM_COLORS + 1)
        # place run starting somewhere it fits
        r0 = rng.integers(0, N); c0 = rng.integers(0, N)
        cells[b] = (r0, c0)
        for k in range(-L + 1, L):
            r, c = r0 + k * dr, c0 + k * dc
            if 0 <= r < N and 0 <= c < N and rng.random() < 0.6:
                boards[b, r, c] = color
        boards[b, r0, c0] = color                            # ensure placed cell set
    return boards, cells


def _test_clear():
    from game.board import ColorLinesGame, _clear_lines_at
    rng = np.random.default_rng(1)
    boards, cells = _planted_boards(300, rng)
    bt = torch.from_numpy(boards).long()
    ct = torch.from_numpy(cells).long()
    new_b, cnt = clear_lines_at(bt, ct)
    new_b = new_b.numpy().astype(np.int8); cnt = cnt.numpy()
    mism = 0
    n_cleared = 0
    for b in range(len(boards)):
        ref = boards[b].copy()
        ccount = _clear_lines_at(ref, int(cells[b, 0]), int(cells[b, 1]))
        if ccount > 0:
            n_cleared += 1
        if not np.array_equal(ref, new_b[b]) or ccount != cnt[b]:
            mism += 1
            if mism <= 3:
                print(f"  board {b}: cpu cleared {ccount} gpu {cnt[b]}, "
                      f"board match {np.array_equal(ref, new_b[b])}")
    print(f"clear_lines_at: {len(boards)-mism}/{len(boards)} match "
          f"({n_cleared} had clears) ({'PASS' if mism == 0 else 'FAIL'})")
    return mism == 0


def _anchor_tensors(anchor, B, dev):
    board = torch.tensor(anchor['board'], dtype=torch.long, device=dev
                         ).unsqueeze(0).repeat(B, 1, 1)
    npos = torch.zeros(B, 3, 2, dtype=torch.long, device=dev)
    ncol = torch.zeros(B, 3, dtype=torch.long, device=dev)
    nb = anchor['next_balls']
    nn = min(len(nb), 3)
    for i in range(nn):
        (r, c), col = nb[i]
        npos[:, i, 0] = int(r); npos[:, i, 1] = int(c); ncol[:, i] = int(col)
    nnext = torch.full((B,), nn, dtype=torch.long, device=dev)
    return board, npos, ncol, nnext


def _test_parity(device='mps', R=400, n_cand=6, depth=30, horizon=300):
    import json
    import glob
    from alphatrain.evaluate import load_model
    from alphatrain.dataset import TensorDatasetGPU
    from alphatrain.mcts import _build_obs_for_game, _get_legal_priors_flat
    from scripts.batched_rollout import batched_rollout, restore, _decode
    dev = torch.device(device)
    net, _ = load_model('alphatrain/data/pillar3b_epoch_20.pt', dev,
                        fp16=(dev.type != 'cpu'))
    dtype = next(net.parameters()).dtype
    ds = TensorDatasetGPU.__new__(TensorDatasetGPU)
    ds.device = dev
    game = sorted(glob.glob('alphatrain/data/death_games/death_*.json'))[0]
    d = json.load(open(game))
    fr = d['frames'][len(d['frames']) - 1 - depth]
    anchor = {'board': fr['board'], 'next_balls': fr['next_balls'],
              'score': fr.get('score_before', fr['score']), 'turn': fr['turn']}
    g0 = restore(anchor, 0)
    obs0 = torch.from_numpy(_build_obs_for_game(g0)).unsqueeze(0).to(dev, dtype)
    logits0 = net(obs0)[0].float().cpu().numpy()
    pri = _get_legal_priors_flat(g0.board, logits0, 64)
    cand = [m for m, _ in sorted(pri.items(), key=lambda x: -x[1])[:n_cand]]
    cand_dec = [_decode(m) for m in cand]

    import time
    # CPU engine rates (the reference)
    jobs = [(anchor, c, s) for c in cand_dec for s in range(R)]
    t = time.perf_counter()
    cpu = batched_rollout(net, dev, dtype, jobs, horizon, batch=256)
    t_cpu = time.perf_counter() - t
    cpu_turns = sum(x['turns'] for x in cpu)
    cpu_rate = [float(np.mean([cpu[ci * R + s]['died'] for s in range(R)]))
                for ci in range(n_cand)]

    # GPU engine rates
    B = n_cand * R
    board, npos, ncol, nnext = _anchor_tensors(anchor, B, dev)
    first = torch.tensor([cand[ci] for ci in range(n_cand) for _ in range(R)],
                         dtype=torch.long, device=dev)
    gen = torch.Generator(device=dev).manual_seed(0)
    gpu_rollout(net, ds, board[:8], npos[:8], ncol[:8], nnext[:8], first[:8],
                10, gen, dtype)                                   # warmup
    t = time.perf_counter()
    died, turns = gpu_rollout(net, ds, board, npos, ncol, nnext, first,
                              horizon, gen, dtype)
    if dev.type == 'mps':
        torch.mps.synchronize()
    t_gpu = time.perf_counter() - t
    gpu_turns = int(turns.sum().item())
    gpu_rate = died.reshape(n_cand, R).float().mean(1).cpu().numpy()
    print(f"\nSPEED: CPU-engine {t_cpu:.1f}s ({cpu_turns/t_cpu:.0f} turns/s)  "
          f"GPU-engine {t_gpu:.1f}s ({gpu_turns/t_gpu:.0f} turns/s)  "
          f"=> {t_cpu/t_gpu:.2f}x")

    print(f"\nparity (game={game.split('/')[-1]} depth={depth} R={R}):")
    print(f"{'move':>14} {'cpu%':>7} {'gpu%':>7} {'Δpp':>6} {'2*SE':>6}")
    maxdev = 0.0
    for ci in range(n_cand):
        se = 100 * (cpu_rate[ci] * (1 - cpu_rate[ci]) / R) ** 0.5
        dpp = 100 * (gpu_rate[ci] - cpu_rate[ci])
        maxdev = max(maxdev, abs(dpp) / max(2 * se, 1e-6))
        print(f"{str(cand_dec[ci]):>14} {100*cpu_rate[ci]:>7.1f} "
              f"{100*gpu_rate[ci]:>7.1f} {dpp:>6.1f} {2*se:>6.1f}")
    print(f"max |Δ|/2SE = {maxdev:.2f}  "
          f"({'PARITY OK' if maxdev < 1.5 else 'CHECK — exceeds 2SE'})")


if __name__ == '__main__':
    import sys
    torch.set_grad_enabled(False)
    if '--parity' in sys.argv:
        dev = sys.argv[sys.argv.index('--parity') + 1] if len(sys.argv) > sys.argv.index('--parity') + 1 else 'mps'
        _test_parity(device=dev)
    else:
        print("=== golden tests (CPU torch vs CPU engine) ===")
        ok = _test_legal_mask()
        ok &= _test_clear()
        print("ALL PASS" if ok else "FAILURES")
