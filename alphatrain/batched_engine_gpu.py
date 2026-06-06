"""Torch (GPU) port of the batched engine primitives — Stage-5 GPU MCTS.

Same algorithms as alphatrain.batched_engine (numpy), as torch tensor ops so the whole
search can live on the GPU (one process, K=512+ trees). Fixed-iteration label propagation
(no convergence sync). Validated against the numpy/scalar versions; first we DE-RISK whether
mps kernel-launch overhead lets the GPU actually beat numpy on these tiny 9x9 boards.
"""
import torch
from game.config import NUM_COLORS, BALLS_PER_TURN

BOARD = 9
_DIAM = BOARD * BOARD              # worst-case propagation (snake component) for correctness


def make_cell_id(dev):
    return (torch.arange(BOARD * BOARD, device=dev, dtype=torch.int32).reshape(BOARD, BOARD) + 1)


def label_components_t(boards, cell_id, big=32767):
    """Connected components of empty cells, batched. boards int [K,9,9] -> labels int [K,9,9]
    (0 at balls; component id = min cell-id in the region). Fixed-iteration min propagation."""
    empty = boards == 0
    labels = torch.where(empty, cell_id.to(boards.dtype), torch.zeros_like(boards))
    bigv = torch.tensor(big, dtype=boards.dtype, device=boards.device)
    for _ in range(_DIAM):
        work = torch.where(empty, labels, bigv)
        m = work.clone()
        m[:, 1:, :] = torch.minimum(m[:, 1:, :], work[:, :-1, :])
        m[:, :-1, :] = torch.minimum(m[:, :-1, :], work[:, 1:, :])
        m[:, :, 1:] = torch.minimum(m[:, :, 1:], work[:, :, :-1])
        m[:, :, :-1] = torch.minimum(m[:, :, :-1], work[:, :, 1:])
        labels = torch.where(empty, m, torch.zeros_like(boards))
    return labels


def label_components_pj(boards, iters=12):
    """Connected components via label-propagation + POINTER JUMPING (O(log) rounds,
    ~7 enough for a 9x9 grid; far fewer than the O(diameter)~81 of naive propagation).
    boards int [K,9,9] -> labels [K,9,9] (component rep id at empty cells; ball cells
    carry junk and must be ignored). Empty neighbours only propagate."""
    K = boards.shape[0]
    dev = boards.device
    empty = boards == 0
    idx = torch.arange(BOARD * BOARD, device=dev, dtype=torch.int64).reshape(BOARD, BOARD)
    label = idx.expand(K, BOARD, BOARD).clone()
    BIG = BOARD * BOARD                     # sentinel > any id, so ball neighbours don't lower min
    for _ in range(iters):
        lab_e = torch.where(empty, label, torch.full_like(label, BIG))  # balls -> BIG
        m = label.clone()                   # center uses its own label
        m[:, 1:, :] = torch.minimum(m[:, 1:, :], lab_e[:, :-1, :])      # up
        m[:, :-1, :] = torch.minimum(m[:, :-1, :], lab_e[:, 1:, :])     # down
        m[:, :, 1:] = torch.minimum(m[:, :, 1:], lab_e[:, :, :-1])      # left
        m[:, :, :-1] = torch.minimum(m[:, :, :-1], lab_e[:, :, 1:])     # right
        mf = m.reshape(K, BOARD * BOARD)
        label = torch.gather(mf, 1, mf).reshape(K, BOARD, BOARD)        # jump: m[m[i]]
    return torch.where(empty, label, torch.full_like(label, -1))        # balls -> -1


def label_components_sv(boards, iters=8):
    """Connected components via Shiloach-Vishkin hooking + pointer-jumping (TRUE O(log
    diameter): ~8 rounds for a 9x9 grid vs the ~45 of 2-hop propagation in label_components_pj).

    Maintains a parent forest f[i] (flat). Each round: (1) neighbour-min of the PARENT field
    over empty neighbours -> g[i] (best root reachable in one hop); (2) HOOK via scatter-min:
    lower each cell's current root f[i] toward g[i] (this is what makes it logarithmic, not
    linear in diameter); (3) SHORTCUT f[i]=f[f[i]] (path compression). boards int [K,9,9] ->
    labels [K,9,9] (component min-id at empty cells; -1 at balls)."""
    K = boards.shape[0]
    dev = boards.device
    N = BOARD * BOARD
    empty2d = boards == 0
    emf = empty2d.reshape(K, N)
    # ids fit in int32 (<=80); int32 keeps scatter_reduce(amin) supported on mps (int64 isn't).
    idx = torch.arange(N, device=dev, dtype=torch.int32)
    f = idx.expand(K, N).clone()                                # parent forest (flat, int32 ids)
    BIG = N                                                     # > any id; ball cells / no-hook
    for _ in range(iters):
        f2d = f.reshape(K, BOARD, BOARD)
        fe = torch.where(empty2d, f2d, torch.full_like(f2d, BIG))   # ball neighbours don't lower
        g = f2d.clone()                                         # center's own parent
        g[:, 1:, :] = torch.minimum(g[:, 1:, :], fe[:, :-1, :])      # up
        g[:, :-1, :] = torch.minimum(g[:, :-1, :], fe[:, 1:, :])     # down
        g[:, :, 1:] = torch.minimum(g[:, :, 1:], fe[:, :, :-1])      # left
        g[:, :, :-1] = torch.minimum(g[:, :, :-1], fe[:, :, 1:])     # right
        g = torch.where(empty2d, g, torch.full_like(g, BIG)).reshape(K, N)  # balls don't hook
        parent = f.clone().long()                              # frozen int64 index for scatter/gather
        f.scatter_reduce_(1, parent, g, reduce='amin', include_self=True)   # hook roots down
        f = torch.gather(f, 1, f.long())                       # shortcut: f[i] = f[f[i]]
    return torch.where(emf, f.long(), torch.full((K, N), -1, dtype=torch.int64, device=dev)
                       ).reshape(K, BOARD, BOARD)


def reachable_many_t(labels, src, tgt):
    """Torch reachable_many. labels [K,9,9] (empty=component id>=0, ball=-1),
    src/tgt int [K,W,2]. Returns bool [K,W]."""
    K, W = src.shape[0], src.shape[1]
    kk = torch.arange(K, device=labels.device).unsqueeze(1)             # [K,1]
    tr, tc = tgt[..., 0], tgt[..., 1]
    tgt_label = labels[kk, tr, tc]                                      # [K,W]
    valid = tgt_label >= 0                                              # tgt is empty
    sr, sc = src[..., 0], src[..., 1]
    out = torch.zeros((K, W), dtype=torch.bool, device=labels.device)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nr, nc = sr + dr, sc + dc
        inb = (nr >= 0) & (nr < BOARD) & (nc >= 0) & (nc < BOARD)
        nrc = nr.clamp(0, BOARD - 1); ncc = nc.clamp(0, BOARD - 1)
        out |= inb & valid & (labels[kk, nrc, ncc] == tgt_label)
    return out


def clear_lines_at_t(boards, rows, cols, active=None, min_line=5):
    """Torch clear_lines_at (padded board + cumprod run), IN PLACE. boards int [K,9,9].
    Returns cleared-count [K]. Same rule as scalar (validated against numpy partition)."""
    K = boards.shape[0]
    dev = boards.device
    kar = torch.arange(K, device=dev)
    color = boards[kar, rows, cols]                                    # [K]
    act = (color > 0) if active is None else (active & (color > 0))
    P = BOARD - 1
    SZP = BOARD + 2 * P
    pad = torch.full((K, SZP, SZP), -1, dtype=boards.dtype, device=dev)
    pad[:, P:P + BOARD, P:P + BOARD] = boards
    rp, cp = rows + P, cols + P
    clear_pad = torch.zeros((K, SZP, SZP), dtype=torch.bool, device=dev)
    kar2 = kar.unsqueeze(1)
    color2 = color.unsqueeze(1)
    act2 = act.unsqueeze(1)
    offs = torch.arange(1, BOARD, device=dev).unsqueeze(0)             # [1,8]
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        length = act.to(torch.int32)
        cells = []
        for sign in (1, -1):
            rr = rp.unsqueeze(1) + (sign * dr) * offs                   # [K,8]
            cc = cp.unsqueeze(1) + (sign * dc) * offs
            same = (pad[kar2, rr, cc] == color2) & act2
            run = torch.cumprod(same.to(torch.int32), dim=1) > 0        # cumulative AND
            length = length + run.sum(dim=1)
            cells.append((rr, cc, run))
        long = length >= min_line
        # per-direction mark with NON-atomic overwrite (cells distinct within a dir),
        # then OR across dirs (a cell may belong to lines in several directions).
        dirp = torch.zeros((K, SZP, SZP), dtype=torch.bool, device=dev)
        dirp.index_put_((kar, rp, cp), long, accumulate=False)
        long2 = long.unsqueeze(1)
        for rr, cc, run in cells:
            dirp.index_put_((kar2.expand_as(rr), rr, cc), run & long2, accumulate=False)
        clear_pad |= dirp
    cmask = clear_pad[:, P:P + BOARD, P:P + BOARD]
    n_clear = cmask.reshape(K, -1).sum(dim=1)
    boards[cmask] = 0
    return n_clear


def _rand_empty_order(boards, generator=None):
    """Per tree, cell indices ranked by uniform-random key with non-empty cells last.
    Returns order [K,81] (first n_empty entries are random distinct empty cells), n_empty [K]."""
    K = boards.shape[0]
    dev = boards.device
    empty = (boards == 0).reshape(K, BOARD * BOARD)
    keys = torch.where(empty, torch.rand(K, BOARD * BOARD, device=dev, generator=generator),
                       torch.full((K, BOARD * BOARD), -1.0, device=dev))
    order = keys.argsort(dim=1, descending=True)
    return order, empty.sum(dim=1)


def apply_move_t(boards, next_pos, next_col, next_n, src, tgt,
                 active=None, num_colors=NUM_COLORS, generator=None):
    """Torch port of batched_engine.apply_move (move ball -> clear at tgt -> if no clear,
    spawn this turn's balls / displace blocked ones / clear at landings / regen / flag full).
    IN PLACE on boards. Deterministic move+clear identical to scalar; spawn draws use torch
    RNG (open-loop). Returns (game_over[K] bool, new_pos, new_col, new_n)."""
    import torch.nn.functional as _F  # noqa
    K = boards.shape[0]
    dev = boards.device
    kar = torch.arange(K, device=dev)
    act0 = torch.ones(K, dtype=torch.bool, device=dev) if active is None else active
    sr, sc, tr, tc = src[:, 0], src[:, 1], tgt[:, 0], tgt[:, 1]
    color = torch.where(act0, boards[kar, sr, sc], torch.zeros_like(boards[:, 0, 0]))
    mv = kar[act0]
    boards[mv, sr[act0], sc[act0]] = 0
    boards[mv, tr[act0], tc[act0]] = color[act0]
    cleared = clear_lines_at_t(boards, tr, tc, active=act0)
    spawn = act0 & (cleared == 0)

    new_pos = next_pos.clone(); new_col = next_col.clone(); new_n = next_n.clone()
    for i in range(next_pos.shape[1]):
        act = spawn & (i < next_n)
        if not bool(act.any()):
            continue
        pr = next_pos[:, i, 0].long(); pc = next_pos[:, i, 1].long()
        pcol = next_col[:, i]
        intended_empty = boards[kar, pr, pc] == 0
        land_r, land_c = pr.clone(), pc.clone()
        disp = act & ~intended_empty
        if bool(disp.any()):
            order, n_emp = _rand_empty_order(boards, generator)
            cell = order[:, 0]
            use = disp & (n_emp > 0)
            land_r = torch.where(use, cell // BOARD, land_r)
            land_c = torch.where(use, cell % BOARD, land_c)
            disp = use
        place = (act & intended_empty) | disp
        pk = kar[place]
        boards[pk, land_r[place], land_c[place]] = pcol[place].to(boards.dtype)
        clear_lines_at_t(boards, land_r, land_c, active=place)

    game_over = torch.zeros(K, dtype=torch.bool, device=dev)
    if bool(spawn.any()):
        filled = (boards.reshape(K, -1) != 0).all(dim=1)
        game_over = spawn & filled
        order, n_emp = _rand_empty_order(boards, generator)
        want = torch.minimum(torch.tensor(BALLS_PER_TURN, device=dev), n_emp)
        cols = torch.randint(1, num_colors + 1, (K, BALLS_PER_TURN), device=dev,
                             generator=generator)
        for i in range(BALLS_PER_TURN):
            sel = spawn & (i < want)
            cell = order[:, i]
            new_pos[:, i, 0] = torch.where(sel, (cell // BOARD).to(new_pos.dtype), new_pos[:, i, 0])
            new_pos[:, i, 1] = torch.where(sel, (cell % BOARD).to(new_pos.dtype), new_pos[:, i, 1])
            new_col[:, i] = torch.where(sel, cols[:, i].to(new_col.dtype), new_col[:, i])
        new_n = torch.where(spawn, want.to(new_n.dtype), new_n)
    return game_over, new_pos, new_col, new_n


def _bench(device=None, K=512, W=300, pj_iters=45):
    """Validate (vs numpy) + benchmark the hot per-descent-step primitives GPU vs numpy.
    Run on Colab CUDA:  PYTHONPATH=. python alphatrain/batched_engine_gpu.py cuda"""
    import time, numpy as np
    from alphatrain import batched_engine as be
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else \
                 'mps' if torch.backends.mps.is_available() else 'cpu'
    dev = torch.device(device)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else \
           (lambda: torch.mps.synchronize()) if dev.type == 'mps' else (lambda: None)
    print(f"device={dev.type}  K={K}  W={W}", flush=True)
    rng = np.random.default_rng(0)
    boards_np = np.where(rng.random((K, 9, 9)) < 0.5,
                         rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
    boards_t = torch.from_numpy(boards_np.astype(np.int32)).to(dev)
    ln = be.label_components(boards_np)

    def partition_ok(lab_t):
        lp = lab_t.cpu().numpy(); bad = 0
        for k in range(K):
            e = boards_np[k] == 0
            a = ln[k][e]; b = lp[k][e]
            if not np.array_equal((a[:, None] == a[None, :]), (b[:, None] == b[None, :])):
                bad += 1
        return K - bad

    print(f"correctness: pj-label partition {partition_ok(label_components_pj(boards_t, pj_iters))}/{K}"
          f"  sv-label(8) {partition_ok(label_components_sv(boards_t, 8))}/{K}", flush=True)
    # reachable_many correctness vs numpy
    src = np.zeros((K, W, 2), dtype=np.int64); tgt = np.zeros((K, W, 2), dtype=np.int64)
    for k in range(K):
        b = np.argwhere(boards_np[k] != 0); e = np.argwhere(boards_np[k] == 0)
        for w in range(W):
            src[k, w] = b[rng.integers(len(b))] if len(b) else (0, 0)
            tgt[k, w] = e[rng.integers(len(e))] if len(e) else (0, 0)
    rn = be.reachable_many(be.label_components(boards_np), src, tgt)
    rt = reachable_many_t(label_components_pj(boards_t, pj_iters),
                          torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev)).cpu().numpy()
    print(f"correctness: reachable_many matches {int((rn == rt).all())} "
          f"({(rn == rt).mean()*100:.1f}% of {K*W})", flush=True)
    # clear correctness
    rows = np.random.default_rng(1).integers(0, 9, K); cols = np.random.default_rng(2).integers(0, 9, K)
    bn = boards_np.copy(); cn = be.clear_lines_at(bn, rows, cols)
    bt = boards_t.clone(); ct = clear_lines_at_t(bt, torch.from_numpy(rows).to(dev),
                                                 torch.from_numpy(cols).to(dev)).cpu().numpy()
    print(f"correctness: clear_lines boards-match {int((bn == bt.cpu().numpy()).all())} "
          f"counts-match {int((cn == ct).all())}", flush=True)

    src_t = torch.from_numpy(src).to(dev); tgt_t = torch.from_numpy(tgt).to(dev)
    rows_t = torch.from_numpy(rows).to(dev); cols_t = torch.from_numpy(cols).to(dev)

    def timeit(fn, n=1000):
        for _ in range(8):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        sync()
        return (time.perf_counter() - t0) * 1e3 / n

    print(f"\n--- per-call ms (GPU vs numpy), K={K} ---", flush=True)
    res = {}
    for name, gpu_fn, np_fn in [
        ('label_components',
         lambda: label_components_sv(boards_t, 8),
         lambda: be.label_components(boards_np)),
        ('reachable_many',
         lambda: reachable_many_t(label_components_sv(boards_t, 8), src_t, tgt_t),
         lambda: be.reachable_many(be.label_components(boards_np), src, tgt)),
        ('clear_lines_at',
         lambda: clear_lines_at_t(boards_t.clone(), rows_t, cols_t),
         lambda: be.clear_lines_at(boards_np.copy(), rows, cols)),
    ]:
        g = timeit(gpu_fn); n = timeit(np_fn)
        res[name] = (g, n)
        print(f"  {name:18s} GPU {g:6.2f} | numpy {n:6.2f} | {n/g:.2f}x", flush=True)
    # net descent-step proxy: 1 label + 1 reachable + ~4 clears (move + 3 spawn) per step
    gs = res['label_components'][0] + res['reachable_many'][0] + 4 * res['clear_lines_at'][0]
    ns = res['label_components'][1] + res['reachable_many'][1] + 4 * res['clear_lines_at'][1]
    print(f"  {'DESCENT STEP 1L+1R+4C':18s} GPU {gs:6.2f} | numpy {ns:6.2f} | {ns/gs:.2f}x",
          flush=True)
    return res


if __name__ == '__main__':
    import sys
    devarg = sys.argv[1] if len(sys.argv) > 1 else None
    ks = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [512]
    for kk in ks:
        _bench(device=devarg, K=kk)
        print(flush=True)
