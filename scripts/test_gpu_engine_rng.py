"""Golden tests 3-4: production SplitMix64 spawn streams.

3. UNIFORMITY — displacement/regen draws are uniform over empty cells and colors
   uniform over 1..7 (chi-square against 4σ bounds, 4000 independent game streams).
4. BATCH-COMPOSITION INDEPENDENCE (protocol v2's headline property) — the same seed
   played in different batch sizes/orders produces the IDENTICAL game, because each
   game owns its SplitMix64 stream. Played with a fixed constant policy (no net) to
   isolate the ENGINE side; the net's own fp16 batch behavior is measured separately
   in Stage 4.

    PYTHONPATH=. python scripts/test_gpu_engine_rng.py --device mps
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch

from game.board import ColorLinesGame
from alphatrain.gpu_eval_engine import (
    seed_stream,
    GpuGames, choose_moves, _rand_keys_cells, BOARD, NUM_COLORS, NCELL,
)


def init_from_cpu(st, slot, seed):
    g = ColorLinesGame(seed=seed)
    g.reset()
    st.boards[slot] = torch.from_numpy(g.board).to(st.dev)
    st.next_pos[slot] = 0
    st.next_col[slot] = 0
    for i, ((r, c), col) in enumerate(g.next_balls):
        st.next_pos[slot, i, 0] = r
        st.next_pos[slot, i, 1] = c
        st.next_col[slot, i] = col
    st.n_next[slot] = len(g.next_balls)
    st.score[slot] = 0
    st.turns[slot] = 0
    st.seed[slot] = seed
    st.rng[slot] = seed_stream(seed)
    st.alive[slot] = True


def test_uniformity(dev):
    B = 4000
    st = GpuGames(B, dev)
    # one fixed half-full board for all games; independent streams per game
    g = ColorLinesGame(seed=11)
    g.reset()
    rs = np.random.RandomState(7)
    board = g.board.copy()
    filled = rs.choice(81, 40, replace=False)
    board.reshape(-1)[filled] = rs.randint(1, 8, 40)
    bt = torch.from_numpy(board).to(dev)
    st.boards[:] = bt.unsqueeze(0)
    st.rng[:] = torch.arange(B, dtype=torch.int64, device=dev) * 1234567 + 99

    order, n_emp = st._empty_order_rand(purpose=0)
    first = order[:, 0].cpu().numpy()
    empty_cells = np.flatnonzero(board.reshape(-1) == 0)
    E = len(empty_cells)
    counts = np.bincount(first, minlength=81)[empty_cells]
    exp = B / E
    sd = (B * (1 / E) * (1 - 1 / E)) ** 0.5
    worst = np.abs(counts - exp).max() / sd
    print(f"[uniformity] E={E} empties, counts {counts.min()}..{counts.max()} "
          f"(exp {exp:.0f}), worst dev {worst:.2f}σ")
    assert (counts > 0).all() and worst < 4.5, "spawn cell distribution skewed"

    keys = _rand_keys_cells(st.rng, st.turns, 4, dev)[:, :3]
    cols = (keys.abs() % NUM_COLORS + 1).cpu().numpy().ravel()
    ccounts = np.bincount(cols, minlength=NUM_COLORS + 1)[1:]
    cexp = len(cols) / NUM_COLORS
    csd = (len(cols) * (1 / NUM_COLORS) * (1 - 1 / NUM_COLORS)) ** 0.5
    cworst = np.abs(ccounts - cexp).max() / csd
    print(f"[uniformity] colors {ccounts.tolist()} (exp {cexp:.0f}), "
          f"worst dev {cworst:.2f}σ")
    assert cworst < 4.5, "color distribution skewed"
    print("[uniformity] PASS")


def play_to_death(seeds, batch, dev, const_logits, max_turns=4000):
    """Play each seed to death with a constant policy; refill slots from the list.
    Returns {seed: (score, turns)}. Flat action = src_cell*81 + tgt_cell."""
    res = {}
    st = GpuGames(batch, dev)
    queue = list(seeds)
    slot_seed = [None] * batch
    for s in range(batch):
        if queue:
            sd = queue.pop(0)
            init_from_cpu(st, s, sd)
            slot_seed[s] = sd
    L = const_logits.to(dev)
    while any(s is not None for s in slot_seed):
        logits = L.unsqueeze(0).expand(batch, -1)
        moves, has = choose_moves(st.boards, logits)
        st.alive &= has                       # no legal move == death
        mv = moves.clamp(min=0)
        st.step(mv // NCELL, mv % NCELL)
        st.alive &= st.turns < max_turns      # cap guard (test hygiene)
        alive_cpu = st.alive.cpu().numpy()
        for s in range(batch):
            if slot_seed[s] is not None and not alive_cpu[s]:
                res[slot_seed[s]] = (int(st.score[s]), int(st.turns[s]))
                if queue:
                    sd = queue.pop(0)
                    init_from_cpu(st, s, sd)
                    slot_seed[s] = sd
                else:
                    slot_seed[s] = None
    return res


def test_batch_independence(dev):
    seeds = list(range(880000, 880048))
    rng = np.random.RandomState(3)
    L = torch.from_numpy(rng.randn(NCELL * NCELL).astype(np.float32))
    runs = {}
    for batch in (48, 16, 7):
        runs[batch] = play_to_death(seeds, batch, dev, L)
    a, b, c = runs[48], runs[16], runs[7]
    diffs = [s for s in seeds if not (a[s] == b[s] == c[s])]
    lens = [a[s][1] for s in seeds]
    print(f"[batch-independence] 48 seeds, turns {min(lens)}..{max(lens)}, "
          f"batch sizes 48/16/7 -> {len(diffs)} diverging seeds")
    if diffs:
        s = diffs[0]
        raise SystemExit(f"FAIL: seed {s}: B48={a[s]} B16={b[s]} B7={c[s]}")
    print("[batch-independence] PASS: identical games at every batch size")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='mps')
    a = p.parse_args()
    dev = torch.device(a.device)
    test_uniformity(dev)
    test_batch_independence(dev)


if __name__ == '__main__':
    main()
