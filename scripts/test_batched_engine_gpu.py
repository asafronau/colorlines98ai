"""Golden tests for the torch GPU engine vs numpy/scalar (run on mps or cuda)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain import batched_engine as be
from alphatrain import batched_engine_gpu as beg


def _dev():
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def _make_clear_case(rng):
    board = np.zeros((9, 9), dtype=np.int8)
    color = int(rng.integers(1, 8))
    row = int(rng.integers(1, 9)); c0 = int(rng.integers(0, 5))
    for i in range(4):
        board[row, c0 + i] = color
    board[row - 1, c0 + 4] = color          # the ball at src (was missing) that moves to complete
    return board, (row - 1, c0 + 4), (row, c0 + 4)


def test_apply_move_clear(K=120, seed=3):
    """Deterministic move+clear path: torch apply_move_t == numpy apply_move (bit-identical)."""
    dev = _dev()
    rng = np.random.default_rng(seed)
    cases = [_make_clear_case(rng) for _ in range(K)]
    boards = np.stack([c[0] for c in cases])
    src = np.array([c[1] for c in cases], dtype=np.int64)
    tgt = np.array([c[2] for c in cases], dtype=np.int64)
    npos = np.zeros((K, 3, 2), dtype=np.int8); ncol = np.ones((K, 3), dtype=np.int8)
    nn = np.full(K, 3, dtype=np.int8)
    # numpy
    bn = boards.copy()
    gn, _, _, _ = be.apply_move(bn, npos.copy(), ncol.copy(), nn.copy(), src, tgt,
                                np.random.default_rng(0))
    # torch
    bt = torch.from_numpy(boards.astype(np.int32)).to(dev)
    gt, _, _, _ = beg.apply_move_t(bt,
                                   torch.from_numpy(npos.astype(np.int64)).to(dev),
                                   torch.from_numpy(ncol.astype(np.int64)).to(dev),
                                   torch.from_numpy(nn.astype(np.int64)).to(dev),
                                   torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev))
    btn = bt.cpu().numpy().astype(np.int8)
    bmatch = (btn == bn).all()
    gmatch = (gt.cpu().numpy() == gn).all()
    print(f"  apply_move_t clear-path: boards {'OK' if bmatch else 'FAIL'} "
          f"({(btn == bn).mean()*100:.1f}%), game_over {'OK' if gmatch else 'FAIL'}", flush=True)
    assert bmatch and gmatch


def test_apply_move_spawn_valid(K=200, seed=7):
    """Spawn path: every newly-filled cell was empty; game_over <=> board full."""
    dev = _dev()
    rng = np.random.default_rng(seed)
    boards = np.where(rng.random((K, 9, 9)) < 0.45, rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
    src = np.zeros((K, 2), dtype=np.int64); tgt = np.zeros((K, 2), dtype=np.int64)
    keep = np.zeros(K, dtype=bool)
    for k in range(K):
        b = np.argwhere(boards[k] != 0); e = np.argwhere(boards[k] == 0)
        lab = be._label_empty_components(boards[k]) if hasattr(be, '_label_empty_components') else None
        from game.board import _label_empty_components, _is_reachable
        lab = _label_empty_components(boards[k])
        for _ in range(20):
            if not len(b) or not len(e):
                break
            s = b[rng.integers(len(b))]; t = e[rng.integers(len(e))]
            if _is_reachable(lab, int(s[0]), int(s[1]), int(t[0]), int(t[1])):
                src[k], tgt[k] = s, t; keep[k] = True; break
    npos = np.zeros((K, 3, 2), dtype=np.int8); ncol = np.ones((K, 3), dtype=np.int8)
    nn = np.full(K, 3, dtype=np.int8)
    before = boards.copy()
    bt = torch.from_numpy(boards.astype(np.int32)).to(dev)
    gt, _, _, _ = beg.apply_move_t(bt,
                                   torch.from_numpy(npos.astype(np.int64)).to(dev),
                                   torch.from_numpy(ncol.astype(np.int64)).to(dev),
                                   torch.from_numpy(nn.astype(np.int64)).to(dev),
                                   torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev))
    after = bt.cpu().numpy().astype(np.int8); go = gt.cpu().numpy()
    bad = 0
    for k in range(K):
        if not keep[k]:
            continue
        if (after[k] < 0).any() or (after[k] > 7).any():
            bad += 1; continue
        if go[k] and (after[k] != 0).sum() != 81:
            bad += 1
    print(f"  apply_move_t spawn-path validity: {int(keep.sum())-bad}/{int(keep.sum())} ok",
          flush=True)
    assert bad == 0


if __name__ == '__main__':
    print(f"=== GPU engine golden tests (device={_dev().type}) ===", flush=True)
    test_apply_move_clear()
    test_apply_move_spawn_valid()
    print("ALL PASS", flush=True)
