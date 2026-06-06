"""Golden test: apply_move_nosync_t (capture-safe) vs apply_move_t. Move+clear must be bit-identical
(deterministic); spawn path must stay valid (every filled cell was empty; game_over <=> board full).

    PYTHONPATH=. python scripts/test_apply_move_nosync.py [device]
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from alphatrain import batched_engine_gpu as beg


def _dev(a):
    if a:
        return torch.device(a)
    return torch.device('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available() else 'cpu')


def _clear_cases(K, seed):
    rng = np.random.default_rng(seed)
    boards = np.zeros((K, 9, 9), dtype=np.int8)
    src = np.zeros((K, 2), dtype=np.int64); tgt = np.zeros((K, 2), dtype=np.int64)
    for k in range(K):
        col = int(rng.integers(1, 8)); row = int(rng.integers(1, 9)); c0 = int(rng.integers(0, 5))
        for i in range(4):
            boards[k, row, c0 + i] = col
        boards[k, row - 1, c0 + 4] = col
        src[k] = (row - 1, c0 + 4); tgt[k] = (row, c0 + 4)
    return boards, src, tgt


def main():
    dev = _dev(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"device={dev.type}", flush=True)
    K = 200
    boards, src, tgt = _clear_cases(K, 3)
    npos = np.zeros((K, 3, 2), dtype=np.int8); ncol = np.ones((K, 3), dtype=np.int8)
    nn = np.full(K, 3, dtype=np.int8)

    def mk():
        return (torch.from_numpy(npos.astype(np.int64)).to(dev),
                torch.from_numpy(ncol.astype(np.int64)).to(dev),
                torch.from_numpy(nn.astype(np.int64)).to(dev),
                torch.from_numpy(src).to(dev), torch.from_numpy(tgt).to(dev))

    b1 = torch.from_numpy(boards.astype(np.int32)).to(dev)
    g1, *_ = beg.apply_move_t(b1, *mk())
    b2 = torch.from_numpy(boards.astype(np.int32)).to(dev)
    g2, *_ = beg.apply_move_nosync_t(b2, *mk())
    bmatch = bool((b1 == b2).all()); gmatch = bool((g1 == g2).all())
    print(f"move+clear bit-identical: boards {'OK' if bmatch else 'FAIL'} "
          f"({(b1 == b2).float().mean()*100:.1f}%), game_over {'OK' if gmatch else 'FAIL'}", flush=True)

    # spawn validity on dense random boards (no clear -> spawns happen)
    rng = np.random.default_rng(7)
    bd = np.where(rng.random((K, 9, 9)) < 0.5, rng.integers(1, 8, (K, 9, 9)), 0).astype(np.int8)
    from game.board import _label_empty_components, _is_reachable
    s2 = np.zeros((K, 2), dtype=np.int64); t2 = np.zeros((K, 2), dtype=np.int64); keep = np.zeros(K, bool)
    for k in range(K):
        ball = np.argwhere(bd[k] != 0); emp = np.argwhere(bd[k] == 0)
        lab = _label_empty_components(bd[k])
        for _ in range(20):
            if not len(ball) or not len(emp):
                break
            s = ball[rng.integers(len(ball))]; t = emp[rng.integers(len(emp))]
            if _is_reachable(lab, int(s[0]), int(s[1]), int(t[0]), int(t[1])):
                s2[k], t2[k], keep[k] = s, t, True; break
    bt = torch.from_numpy(bd.astype(np.int32)).to(dev)
    go, *_ = beg.apply_move_nosync_t(bt, torch.from_numpy(npos.astype(np.int64)).to(dev),
                                     torch.from_numpy(ncol.astype(np.int64)).to(dev),
                                     torch.from_numpy(nn.astype(np.int64)).to(dev),
                                     torch.from_numpy(s2).to(dev), torch.from_numpy(t2).to(dev))
    after = bt.cpu().numpy().astype(np.int8); gov = go.cpu().numpy()
    bad = 0
    for k in range(K):
        if not keep[k]:
            continue
        if (after[k] < 0).any() or (after[k] > 7).any():
            bad += 1
        elif gov[k] and (after[k] != 0).sum() != 81:
            bad += 1
    print(f"spawn-path validity: {int(keep.sum())-bad}/{int(keep.sum())} ok", flush=True)
    assert bmatch and gmatch and bad == 0, "apply_move_nosync_t mismatch"
    print("APPLY_MOVE_NOSYNC_T OK", flush=True)


if __name__ == '__main__':
    main()
