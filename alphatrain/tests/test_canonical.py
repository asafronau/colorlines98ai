import numpy as np
from alphatrain.canonical import canonicalize, CELL, ACT


def _random_state(rng):
    b = np.zeros(81, np.int8); occ = rng.choice(81, rng.integers(5, 60), replace=False); b[occ] = rng.integers(1, 8, len(occ))
    empty = np.flatnonzero(b == 0); pv = rng.choice(empty, min(3, len(empty)), replace=False)
    return b, [(int(p) // 9, int(p) % 9, int(rng.integers(1, 8))) for p in pv]


def test_canonical_form_is_invariant_to_all_rotations_and_color_relabelings():
    rng = np.random.default_rng(0)
    for _ in range(200):
        b, nb = _random_state(rng)
        _, cb0, cn0 = canonicalize(b, nb)
        for v in range(8):
            perm = np.concatenate([[0], rng.permutation(7) + 1])
            b2 = np.zeros(81, np.int8); b2[CELL[v]] = perm[b]
            nb2 = [(int(CELL[v][r * 9 + c]) // 9, int(CELL[v][r * 9 + c]) % 9, int(perm[col])) for r, c, col in nb]
            rng.shuffle(nb2)
            _, cb, cn = canonicalize(b2, nb2)
            assert np.array_equal(cb, cb0) and cn == cn0


def test_canonical_view_maps_board_and_moves_consistently():
    rng = np.random.default_rng(1)
    b, nb = _random_state(rng)
    v, cb, _ = canonicalize(b, nb)
    occ = (b != 0); cocc = np.zeros(81, bool); cocc[CELL[v]] = occ
    assert np.array_equal(cocc, cb.reshape(81) != 0)          # same occupancy after the chosen transform
    s = int(np.flatnonzero(b)[0]); d = int(np.flatnonzero(b == 0)[0])
    a = s * 81 + d; ca = ACT[v][a]
    assert cb.reshape(81)[ca // 81] != 0 and cb.reshape(81)[ca % 81] == 0
