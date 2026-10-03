"""p4m trunk (HISTORY 278): group tables, input-channel permutation, EXACT D4 equivariance of the whole policy on
real observations, freeze() equivalence + tracing, and the state-dict round trip."""
import numpy as np
import torch

from alphatrain.model_p4m import (G, MULT, INV, PolicyNetP4M, apply_g, input_permutation, is_p4m_state,
                                  p4m_kwargs_from_state)
from alphatrain.observation import build_observation

M = 8
F = [lambda r, c: (r, c), lambda r, c: (c, M - r), lambda r, c: (M - r, M - c), lambda r, c: (M - c, r),
     lambda r, c: (r, M - c), lambda r, c: (M - r, c), lambda r, c: (c, r), lambda r, c: (M - c, M - r)]
CELL = np.array([[f(r, c)[0] * 9 + f(r, c)[1] for r in range(9) for c in range(9)] for f in F])
ACT = torch.from_numpy((CELL[:, :, None] * 81 + CELL[:, None, :]).reshape(8, 6561))


def _views(board, nb):
    """Observations of all 8 dihedral views, built from the TRANSFORMED board and preview (no channel tricks)."""
    out = []
    for v, f in enumerate(F):
        b2 = np.zeros(81, np.int8); b2[CELL[v]] = board.reshape(81)
        rr = np.array([f(r, c)[0] for r, c, _ in nb], np.int64); cc = np.array([f(r, c)[1] for r, c, _ in nb], np.int64)
        col = np.array([k for _, _, k in nb], np.int64)
        out.append(build_observation(b2.reshape(9, 9), rr, cc, col, len(nb)))
    return torch.from_numpy(np.stack(out))


def _random_state(rng, fill):
    board = rng.integers(1, 8, (9, 9)).astype(np.int8)
    board[rng.random((9, 9)) >= fill] = 0
    empty = np.flatnonzero(board.reshape(81) == 0)
    cells = rng.choice(empty, 3, replace=False)
    return board, [(int(c) // 9, int(c) % 9, int(rng.integers(1, 8))) for c in cells]


def _net(blocks=2, gc=4, expand=8):
    torch.manual_seed(0)
    net = PolicyNetP4M(num_blocks=blocks, group_channels=gc, expand=expand, policy_channels=16, head='pair2', pair_dim=8)
    with torch.no_grad():                       # non-trivial BN statistics/affines so inference mode is exercised
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2); m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.2, 0.2)
    return net


def test_group_tables_form_d4():
    assert MULT[0] == list(range(G)) and [MULT[g][0] for g in range(G)] == list(range(G))
    for a in range(G):
        assert MULT[a][INV[a]] == 0 and MULT[INV[a]][a] == 0
        for b in range(G):
            for c in range(G):
                assert MULT[MULT[a][b]][c] == MULT[a][MULT[b][c]]
    x = torch.randn(2, 5, 9, 9)
    for a in range(G):
        for b in range(G):
            assert torch.equal(apply_g(apply_g(x, b), a), apply_g(x, MULT[a][b]))


def test_input_permutation_swaps_only_line_directions():
    perm = input_permutation()
    assert perm.shape == (G, 18) and perm[0].tolist() == list(range(18))
    rot90 = [g for g in range(G) if g == 1][0]                  # (k=1, no mirror): H<->V and D1<->D2
    assert perm[rot90, 13:17].tolist() == [14, 13, 16, 15]
    for g in range(G):
        assert sorted(perm[g].tolist()) == list(range(18))


def test_policy_is_exactly_d4_equivariant_on_real_observations():
    net = _net()
    rng = np.random.default_rng(7)
    for train_mode in (False, True):
        net.train(train_mode)
        for fill in (0.2, 0.5, 0.8):
            board, nb = _random_state(rng, fill)
            obs = _views(board, nb)
            with torch.no_grad():
                if train_mode:                   # batch stats of one view are invariant: run each view alone
                    out = torch.cat([net(obs[v:v + 1]) for v in range(8)])
                else:
                    out = net(obs)
            base = out[0]
            for v in range(1, 8):
                torch.testing.assert_close(out[v, ACT[v]], base, atol=2e-4, rtol=1e-4, msg=f'view {v} fill {fill}')


def test_backbone_features_are_invariant_scalar_fields():
    net = _net()
    net.train(False)
    board, nb = _random_state(np.random.default_rng(3), 0.5)
    obs = _views(board, nb)
    with torch.no_grad():
        feats = net.backbone_features(obs)
    assert feats.shape == (8, 16, 9, 9)
    for v in range(8):
        f0 = torch.zeros(16, 81); f0[:, torch.from_numpy(CELL[v])] = feats[0].reshape(16, 81)
        torch.testing.assert_close(feats[v], f0.reshape(16, 9, 9), atol=1e-4, rtol=1e-4)


def test_freeze_and_bn_fold_match_and_trace():
    from alphatrain.inference_cpp.export_ts import fold_batchnorm
    from alphatrain.model_p4m import GBN, GConv, LiftConv
    net = _net()
    net.train(False)
    board, nb = _random_state(np.random.default_rng(11), 0.6)
    x = _views(board, nb)
    with torch.no_grad():
        ref = net(x)
        net.freeze()
        assert not any(isinstance(m, (GBN, GConv, LiftConv)) for m in net.modules())
        torch.testing.assert_close(net(x), ref, atol=1e-5, rtol=1e-5)
        assert fold_batchnorm(net) == 2 + 3                 # 2 blocks + stem + expand + policy head
        torch.testing.assert_close(net(x), ref, atol=1e-5, rtol=1e-5)
        traced = torch.jit.trace(net, x[:1])
        torch.testing.assert_close(traced(x), ref, atol=1e-5, rtol=1e-5)


def test_state_dict_round_trip():
    net = _net(blocks=3, gc=5, expand=6)
    state = net.state_dict()
    assert is_p4m_state(state)
    kw = p4m_kwargs_from_state(state)
    assert kw == dict(in_channels=18, num_blocks=3, group_channels=5, expand=6, head='pair2', pair_dim=8,
                      policy_channels=16)
    net2 = PolicyNetP4M(**kw)
    net2.load_state_dict(state, strict=True)
    x = torch.randn(2, 18, 9, 9)
    net.train(False); net2.train(False)
    with torch.no_grad():
        torch.testing.assert_close(net2(x), net(x))
