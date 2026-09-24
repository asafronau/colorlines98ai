"""Pair policy head: exact cell-permutation equivariance, move-index convention, loader round trip,
and no change to the historical 'abs' head."""
import numpy as np
import torch

from alphatrain.model import PolicyNet, head_kwargs_from_state

M = 8
F = [lambda r, c: (r, c), lambda r, c: (c, M - r), lambda r, c: (M - r, M - c), lambda r, c: (M - c, r),
     lambda r, c: (r, M - c), lambda r, c: (M - r, c), lambda r, c: (c, r), lambda r, c: (M - c, M - r)]
CELL = np.array([[f(r, c)[0] * 9 + f(r, c)[1] for r in range(9) for c in range(9)] for f in F])


def _pair_net():
    torch.manual_seed(0)
    net = PolicyNet(num_blocks=2, channels=16, head='pair', pair_dim=8)
    net.train(False)
    return net


def test_pair_head_is_exactly_equivariant_under_all_d4_views():
    net = _pair_net()
    feats = torch.randn(3, 16, 9, 9)
    with torch.no_grad():
        base = net._policy_from_features(feats)
        for v in range(8):
            f2 = torch.zeros_like(feats).reshape(3, 16, 81)
            f2[:, :, torch.from_numpy(CELL[v])] = feats.reshape(3, 16, 81)
            out = net._policy_from_features(f2.reshape(3, 16, 9, 9))
            act = torch.from_numpy((CELL[v][:, None] * 81 + CELL[v][None, :]).reshape(-1))
            assert torch.allclose(out[:, act], base, atol=1e-5), v


def test_pair_head_index_is_source_times_81_plus_destination():
    net = _pair_net()
    with torch.no_grad():
        for mod in (net.pair_src, net.pair_dst, net.pair_sbias, net.pair_dbias):
            mod.weight.zero_(); mod.bias.zero_()
        net.pair_sbias.weight[0, 0] = 1.0     # source score = channel 0 of the head input
        net.pair_dbias.weight[0, 1] = 1.0     # destination score = channel 1
        p = torch.zeros(1, net.pair_src.in_channels, 9, 9)
        p[0, 0, 2, 3] = 5.0                   # source cell (2,3) = 21
        p[0, 1, 7, 1] = 7.0                   # destination cell (7,1) = 64
        u = net.pair_src(p).reshape(1, net.pair_dim, 81); w = net.pair_dst(p).reshape(1, net.pair_dim, 81)
        logits = (torch.bmm(u.transpose(1, 2), w) * net.pair_dim ** -0.5 + net.pair_sbias(p).reshape(1, 81, 1)
                  + net.pair_dbias(p).reshape(1, 1, 81)).reshape(1, -1)
    assert int(logits.argmax()) == 21 * 81 + 64


def test_head_kwargs_round_trip_and_abs_unchanged():
    net = _pair_net()
    kw = head_kwargs_from_state(net.state_dict())
    assert kw == {'head': 'pair', 'pair_dim': 8, 'policy_channels': 128}
    net2 = PolicyNet(num_blocks=2, channels=16, **kw); net2.load_state_dict(net.state_dict())
    torch.manual_seed(1)
    old = PolicyNet(num_blocks=2, channels=16)
    assert head_kwargs_from_state(old.state_dict())['head'] == 'abs'
    assert not any(k.startswith('pair_') for k in old.state_dict())
    x = torch.randn(2, 18, 9, 9)
    old.train(False)
    with torch.no_grad():
        assert old(x).shape == (2, 6561) and net2(x).shape == (2, 6561)


def test_pair2_is_d4_equivariant_but_knows_geometry():
    torch.manual_seed(0)
    net = PolicyNet(num_blocks=2, channels=16, head='pair2', pair_dim=8); net.train(False)
    feats = torch.randn(2, 16, 9, 9)
    with torch.no_grad():
        base = net._policy_from_features(feats)
        for v in range(8):
            f2 = torch.zeros_like(feats).reshape(2, 16, 81)
            f2[:, :, torch.from_numpy(CELL[v])] = feats.reshape(2, 16, 81)
            out = net._policy_from_features(f2.reshape(2, 16, 9, 9))
            act = torch.from_numpy((CELL[v][:, None] * 81 + CELL[v][None, :]).reshape(-1))
            assert torch.allclose(out[:, act], base, atol=1e-5), v
        perm = torch.randperm(81, generator=torch.Generator().manual_seed(3))   # arbitrary cell shuffle
        f3 = feats.reshape(2, 16, 81)[:, :, perm].reshape(2, 16, 9, 9)
        out = net._policy_from_features(f3)
        inv = torch.argsort(perm)
        act = (inv[:, None] * 81 + inv[None, :]).reshape(-1)
        assert not torch.allclose(out[:, act], base, atol=1e-3)            # gates break arbitrary shuffles
    assert head_kwargs_from_state(net.state_dict())['head'] == 'pair2'
