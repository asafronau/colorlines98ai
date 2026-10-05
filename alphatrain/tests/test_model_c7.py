"""Color-equivariant policy (HISTORY 279): exact invariance to renaming the colors on real observations, the move
index convention, tracing and the state-dict round trip."""
import itertools

import numpy as np
import torch

from alphatrain.model_c7 import PolicyNetC7, c7_kwargs_from_state, is_c7_state, split_observation
from alphatrain.observation import build_observation


def _random_state(rng, fill):
    board = rng.integers(1, 8, (9, 9)).astype(np.int8)
    board[rng.random((9, 9)) >= fill] = 0
    empty = np.flatnonzero(board.reshape(81) == 0)
    cells = rng.choice(empty, 3, replace=False)
    return board, [(int(c) // 9, int(c) % 9, int(rng.integers(1, 8))) for c in cells]


def _obs(board, nb, perm=None):
    lut = np.arange(8) if perm is None else np.concatenate([[0], np.asarray(perm) + 1])
    return build_observation(lut[board].astype(np.int8), np.array([r for r, _, _ in nb]), np.array([c for _, c, _ in nb]),
                             np.array([lut[k] for _, _, k in nb]), len(nb))


def _net(blocks=2, k=4, s=6):
    torch.manual_seed(0)
    net = PolicyNetC7(num_blocks=blocks, slot_channels=k, shared_channels=s, policy_channels=8, pair_dim=8)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2); m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.2, 0.2)
    return net


def test_split_observation_decodes_preview_colors():
    board, nb = _random_state(np.random.default_rng(1), 0.5)
    for dtype in (torch.float32, torch.float16):
        slots, shared, onehot = split_observation(torch.from_numpy(_obs(board, nb))[None].to(dtype))
        for r, c, k in nb:
            assert slots[0, :, 1, r, c].tolist() == [1.0 if j == k - 1 else 0.0 for j in range(7)]
        assert int(slots[0, :, 1].sum()) == len(nb)
        assert torch.equal(onehot[0].argmax(0)[torch.from_numpy(board > 0)] + 1,
                           torch.from_numpy(board[board > 0]).long())
        assert shared.shape == (1, 8, 9, 9)


def test_logits_exactly_invariant_to_renaming_colors():
    net = _net()
    rng = np.random.default_rng(7)
    perms = [None] + [rng.permutation(7) for _ in range(6)] + [np.roll(np.arange(7), 1)]
    for train_mode in (False, True):
        net.train(train_mode)
        for fill in (0.2, 0.5, 0.8):
            board, nb = _random_state(rng, fill)
            obs = torch.from_numpy(np.stack([_obs(board, nb, p) for p in perms]))
            with torch.no_grad():
                out = net(obs)
            for i in range(1, len(perms)):
                torch.testing.assert_close(out[i], out[0], atol=2e-5, rtol=1e-5, msg=f'perm {i} fill {fill}')


def test_index_is_source_times_81_plus_destination():
    """An empty source has no color slot: its row s*81 + (0..80) is constant over destinations."""
    net = _net()
    net.train(False)
    board, nb = _random_state(np.random.default_rng(3), 0.5)
    with torch.no_grad():
        out = net(torch.from_numpy(_obs(board, nb))[None])[0].reshape(81, 81)
    empty = np.flatnonzero(board.reshape(81) == 0)
    balls = np.flatnonzero(board.reshape(81) > 0)
    rows = out[torch.from_numpy(empty)]
    assert torch.allclose(rows, rows[:, :1].expand_as(rows), atol=1e-6)
    assert (out[torch.from_numpy(balls)].std(1) > 1e-4).all()


def test_traces_and_round_trips():
    net = _net(blocks=3, k=5, s=7)
    net.train(False)
    state = net.state_dict()
    assert is_c7_state(state)
    kw = c7_kwargs_from_state(state)
    assert kw == dict(num_blocks=3, slot_channels=5, shared_channels=7, policy_channels=8, pair_dim=8)
    net2 = PolicyNetC7(**kw)
    net2.load_state_dict(state, strict=True)
    net2.train(False)
    rng = np.random.default_rng(11)
    x = torch.from_numpy(np.stack([_obs(*_random_state(rng, f)) for f in (0.3, 0.6)]))
    with torch.no_grad():
        ref = net(x)
        torch.testing.assert_close(net2(x), ref)
        torch.testing.assert_close(torch.jit.trace(net, x[:1])(x), ref, atol=1e-5, rtol=1e-5)
        logits, feats = net.forward_with_features(x)
        torch.testing.assert_close(logits, ref)
        assert feats.shape == (2, 2 * 5 + 7, 9, 9)


def test_bn_fold_matches():
    from alphatrain.inference_cpp.export_ts import fold_batchnorm
    net = _net(blocks=3)
    net.train(False)
    rng = np.random.default_rng(13)
    x = torch.from_numpy(np.stack([_obs(*_random_state(rng, f)) for f in (0.2, 0.5, 0.8)]))
    with torch.no_grad():
        ref = net(x)
        net.freeze()
        torch.testing.assert_close(net(x), ref, atol=1e-5, rtol=1e-5)
        assert fold_batchnorm(net) == 1 + 3 + 2
        torch.testing.assert_close(net(x), ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.jit.trace(net, x[:1])(x), ref, atol=1e-5, rtol=1e-5)


def test_dense_slot_conv_is_the_deepsets_map():
    """SlotMix's single dense convolution == h'_c = A*h_c + B*mean(h) + C*z, z' = D*z + E*mean(h)."""
    from alphatrain.model_c7 import SlotMix
    torch.manual_seed(1)
    k, s = 3, 4
    mix = SlotMix(k, s, 5, 6)
    h = torch.randn(2, 7, k, 9, 9); z = torch.randn(2, s, 9, 9)
    with torch.no_grad():
        out = mix(torch.cat([h.reshape(2, 7 * k, 9, 9), z], 1))
        g = mix.g.weight
        mean = h.mean(1)
        conv = lambda inp, w: torch.nn.functional.conv2d(inp, w, padding=1)
        ref_h = torch.stack([conv(h[:, c], mix.a.weight) + conv(mean, g[:5, :k]) + conv(z, g[:5, k:]) for c in range(7)], 1)
        ref_z = conv(mean, g[5:, :k]) + conv(z, g[5:, k:])
    torch.testing.assert_close(out[:, :35].reshape(2, 7, 5, 9, 9), ref_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, 35:], ref_z, atol=1e-5, rtol=1e-5)


def test_smax_invariance_and_bn_fold():
    """PolicyNetC7 with use_smax=True is S7-invariant, round-trips state_dict, and folds BatchNorms."""
    from alphatrain.inference_cpp.export_ts import fold_batchnorm
    torch.manual_seed(2)
    net = PolicyNetC7(num_blocks=2, slot_channels=4, shared_channels=6,
                      policy_channels=8, pair_dim=8, use_smax=True)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2); m.running_var.uniform_(0.5, 2.0)
                m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.2, 0.2)
    net.train(False)
    rng = np.random.default_rng(17)
    board, nb = _random_state(rng, 0.5)
    perms = [None] + [rng.permutation(7) for _ in range(4)]
    obs = torch.from_numpy(np.stack([_obs(board, nb, p) for p in perms]))
    with torch.no_grad():
        ref = net(obs)
        for i in range(1, len(perms)):
            torch.testing.assert_close(ref[i], ref[0], atol=2e-5, rtol=1e-5)
        kw = c7_kwargs_from_state(net.state_dict())
        assert kw == dict(num_blocks=2, slot_channels=4, shared_channels=6,
                          policy_channels=8, pair_dim=8, use_smax=True)
        net2 = PolicyNetC7(**kw)
        net2.load_state_dict(net.state_dict(), strict=True)
        net2.train(False)
        torch.testing.assert_close(net2(obs), ref, atol=1e-5, rtol=1e-5)
        assert fold_batchnorm(net) == 1 + 2 + 2
        torch.testing.assert_close(net(obs), ref, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.jit.trace(net, obs[:1])(obs), ref, atol=1e-5, rtol=1e-5)
