import numpy as np
import torch

from alphatrain.mcts import _legal_priors_jit
from alphatrain.scripts.distill_relabel import legal_topk_batch


def test_legal_topk_batch_matches_reference():
    rng = np.random.default_rng(20260808)
    boards = np.zeros((5, 9, 9), dtype=np.int8)
    # Several connectedness/density regimes plus one terminal board.
    for i, occupied in enumerate((3, 18, 45, 72)):
        cells = rng.choice(81, size=occupied, replace=False)
        boards[i].flat[cells] = rng.integers(1, 8, size=occupied)
    boards[4].fill(1)
    logits = rng.normal(size=(5, 6561)).astype(np.float32)

    idx, probs, counts = legal_topk_batch(
        torch.from_numpy(boards), torch.from_numpy(logits), 5)
    idx = idx.numpy()
    probs = probs.numpy()
    counts = counts.numpy()

    for i in range(5):
        k, ref_idx, ref_prob = _legal_priors_jit(boards[i], logits[i], 5)
        order = np.argsort(-ref_prob[:k])
        assert counts[i] == k
        np.testing.assert_array_equal(idx[i, :k], ref_idx[:k][order])
        np.testing.assert_allclose(
            probs[i, :k], ref_prob[:k][order], rtol=2e-6, atol=2e-7)
        assert np.all(idx[i, k:] == 0)
        assert np.all(probs[i, k:] == 0)
