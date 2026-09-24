import numpy as np

from alphatrain.scripts.build_shared_edit_sidecar import edit_partition


def test_edit_partition_is_exhaustive_and_disjoint():
    reference = np.array([0, 0, 1, 1], dtype=bool)
    candidate = np.array([0, 1, 0, 1], dtype=bool)
    parts = edit_partition(reference, candidate)
    assert {name: mask.tolist() for name, mask in parts.items()} == {
        'shared': [False, False, False, True],
        'reference_only': [False, False, True, False],
        'candidate_only': [False, True, False, False],
        'neither': [True, False, False, False],
    }
    total = sum(mask.astype(np.int8) for mask in parts.values())
    np.testing.assert_array_equal(total, np.ones(4, dtype=np.int8))


def test_edit_partition_rejects_shape_mismatch():
    try:
        edit_partition(np.zeros(2), np.zeros(3))
    except ValueError as exc:
        assert 'shapes' in str(exc)
    else:
        raise AssertionError('shape mismatch was accepted')
