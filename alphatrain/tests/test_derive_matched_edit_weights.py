import numpy as np
import pytest

from alphatrain.scripts.derive_matched_edit_weights import derive_weights


def test_matches_reference_agree_and_edit_mass_exactly():
    weight = np.array([1.0, 2.0, 3.0, 4.0])
    reference = np.array([False, False, True, False])
    candidate = np.array([False, True, True, False])
    out = derive_weights(weight, reference, candidate, 1.0, 5.0)
    assert out['reference_effective_agree_mass'] == 7.0
    assert out['reference_effective_edit_mass'] == 15.0
    assert out['candidate_agree_weight'] == pytest.approx(7.0 / 5.0)
    assert out['candidate_disagree_weight'] == pytest.approx(15.0 / 5.0)
    assert out['candidate_effective_agree_mass'] == pytest.approx(7.0)
    assert out['candidate_effective_edit_mass'] == pytest.approx(15.0)


def test_rejects_missing_candidate_stratum():
    with pytest.raises(ValueError, match='agree and edit'):
        derive_weights(np.ones(3), [False, True, False],
                       [False, False, False], 1.0, 4.0)
