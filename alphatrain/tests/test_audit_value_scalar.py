import numpy as np

from alphatrain.scripts.audit_value_scalar import auc


def test_auc_is_tie_correct_and_order_sensitive():
    labels = np.array([0, 0, 1, 1])
    assert auc([0, 0, 1, 1], labels) == 1.0
    assert auc([1, 1, 0, 0], labels) == 0.0
    assert auc([0, 0, 0, 0], labels) == 0.5


def test_auc_one_class_is_nan():
    assert np.isnan(auc([0, 1], [1, 1]))
