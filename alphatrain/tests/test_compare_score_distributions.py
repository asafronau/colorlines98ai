import numpy as np

from alphatrain.scripts.compare_score_distributions import (
    bootstrap_differences, bootstrap_mean_difference,
    observed_survival,
)


def test_capped_game_is_unknown_beyond_cap_but_observed_at_cap():
    run = {'turns': np.array([300, 1000]), 'capped': np.array([0, 1])}
    np.testing.assert_array_equal(observed_survival(run, 1000), [0, 1])
    assert observed_survival(run, 2000) is None


def test_completed_games_have_known_survival_at_later_horizons():
    run = {'turns': np.array([300, 1000]), 'capped': np.array([0, 0])}
    np.testing.assert_array_equal(observed_survival(run, 2000), [0, 0])


def test_independent_bootstrap_does_not_pair_rows():
    # B is A in reverse order. A paired-row delta would have large variance;
    # distributionally they are identical and the independent bootstrap is
    # centered near zero for every symmetric statistic.
    a = np.arange(100, dtype=np.float64)
    b = a[::-1].copy()
    d = bootstrap_differences(a, b, n_boot=2000, seed=7, chunk=50)
    assert abs(d['mean'].mean()) < 0.2
    assert abs(d['median'].mean()) < 0.3


def test_independent_mean_bootstrap_is_centered_for_same_distribution():
    a = np.arange(100, dtype=np.float64)
    b = a[::-1].copy()
    d = bootstrap_mean_difference(a, b, n_boot=2000, seed=9, chunk=50)
    assert abs(d.mean()) < 0.2
