"""Unit tests for the survive_H labelling logic.

Censoring is the subtle part — capped games can't generate negative
labels for horizons longer than the remaining-to-cap count, because
we don't know whether the game would have survived past the cap.
Treating those as deaths would teach the head false negatives.
"""

import numpy as np
import pytest

from alphatrain.scripts.build_value_targets import survive_labels_for_game
from alphatrain.value_head import SURVIVAL_HORIZONS


HORIZONS = (25, 50, 100, 200)


class TestNaturalDeath:
    """Game that died naturally (capped=False). All horizons get full
    {0, 1} labels; nothing is censored."""

    def test_long_game_all_positive_for_short_horizons(self):
        # 500-move game; first ~300 moves should all have label=1 for H=200
        labels, masks = survive_labels_for_game(500, capped=False,
                                                 horizons=HORIZONS)
        assert labels.shape == (500, 4)
        # Mask is all 1 — natural death never censors
        assert masks.all()
        # First move: 500 remaining → all 4 horizons label 1
        assert np.array_equal(labels[0], [1, 1, 1, 1])
        # Move at t=299 (remaining=201): all 4 horizons still label 1
        assert np.array_equal(labels[299], [1, 1, 1, 1])
        # Move at t=300 (remaining=200): exactly meets H=200 → label 1
        assert np.array_equal(labels[300], [1, 1, 1, 1])
        # Move at t=301 (remaining=199): H=200 fails → label 0
        assert np.array_equal(labels[301], [1, 1, 1, 0])

    def test_short_game_partial_labels(self):
        # 30-move game: H=25 gets 5 positives + 25 negatives,
        # all longer horizons are 0.
        labels, masks = survive_labels_for_game(30, capped=False,
                                                 horizons=HORIZONS)
        assert masks.all()
        # t=0: remaining=30, only H=25 fires
        assert np.array_equal(labels[0], [1, 0, 0, 0])
        # t=4: remaining=26, H=25 still fires
        assert np.array_equal(labels[4], [1, 0, 0, 0])
        # t=5: remaining=25, exactly = H=25 → still fires
        assert np.array_equal(labels[5], [1, 0, 0, 0])
        # t=6: remaining=24, H=25 fails → all 0
        assert np.array_equal(labels[6], [0, 0, 0, 0])
        # t=29: remaining=1 → all 0
        assert np.array_equal(labels[29], [0, 0, 0, 0])

    def test_very_short_game_all_negative(self):
        # 10-move game: never reaches even the shortest horizon
        labels, masks = survive_labels_for_game(10, capped=False,
                                                 horizons=HORIZONS)
        assert masks.all()
        for t in range(10):
            assert np.array_equal(labels[t], [0, 0, 0, 0])


class TestCapped:
    """Game that hit the cap (capped=True). Horizons beyond the
    remaining-to-cap count are CENSORED, not negative."""

    def test_capped_long_game(self):
        # 500-move capped game
        labels, masks = survive_labels_for_game(500, capped=True,
                                                 horizons=HORIZONS)
        # Move at t=300 (remaining=200): all 4 fit → mask=1, all label=1
        assert np.array_equal(masks[300], [1, 1, 1, 1])
        assert np.array_equal(labels[300], [1, 1, 1, 1])
        # Move at t=301 (remaining=199): H=200 doesn't fit; for capped,
        # we DON'T know if game would have survived → mask=0
        assert np.array_equal(masks[301], [1, 1, 1, 0])
        assert labels[301, 3] == 0  # masked-out value is 0 by convention
        # Other horizons still fit
        assert np.array_equal(labels[301, :3], [1, 1, 1])

    def test_capped_short_game(self):
        # 30-move capped game (unusual — typical caps are much longer
        # but we test the logic): all longer horizons should be censored
        labels, masks = survive_labels_for_game(30, capped=True,
                                                 horizons=HORIZONS)
        # t=0: remaining=30, only H=25 fits cleanly. H=50, H=100, H=200
        # all extend past cap → censored
        assert masks[0, 0] == 1   # H=25 fits
        assert labels[0, 0] == 1
        assert np.array_equal(masks[0, 1:], [0, 0, 0])  # rest censored
        # t=29 (remaining=1): no horizon fits → all censored for capped
        assert np.array_equal(masks[29], [0, 0, 0, 0])

    def test_cap_at_exact_horizon_boundary(self):
        # Cap = exactly H positions in. The position at t=0 has
        # remaining = num_moves; if num_moves == H, label should be 1
        # (>=H). Capped doesn't matter here because the relation holds.
        labels, masks = survive_labels_for_game(100, capped=True,
                                                 horizons=(100,))
        assert masks[0, 0] == 1
        assert labels[0, 0] == 1   # remaining=100 == H=100
        # t=1: remaining=99 < 100. Capped → censored.
        assert masks[1, 0] == 0


class TestCensoringConvention:
    """Where masks[..., h]==0, labels[..., h] must be 0 by convention.
    The training loop must respect the mask; if it accidentally treats
    masked values as labels, censored capped positions would be coded
    as deaths."""

    def test_capped_masked_labels_are_zero(self):
        labels, masks = survive_labels_for_game(60, capped=True,
                                                 horizons=HORIZONS)
        # Wherever mask=0, label must be 0
        masked_out = (masks == 0)
        assert (labels[masked_out] == 0).all()


class TestProductionHorizons:
    """Sanity: the production SURVIVAL_HORIZONS constant matches what
    we test."""

    def test_horizon_constant(self):
        assert SURVIVAL_HORIZONS == HORIZONS
