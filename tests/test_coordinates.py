import numpy as np
from astropy.time import Time

import tdpy


def test_retr_xposypos_scalar():
    """Polar coordinates convert to the expected Cartesian axes."""

    xpos, ypos = tdpy.retr_xposypos(2., np.pi / 2.)

    np.testing.assert_allclose(xpos, 0., atol=1e-12)
    np.testing.assert_allclose(ypos, 2.)


def test_retr_xposypos_array_round_trip():
    """Array inputs preserve radial distance and wrapped position angle."""

    gang = np.array([0.5, 1., 2.])
    aang = np.array([0., np.pi / 2., 3. * np.pi / 2.])
    xpos, ypos = tdpy.retr_xposypos(gang, aang)

    np.testing.assert_allclose(np.sqrt(xpos**2 + ypos**2), gang)
    np.testing.assert_allclose(np.mod(np.arctan2(ypos, xpos), 2. * np.pi), aang)


def test_counter_returns_previous_index():
    """The shared counter returns the previous value and supports increments."""

    counter = tdpy.cntr()

    assert counter.incr() == 0
    assert counter.incr(3) == 1
    assert counter.gets() == 4


def test_find_apparent_retrograde_runs_returns_sustained_ordered_intervals():
    """Retrograde intervals are chronological, sustained, and inside the scan."""

    runs, times = tdpy.find_apparent_retrograde_runs(
        Time('2025-01-01'), years_back=5, ephemeris='builtin'
    )

    assert len(runs) >= 2
    assert all(end - start + 1 >= 5 for start, end in runs)
    assert all(first[1] < second[0] for first, second in zip(runs, runs[1:]))
    assert 0 <= runs[0][0] <= runs[-1][1] < len(times)