import numpy as np
import pytest
from astropy.time import Time

import tdpy


@pytest.mark.parametrize('calling_form', ['tuple', 'list', 'array', 'keyword', 'positional'])
@pytest.mark.parametrize('components,right_ascension,declination', [
    ((8, 35, 20.655, -45, 10, 35.15), '08h35m20.655s', '-45d10m35.15s'),
    ((12, 34, 56.7, 12, 34, 56.7), '12h34m56.7s', '+12d34m56.7s'),
    ((0, 0, 0, -0.0, 12, 30), '00h00m00s', '-00d12m30s'),
    ((23, 59, 59.9, 0, 0, 0), '23h59m59.9s', '+00d00m00s'),
    ((0, 0, 0, 90, 0, 0), '00h00m00s', '+90d00m00s'),
    ((0, 0, 0, -90, 0, 0), '00h00m00s', '-90d00m00s'),
])
def test_sexagesimal_coordinates_match_astropy(components, right_ascension, declination, calling_form):
    from astropy.coordinates import SkyCoord

    reference = SkyCoord(right_ascension, declination, frame='icrs')
    sequences = {'tuple': components, 'list': list(components), 'array': np.array(components)}
    if calling_form == 'positional':
        actual = tdpy.conv_rascdecl(*components)
    elif calling_form == 'keyword':
        actual = tdpy.conv_rascdecl(args=components)
    else:
        actual = tdpy.conv_rascdecl(sequences[calling_form])

    np.testing.assert_allclose(actual, [reference.ra.deg, reference.dec.deg], rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(sequences['array'], components)
    assert sequences['list'] == list(components)


def test_healgrid_preserves_ring_pixels_and_existing_longitude_convention():
    pytest.importorskip('healpy')

    longitude, latitude, pixel_count, pixel_area = tdpy.retr_healgrid(1)

    expected_longitude = np.array([-45., -135., 135., 45., 0., -90., 180., 90.,
                                   -45., -135., 135., 45.])  # [deg]
    polar_latitude = np.rad2deg(np.arcsin(2. / 3.))  # [deg]
    expected_latitude = np.repeat([polar_latitude, 0., -polar_latitude], 4)  # [deg]
    assert pixel_count == 12
    assert pixel_area == pytest.approx(np.pi / 3.)
    np.testing.assert_allclose(longitude, expected_longitude, atol=1e-12)
    np.testing.assert_allclose(latitude, expected_latitude, atol=1e-12)


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('channels', [False, True])
@pytest.mark.parametrize('grid_shape', [(7, 5), (5, 7)])
def test_healpix_cartesian_conversion_matches_native_lookup(nested, channels, grid_shape):
    healpy = pytest.importorskip('healpy')
    side = 4
    values = np.arange(12 * side**2, dtype=float)
    if nested:
        values = healpy.reorder(values, r2n=True)
    if channels:
        values = np.column_stack((values, values + 1000.0))
    original = values.copy()
    longitude = np.linspace(-130.0, 140.0, grid_shape[0])  # [deg]
    latitude = np.linspace(-50.0, 60.0, grid_shape[1])  # [deg]
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    indices = healpy.ang2pix(side, longitude_grid, latitude_grid, lonlat=True, nest=nested)
    expected = np.fliplr(values[indices]).T

    result = tdpy.retr_cart(
        values, minmlgal=-130.0, maxmlgal=140.0, minmbgal=-50.0, maxmbgal=60.0,
        numbsidelgal=grid_shape[0], numbsidebgal=grid_shape[1], nest=nested,
    )

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(values, original)


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

    np.testing.assert_allclose(tdpy.retr_gang(xpos, ypos), gang)
    np.testing.assert_allclose(tdpy.retr_aang(xpos, ypos), aang)


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