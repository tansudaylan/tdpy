import numpy as np
import pytest
from types import SimpleNamespace

import tdpy
import tdpy.util


@pytest.mark.parametrize('width', [0.05, 0.2, 1.0, 2.0])
def test_spherical_gaussian_normalization_has_unit_solid_angle_integral(width):
    from scipy.integrate import quad

    with np.errstate(all='raise'):
        normalization = tdpy.retr_psfngausnorm(width)
    integral, _ = quad(
        lambda angle: np.exp(-0.5 * (angle / width)**2) * np.sin(angle),
        0.0, np.pi, epsabs=1e-12, epsrel=1e-12,
    )

    assert np.isfinite(normalization)
    assert normalization > 0.0
    assert normalization * 2.0 * np.pi * integral == pytest.approx(1.0, rel=1e-12)


def test_spherical_gaussian_normalization_preserves_array_shape_and_values():
    widths = np.array([[0.05, 0.2], [1.0, 2.0]])  # [rad]
    original = widths.copy()

    with np.errstate(all='raise'):
        result = tdpy.retr_psfngausnorm(widths)

    expected = np.array([tdpy.retr_psfngausnorm(width) for width in widths.ravel()]).reshape(widths.shape)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(widths, original)


def test_spherical_point_map_matches_great_circle_gaussians():
    pytest.importorskip('healpy')
    longitude = np.array([10.0, -30.0])  # [deg]
    latitude = np.array([20.0, -15.0])  # [deg]
    width = 12.0  # [deg]
    flux = np.array([3.0, 7.0])
    originals = [values.copy() for values in (longitude, latitude, flux)]
    grid_longitude, grid_latitude, pixel_count, pixel_area = tdpy.retr_healgrid(16)
    longitude_difference = np.deg2rad(grid_longitude[:, None] - longitude[None, :])
    grid_latitude_radians = np.deg2rad(grid_latitude[:, None])
    source_latitude_radians = np.deg2rad(latitude[None, :])
    cosine_distance = (
        np.sin(grid_latitude_radians) * np.sin(source_latitude_radians)
        + np.cos(grid_latitude_radians) * np.cos(source_latitude_radians) * np.cos(longitude_difference)
    )
    separation = np.arccos(np.clip(cosine_distance, -1.0, 1.0))  # [rad]
    width_radians = np.deg2rad(width)  # [rad]
    expected = pixel_area * tdpy.retr_psfngausnorm(width_radians) * np.sum(
        flux[None, :] * np.exp(-0.5 * (separation / width_radians)**2), axis=1
    )

    with np.errstate(all='raise'):
        result = tdpy.retr_mapspnts(longitude, latitude, width, flux, numbside=16)

    assert result.shape == (pixel_count,)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=1e-16)
    for values, original in zip((longitude, latitude, flux), originals):
        np.testing.assert_array_equal(values, original)


def _assert_profile_formulas(scaldevi, frac, sigc, gamc, sigt, gamt):
    gaussian_core = np.exp(-0.5 * (scaldevi / sigc)**2) / (2.0 * np.pi * sigc**2)
    gaussian_tail = np.exp(-0.5 * (scaldevi / sigt)**2) / (2.0 * np.pi * sigt**2)
    king_core = (1.0 - 1.0 / gamc) * (1.0 + scaldevi**2 / (2.0 * gamc * sigc**2))**(-gamc) / (2.0 * np.pi * sigc**2)
    king_tail = (1.0 - 1.0 / gamt) * (1.0 + scaldevi**2 / (2.0 * gamt * sigt**2))**(-gamt) / (2.0 * np.pi * sigt**2)
    np.testing.assert_allclose(tdpy.retr_singgaus(scaldevi, sigc), gaussian_core)
    np.testing.assert_allclose(tdpy.retr_singking(scaldevi, sigc, gamc), king_core)
    np.testing.assert_allclose(tdpy.retr_doubgaus(scaldevi, frac, sigc, sigt), frac * gaussian_core + (1.0 - frac) * gaussian_tail)
    np.testing.assert_allclose(tdpy.retr_gausking(scaldevi, frac, sigc, sigt, gamt), frac * gaussian_core + (1.0 - frac) * king_tail)
    np.testing.assert_allclose(tdpy.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt), frac * king_core + (1.0 - frac) * king_tail)


def test_kernel_functions_match_analytic_profiles_scalar():
    scaldevi = np.array([0.0, 0.2, 0.8, 1.5])
    frac = 0.35
    sigc = 0.8
    gamc = 2.2
    sigt = 1.7
    gamt = 3.4

    _assert_profile_formulas(scaldevi, frac, sigc, gamc, sigt, gamt)


def test_psf_width_interpolates_each_energy_and_event_class():
    angles = np.linspace(0.0, 4.0, 401)
    sigmas = np.array([[0.5, 1.0], [1.5, 2.0]])
    profiles = np.exp(-angles[None, :, None] ** 2 / (2.0 * sigmas[:, None, :] ** 2))

    widths = tdpy.retr_psfnwdth(profiles, angles)

    np.testing.assert_allclose(widths, sigmas * np.sqrt(2.0 * np.log(2.0)), atol=1e-4)


def test_kernel_functions_match_analytic_profiles_broadcast():
    scaldevi = np.linspace(0.0, 2.0, 5)[None, :, None]
    frac = np.array([0.25, 0.75])[:, None, None]
    sigc = np.array([0.7, 1.0])[:, None, None]
    gamc = np.array([2.0, 2.5])[:, None, None]
    sigt = np.array([1.2, 1.8])[:, None, None]
    gamt = np.array([3.0, 4.0])[:, None, None]

    _assert_profile_formulas(scaldevi, frac, sigc, gamc, sigt, gamt)


def test_fermi_psf_parameters_support_pcat_schema(monkeypatch):
    response = {
        'energ_lo': np.array([[100., 1000.]]),
        'energ_hi': np.array([[1000., 10000.]]),
    }
    for name, values in {
        'score': [1., 2.],
        'gcore': [3., 4.],
        'stail': [5., 6.],
        'gtail': [7., 8.],
        'ntail': [9., 10.],
    }.items():
        response[name] = np.array([values, values])
    scale = {'PSFSCALE': np.array([1., 2., 0.])}
    monkeypatch.setattr(tdpy.util.astropy.io.fits, 'getdata', lambda path, extension: response if extension == 1 else scale)

    energy = np.sqrt(np.array([.1, 10.]))  # [GeV]
    legacy = SimpleNamespace(
        exproaxitype=True, recotype='rec7', pathdata='/tmp/', numbener=2,
        numbevtt=2, indxevtt=np.arange(2), indxener=np.arange(2), meanener=energy,
    )
    tdpy.retr_psfpferm(legacy)

    modern = SimpleNamespace(
        anlytype='rec7', pathdata='/tmp/', numbener=2, numbdqlt=2,
        indxdqlt=np.arange(2), indxdqltincl=np.arange(2), indxener=np.arange(2),
        bctrpara=SimpleNamespace(ener=energy),
    )
    model = SimpleNamespace()
    tdpy.retr_psfpferm(modern, model)

    np.testing.assert_allclose(model.psfpexpr, legacy.psfpexpr)
    np.testing.assert_allclose(modern.fermscalfact, legacy.fermscalfact)
