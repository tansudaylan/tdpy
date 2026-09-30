import numpy as np
import pytest

from tdpy.exoplanet import (
    empirical_interval_contrasts,
    keplerian_radial_velocity,
    kipping_to_quadratic_limb_darkening,
    paired_interval_contrast_forecast,
    plot_atmosphere_spectra,
    plot_paired_band_contrasts,
    quadratic_limb_darkened_stellar_grid,
    quadratic_limb_darkening,
    quadratic_to_kipping_limb_darkening,
)


def test_keplerian_radial_velocity_reduces_to_a_sinusoid_for_circular_orbits():
    time = np.linspace(0.0, 30.0, 200)  # [day]
    velocity = keplerian_radial_velocity(time, 7.0, 3.0, 0.0, 0.4, 1.1)
    expected = 3.0 * np.cos(0.4 + 1.1 + 2.0 * np.pi * time / 7.0)
    assert velocity == pytest.approx(expected, abs=1e-10)


def test_keplerian_radial_velocity_has_zero_time_average_and_broadcasts():
    time = np.linspace(0.0, 10.0, 20001)[:-1]  # [day]
    eccentricity = np.array([0.0, 0.3, 0.7])
    velocity = keplerian_radial_velocity(time[:, None], 10.0, 5.0, eccentricity, 1.0, 0.2)
    assert velocity.shape == (time.size, 3)
    assert np.mean(velocity, axis=0) == pytest.approx(np.zeros(3), abs=1e-6)
    assert np.ptp(velocity, axis=0) == pytest.approx(
        2.0 * 5.0 * np.ones(3), rel=1e-3
    )


def test_kipping_limb_darkening_transforms_round_trip():
    coefficients = np.array((0.4, 0.25))
    kipping_parameters = quadratic_to_kipping_limb_darkening(*coefficients)

    assert kipping_parameters == pytest.approx((0.4225, 0.3076923077))
    assert kipping_to_quadratic_limb_darkening(*kipping_parameters) == pytest.approx(coefficients)


def test_quadratic_limb_darkening_and_grid_preserve_disk_geometry():
    intensity = quadratic_limb_darkening(np.array([0.0, 1.0]), (0.4, 0.25))
    image_x, image_y, radial_distance, brightness = quadratic_limb_darkened_stellar_grid(
        5, (0.4, 0.25), image_limit=2.0
    )

    assert intensity == pytest.approx((0.35, 1.0))
    assert image_x.shape == image_y.shape == radial_distance.shape == brightness.shape == (5, 5)
    assert brightness[2, 2] == pytest.approx(1.0)
    assert brightness[2, 3] == pytest.approx(0.35)
    assert brightness[0, 0] == 0.0


def test_interval_contrast_forecasts_preserve_paired_differences():
    centers = np.array([1.0, 1.1, 1.2, 1.3])
    definitions = {"feature": {"feature": (1.0, 1.2), "continuum": ((1.2, 1.4),)}}
    empirical = empirical_interval_contrasts(
        centers, np.array([2.0, 2.0, 0.0, 0.0]), np.eye(4), definitions
    )
    assert empirical["contrasts"]["feature"]["injected_contrast_ppm"] == 2.0
    paired = paired_interval_contrast_forecast(
        {"d": centers, "e": centers},
        {
            "d": {"model": [2.0, 2.0, 0.0, 0.0]},
            "e": {"model": [1.0, 1.0, 0.0, 0.0]},
        },
        {"d": np.eye(4), "e": np.eye(4)},
        definitions,
    )
    assert paired["models"]["model"]["d_minus_e_contrast_ppm"] == [1.0]


def test_exoplanet_forecast_plots_write_requested_files(tmp_path):
    bands = {"feature": {"label": "X", "feature": (1.0, 1.2)}}
    centers = {"b": np.array([1.0, 1.1]), "c": np.array([1.0, 1.1])}
    spectra = {name: {"model": np.array([-1.0, 1.0])} for name in centers}
    atmosphere_path = plot_atmosphere_spectra(
        centers,
        spectra,
        {name: np.zeros(2) for name in centers},
        {name: np.ones(2) for name in centers},
        bands,
        (("model", "-"),),
        tmp_path / "atmosphere.png",
    )
    contrast_path = plot_paired_band_contrasts(
        {
            "difference_uncertainty_ppm": [1.0],
            "models": {"model": {"d_minus_e_contrast_ppm": [2.0]}},
        },
        bands,
        (("model", "model", "black", "o"),),
        tmp_path / "contrast.pdf",
    )
    assert atmosphere_path.is_file()
    assert contrast_path.is_file()
