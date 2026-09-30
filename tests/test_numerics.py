import numpy as np
import pytest

from tdpy.numerics import (
    bin_spectrum,
    cdfn_gaus,
    cdfn_logt,
    cdfn_self,
    exponential_noise_covariance,
    finite_difference_jacobian,
    interval_contrast_weights,
    icdf_gaus,
    icdf_logt,
    icdf_self,
    linearized_gaussian_retrieval,
    paired_shared_parameter_forecast,
    profile_grid_models,
    simulate_observation,
    template_amplitude_forecast,
    template_goodness_of_fit,
)


def test_prior_transforms_round_trip():
    assert icdf_self(cdfn_self(3.0, 1.0, 5.0), 1.0, 5.0) == pytest.approx(3.0)
    assert icdf_logt(cdfn_logt(10.0, 1.0, 100.0), 1.0, 100.0) == pytest.approx(10.0)
    assert icdf_gaus(cdfn_gaus(3.0, 2.0, 4.0), 2.0, 4.0) == pytest.approx(3.0)


def test_bin_spectrum_averages_samples_within_each_bin():
    wavelength = np.array([1.00, 1.02, 1.10, 1.12])
    values = np.array([2.0, 4.0, 10.0, 14.0])
    centers = np.array([1.01, 1.11])
    np.testing.assert_allclose(
        bin_spectrum(wavelength, values, centers, half_width=0.03),
        [3.0, 12.0],
    )


def test_finite_difference_jacobian_matches_linear_model():
    matrix = np.array([[2.0, 3.0], [-1.0, 4.0]])
    jacobian = finite_difference_jacobian(
        np.array([1.0, 2.0]),
        np.array([0.1, 0.2]),
        lambda parameters: matrix @ parameters,
    )
    np.testing.assert_allclose(jacobian, matrix, atol=1.0e-12)


def test_linearized_gaussian_retrieval_returns_covariance_and_recovery():
    parameter_covariance, recovery, jacobian = linearized_gaussian_retrieval(
        np.array([1.0, 2.0]),
        np.array([0.1, 0.1]),
        lambda values: np.array([values[0], values[1]]),
        np.diag([4.0, 9.0]),
        np.array([np.inf, 3.0]),
    )
    np.testing.assert_allclose(jacobian, np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(parameter_covariance, np.diag([4.0, 4.5]))
    np.testing.assert_allclose(recovery, np.diag([1.0, 0.5]))


def test_profile_grid_models_selects_best_nuisance_model_per_group():
    profiles = profile_grid_models(
        np.array([3.0, 5.0]),
        np.array([[1.0, 3.0], [2.0, 6.0], [5.0, 7.0]]),
        np.eye(2),
        np.array([0.0, 0.0, 0.0]),
        [{"group": 0, "nuisance": 0}, {"group": 0, "nuisance": 1}, {"group": 1, "nuisance": 0}],
        ("group",),
    )
    assert profiles[(0,)]["parameters"]["nuisance"] == 0
    assert profiles[(0,)]["depth_offset_ppm"] == pytest.approx(2.0)
    assert profiles[(1,)]["score"] == pytest.approx(0.0)


def test_paired_shared_parameter_forecast_matches_analytical_difference():
    result = paired_shared_parameter_forecast(
        {"first": np.eye(2), "second": np.eye(2)},
        {"first": np.eye(2), "second": np.eye(2)},
        {"first": np.full(2, np.inf), "second": np.full(2, np.inf)},
        (0,),
        {0: (0.0, 1.0)},
    )
    grid = result["difference_injection_grids"][0]
    assert result["degrees_freedom"] == 1
    assert grid["0.0"]["noncentrality"] == pytest.approx(0.0)
    assert grid["1.0"]["noncentrality"] == pytest.approx(0.5)


def test_exponential_noise_covariance_adds_correlated_component():
    covariance = exponential_noise_covariance(
        np.array([0.0, 1.0]),
        np.array([2.0, 3.0]),
        systematic_noise=1.0,
        correlation_length=2.0,
    )
    np.testing.assert_allclose(
        covariance,
        [[5.0, np.exp(-0.5)], [np.exp(-0.5), 10.0]],
    )


def test_exponential_noise_covariance_requires_scale_for_systematic_noise():
    with pytest.raises(ValueError, match="correlation_length"):
        exponential_noise_covariance(
            np.array([0.0]), np.array([1.0]), systematic_noise=1.0
        )


def test_simulate_observation_returns_noise_and_noisy_signal():
    signal = np.array([2.0, 3.0])
    noise, observed = simulate_observation(
        signal, np.eye(2), np.random.default_rng(123)
    )
    np.testing.assert_allclose(observed, signal + noise)


def test_interval_contrast_weights_average_feature_and_reference_bins():
    weights = interval_contrast_weights(
        np.array([1.0, 1.1, 1.2, 1.3]),
        [((1.0, 1.2), ((1.2, 1.4),))],
    )
    np.testing.assert_allclose(weights, [[0.5, 0.5, -0.5, -0.5]])


def test_interval_contrast_weights_require_both_intervals():
    with pytest.raises(ValueError, match="feature and reference"):
        interval_contrast_weights(np.array([1.0]), [((1.0, 1.1), ((2.0, 2.1),))])


def test_template_amplitude_forecast_is_deterministic_and_covariance_weighted():
    result = template_amplitude_forecast(
        np.array([-1.0, 0.0, 1.0]),
        np.eye(3),
        np.random.default_rng(123),
        1000,
    )
    assert result["amplitude_uncertainty"] == pytest.approx(2.0**-0.5)
    assert result["true_sampling_uncertainty"] == pytest.approx(2.0**-0.5)
    assert result["injection_grid"] == {"0.0": 0.0, "0.5": 0.002, "1.0": 0.022}


def test_template_goodness_of_fit_accepts_affine_template():
    result = template_goodness_of_fit(
        3.0 + 2.0 * np.arange(4.0),
        np.arange(4.0),
        np.eye(4),
    )
    assert result["degrees_freedom"] == 2
    assert result["noncentrality"] == pytest.approx(0.0, abs=1.0e-28)
    assert result["probability_rejecting_candidate_at_95_percent"] == pytest.approx(0.05)