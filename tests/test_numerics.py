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
    find_nearest,
    find_nearest_index,
    rescale_range,
    range_shift,
    linearized_gaussian_retrieval,
    paired_shared_parameter_forecast,
    profile_grid_models,
    simulate_observation,
    template_amplitude_forecast,
    template_goodness_of_fit,
)
from tdpy.util import cdfn_expo, gdatstrtpcat


def test_prior_transforms_round_trip():
    assert icdf_self(cdfn_self(3.0, 1.0, 5.0), 1.0, 5.0) == pytest.approx(3.0)
    assert icdf_logt(cdfn_logt(10.0, 1.0, 100.0), 1.0, 100.0) == pytest.approx(10.0)
    assert icdf_gaus(cdfn_gaus(3.0, 2.0, 4.0), 2.0, 4.0) == pytest.approx(3.0)


def test_exponential_cdf_uses_scale_and_reaches_upper_endpoint():
    assert cdfn_expo(2.0, 10.0, 2.0) == pytest.approx(
        (1.0 - np.exp(-1.0)) / (1.0 - np.exp(-5.0))
    )
    assert cdfn_expo(10.0, 10.0, 2.0) == pytest.approx(1.0)


def test_pcat_state_accepts_valid_attribute_assignment():
    state = gdatstrtpcat()
    state.testname = "value"
    assert state.testname == "value"


def test_time_func_returns_timing_statistics(monkeypatch):
    import tdpy.util as util

    times = iter(np.arange(200, dtype=float) * 0.5)
    monkeypatch.setattr(util.time, "time", lambda: next(times))

    mean, standard_deviation = util.time_func(lambda: None)

    assert mean == pytest.approx(0.5)
    assert standard_deviation == pytest.approx(0.0)


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


def test_linearized_gaussian_retrieval_rejects_unconstrained_null_direction():
    model = lambda values: np.array([values[0] + values[1]])

    with pytest.raises(ValueError, match="rank deficient"):
        linearized_gaussian_retrieval(
            np.zeros(2), np.ones(2), model, np.eye(1), np.full(2, np.inf)
        )


def test_find_nearest_index():
    array = np.array([1.0, 2.5, 5.0, 10.0])
    assert find_nearest_index(array, 2.4) == 1
    assert find_nearest(array, 9.8) == 3
    assert find_nearest(array, 0.0) == 0


def test_rescale_range():
    val = np.array([0.0, 0.5, 1.0])
    rescaled = rescale_range(val, 0.0, 1.0, 10.0, 20.0)
    np.testing.assert_allclose(rescaled, [10.0, 15.0, 20.0])
    assert range_shift(0.5, 0.0, 1.0, -np.pi, np.pi) == pytest.approx(0.0)


def test_analytical_probability_densities():
    from tdpy.numerics import (
        cdfn_dexp,
        cdfn_expo,
        icdf_dexp,
        icdf_expo,
        pdfn_atan,
        pdfn_dexp,
        pdfn_dpow,
        pdfn_expo,
        pdfn_gaus,
        pdfn_igam,
        pdfn_lgau,
        pdfn_lnor,
        pdfn_logt,
        pdfn_powr,
        pdfn_self,
    )

    # Uniform PDF
    assert pdfn_self(2.0, 1.0, 5.0) == pytest.approx(0.25)
    np.testing.assert_allclose(pdfn_self(np.array([2.0, 3.0]), 1.0, 5.0), [0.25, 0.25])

    # Log-uniform PDF
    assert pdfn_logt(np.e, 1.0, np.e**2) == pytest.approx(1.0 / (2.0 * np.e))

    # Gaussian and Log-normal PDF
    assert pdfn_gaus(0.0, 0.0, 1.0) == pytest.approx(1.0 / np.sqrt(2.0 * np.pi))
    assert pdfn_lnor(np.e, 1.0, 1.0) == pytest.approx(pdfn_gaus(1.0, 0.0, 1.0))
    assert pdfn_lgau(np.e, 1.0, 1.0) == pytest.approx(pdfn_lnor(np.e, 1.0, 1.0))

    # Exponential PDF and CDF/ICDF round-trip
    prob_exp = cdfn_expo(2.0, 10.0, 2.0)
    assert icdf_expo(prob_exp, 10.0, 2.0) == pytest.approx(2.0)
    assert pdfn_expo(0.0, 10.0, 2.0) > 0.0
    assert pdfn_expo(15.0, 10.0, 2.0) == 0.0

    # Double exponential
    assert pdfn_dexp(0.0, 10.0, 2.0) == pytest.approx(0.5 * pdfn_expo(0.0, 10.0, 2.0))
    prob_dexp = cdfn_dexp(1.5, 10.0, 2.0)
    assert icdf_dexp(prob_dexp, 10.0, 2.0) == pytest.approx(1.5)

    # Power law
    assert pdfn_powr(2.0, 1.0, 4.0, 2.0) > 0.0

    # Broken power law
    dpow_val = pdfn_dpow(2.0, 1.0, 10.0, 3.0, 1.5, 2.5)
    assert dpow_val > 0.0

    # Inverse gamma and arctan
    assert pdfn_igam(2.0, 3.0, 1.0) > 0.0
    assert pdfn_atan(0.0, -1.0, 1.0) == pytest.approx(1.0 / (np.pi / 2.0))


def test_linearized_gaussian_retrieval_proper_prior_constrains_null_direction():
    model = lambda values: np.array([values[0] + values[1]])

    parameter_covariance, recovery, jacobian = linearized_gaussian_retrieval(
        np.zeros(2), np.ones(2), model, np.eye(1), np.array([np.inf, 1.0])
    )

    np.testing.assert_allclose(jacobian, [[1.0, 1.0]])
    assert np.linalg.eigvalsh(parameter_covariance).min() > 0.0
    assert np.isfinite(recovery).all()


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


def test_retr_llik_corr_evaluates_bivariate_log_likelihood():
    from tdpy.util import retr_llik_corr, gdatstrt

    gdat = gdatstrt()
    gdat.tempfrst = np.array([0.0, 1.0, 2.0])
    gdat.tempfrststdv = np.array([0.1, 0.1, 0.1])
    gdat.tempseco = np.array([0.0, 1.0, 2.0])
    gdat.tempsecostdv = np.array([0.1, 0.1, 0.1])

    # angle = 3*pi/4 gives slope = -1/tan(3*pi/4) = 1.0, intercept = 0.0
    angle = 3.0 * np.pi / 4.0
    intercept = 0.0
    llik_good = retr_llik_corr((angle, intercept), gdat)
    assert np.isfinite(llik_good)

    # Offset intercept should decrease log likelihood
    llik_bad = retr_llik_corr((angle, 5.0), gdat)
    assert llik_good > llik_bad


def test_weighted_linear_regression_recovers_known_parameters():
    from tdpy.numerics import weighted_linear_regression, retr_mlikregr

    x = np.linspace(0.0, 10.0, 50)
    true_params = np.array([2.5, 1.8])
    design = np.column_stack([np.ones_like(x), x])
    variance = np.full_like(x, 0.25)
    data = design @ true_params

    params, cov = weighted_linear_regression(data, design, variance)
    np.testing.assert_allclose(params, true_params, rtol=1e-10)
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) > 0)

    # Test alias retr_mlikregr
    params_alias, cov_alias = retr_mlikregr(data, design, variance)
    np.testing.assert_allclose(params_alias, true_params, rtol=1e-10)

    # Test 2D covariance matrix input
    params_2d, cov_2d = weighted_linear_regression(data, design, np.diag(variance))
    np.testing.assert_allclose(params_2d, true_params, rtol=1e-10)


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