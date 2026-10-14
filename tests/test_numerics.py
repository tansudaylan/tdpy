import numpy as np
import pytest

from tdpy.numerics import (
    bin_spectrum,
    build_kde_prior_from_samples,
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
    robust_scatter,
    range_shift,
    linearized_gaussian_retrieval,
    midpoint_bin_edges,
    paired_shared_parameter_forecast,
    prepare_kde_prior_spec,
    prepare_parameter_prior_specs,
    profile_grid_models,
    RectilinearInterpolator2D,
    simulate_observation,
    template_amplitude_forecast,
    template_goodness_of_fit,
)
from tdpy.util import cdfn_expo, gdatstrtpcat


@pytest.mark.parametrize("normalization", ["trapezoid", "rectangle"])
def test_kde_prior_preserves_roman_grid_calculations(normalization):
    from scipy.integrate import trapezoid
    from scipy.stats import gaussian_kde

    samples = np.array([-1.0, -0.2, 0.0, 0.4, 2.0])
    prior = build_kde_prior_from_samples(
        np.append(samples, [np.nan, np.inf]), low=-0.5, high=1.5,
        bandwidth=0.3, grid_size=256, normalization=normalization,
    )
    grid = np.linspace(-0.5, 1.5, 256)
    density = gaussian_kde(samples, bw_method=0.3)(grid)
    spacing = grid[1] - grid[0]
    integral = trapezoid(density, grid) if normalization == "trapezoid" else np.sum(density) * spacing
    density /= integral
    cumulative = np.cumsum(density) * spacing
    cumulative[0], cumulative[-1] = 0.0, 1.0
    cumulative, indices = np.unique(cumulative, return_index=True)
    np.testing.assert_array_equal(prior["x_grid"], grid)
    np.testing.assert_allclose(prior["pdf_grid"], density, rtol=1e-14)
    np.testing.assert_allclose(prior["cdf_grid"], cumulative, rtol=1e-14)
    np.testing.assert_array_equal(prior["cdf_x_grid"], grid[indices])
    assert prior["n_samples"] == samples.size


def test_kde_prior_automatic_bounds_and_inverse_endpoints():
    samples = np.array([-1.0, 0.0, 2.0])
    prior = build_kde_prior_from_samples(samples)
    padding = 3.0 * np.std(samples, ddof=1)
    assert prior["low"] == pytest.approx(samples.min() - padding)
    assert prior["high"] == pytest.approx(samples.max() + padding)
    np.testing.assert_allclose(
        np.interp([0.0, 1.0], prior["cdf_grid"], prior["cdf_x_grid"]),
        [prior["low"], prior["high"]],
    )


@pytest.mark.parametrize("kwargs, message", [
    ({"samples": [np.nan, 1.0]}, "at least 2 finite samples"),
    ({"grid_size": 255}, "at least 256"),
    ({"low": 1.0, "high": 1.0}, "finite interval"),
    ({"low": np.nan}, "finite interval"),
    ({"normalization": "unknown"}, "normalization"),
])
def test_kde_prior_rejects_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_kde_prior_from_samples(**({"samples": [-1.0, 0.0, 1.0]} | kwargs))


@pytest.mark.parametrize("use_overrides", [False, True])
def test_kde_spec_preparation_resolves_overrides_without_mutating_inputs(use_overrides):
    spec = {"name": "ampl", "prior_type": "custom", "low": -1.0, "high": 2.0,
            "bandwidth": 0.2, "grid_size": 512, "label": "Amplitude"}
    original_spec = dict(spec)
    samples = np.array([-0.5, 0.0, 0.5, 1.0, np.nan])
    overrides = {"samples": samples, "low": -0.8, "high": 1.5, "bandwidth": 0.4,
                 "grid_size": 256, "custom_kind": " KDE "}
    sample_input = overrides if use_overrides else samples
    prepared = prepare_kde_prior_spec(spec, {"ampl": sample_input})
    options = overrides if use_overrides else spec
    expected = build_kde_prior_from_samples(
        samples, low=options["low"], high=options["high"],
        bandwidth=options["bandwidth"], grid_size=options["grid_size"],
    )
    assert spec == original_spec
    assert np.isnan(samples[-1])
    assert prepared["custom_kind"] == "kde"
    assert prepared["label"] == "Amplitude"
    assert "kind" not in prepared and "n_samples" not in prepared
    for name in ("x_grid", "pdf_grid", "cdf_grid", "cdf_x_grid"):
        np.testing.assert_array_equal(prepared[name], expected[name])
    assert prepared["low"] == expected["low"] and prepared["high"] == expected["high"]


def test_parameter_prior_specs_dispatch_only_custom_specs_and_copy_others():
    from unittest.mock import Mock

    specs = [{"name": "fixed", "prior_type": "fixed", "fixed_value": 1.0},
             {"name": "custom", "prior_type": " CUSTOM "}]
    samples = {"custom": [0.0, 1.0]}
    custom = Mock(return_value={"name": "custom", "prepared": True})
    prepared = prepare_parameter_prior_specs(specs, samples, prepare_custom_prior=custom)
    assert prepared[0] == specs[0] and prepared[0] is not specs[0]
    assert prepared[1] == {"name": "custom", "prepared": True}
    custom.assert_called_once_with(specs[1], custom_prior_samples=samples)
    assert "prepared" not in specs[1]


@pytest.mark.parametrize("spec, samples, message", [
    ({"name": "ampl", "custom_kind": "other"}, None, "Unsupported custom prior kind"),
    ({"name": "ampl"}, None, "requires an entry"),
    ({"name": "ampl"}, {"ampl": {}}, "contain a 'samples' entry"),
    ({"name": "ampl"}, {"ampl": {"samples": [0.0, 1.0], "custom_kind": "other"}}, "Unsupported custom prior kind"),
    ({"name": "ampl"}, {"ampl": [1.0, np.nan]}, "at least 2 finite samples"),
])
def test_kde_spec_preparation_rejects_invalid_metadata(spec, samples, message):
    with pytest.raises(ValueError, match=message):
        prepare_kde_prior_spec(spec, samples)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_robust_scatter_matches_masked_median_definition(axis):
    values = np.array([[1.0, 10.0, np.nan], [3.0, np.inf, np.nan], [100.0, 14.0, np.nan]])
    masked = np.ma.masked_invalid(values)
    center = np.ma.median(masked, axis=axis, keepdims=True)
    expected = np.ma.filled(1.4826 * np.ma.median(np.ma.abs(masked - center), axis=axis), np.nan)
    np.testing.assert_array_equal(robust_scatter(values, axis=axis), expected)


def test_robust_scatter_preserves_explicit_masks_and_scale():
    values = np.ma.array([1, 3, 100], mask=[False, False, True])
    assert robust_scatter(values) == pytest.approx(1.4826)
    assert robust_scatter(values, scale=1.0) == pytest.approx(1.0)
    np.testing.assert_array_equal(values.data, [1, 3, 100])
    np.testing.assert_array_equal(values.mask, [False, False, True])


def test_robust_scatter_all_invalid_slices_are_nan_without_warnings():
    import warnings

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = robust_scatter([[np.nan, np.inf], [np.nan, -np.inf]], axis=0)
    assert np.isnan(result).all()
    assert not captured
    assert robust_scatter([5.0, 5.0, 5.0]) == 0.0


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


@pytest.mark.parametrize("indexing", ["axes", "xy", "ij"])
def test_rectilinear_interpolation_preserves_orientation_and_cubic_values(indexing):
    import pickle

    x_axis = np.linspace(-2.0, 2.0, 5)
    y_axis = np.linspace(-1.0, 2.0, 6)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="ij" if indexing == "ij" else "xy")
    values = x_grid**3 + 2.0 * y_grid**2 + x_grid * y_grid
    coordinates = (x_axis, y_axis) if indexing == "axes" else (x_grid, y_grid)
    interpolate = RectilinearInterpolator2D(*coordinates, values)
    query_x = np.array([1.25, -0.5, 0.2])
    query_y = np.array([0.3, -0.75, 1.8])
    expected = query_x**3 + 2.0 * query_y**2 + query_x * query_y
    np.testing.assert_allclose(interpolate(query_x, query_y, grid=False), expected, atol=1e-12)
    restored = pickle.loads(pickle.dumps(interpolate))
    np.testing.assert_allclose(restored(query_x, query_y, grid=False), expected, atol=1e-12)
    sampled_x, sampled_y = np.meshgrid(query_x, query_y)
    np.testing.assert_allclose(
        interpolate(query_x, query_y), sampled_x**3 + 2.0 * sampled_y**2 + sampled_x * sampled_y,
        atol=1e-12,
    )


def test_rectilinear_interpolation_optional_boundary_extension_matches_spline_engine():
    from scipy.interpolate import RectBivariateSpline

    x_axis = np.array([0.0, 1.0, 3.0])
    y_axis = np.array([-1.0, 1.0, 2.0, 4.0])
    x_grid, y_grid = np.meshgrid(x_axis, y_axis)
    values = 2.0 * x_grid + 3.0 * y_grid
    interpolate = RectilinearInterpolator2D(x_axis, y_axis, values, degree=1, fill_value=None)
    reference = RectBivariateSpline(x_axis, y_axis, values.T, kx=1, ky=1, s=0)
    query_x = np.array([-2.0, 0.5, 5.0])
    query_y = np.array([-3.0, 0.0, 6.0])
    np.testing.assert_array_equal(interpolate(query_x, query_y), reference(query_x, query_y).T)
    np.testing.assert_array_equal(
        interpolate(query_x, query_y, grid=False), reference.ev(query_x, query_y),
    )
    assert interpolate(query_x, query_y).dtype.kind == "f"


def test_rectilinear_interpolation_fill_and_paired_broadcasting():
    axis = np.arange(4.0)
    interpolate = RectilinearInterpolator2D(axis, axis, np.ones((4, 4)), fill_value=-7.0)
    np.testing.assert_allclose(
        interpolate([-0.1, 0.0, 3.0, 3.1, np.nan], 1.0, grid=False), [-7.0, 1.0, 1.0, -7.0, -7.0],
    )
    assert interpolate(1.0, 2.0, grid=False).shape == ()
    assert interpolate([0.0, 1.0], [0.0, 1.0, 2.0]).shape == (3, 2)


@pytest.mark.parametrize("x_coordinates, y_coordinates, values", [
    ([0.0, 1.0], [0.0, 1.0], np.ones((3, 2))),
    ([0.0, 0.0], [0.0, 1.0], np.ones((2, 2))),
    ([1.0, 0.0], [0.0, 1.0], np.ones((2, 2))),
    ([0.0, np.nan], [0.0, 1.0], np.ones((2, 2))),
    ([0.0, 1.0], [0.0, 1.0], [[1.0, np.inf], [1.0, 1.0]]),
    ([], [], np.empty((0, 0))),
])
def test_rectilinear_interpolation_rejects_invalid_grids(x_coordinates, y_coordinates, values):
    with pytest.raises(ValueError):
        RectilinearInterpolator2D(x_coordinates, y_coordinates, values, degree=1)


@pytest.mark.parametrize("coordinates, expected", [
    ([0, 1, 2], [-0.5, 0.5, 1.5, 2.5]),
    ([1.0, 2.0, 6.0], [0.5, 1.5, 4.0, 8.0]),
    ([-3.0, -1.0], [-4.0, -2.0, 0.0]),
])
def test_midpoint_bin_edges_preserves_sample_spacing(coordinates, expected):
    coordinates = np.asarray(coordinates)
    original = coordinates.copy()
    edges = midpoint_bin_edges(coordinates)
    np.testing.assert_array_equal(edges, expected)
    np.testing.assert_array_equal(coordinates, original)
    assert edges.dtype.kind == "f"
    assert np.all(coordinates > edges[:-1])
    assert np.all(coordinates < edges[1:])


@pytest.mark.parametrize("coordinates", [
    [], [1.0], [[1.0, 2.0]], [1.0, 1.0], [2.0, 1.0],
    [0.0, np.nan], [0.0, np.inf],
])
def test_midpoint_bin_edges_rejects_invalid_coordinates(coordinates):
    with pytest.raises(ValueError, match="coordinates"):
        midpoint_bin_edges(coordinates)


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


    def test_weighted_quantile_and_median():
        from tdpy.numerics import weighted_quantile, weighted_median

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 10.0, 1.0, 1.0])

        # Unweighted quantile matches np.quantile
        np.testing.assert_allclose(
            weighted_quantile(values, [0.25, 0.5, 0.75]),
            np.quantile(values, [0.25, 0.5, 0.75]),
        )
        assert weighted_median(values) == pytest.approx(3.0)

        # Weighted quantile pulls median towards value with highest weight
        assert weighted_median(values, weights) == pytest.approx(3.0)

        # 2D samples across axis=0
        samples_2d = np.column_stack([values, values * 2.0])
        medians_2d = weighted_median(samples_2d, weights, axis=0)
        np.testing.assert_allclose(medians_2d, [3.0, 6.0])

        # Edge cases
        with pytest.raises(ValueError, match="values must be a nonempty"):
            weighted_quantile(np.array([]), 0.5)
        with pytest.raises(ValueError, match="quantiles must be within"):
            weighted_quantile(values, 1.5)


    def test_grid_quantiles():
        from tdpy.numerics import grid_quantiles

        grid = np.linspace(-5.0, 5.0, 2001)
        pdf = np.exp(-0.5 * grid**2) / np.sqrt(2.0 * np.pi)
        quantiles = grid_quantiles(grid, pdf, [0.158655, 0.5, 0.841345])
        np.testing.assert_allclose(quantiles, [-1.0, 0.0, 1.0], atol=1e-2)


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