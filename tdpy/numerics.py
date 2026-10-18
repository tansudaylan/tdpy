"""Small numerical building blocks for scientific forecast pipelines."""

from collections.abc import Callable

from astropy.stats import gaussian_fwhm_to_sigma as _gaussian_fwhm_to_sigma
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline
from scipy.special import erf, erfinv
from scipy.special import logsumexp as _logsumexp
from scipy.stats import chi2, gaussian_kde, invgamma, ncx2
from scipy.stats import norm as _norm


def gaussian_sigma_from_fwhm(fwhm):
    """Convert Gaussian full width at half maximum to sigma in the input units.

    Scalars, arrays, masked arrays, and Astropy quantities retain their shape
    and units. Validation of physically positive widths belongs to the caller.
    """
    return np.asanyarray(fwhm) * _gaussian_fwhm_to_sigma


def build_kde_prior_from_samples(samples, low=None, high=None, bandwidth=None,
                                 grid_size=4096, *, normalization="trapezoid"):
    """Build density and inverse-CDF grids for a one-dimensional KDE prior.

    Nonfinite samples are discarded. The CDF uses cumulative grid masses,
    matching the Roman samplers. ``normalization='rectangle'`` retains the
    nested sampler's rectangular density normalization.
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        raise ValueError("KDE prior requires at least 2 finite samples.")
    if int(grid_size) < 256:
        raise ValueError("KDE prior grid_size must be at least 256.")
    if normalization not in ("trapezoid", "rectangle"):
        raise ValueError("KDE prior normalization must be 'trapezoid' or 'rectangle'.")

    sample_std = float(np.std(samples, ddof=1))
    sample_range = float(np.max(samples) - np.min(samples))
    padding_scale = 3.0 * (sample_std if sample_std > 0.0 else max(sample_range, 1.0))
    domain_low = float(low) if low is not None else float(np.min(samples) - padding_scale)
    domain_high = float(high) if high is not None else float(np.max(samples) + padding_scale)
    if not np.isfinite(domain_low) or not np.isfinite(domain_high) or domain_high <= domain_low:
        raise ValueError("KDE prior bounds must define a finite interval with high > low.")

    kde = gaussian_kde(samples, **({} if bandwidth is None else {"bw_method": float(bandwidth)}))
    x_grid = np.linspace(domain_low, domain_high, int(grid_size), dtype=float)
    pdf = np.clip(np.asarray(kde(x_grid), dtype=float), 0.0, None)
    dx = float(x_grid[1] - x_grid[0])
    integral = float(trapezoid(pdf, x_grid) if normalization == "trapezoid" else np.sum(pdf) * dx)
    if not np.isfinite(integral) or integral <= 0.0:
        raise ValueError("KDE prior produced a non-finite or zero integral over the sampling domain.")

    pdf /= integral
    cdf = np.cumsum(pdf) * dx
    cdf[0] = 0.0
    cdf[-1] = 1.0
    unique_cdf, unique_idx = np.unique(cdf, return_index=True)
    if unique_cdf.size < 2:
        raise ValueError("KDE prior CDF is degenerate; broaden bounds or use different samples.")
    return {
        "kind": "kde",
        "x_grid": x_grid,
        "pdf_grid": pdf,
        "cdf_grid": unique_cdf,
        "cdf_x_grid": x_grid[unique_idx],
        "low": domain_low,
        "high": domain_high,
        "bandwidth": None if bandwidth is None else float(bandwidth),
        "grid_size": int(grid_size),
        "n_samples": int(samples.size),
    }


def prepare_kde_prior_spec(spec, custom_prior_samples=None, *,
                           prior_builder=build_kde_prior_from_samples,
                           pipeline_name="sampling pipeline"):
    """Copy a parameter spec and resolve its sample-derived KDE grids.

    Sample entries may be arrays or dictionaries with ``samples`` plus
    bandwidth, grid_size, low, and high overrides. A supplied builder retains
    the consumer's grid convention without storing callbacks in the result.
    """
    prepared = dict(spec)
    parameter_name = str(prepared.get("name", "")).strip()
    kind = str(prepared.get("custom_kind", prepared.get("prior_kind", "kde"))).strip().lower()
    if kind != "kde":
        raise ValueError(f"Unsupported custom prior kind '{kind}' for {prepared.get('name', '?')}.")
    custom_prior_samples = {} if custom_prior_samples is None else custom_prior_samples
    if parameter_name not in custom_prior_samples:
        raise ValueError(
            f"Custom KDE prior for parameter {prepared.get('name', '?')} requires an entry in "
            f"custom_prior_samples passed to {pipeline_name}."
        )
    sample_input = custom_prior_samples[parameter_name]
    overrides = sample_input if isinstance(sample_input, dict) else {"samples": sample_input}
    if "samples" not in overrides:
        raise ValueError(f"custom_prior_samples['{parameter_name}'] must contain a 'samples' entry.")
    kind = str(overrides.get("custom_kind", kind)).strip().lower()
    if kind != "kde":
        raise ValueError(f"Unsupported custom prior kind '{kind}' for {prepared.get('name', '?')}.")
    samples = np.asarray(overrides["samples"], dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if samples.size < 2:
        raise ValueError(
            f"Custom KDE prior for parameter {prepared.get('name', '?')} requires at least 2 finite samples."
        )
    defaults = {"low": None, "high": None, "bandwidth": None, "grid_size": 4096}
    options = {
        name: overrides.get(name, prepared.get(name, default))
        for name, default in defaults.items()
    }
    prior_data = prior_builder(samples, **options)
    grid_fields = ("bandwidth", "grid_size", "low", "high", "x_grid", "pdf_grid", "cdf_grid", "cdf_x_grid")
    prepared.update({name: prior_data[name] for name in grid_fields if name in prior_data})
    prepared["custom_kind"] = kind
    return prepared


def prepare_parameter_prior_specs(param_specs, custom_prior_samples=None, *,
                                  prepare_custom_prior=prepare_kde_prior_spec):
    """Copy ordinary parameter specs and precompute custom prior metadata."""
    return [
        prepare_custom_prior(spec, custom_prior_samples=custom_prior_samples)
        if str(spec.get("prior_type", "uniform")).strip().lower() == "custom"
        else dict(spec)
        for spec in param_specs
    ]


def cdfn_self(value, minimum, maximum):
    """Map a uniform variable from its physical interval to the unit interval."""
    return (value - minimum) / (maximum - minimum)


def icdf_self(probability, minimum, maximum):
    """Map a unit-interval probability to a uniform physical interval."""
    return minimum + probability * (maximum - minimum)


def cdfn_logt(value, minimum, maximum):
    """Map a log-uniform variable to the unit interval."""
    return np.log(value / minimum) / np.log(maximum / minimum)


def icdf_logt(probability, minimum, maximum):
    """Map a unit-interval probability to a log-uniform physical interval."""
    return minimum * np.exp(probability * np.log(maximum / minimum))


def cdfn_gaus(value, mean, standard_deviation):
    """Return the Gaussian cumulative probability of a value."""
    return 0.5 * (1.0 + erf((value - mean) / (np.sqrt(2.0) * standard_deviation)))


def icdf_gaus(probability, mean, standard_deviation):
    """Return the Gaussian quantile for a cumulative probability."""
    return mean + np.sqrt(2.0) * standard_deviation * erfinv(2.0 * probability - 1.0)


def cdfn_expo(value, maximum, scale):
    """Return the exponential cumulative probability."""
    return (1.0 - np.exp(-value / scale)) / (1.0 - np.exp(-maximum / scale))


def icdf_expo(probability, maximum, scale):
    """Return the exponential quantile for a cumulative probability."""
    return -scale * np.log(1.0 - probability * (1.0 - np.exp(-maximum / scale)))


def cdfn_dexp(value, maximum, scale):
    """Return the double-exponential cumulative probability."""
    val = np.asarray(value)
    if val.ndim == 0:
        if val < 0.0:
            return 0.5 * (1.0 - cdfn_expo(-val, maximum, scale))
        return 0.5 * (1.0 + cdfn_expo(val, maximum, scale))
    pos = val >= 0.0
    res = np.empty_like(val, dtype=float)
    res[pos] = 0.5 * (1.0 + cdfn_expo(val[pos], maximum, scale))
    res[~pos] = 0.5 * (1.0 - cdfn_expo(-val[~pos], maximum, scale))
    return res


def icdf_dexp(probability, maximum, scale):
    """Return the double-exponential quantile for a cumulative probability."""
    prob = np.asarray(probability)
    if prob.ndim == 0:
        if prob < 0.5:
            return -icdf_expo(1.0 - 2.0 * prob, maximum, scale)
        return icdf_expo(2.0 * (prob - 0.5), maximum, scale)
    res = np.empty_like(prob, dtype=float)
    lower = prob < 0.5
    res[lower] = -icdf_expo(1.0 - 2.0 * prob[lower], maximum, scale)
    res[~lower] = icdf_expo(2.0 * (prob[~lower] - 0.5), maximum, scale)
    return res


def pdfn_self(value, minimum, maximum):
    """Return the uniform probability density on [minimum, maximum]."""
    density = 1.0 / (maximum - minimum)
    if np.ndim(value) > 0:
        return np.full(np.shape(value), density, dtype=float)
    return density


def pdfn_logt(value, minimum, maximum):
    """Return the log-uniform probability density on [minimum, maximum]."""
    return 1.0 / (np.log(maximum) - np.log(minimum)) / value


def pdfn_gaus(value, mean, standard_deviation):
    """Return the Gaussian probability density."""
    return 1.0 / (np.sqrt(2.0 * np.pi) * standard_deviation) * np.exp(
        -0.5 * ((value - mean) / standard_deviation) ** 2
    )


def logmeanexp(log_values, axis=None, *, keepdims=False):
    """Return the log of a mean in linear space without exponent overflow.

    Reduction axes and retained dimensions follow SciPy's ``logsumexp``.
    Inputs must contain at least one value.
    """
    values = np.asarray(log_values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot average an empty set of log values")
    total = _logsumexp(values, axis=axis, keepdims=keepdims)
    count = values.size // np.size(total)
    return total - np.log(count)


def logpdf_gaus(value, mean, standard_deviation):
    """Return normalized Gaussian log densities without density underflow.

    Inputs broadcast as in SciPy's normal distribution. Values, means, and
    standard deviations must be expressed in the same units.
    """
    return _norm.logpdf(value, loc=mean, scale=standard_deviation)


def pdfn_lnor(value, mean, standard_deviation):
    """Return the log-normal probability density."""
    return pdfn_gaus(np.log(value), np.log(mean), standard_deviation)


pdfn_lgau = pdfn_lnor


def pdfn_expo(value, maximum, scale):
    """Return the truncated exponential probability density on [0, maximum]."""
    arr = np.asarray(value)
    norm = 1.0 / scale / (1.0 - np.exp(-maximum / scale))
    if (arr > maximum).any():
        if arr.ndim == 0:
            return 0.0
        return np.where(arr > maximum, 0.0, norm * np.exp(-arr / scale))
    return norm * np.exp(-value / scale)


def pdfn_dexp(value, maximum, scale):
    """Return the double-exponential probability density on [-maximum, maximum]."""
    return 0.5 * pdfn_expo(np.fabs(value), maximum, scale)


def pdfn_powr(value, minimum, maximum, slope):
    """Return the power-law probability density."""
    norm = (1.0 - slope) / (maximum ** (1.0 - slope) - minimum ** (1.0 - slope))
    return norm * value ** (-slope)


def pdfn_dpow(value, minimum, maximum, break_value, slope_lower, slope_upper):
    """Return the broken power-law probability density."""
    xdat = np.atleast_1d(np.asarray(value))
    faca = 1.0 / (
        break_value ** (slope_lower - slope_upper)
        * (break_value ** (1.0 - slope_lower) - minimum ** (1.0 - slope_lower))
        / (1.0 - slope_lower)
        + (maximum ** (1.0 - slope_upper) - break_value ** (1.0 - slope_upper))
        / (1.0 - slope_upper)
    )
    pdfn = np.empty_like(xdat, dtype=float)
    indxlowr = np.where(xdat <= break_value)[0]
    indxuppr = np.where(xdat > break_value)[0]
    if indxlowr.size > 0:
        pdfn[indxlowr] = faca * break_value ** (slope_lower - slope_upper) * xdat[indxlowr] ** (-slope_lower)
    if indxuppr.size > 0:
        pdfn[indxuppr] = faca * xdat[indxuppr] ** (-slope_upper)
    return pdfn[0] if np.isscalar(value) else pdfn


def pdfn_igam(value, slope, cutoff):
    """Return the inverse-gamma probability density."""
    return invgamma.pdf(value, slope - 1.0, scale=cutoff)


def pdfn_atan(value, minimum, maximum):
    """Return the arctan-transformed probability density."""
    return 1.0 / (value ** 2 + 1.0) / (np.arctan(maximum) - np.arctan(minimum))


def find_nearest_index(array, value):
    """Return the index of the element in array closest to value."""
    arr = np.asarray(array)
    return int((np.abs(arr - value)).argmin())


find_nearest = find_nearest_index


def rescale_range(data, old_min, old_max, new_min, new_max):
    """Linearly map data from interval [old_min, old_max] to [new_min, new_max]."""
    return new_min + (new_max - new_min) / (old_max - old_min) * (data - old_min)


range_shift = rescale_range


class RectilinearInterpolator2D:
    """Interpolate a finite rectangular grid with explicit coordinate ordering.

    One-dimensional axes use values shaped (len(y), len(x)). Aligned 2D
    mesh coordinates may use either NumPy meshgrid indexing convention.
    Calls use Cartesian products by default; ``grid=False`` evaluates paired,
    broadcastable coordinates. Samples outside the grid use ``fill_value``;
    ``None`` retains the spline engine's boundary extension.
    """

    def __init__(self, x_coordinates, y_coordinates, values, *, degree=3, fill_value=0.0):
        x_coordinates, y_coordinates, values = (
            np.asarray(array, dtype=float) for array in (x_coordinates, y_coordinates, values)
        )
        if x_coordinates.size == 0 or y_coordinates.size == 0:
            raise ValueError("coordinate grids must not be empty")
        if x_coordinates.ndim == y_coordinates.ndim == 2 and x_coordinates.shape == y_coordinates.shape == values.shape:
            shape = x_coordinates.shape
            if (np.array_equal(x_coordinates, np.broadcast_to(x_coordinates[0, :], shape))
                    and np.array_equal(y_coordinates, np.broadcast_to(y_coordinates[:, 0, None], shape))):
                x_coordinates, y_coordinates = x_coordinates[0, :], y_coordinates[:, 0]
            elif (np.array_equal(x_coordinates, np.broadcast_to(x_coordinates[:, 0, None], shape))
                    and np.array_equal(y_coordinates, np.broadcast_to(y_coordinates[0, :], shape))):
                x_coordinates, y_coordinates, values = x_coordinates[:, 0], y_coordinates[0, :], values.T
            else:
                raise ValueError("mesh coordinates must describe a rectangular grid")
        if x_coordinates.ndim != 1 or y_coordinates.ndim != 1 or values.shape != (y_coordinates.size, x_coordinates.size):
            raise ValueError("values must match the rectangular x/y coordinate grid")
        if (not np.isfinite(x_coordinates).all() or not np.isfinite(y_coordinates).all() or not np.isfinite(values).all()
                or np.any(np.diff(x_coordinates) <= 0.0) or np.any(np.diff(y_coordinates) <= 0.0)):
            raise ValueError("grid values must be finite and axes strictly increasing")
        if degree not in (1, 2, 3) or min(x_coordinates.size, y_coordinates.size) <= degree:
            raise ValueError("degree must be 1, 2, or 3 with more grid samples than degree")
        self.bounds = (x_coordinates[0], x_coordinates[-1], y_coordinates[0], y_coordinates[-1])
        self.fill_value = fill_value
        self.spline = RectBivariateSpline(x_coordinates, y_coordinates, values.T, kx=degree, ky=degree, s=0)

    def __call__(self, x_coordinates, y_coordinates, *, grid=True):
        x_coordinates, y_coordinates = np.asarray(x_coordinates, dtype=float), np.asarray(y_coordinates, dtype=float)
        if grid:
            if x_coordinates.ndim > 1 or y_coordinates.ndim > 1:
                raise ValueError("grid evaluation requires scalar or one-dimensional coordinates")
            x_coordinates, y_coordinates = np.meshgrid(np.atleast_1d(x_coordinates), np.atleast_1d(y_coordinates))
        else:
            x_coordinates, y_coordinates = np.broadcast_arrays(x_coordinates, y_coordinates)
        values = self.spline.ev(x_coordinates.ravel(), y_coordinates.ravel()).reshape(x_coordinates.shape)
        if self.fill_value is None:
            return values
        xmin, xmax, ymin, ymax = self.bounds
        inside = (np.isfinite(x_coordinates) & np.isfinite(y_coordinates)
                  & (x_coordinates >= xmin) & (x_coordinates <= xmax)
                  & (y_coordinates >= ymin) & (y_coordinates <= ymax))
        return np.where(inside, values, self.fill_value)


def midpoint_bin_edges(coordinates):
    """Infer edges from increasing samples using adjacent arithmetic midpoints.

    Outer edges extend by half the first and last sample separations. Input
    and output coordinates use the same units; irregular spacing is retained.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 1 or coordinates.size < 2:
        raise ValueError("coordinates must be a 1D array with at least two samples")
    if not np.isfinite(coordinates).all() or np.any(np.diff(coordinates) <= 0.0):
        raise ValueError("coordinates must be finite and strictly increasing")
    edges = np.empty(coordinates.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coordinates[:-1] + coordinates[1:])
    edges[0] = coordinates[0] - 0.5 * (coordinates[1] - coordinates[0])
    edges[-1] = coordinates[-1] + 0.5 * (coordinates[-1] - coordinates[-2])
    return edges


def bin_spectrum(
    wavelength: np.ndarray,
    values: np.ndarray,
    centers: np.ndarray,
    half_width: float = 0.05,
) -> np.ndarray:
    """Average sampled values within fixed-width bins around each center."""
    binned = np.full(centers.size, np.nan)
    for index, center in enumerate(centers):
        selected = np.abs(wavelength - center) < half_width
        binned[index] = np.mean(values[selected])
    return binned


def grouped_polynomial_design_matrix(coordinates, groups, degree=1, *, normalize=False):
    """Build centered polynomial columns for each sorted group label.

    Columns are ordered by group, then by increasing power starting at zero.
    Each column vanishes outside its group. With ``normalize=True``, centered
    coordinates are divided by the group's coordinate range.
    """
    coordinates, groups = np.asarray(coordinates, dtype=float), np.asarray(groups)
    if coordinates.ndim != 1 or groups.shape != coordinates.shape or coordinates.size == 0:
        raise ValueError("coordinates and groups must be nonempty matching 1D arrays")
    if not isinstance(degree, (int, np.integer)) or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    columns = []
    for group in np.unique(groups):
        selected = groups == group
        if normalize:
            centered = np.zeros_like(coordinates)
            centered[selected] = (coordinates[selected] - coordinates[selected].mean()) / np.ptp(coordinates[selected])
        else:
            centered = np.where(selected, coordinates - coordinates[selected].mean(), 0.)
        columns.append(selected.astype(float))
        for power in range(1, degree + 1):
            columns.append(np.where(selected, centered**power, 0.))
    return np.column_stack(columns)


def finite_difference_jacobian(
    parameters: np.ndarray,
    steps: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Evaluate centered finite-difference derivatives for a vector model."""
    columns = []
    for index, step in enumerate(steps):
        upper = parameters.copy()
        lower = parameters.copy()
        upper[index] += step
        lower[index] -= step
        columns.append((model(upper) - model(lower)) / (2.0 * step))
    return np.column_stack(columns)


def linearized_gaussian_retrieval(
    parameters: np.ndarray,
    steps: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    covariance: np.ndarray,
    prior_sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return parameter covariance, recovery operator, and local Jacobian."""
    jacobian = finite_difference_jacobian(parameters, steps, model)
    inverse_covariance = np.linalg.inv(covariance)
    fisher = jacobian.T @ inverse_covariance @ jacobian
    prior_sigmas = np.asarray(prior_sigmas, dtype=float)
    if prior_sigmas.shape != (jacobian.shape[1],) or np.isnan(prior_sigmas).any() or np.any(prior_sigmas <= 0.0):
        raise ValueError("prior_sigmas must contain one positive value or infinity per parameter.")
    prior_precision = np.zeros_like(prior_sigmas)
    finite_prior = np.isfinite(prior_sigmas)
    prior_precision[finite_prior] = prior_sigmas[finite_prior] ** -2
    posterior_precision = fisher + np.diag(prior_precision)
    posterior_precision = 0.5 * (posterior_precision + posterior_precision.T)
    try:
        np.linalg.cholesky(posterior_precision)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The local posterior is rank deficient; constrain every null direction with a proper prior."
        ) from error
    parameter_covariance = np.linalg.solve(
        posterior_precision, np.eye(posterior_precision.shape[0])
    )
    recovery = parameter_covariance @ jacobian.T @ inverse_covariance
    return parameter_covariance, recovery, jacobian


def profile_grid_models(
    data: np.ndarray,
    models: np.ndarray,
    covariance: np.ndarray,
    prior_penalties: np.ndarray,
    parameters: list[dict],
    profile_keys: tuple[str, ...],
) -> dict:
    """Profile an additive offset and retain the best model per parameter group."""
    inverse_covariance = np.linalg.inv(covariance)
    ones = np.ones(data.size)
    residuals = data[None, :] - models
    offsets = (residuals @ inverse_covariance @ ones) / (ones @ inverse_covariance @ ones)
    residuals -= offsets[:, None]
    scores = (
        np.einsum("ij,jk,ik->i", residuals, inverse_covariance, residuals)
        + prior_penalties
    )
    profiles = {}
    for index, values in enumerate(parameters):
        group = tuple(values[key] for key in profile_keys)
        if group not in profiles or scores[index] < profiles[group]["score"]:
            profiles[group] = {
                "score": float(scores[index]),
                "parameters": values,
                "depth_offset_ppm": float(offsets[index]),
            }
    return profiles


def paired_shared_parameter_forecast(
    jacobians: dict[str, np.ndarray],
    covariances: dict[str, np.ndarray],
    prior_sigmas: dict[str, np.ndarray],
    shared_parameter_indices: tuple[int, ...],
    difference_grids: dict[int, tuple[float, ...]],
    false_positive_probability: float = 0.05,
) -> dict:
    """Forecast tests of shared parameters for two independent data sets."""
    names = list(jacobians)
    if len(names) != 2:
        raise ValueError("A paired forecast requires exactly two data sets.")
    parameter_count = jacobians[names[0]].shape[1]
    row_count = sum(jacobians[name].shape[0] for name in names)
    independent = np.zeros((row_count, parameter_count * 2))
    row_start = 0
    for dataset_index, name in enumerate(names):
        row_stop = row_start + jacobians[name].shape[0]
        columns = slice(dataset_index * parameter_count, (dataset_index + 1) * parameter_count)
        independent[row_start:row_stop, columns] = np.linalg.solve(
            np.linalg.cholesky(covariances[name]), jacobians[name]
        )
        row_start = row_stop
    prior_rows = []
    for dataset_index, name in enumerate(names):
        for parameter_index, sigma in enumerate(prior_sigmas[name]):
            if np.isfinite(sigma):
                row = np.zeros(independent.shape[1])
                row[dataset_index * parameter_count + parameter_index] = 1.0 / sigma
                prior_rows.append(row)
    if prior_rows:
        independent = np.vstack((independent, prior_rows))
    nuisance_indices = [
        index for index in range(parameter_count) if index not in shared_parameter_indices
    ]
    mapping = np.zeros(
        (independent.shape[1], len(shared_parameter_indices) + 2 * len(nuisance_indices))
    )
    for dataset_index in range(2):
        start = dataset_index * parameter_count
        for shared_column, parameter_index in enumerate(shared_parameter_indices):
            mapping[start + parameter_index, shared_column] = 1.0
        for nuisance_column, parameter_index in enumerate(nuisance_indices):
            mapping[
                start + parameter_index,
                len(shared_parameter_indices) + dataset_index * len(nuisance_indices) + nuisance_column,
            ] = 1.0
    shared = independent @ mapping
    degrees_freedom = independent.shape[1] - shared.shape[1]
    critical_value = chi2.ppf(1.0 - false_positive_probability, degrees_freedom)
    grids = {}
    for parameter_index, differences in difference_grids.items():
        grid = {}
        for difference in differences:
            parameters = np.zeros(independent.shape[1])
            parameters[parameter_index] = difference / 2.0
            parameters[parameter_count + parameter_index] = -difference / 2.0
            signal = independent @ parameters
            residual = signal - shared @ (np.linalg.pinv(shared) @ signal)
            noncentrality = float(residual @ residual)
            grid[str(difference)] = {
                "noncentrality": noncentrality,
                "expected_delta_chi_squared": degrees_freedom + noncentrality,
                "probability_rejecting_shared_model_at_95_percent": float(
                    ncx2.sf(critical_value, degrees_freedom, noncentrality)
                ),
            }
        grids[parameter_index] = grid
    return {"degrees_freedom": degrees_freedom, "difference_injection_grids": grids}


def exponential_noise_covariance(
    locations: np.ndarray,
    errors: np.ndarray,
    systematic_noise: float = 0.0,
    correlation_length: float | None = None,
) -> np.ndarray:
    """Return diagonal measurement covariance plus exponential covariance."""
    covariance = np.diag(errors**2)
    if systematic_noise > 0.0:
        if correlation_length is None or correlation_length <= 0.0:
            raise ValueError("correlation_length must be positive")
        separation = np.abs(locations[:, None] - locations[None, :])
        covariance += systematic_noise**2 * np.exp(-separation / correlation_length)
    return covariance


def simulate_observation(
    signal: np.ndarray,
    covariance: np.ndarray,
    random_generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw multivariate noise and add it to a model signal."""
    noise = random_generator.multivariate_normal(np.zeros(signal.size), covariance)
    return noise, signal + noise


def interval_contrast_weights(
    locations: np.ndarray,
    definitions: list[tuple[tuple[float, float], tuple[tuple[float, float], ...]]],
) -> np.ndarray:
    """Build equal-weight feature-minus-reference contrasts over intervals."""
    weights = np.zeros((len(definitions), locations.size))
    for row, (feature_interval, reference_intervals) in enumerate(definitions):
        feature = (locations >= feature_interval[0]) & (locations < feature_interval[1])
        reference = np.zeros(locations.size, dtype=bool)
        for lower, upper in reference_intervals:
            reference |= (locations >= lower) & (locations < upper)
        if not np.any(feature) or not np.any(reference):
            raise ValueError("Every contrast requires sampled feature and reference intervals.")
        weights[row, feature] = 1.0 / np.count_nonzero(feature)
        weights[row, reference] = -1.0 / np.count_nonzero(reference)
    return weights


def template_amplitude_forecast(
    template: np.ndarray,
    true_covariance: np.ndarray,
    random_generator: np.random.Generator,
    number_trials: int,
    analysis_covariance: np.ndarray | None = None,
    injected_amplitudes: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> dict:
    """Forecast a covariance-weighted template-amplitude test against a constant."""
    if analysis_covariance is None:
        analysis_covariance = true_covariance
    design = np.column_stack((np.ones(template.size), template))
    inverse_covariance = np.linalg.inv(analysis_covariance)
    parameter_covariance = np.linalg.inv(design.T @ inverse_covariance @ design)
    amplitude_uncertainty = np.sqrt(parameter_covariance[1, 1])
    recovery = parameter_covariance @ design.T @ inverse_covariance
    sampling_uncertainty = np.sqrt(recovery[1] @ true_covariance @ recovery[1])
    threshold = np.sqrt(10.0 + np.log(template.size))
    noise = random_generator.normal(scale=sampling_uncertainty, size=number_trials)
    return {
        "amplitude_uncertainty": float(amplitude_uncertainty),
        "true_sampling_uncertainty": float(sampling_uncertainty),
        "injection_grid": {
            str(amplitude): float(
                np.mean(np.abs(amplitude + noise) / amplitude_uncertainty > threshold)
            )
            for amplitude in injected_amplitudes
        },
        "criterion": "ln K_A,flat > 5, with ln K approximated as Delta BIC / 2",
    }


def template_goodness_of_fit(
    injected_template: np.ndarray,
    candidate_template: np.ndarray,
    covariance: np.ndarray,
    false_positive_probability: float = 0.05,
) -> dict:
    """Forecast lack-of-fit power after fitting an offset and template amplitude."""
    design = np.column_stack((np.ones(candidate_template.size), candidate_template))
    inverse_covariance = np.linalg.inv(covariance)
    fitted = np.linalg.solve(
        design.T @ inverse_covariance @ design,
        design.T @ inverse_covariance @ injected_template,
    )
    residual = injected_template - design @ fitted
    noncentrality = float(residual @ inverse_covariance @ residual)
    degrees_freedom = candidate_template.size - design.shape[1]
    critical_value = chi2.ppf(1.0 - false_positive_probability, degrees_freedom)
    return {
        "noncentrality": noncentrality,
        "expected_reduced_chi_squared": (degrees_freedom + noncentrality) / degrees_freedom,
        "probability_rejecting_candidate_at_95_percent": float(
            ncx2.sf(critical_value, degrees_freedom, noncentrality)
        ),
        "degrees_freedom": degrees_freedom,
    }


def weighted_linear_regression(
    data: np.ndarray,
    design_matrix: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum likelihood estimates and parameter covariance for linear regression.

    Parameters
    ----------
    data : np.ndarray
        Observed 1D data vector of shape (N,).
    design_matrix : np.ndarray
        Linear model design matrix of shape (N, M).
    variance : np.ndarray
        Data noise variance vector of shape (N,) or full covariance matrix (N, N).

    Returns
    -------
    parameters : np.ndarray
        Best-fit regression parameters of shape (M,).
    covariance : np.ndarray
        Parameter covariance matrix of shape (M, M).
    """
    data = np.asarray(data)
    design_matrix = np.asarray(design_matrix)
    variance = np.asarray(variance)

    if variance.ndim == 1:
        inv_variance = 1.0 / variance
        design_weighted = design_matrix * inv_variance[:, None]
        normal_matrix = np.matmul(design_matrix.T, design_weighted)
        covariance = np.linalg.inv(normal_matrix)
        rhs = np.matmul(design_weighted.T, data)
        parameters = np.matmul(covariance, rhs)
    else:
        inv_variance = np.linalg.inv(variance)
        normal_matrix = np.matmul(design_matrix.T, np.matmul(inv_variance, design_matrix))
        covariance = np.linalg.inv(normal_matrix)
        rhs = np.matmul(design_matrix.T, np.matmul(inv_variance, data))
        parameters = np.matmul(covariance, rhs)

    return parameters, covariance


retr_mlikregr = weighted_linear_regression


def robust_scatter(values, axis=None, scale=1.4826):
    """Return scaled median absolute deviation, ignoring masked/nonfinite values.

    The output uses the input's units. The dimensionless default scale is the
    existing approximate Gaussian scatter conversion; all-invalid slices
    return NaN without warnings.
    """
    masked = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    center = np.ma.median(masked, axis=axis, keepdims=True)
    deviation = np.ma.median(np.ma.abs(masked - center), axis=axis)
    return np.ma.filled(scale * deviation, np.nan)


def weighted_quantile(
    values: np.ndarray,
    quantiles: float | list[float] | np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float | np.ndarray:
    """Compute 1D quantiles with optional sample weights."""
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1 or vals.size == 0 or not np.isfinite(vals).all():
        raise ValueError("values must be a nonempty 1D array of finite numbers.")
    quants = np.asarray(quantiles, dtype=float)
    if np.any((quants < 0.0) | (quants > 1.0)) or not np.isfinite(quants).all():
        raise ValueError("quantiles must be within [0, 1].")

    if sample_weight is None:
        result = np.quantile(vals, quants)
    else:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != vals.shape or not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("sample_weight must match values shape with finite non-negative values.")
        total_weight = weights.sum()
        if total_weight <= 0.0:
            raise ValueError("sample_weight must have positive sum.")
        sorter = np.argsort(vals)
        vals_sorted = vals[sorter]
        weights_sorted = weights[sorter]
        cdf = np.cumsum(weights_sorted) / total_weight
        result = np.interp(quants, cdf, vals_sorted)

    if np.ndim(quantiles) == 0:
        return float(result)
    return result


def weighted_median(
    values: np.ndarray,
    sample_weight: np.ndarray | None = None,
    axis: int | None = None,
) -> float | np.ndarray:
    """Compute median or weighted median along specified axis."""
    vals = np.asarray(values, dtype=float)
    if vals.ndim == 1 or axis is None:
        return weighted_quantile(vals.ravel(), 0.50, sample_weight=sample_weight)
    if axis == 0 and vals.ndim == 2:
        if sample_weight is None:
            return np.median(vals, axis=0)
        return np.array([
            weighted_quantile(vals[:, col_idx], 0.50, sample_weight=sample_weight)
            for col_idx in range(vals.shape[1])
        ])
    raise NotImplementedError("weighted_median currently supports 1D arrays or 2D arrays with axis=0.")


def grid_quantiles(
    grid: np.ndarray,
    probability_density: np.ndarray,
    quantiles: float | list[float] | np.ndarray = (0.16, 0.50, 0.84),
) -> float | np.ndarray:
    """Compute credible intervals/quantiles from a 1D probability density on a grid."""
    grid_arr = np.asarray(grid, dtype=float)
    pdf_arr = np.asarray(probability_density, dtype=float)
    if grid_arr.ndim != 1 or pdf_arr.ndim != 1 or grid_arr.size != pdf_arr.size:
        raise ValueError("grid and probability_density must be 1D arrays of matching length.")
    if grid_arr.size < 2:
        raise ValueError("grid must contain at least 2 points.")
    if np.any(np.diff(grid_arr) <= 0.0):
        raise ValueError("grid must be strictly increasing.")
    if np.any(pdf_arr < 0.0) or not np.isfinite(pdf_arr).all():
        raise ValueError("probability_density must be finite and non-negative.")

    total_prob = trapezoid(pdf_arr, grid_arr)
    if total_prob <= 0.0:
        raise ValueError("probability_density must have positive integral.")
    normalized_pdf = pdf_arr / total_prob
    dx = np.diff(grid_arr)
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (normalized_pdf[1:] + normalized_pdf[:-1]) * dx)))
    cumulative = np.clip(cumulative, 0.0, 1.0)
    cumulative[-1] = 1.0
    quants = np.asarray(quantiles, dtype=float)
    result = np.interp(quants, cumulative, grid_arr)
    if np.ndim(quantiles) == 0:
        return float(result)
    return result
