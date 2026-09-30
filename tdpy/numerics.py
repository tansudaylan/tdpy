"""Small numerical building blocks for scientific forecast pipelines."""

from collections.abc import Callable

import numpy as np
from scipy.special import erf, erfinv
from scipy.stats import chi2, ncx2


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
    prior_precision = np.where(np.isfinite(prior_sigmas), prior_sigmas**-2, 0.0)
    parameter_covariance = np.linalg.pinv(fisher + np.diag(prior_precision))
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