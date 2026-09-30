"""Reusable exoplanet forecast calculations and figures."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, ncx2

from .numerics import bin_spectrum, interval_contrast_weights


def quadratic_to_kipping_limb_darkening(
    linear_coefficient: float | np.ndarray,
    quadratic_coefficient: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert quadratic limb-darkening coefficients to Kipping parameters."""

    coefficient_sum = linear_coefficient + quadratic_coefficient
    q1 = coefficient_sum**2
    q2 = linear_coefficient / (2.0 * coefficient_sum)
    return q1, q2


def kipping_to_quadratic_limb_darkening(
    q1: float | np.ndarray,
    q2: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Convert Kipping parameters to quadratic limb-darkening coefficients."""

    coefficient_sum = np.sqrt(q1)
    linear_coefficient = 2.0 * coefficient_sum * q2
    quadratic_coefficient = coefficient_sum * (1.0 - 2.0 * q2)
    return linear_coefficient, quadratic_coefficient


def quadratic_limb_darkening(
    cosine_emission_angle: np.ndarray,
    coefficients: tuple[float, float],
) -> np.ndarray:
    """Evaluate quadratic limb-darkened intensity relative to the disk center."""

    cosine_emission_angle = np.asarray(cosine_emission_angle, dtype=float)
    linear_coefficient, quadratic_coefficient = coefficients
    return (
        1.0
        - linear_coefficient * (1.0 - cosine_emission_angle)
        - quadratic_coefficient * (1.0 - cosine_emission_angle) ** 2
    )


def quadratic_limb_darkened_stellar_grid(
    grid_size: int,
    coefficients: tuple[float, float],
    *,
    image_limit: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sky coordinates, radius, and brightness for a unit stellar disk."""

    if not isinstance(grid_size, int) or grid_size < 3 or grid_size % 2 == 0:
        raise ValueError("grid_size must be an odd integer of at least 3")
    if not np.isfinite(image_limit) or image_limit < 1.0:
        raise ValueError("image_limit must be finite and at least 1")

    coordinates = np.linspace(-image_limit, image_limit, grid_size)
    image_x, image_y = np.meshgrid(coordinates, coordinates)
    radial_distance = np.hypot(image_x, image_y)
    stellar_disk = radial_distance <= 1.0
    cosine_emission_angle = np.sqrt(np.clip(1.0 - radial_distance**2, 0.0, 1.0))
    stellar_brightness = np.zeros_like(radial_distance)
    stellar_brightness[stellar_disk] = quadratic_limb_darkening(
        cosine_emission_angle[stellar_disk], coefficients
    )
    return image_x, image_y, radial_distance, stellar_brightness


def calculate_binned_transmission_spectrum(
    calculator: Any,
    stellar_radius_m: float,
    planet_mass_kg: float,
    planet_radius_m: float,
    temperature_kelvin: float,
    centers_micrometer: np.ndarray,
    *,
    log_metallicity: float,
    carbon_oxygen_ratio: float,
    log10_cloudtop_pressure_pascal: float,
) -> np.ndarray:
    """Calculate a binned equilibrium-chemistry transmission spectrum."""
    wavelength_m, depth, _ = calculator.compute_depths(
        stellar_radius_m,
        planet_mass_kg,
        planet_radius_m,
        temperature_kelvin,
        logZ=log_metallicity,
        CO_ratio=carbon_oxygen_ratio,
        cloudtop_pressure=10.0**log10_cloudtop_pressure_pascal,
    )
    return bin_spectrum(
        wavelength_m * 1.0e6,
        depth * 1.0e6,
        np.asarray(centers_micrometer),
    )  # [ppm]


def empirical_interval_contrasts(
    centers: np.ndarray,
    spectrum: np.ndarray,
    covariance_noise: np.ndarray,
    definitions: Mapping[str, Mapping[str, Any]],
) -> dict:
    """Measure feature-minus-reference contrasts and their covariance."""
    weights = interval_contrast_weights(
        centers,
        [(item["feature"], item["continuum"]) for item in definitions.values()],
    )
    values = weights @ spectrum
    covariance = weights @ covariance_noise @ weights.T
    contrasts = {}
    for index, (name, definition) in enumerate(definitions.items()):
        contrast = float(values[index])
        uncertainty = float(np.sqrt(covariance[index, index]))
        contrasts[name] = {
            "feature_range_micrometer": list(definition["feature"]),
            "reference_ranges_micrometer": [
                list(interval) for interval in definition["continuum"]
            ],
            "weights_by_wavelength_bin": weights[index].tolist(),
            "injected_contrast_ppm": contrast,
            "uncertainty_ppm": uncertainty,
            "injected_signal_to_noise": contrast / uncertainty,
        }
    return {
        "definition": "B = W D with equal positive feature-bin weights and equal negative reference-bin weights",
        "contrasts": contrasts,
        "covariance_ppm2": covariance.tolist(),
    }


def paired_interval_contrast_forecast(
    centers_by_dataset: Mapping[str, np.ndarray],
    model_spectra_by_dataset: Mapping[str, Mapping[str, Sequence[float]]],
    covariance_by_dataset: Mapping[str, np.ndarray],
    definitions: Mapping[str, Mapping[str, Any]],
    *,
    false_positive_probability: float = 0.05,
) -> dict:
    """Forecast paired differences in a vector of interval contrasts."""
    names = list(centers_by_dataset)
    if len(names) != 2:
        raise ValueError("Paired contrasts require exactly two data sets.")
    centers = np.asarray(centers_by_dataset[names[0]])
    if not np.allclose(centers, centers_by_dataset[names[1]]):
        raise ValueError("Paired contrasts require identical wavelength bins.")
    weights = interval_contrast_weights(
        centers,
        [(item["feature"], item["continuum"]) for item in definitions.values()],
    )
    covariance = sum(
        weights @ covariance_by_dataset[name] @ weights.T for name in names
    )
    inverse_covariance = np.linalg.pinv(covariance)
    degrees_freedom = int(np.linalg.matrix_rank(covariance))
    critical_value = chi2.ppf(1.0 - false_positive_probability, degrees_freedom)
    models = {}
    for label in model_spectra_by_dataset[names[0]]:
        values = {
            name: weights @ np.asarray(model_spectra_by_dataset[name][label])
            for name in names
        }
        difference = values[names[0]] - values[names[1]]
        noncentrality = float(difference @ inverse_covariance @ difference)
        models[label] = {
            f"planet_{names[0]}_contrast_ppm": values[names[0]].tolist(),
            f"planet_{names[1]}_contrast_ppm": values[names[1]].tolist(),
            f"{names[0]}_minus_{names[1]}_contrast_ppm": difference.tolist(),
            "global_noncentrality": noncentrality,
            "probability_rejecting_equal_contrast_vector_at_95_percent": float(
                ncx2.sf(critical_value, degrees_freedom, noncentrality)
            ),
        }
    return {
        "band_order": list(definitions),
        "weights_by_wavelength_bin": weights.tolist(),
        "difference_covariance_ppm2": covariance.tolist(),
        "difference_uncertainty_ppm": np.sqrt(np.diag(covariance)).tolist(),
        "global_test": {
            "statistic": "(B_d-B_e)^T Cov(B_d-B_e)^-1 (B_d-B_e)",
            "degrees_freedom": degrees_freedom,
            "threshold": float(critical_value),
            "false_positive_probability": false_positive_probability,
        },
        "models": models,
    }


def plot_atmosphere_spectra(
    centers_by_planet: Mapping[str, np.ndarray],
    model_spectra_by_planet: Mapping[str, Mapping[str, Sequence[float]]],
    measurements_by_planet: Mapping[str, Sequence[float]],
    errors_by_planet: Mapping[str, Sequence[float]],
    band_definitions: Mapping[str, Mapping[str, Any]],
    model_styles: Sequence[tuple[str, Any]],
    output_path: str | Path,
    *,
    planet_labels: Mapping[str, str] | None = None,
    font_size: float = 10.0,  # [point]
    x_limits: tuple[float, float] = (2.87, 5.18),  # [micrometer]
) -> Path:
    """Plot modeled and simulated transmission spectra for multiple planets."""
    output_path = Path(output_path)
    planet_names = list(centers_by_planet)
    labels = planet_labels or {name: name for name in planet_names}
    figure, axes = plt.subplots(
        len(planet_names),
        1,
        figsize=(6.5, 2.6),
        sharex=True,
        facecolor="white",
        squeeze=False,
    )
    axes = axes[:, 0]
    for axis, name in zip(axes, planet_names):
        centers = np.asarray(centers_by_planet[name])  # [micrometer]
        errors = np.asarray(errors_by_planet[name])  # [ppm]
        measurement = np.asarray(measurements_by_planet[name])  # [ppm]
        for definition in band_definitions.values():
            lower, upper = definition["feature"]
            axis.axvspan(lower, upper, color="#ddd8cc", alpha=0.35, zorder=0)
            if name == planet_names[0]:
                axis.text(
                    0.5 * (lower + upper),
                    0.03,
                    definition["label"],
                    transform=axis.get_xaxis_transform(),
                    color="#4b4b4b",
                    fontsize=font_size,
                    ha="center",
                    va="bottom",
                )
        spectra = model_spectra_by_planet[name]
        plotted_values = []
        for model_name, linestyle in model_styles:
            spectrum = np.asarray(spectra[model_name])  # [ppm]
            plotted_values.append(spectrum)
            axis.plot(
                centers,
                spectrum,
                linestyle=linestyle,
                linewidth=1.8,
                label=model_name,
            )
        axis.errorbar(
            centers,
            measurement,
            yerr=errors,
            color="black",
            marker="s",
            markerfacecolor="black",
            linestyle="none",
            markersize=3.0,
            linewidth=0.8,
            capsize=1.5,
            label=r"simulated data (1$\sigma$)",
        )
        plotted_values.extend((measurement - errors, measurement + errors))
        values = np.concatenate(plotted_values)  # [ppm]
        padding = 0.12 * np.ptp(values)  # [ppm]
        axis.set_ylim(np.min(values) - padding, np.max(values) + padding)
        axis.text(
            0.99,
            0.92,
            labels[name],
            transform=axis.transAxes,
            color="black",
            fontsize=font_size,
            fontweight="bold",
            ha="right",
            va="top",
        )
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["bottom", "left"]].set_color("black")
        axis.tick_params(colors="black", labelsize=font_size)
        axis.set_xlim(*x_limits)
        axis.margins(x=0.03, y=0.10)
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.5,
    )
    figure.text(
        0.035,
        0.40,
        "Depth offset [ppm]",
        color="black",
        fontsize=font_size,
        rotation="vertical",
        ha="center",
        va="center",
    )
    axes[-1].set_xlabel("Wavelength [micrometer]", color="black", fontsize=font_size)
    figure.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.74, hspace=0.18)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(figure)
    return output_path


def plot_paired_band_contrasts(
    paired_forecast: Mapping[str, Any],
    band_definitions: Mapping[str, Mapping[str, Any]],
    plotted_cases: Sequence[tuple[str, str, str, str]],
    output_path: str | Path,
    *,
    font_size: float = 12.0,  # [point]
) -> Path:
    """Plot paired feature-minus-reference contrasts for atmosphere models."""
    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(6.5, 2.2), facecolor="white")
    positions = np.arange(len(band_definitions))
    offsets = np.linspace(-0.24, 0.24, len(plotted_cases))
    uncertainties = np.asarray(paired_forecast["difference_uncertainty_ppm"])  # [ppm]
    for offset, (model_name, display_label, color, marker) in zip(
        offsets, plotted_cases
    ):
        values = np.asarray(
            paired_forecast["models"][model_name]["d_minus_e_contrast_ppm"]
        )  # [ppm]
        axis.errorbar(
            positions + offset,
            values,
            yerr=uncertainties,
            color=color,
            marker=marker,
            linestyle="none",
            capsize=2.0,
            label=display_label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(
        positions, [definition["label"] for definition in band_definitions.values()]
    )
    figure.text(
        0.05,
        0.48,
        "Shared-model\n" r"$B_{\rm d}-B_{\rm e}$ [ppm]",
        color="black",
        fontsize=font_size,
        rotation="vertical",
        ha="center",
        va="center",
    )
    axis.tick_params(colors="black", labelsize=font_size)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("black")
    legend_handles, legend_labels = axis.get_legend_handles_labels()
    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.95),
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        ncol=2,
        borderpad=0.35,
        labelspacing=0.25,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
    )
    figure.subplots_adjust(left=0.18, right=0.98, bottom=0.29, top=0.54)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(figure)
    return output_path