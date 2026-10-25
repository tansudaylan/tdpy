"""Reusable exoplanet forecast calculations and figures."""

from tdpy.verbosity import print

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, ncx2

from .numerics import bin_spectrum, interval_contrast_weights
from .plotting import WASHU_GRAY, WASHU_GREEN, WASHU_RED


_DISCOVERY_TARGET_COLORS = (
    WASHU_RED,
    WASHU_GREEN,
    WASHU_GRAY,
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#E69F00",
)


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


def stellar_limb_darkened_brightness(
    cosine_emission_angle: float | np.ndarray,
    coefficients: tuple[float, ...] | list[float] | np.ndarray,
    baseline_brightness: float | np.ndarray = 1.0,
    law: str = "quad",
    *,
    brgtraww: float | np.ndarray | None = None,
    typelmdk: str | None = None,
) -> float | np.ndarray:
    """Evaluate limb-darkened stellar surface brightness for common laws."""

    if brgtraww is not None:
        baseline_brightness = brgtraww
    if typelmdk is not None:
        law = typelmdk

    cosg = np.asarray(cosine_emission_angle, dtype=float)
    if law == "linr":
        factor = 1.0 - coefficients[0] * (1.0 - cosg)
    elif law in ("quad", "quadkipp"):
        factor = quadratic_limb_darkening(cosg, coefficients)
    elif law == "nlin":
        if len(coefficients) != 3:
            raise ValueError("The nlin limb-darkening law requires three coefficients.")
        factor = (
            1.0
            - coefficients[0] * (1.0 - np.sqrt(np.maximum(cosg, 0.0)))
            - coefficients[1] * (1.0 - cosg)
            - coefficients[2] * (1.0 - cosg**1.5)
        )
    elif law == "none":
        factor = np.ones_like(cosg)
    else:
        raise ValueError(f"Unsupported limb-darkening law {law!r}.")

    result = np.asarray(baseline_brightness) * factor
    if cosg.ndim == 0 and np.ndim(baseline_brightness) == 0:
        return float(result)
    return result


retr_brgtlmdk = stellar_limb_darkened_brightness


def relative_flux_from_delta_magnitude(
    delta_magnitude: float | np.ndarray,
    uncertainty: float | np.ndarray | None = None,
    *,
    stdvdmag: float | np.ndarray | None = None,
) -> float | np.ndarray | tuple[float | np.ndarray, float | np.ndarray]:
    """Convert delta magnitude to relative flux and optionally propagate error."""

    if uncertainty is None and stdvdmag is not None:
        uncertainty = stdvdmag

    dmag = np.asarray(delta_magnitude, dtype=float)
    flux = 10.0 ** (-dmag / 2.5)
    if uncertainty is not None:
        flux_err = np.log(10.0) / 2.5 * flux * np.asarray(uncertainty, dtype=float)
        return flux, flux_err
    if dmag.ndim == 0:
        return float(flux)
    return flux


retr_rflxfromdmag = relative_flux_from_delta_magnitude


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


def solve_kepler_equation(mean_anomaly, eccentricity):
    """Solve E - e sin(E) = M using the maintained Newton iteration.

    Anomalies use radians, eccentricity is dimensionless, and inputs
    broadcast. The mean anomaly is not wrapped, preserving orbital cycles.
    """
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    eccentricity = np.asarray(eccentricity, dtype=float)
    eccentric_anomaly = mean_anomaly + eccentricity * np.sin(mean_anomaly)
    for _ in range(100):
        residual = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
        eccentric_anomaly = eccentric_anomaly - residual / (1.0 - eccentricity * np.cos(eccentric_anomaly))
        if np.all(np.abs(residual) < 1e-12):
            return eccentric_anomaly
    raise RuntimeError("Kepler solver did not converge.")


def keplerian_radial_velocity(
    time: np.ndarray,
    period: float | np.ndarray,
    semi_amplitude: float | np.ndarray,
    eccentricity: float | np.ndarray,
    argument_periastron: float | np.ndarray,
    mean_anomaly_reference: float | np.ndarray,
    time_reference: float = 0.0,
) -> np.ndarray:
    """Return the stellar Keplerian radial velocity [m/s].

    Units are time and period [day], semi-amplitude [m/s], and angles [rad].
    All inputs broadcast, so ``time[:, None]`` with per-planet arrays of
    shape ``(M,)`` returns one curve per planet with shape ``(N, M)``.
    """

    eccentricity = np.asarray(eccentricity, dtype=float)
    mean_anomaly = mean_anomaly_reference + 2.0 * np.pi * (np.asarray(time) - time_reference) / period

    eccentric_anomaly = solve_kepler_equation(mean_anomaly, eccentricity)
    true_anomaly = 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )
    return semi_amplitude * (
        np.cos(argument_periastron + true_anomaly) + eccentricity * np.cos(argument_periastron)
    )


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
            "reference_ranges_micrometer": [list(interval) for interval in definition["continuum"]],
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
    covariance = sum(weights @ covariance_by_dataset[name] @ weights.T for name in names)
    inverse_covariance = np.linalg.pinv(covariance)
    degrees_freedom = int(np.linalg.matrix_rank(covariance))
    critical_value = chi2.ppf(1.0 - false_positive_probability, degrees_freedom)
    models = {}
    for label in model_spectra_by_dataset[names[0]]:
        values = {
            name: weights @ np.asarray(model_spectra_by_dataset[name][label]) for name in names
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
    for offset, (model_name, display_label, color, marker) in zip(offsets, plotted_cases):
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
    axis.set_xticks(positions, [definition["label"] for definition in band_definitions.values()])
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


def _prepare_discovery_planets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep confirmed-planet rows with finite discovery, period, and radius data."""
    planets = []
    for row in rows:
        try:
            discovery_year = int(float(row["disc_year"]))
            orbital_period = float(row["pl_orbper"])
            planet_radius = float(row["pl_rade"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite((discovery_year, orbital_period, planet_radius)).all():
            continue
        if orbital_period <= 0.0 or planet_radius <= 0.0:
            continue
        hostname = str(row.get("hostname", "")).strip()
        planet_name = str(row.get("pl_name", "")).strip()
        if not hostname or not planet_name:
            continue
        planets.append(
            {
                "hostname": hostname,
                "pl_name": planet_name,
                "discoverymethod": str(row.get("discoverymethod") or "Unknown"),
                "disc_year": discovery_year,
                "pl_orbper": orbital_period,
                "pl_rade": planet_radius,
            }
        )
    if not planets:
        raise ValueError("rows must contain confirmed planets with discovery year, period, and radius")
    return planets


def _discovery_target_colors(
    highlighted_targets: Mapping[str, str],
    target_colors: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return stable colors for highlighted host names."""
    supplied_colors = target_colors or {}
    return {
        hostname: supplied_colors.get(hostname, _DISCOVERY_TARGET_COLORS[index % len(_DISCOVERY_TARGET_COLORS)])
        for index, hostname in enumerate(highlighted_targets)
    }


def plot_exoplanet_discovery_timeline(
    rows: Sequence[Mapping[str, Any]],
    highlighted_targets: Mapping[str, str],
    output_path: str | Path,
    *,
    target_colors: Mapping[str, str] | None = None,
) -> Path:
    """Plot confirmed-planet discoveries by year and method, marking selected systems."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    planets = _prepare_discovery_planets(rows)
    output_path = Path(output_path)
    discovery_years = np.asarray([planet["disc_year"] for planet in planets])
    years = np.arange(discovery_years.min(), discovery_years.max() + 1)
    method_totals: dict[str, int] = {}
    for planet in planets:
        method = planet["discoverymethod"]
        method_totals[method] = method_totals.get(method, 0) + 1
    ranked_methods = sorted(method_totals, key=method_totals.get, reverse=True)
    displayed_methods = ranked_methods[:5]
    if len(ranked_methods) > len(displayed_methods):
        displayed_methods.append("Other methods")
    method_colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(10.0, 4.8), facecolor="white")
    bottom = np.zeros(years.size, dtype=int)
    method_handles = []
    for method_index, method in enumerate(displayed_methods):
        selected = [
            planet for planet in planets
            if planet["discoverymethod"] == method
            or (method == "Other methods" and planet["discoverymethod"] not in ranked_methods[:5])
        ]
        counts = np.asarray(
            [sum(planet["disc_year"] == year for planet in selected) for year in years],
            dtype=int,
        )
        color = method_colors(method_index / max(len(displayed_methods), 1))
        axis.bar(years, counts, bottom=bottom, width=0.86, color=color, linewidth=0.0)
        bottom += counts
        method_handles.append(Patch(facecolor=color, label=method))

    target_styles = _discovery_target_colors(highlighted_targets, target_colors)
    target_handles = []
    for hostname, label in highlighted_targets.items():
        target_planets = [planet for planet in planets if planet["hostname"] == hostname]
        if not target_planets:
            continue
        discovery_year = min(planet["disc_year"] for planet in target_planets)
        color = target_styles[hostname]
        axis.axvline(discovery_year, color=color, linewidth=1.4, linestyle="--", alpha=0.9)
        target_handles.append(
            Line2D([], [], color=color, linestyle="--", linewidth=1.6,
                   label=f"{label} ({discovery_year})")
        )

    axis.set_xlabel("Discovery year")
    axis.set_ylabel("Confirmed planets discovered")
    axis.set_xlim(years[0] - 0.6, years[-1] + 0.6)
    tick_step = 1 if years.size <= 16 else 2
    axis.set_xticks(years[::tick_step])
    axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        method_handles + target_handles,
        [handle.get_label() for handle in method_handles + target_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    figure.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300 if output_path.suffix.lower() == ".png" else None,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_exoplanet_period_radius(
    rows: Sequence[Mapping[str, Any]],
    highlighted_targets: Mapping[str, str],
    output_path: str | Path,
    *,
    target_colors: Mapping[str, str] | None = None,
) -> Path:
    """Plot confirmed-planet radius against orbital period and highlight target systems."""
    planets = _prepare_discovery_planets(rows)
    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(8.2, 5.2), facecolor="white")
    axis.scatter(
        [planet["pl_orbper"] for planet in planets],
        [planet["pl_rade"] for planet in planets],
        s=12,
        color="#9BA3A5",
        alpha=0.28,
        linewidths=0.0,
        rasterized=True,
        label="Other confirmed planets",
    )
    target_styles = _discovery_target_colors(highlighted_targets, target_colors)
    for hostname, label in highlighted_targets.items():
        target_planets = [planet for planet in planets if planet["hostname"] == hostname]
        if not target_planets:
            continue
        axis.scatter(
            [planet["pl_orbper"] for planet in target_planets],
            [planet["pl_rade"] for planet in target_planets],
            s=58,
            color=target_styles[hostname],
            edgecolors="white",
            linewidths=0.7,
            label=f"{label} ({len(target_planets)})",
            zorder=3,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Orbital period [day]")
    axis.set_ylabel("Planet radius [Earth radii]")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300 if output_path.suffix.lower() == ".png" else None,
                   bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def orbital_phase(time, period, epoch=0.0, *, center=0.5, time_units=False):
    """Fold times into [center - 0.5, center + 0.5) without sorting.

    Time, period, and epoch must use the same units. The epoch has phase
    zero modulo one. ``time_units=True`` scales the phase by the period.
    Scalar and array inputs retain their shape.
    """
    initial_phase = ((np.asarray(time) - epoch) % period) / period
    shift = 0.5 - center
    phase = (initial_phase + shift) % 1.0 - shift
    return phase * period if time_units else phase


def fold_tser(
    arry: np.ndarray,
    epoc: float,
    peri: float,
    boolxdattime: bool = False,
    boolsort: bool = True,
    phascntr: float = 0.5,
    booldiag: bool = True,
) -> np.ndarray:
    """Fold a 2D or 3D time series on period and reference epoch."""

    arry = np.asarray(arry)
    if arry.ndim == 3:
        time = arry[:, 0, 0]
    elif arry.ndim == 2:
        time = arry[:, 0]
    else:
        raise ValueError('arry should be two- or three-dimensional.')

    arryfold = np.empty_like(arry)

    xdat = orbital_phase(time, peri, epoc, center=phascntr, time_units=boolxdattime)

    if arry.ndim == 3:
        arryfold[:, 0, 0] = xdat
    else:
        arryfold[:, 0] = xdat

    arryfold[:, ..., 1:] = arry[:, ..., 1:]

    if boolsort:
        indx = np.argsort(xdat)
        arryfold = arryfold[indx]

    return arryfold


def retr_lcurmodl_flarsing(
    meantime: np.ndarray,
    timeflar: float,
    amplflar: float,
    scalrise: float,
    scalfall: float,
) -> np.ndarray:
    """Evaluate a fast-rise exponential-decay stellar flare light curve."""

    time_array = np.asarray(meantime, dtype=float)
    if time_array.size == 0:
        raise ValueError('meantime must not be empty.')
    lcur = np.empty_like(time_array)
    mask_rise = time_array < timeflar
    lcur[mask_rise] = np.exp((time_array[mask_rise] - timeflar) / scalrise)
    lcur[~mask_rise] = np.exp(-(time_array[~mask_rise] - timeflar) / scalfall)
    peak_flux = np.amax(lcur)
    if peak_flux > 0.0:
        lcur *= amplflar / peak_flux
    return lcur


def evaluate_davenport_flare(
    time: float | np.ndarray,
    peak_time: float,
    fwhm: float,
    amplitude: float,
) -> float | np.ndarray:
    """Evaluate the classical Davenport et al. (2014) stellar flare model.

    Parameters
    ----------
    time : float or np.ndarray
        Time array to evaluate the flare model over.
    peak_time : float
        Time of peak flare flux.
    fwhm : float
        Full width at half maximum of the flare in units of time.
    amplitude : float
        Relative peak flare amplitude.

    Returns
    -------
    float or np.ndarray
        Flare flux model on the time grid (zero outside flare).
    """
    t = np.asarray(time, dtype=float)
    phase = (t - peak_time) / fwhm
    rise = (
        1.00000
        + 1.94053 * phase
        - 0.175084 * phase**2
        - 2.24588 * phase**3
        - 1.12498 * phase**4
    )
    decay = 0.689008 * np.exp(-1.60053 * phase) + 0.302963 * np.exp(-0.278318 * phase)
    shape = np.where(phase < -1.0, 0.0, np.where(phase <= 0.0, rise, decay))
    shape = np.clip(shape, 0.0, None)
    flux = shape * np.abs(amplitude)
    if np.ndim(time) == 0:
        return float(flux)
    return flux


def davenport_flare_area(
    fwhm: float | np.ndarray,
    amplitude: float | np.ndarray,
) -> float | np.ndarray:
    """Return the analytical integrated area of a Davenport et al. (2014) flare."""
    area_factor = 1.8268848560866364
    area = np.abs(amplitude) * np.abs(fwhm) * area_factor
    if np.ndim(amplitude) == 0 and np.ndim(fwhm) == 0:
        return float(area)
    return area


def retr_smaxkepl(
    peri: float | np.ndarray,
    masstotl: float | np.ndarray,
    factnonk: float = 1.0,
) -> float | np.ndarray:
    """Calculate the semi-major axis in AU using Kepler's third law."""

    kepler_factor = 7.496e-6  # [AU^3 / (M_sun day^2)]
    smax = (kepler_factor * masstotl * (peri / factnonk) ** 2) ** (1.0 / 3.0)  # [AU]
    return smax


def retr_perikepl(
    smax: float | np.ndarray,
    masstotl: float | np.ndarray,
    factnonk: float = 1.0,
) -> float | np.ndarray:
    """Calculate the orbital period in days using Kepler's third law."""

    kepler_factor = 7.496e-6  # [AU^3 / (M_sun day^2)]
    peri = np.sqrt(smax ** 3 / kepler_factor / masstotl) * factnonk  # [day]
    return peri


def retr_vesc(
    massplan: float | np.ndarray,
    radicomp: float | np.ndarray,
) -> float | np.ndarray:
    """Return the planetary surface escape velocity in km/s."""

    escape_velocity_earth = 11.2  # [km / s]
    return escape_velocity_earth * np.sqrt(np.asarray(massplan) / np.asarray(radicomp))


planetary_escape_velocity = retr_vesc


def energy_limited_mass_loss(planet_mass, planet_radius, xuv_flux, efficiency=0.1,
                             absorption_radius=None, roche_factor=1.0):
    """Return a static energy-limited escape rate as an Astropy mass/time quantity.

    Physical inputs require units. The calculation is
    eta * pi * R_abs**2 * F_XUV / (G * M_p / R_p * K).
    R_abs defaults to R_p, and the dimensionless Roche factor K defaults to
    unity. The caller supplies K, rather than inferring a Roche geometry here.
    Inputs broadcast. Masked observations, nonfinite or nonphysical inputs,
    efficiencies outside [0, 1], K outside (0, 1], and R_abs < R_p are rejected.
    This energy budget does not model radiative cooling, hydrodynamic escape,
    atmospheric supply, evolving irradiation, or changing planet structure.
    """
    from astropy import units
    from astropy.constants import G

    def converted(value, unit):
        if np.ma.is_masked(value) or np.any(getattr(value, 'mask', False)):
            raise ValueError('Escape inputs must be unmasked')
        array = np.asarray(units.Quantity(value).to_value(unit))
        if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
            raise ValueError('Escape inputs must be finite real values')
        return array.astype(float)

    if absorption_radius is None:
        absorption_radius = planet_radius
    mass, radius, flux, eta, radius_abs, factor = np.broadcast_arrays(
        converted(planet_mass, units.kg), converted(planet_radius, units.m),
        converted(xuv_flux, units.W / units.m**2),
        converted(efficiency, units.dimensionless_unscaled),
        converted(absorption_radius, units.m), converted(roche_factor, units.dimensionless_unscaled),
    )
    if (np.any(mass <= 0) or np.any(radius <= 0) or np.any(flux < 0)
            or np.any((eta < 0) | (eta > 1)) or np.any((factor <= 0) | (factor > 1))
            or np.any(radius_abs < radius)):
        raise ValueError('Require positive mass and radius, nonnegative flux, physical efficiency and Roche factor, and R_abs >= R_p')
    with np.errstate(over='raise', divide='raise', invalid='raise'):
        rate = eta * np.pi * radius_abs**2 * flux * radius / (G.to_value(units.m**3 / units.kg / units.s**2) * mass * factor)
    return rate * units.kg / units.s


def keplerian_orbital_speed(
    stellar_mass_solar: float | np.ndarray,
    semimajor_axis_au: float | np.ndarray,
) -> float | np.ndarray:
    """Return circular Keplerian orbital speed in km/s."""

    gravitational_constant_au_km = 887.129  # [AU (km/s)^2 / M_sun]
    return np.sqrt(gravitational_constant_au_km * np.asarray(stellar_mass_solar) / np.asarray(semimajor_axis_au))


retr_vorbkepl = keplerian_orbital_speed
retr_vorb = keplerian_orbital_speed


def retr_rs2a(
    rsmacomp: float | np.ndarray,
    rrat: float | np.ndarray,
) -> float | np.ndarray:
    """Return the stellar radius divided by the semi-major axis."""

    return np.asarray(rsmacomp) / (1.0 + np.asarray(rrat))


def retr_rsmacomp(
    peri: float | np.ndarray,
    dura: float | np.ndarray,
    cosi: float | np.ndarray,
) -> float | np.ndarray:
    """Return the sum of stellar and companion radii divided by semi-major axis."""

    hours_per_day = 24.0  # [hour / day]
    return np.sqrt(
        np.sin(np.asarray(dura) * np.pi / np.asarray(peri) / hours_per_day) ** 2
        + np.asarray(cosi) ** 2
    )


def retr_duratrantotl(
    pericomp: float | np.ndarray,
    rsmacomp: float | np.ndarray,
    cosicomp: float | np.ndarray,
    booldiag: bool = True,
) -> float | np.ndarray:
    """Return the total transit duration (first to fourth contact) in hours."""

    pericomp_arr = np.asarray(pericomp, dtype=float)
    rsmacomp_arr = np.asarray(rsmacomp, dtype=float)
    cosicomp_arr = np.asarray(cosicomp, dtype=float)
    hours_per_day = 24.0  # [hour / day]

    fact = rsmacomp_arr ** 2 - cosicomp_arr ** 2
    duratrantotl = np.zeros_like(pericomp_arr, dtype=float)
    mask = fact >= 0.0
    if np.any(mask):
        cos_sub = np.where(mask, cosicomp_arr, 0.0)
        sin_sub = np.sqrt(np.maximum(1.0 - cos_sub ** 2, 1e-12))
        fact_sub = np.where(mask, fact, 0.0)
        ratio = np.clip(np.sqrt(fact_sub) / sin_sub, -1.0, 1.0)
        duration = hours_per_day * pericomp_arr / np.pi * np.arcsin(ratio)  # [hour]
        duratrantotl = np.where(mask, duration, 0.0)

    if pericomp_arr.ndim == 0:
        return float(duratrantotl)
    return duratrantotl


def retr_duratranfull(
    pericomp: float | np.ndarray,
    rsmacomp: float | np.ndarray,
    cosicomp: float | np.ndarray,
    rratcomp: float | np.ndarray,
) -> float | np.ndarray:
    """Return the full transit duration (second to third contact) in hours."""

    pericomp_arr = np.asarray(pericomp, dtype=float)
    rsmacomp_arr = np.asarray(rsmacomp, dtype=float)
    cosicomp_arr = np.asarray(cosicomp, dtype=float)
    rratcomp_arr = np.asarray(rratcomp, dtype=float)
    hours_per_day = 24.0  # [hour / day]

    rdiacomp = rsmacomp_arr * (1.0 - 2.0 / (1.0 + rratcomp_arr))
    fact = rdiacomp ** 2 - cosicomp_arr ** 2
    duratranfull = np.full_like(pericomp_arr, np.nan, dtype=float)
    mask = fact > 0.0
    if np.any(mask):
        cos_sub = np.where(mask, cosicomp_arr, 0.0)
        sin_sub = np.sqrt(np.maximum(1.0 - cos_sub ** 2, 1e-12))
        fact_sub = np.where(mask, fact, 0.0)
        ratio = np.clip(np.sqrt(fact_sub) / sin_sub, -1.0, 1.0)
        duration = hours_per_day * pericomp_arr / np.pi * np.arcsin(ratio)  # [hour]
        duratranfull = np.where(mask, duration, np.nan)

    if pericomp_arr.ndim == 0:
        return float(duratranfull)
    return duratranfull


def retr_tsmm(
    radicomp: float | np.ndarray,
    tmptplan: float | np.ndarray,
    massplan: float | np.ndarray,
    radistar: float | np.ndarray,
    jmag: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Transmission Spectroscopy Metric (Kempton et al. 2018)."""

    radius_arr = np.asarray(radicomp, dtype=float)
    scale_factor = np.select(
        [radius_arr < 1.5, radius_arr < 2.75, radius_arr < 4.0, radius_arr <= 10.0],
        [0.190, 1.26, 1.28, 1.15],
        default=np.nan,
    )
    return scale_factor * radius_arr ** 3 * np.asarray(tmptplan) / np.asarray(massplan) / np.asarray(radistar) ** 2 * 10.0 ** (-np.asarray(jmag) / 5.0)


def retr_scalheig(
    tmptplan: float | np.ndarray,
    massplan: float | np.ndarray,
    radicomp: float | np.ndarray,
) -> float | np.ndarray:
    """Return the atmospheric scale height in Jupiter radii."""

    reference_scale = 27.0  # [R_J]
    reference_temp = 160.0  # [K]
    radius_jupiter_km = 71398.0  # [km]
    return reference_scale * (np.asarray(tmptplan) / reference_temp) / (np.asarray(massplan) / np.asarray(radicomp) ** 2) / radius_jupiter_km  # [R_J]


def retr_radiroch(
    radistar: float | np.ndarray,
    densstar: float | np.ndarray,
    denscomp: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Roche limit radius for tidal disruption."""

    return np.asarray(radistar) * (2.0 * np.asarray(densstar) / np.asarray(denscomp)) ** (1.0 / 3.0)


def roche_distance(
    companion_radius: float | np.ndarray,
    stellar_mass: float | np.ndarray,
    companion_mass: float | np.ndarray,
    fluid: bool = True,
) -> float | np.ndarray:
    """Calculate the Roche disruption distance for a companion orbiting a primary host.

    Parameters
    ----------
    companion_radius : float or np.ndarray
        Radius of the companion (same distance units as returned distance).
    stellar_mass : float or np.ndarray
        Mass of the primary host star.
    companion_mass : float or np.ndarray
        Mass of the secondary companion (same mass units as stellar_mass).
    fluid : bool, optional
        Whether to evaluate the fluid Roche limit (coefficient ~2.44) or rigid limit (~1.26).
        Default is True (fluid).

    Returns
    -------
    float or np.ndarray
        Roche disruption distance in the same distance units as companion_radius.
    """
    coefficient = 2.44 if fluid else 1.26
    mass_ratio = np.asarray(stellar_mass, dtype=float) / np.asarray(companion_mass, dtype=float)
    dist = coefficient * np.asarray(companion_radius, dtype=float) * (mass_ratio ** (1.0 / 3.0))
    if np.ndim(companion_radius) == 0 and np.ndim(stellar_mass) == 0 and np.ndim(companion_mass) == 0:
        return float(dist)
    return dist


def planetary_bulk_density(
    mass_earth: float | np.ndarray,
    radius_earth: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the planetary bulk mean density in g / cm^3 from Earth units."""
    dens = 5.5134 * np.asarray(mass_earth, dtype=float) / (np.asarray(radius_earth, dtype=float) ** 3)
    if np.ndim(mass_earth) == 0 and np.ndim(radius_earth) == 0:
        return float(dens)
    return dens


def stellar_bulk_density(
    mass_solar: float | np.ndarray,
    radius_solar: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the stellar bulk mean density in g / cm^3 from Solar units."""
    dens = 1.408 * np.asarray(mass_solar, dtype=float) / (np.asarray(radius_solar, dtype=float) ** 3)
    if np.ndim(mass_solar) == 0 and np.ndim(radius_solar) == 0:
        return float(dens)
    return dens


def retr_radihill(
    smax: float | np.ndarray,
    masscomp: float | np.ndarray,
    massstar: float | np.ndarray,
) -> float | np.ndarray:
    """Return the Hill sphere radius in the same units as semi-major axis."""

    return np.asarray(smax) * (np.asarray(masscomp) / (3.0 * np.asarray(massstar))) ** (1.0 / 3.0)


def retr_logg(
    radi: float | np.ndarray,
    mass: float | np.ndarray,
) -> float | np.ndarray:
    """Return surface gravity log10(g) in cgs [cm / s^2] from Solar units."""

    solar_logg = 4.437  # [cgs]
    return np.log10(np.asarray(mass) / np.asarray(radi) ** 2) + solar_logg


def retr_albg(
    amplplanrefl: float | np.ndarray,
    radicomp: float | np.ndarray,
    smax: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the planetary geometric albedo from reflected light amplitude."""

    from .util import retr_factconv

    au_to_earth_radius = retr_factconv()['asre']
    return np.asarray(amplplanrefl) * (np.asarray(smax) / np.asarray(radicomp) * au_to_earth_radius) ** 2


def retr_imfa(
    cosi: float | np.ndarray,
    rs2a: float | np.ndarray,
    ecce: float | np.ndarray = 0.0,
    sinw: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Return the transit impact parameter."""

    return (
        np.asarray(cosi)
        / np.asarray(rs2a)
        * (1.0 - np.asarray(ecce) ** 2)
        / (1.0 + np.asarray(ecce) * np.asarray(sinw))
    )


def retr_deptbeam(
    peri: float | np.ndarray,
    massstar: float | np.ndarray,
    masscomp: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the Doppler beaming photometric amplitude in parts per thousand."""

    beaming_constant = 2.8  # [ppt]
    return (
        beaming_constant
        * np.asarray(peri) ** (-1.0 / 3.0)
        * (np.asarray(massstar) + np.asarray(masscomp)) ** (-2.0 / 3.0)
        * np.asarray(masscomp)
    )  # [ppt]


def retr_deptelli(
    peri: float | np.ndarray,
    densstar: float | np.ndarray,
    massstar: float | np.ndarray,
    masscomp: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the ellipsoidal variation photometric amplitude in parts per thousand."""

    ellipsoidal_constant = 18.9  # [ppt]
    return (
        ellipsoidal_constant
        * np.asarray(peri) ** (-2.0)
        / np.asarray(densstar)
        * (1.0 / (1.0 + np.asarray(massstar) / np.asarray(masscomp)))
    )  # [ppt]


def retr_alphelli(
    linear_limb_darkening: float | np.ndarray,
    gravity_darkening: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the gravity and limb darkening factor for ellipsoidal variations."""

    u = np.asarray(linear_limb_darkening)
    g = np.asarray(gravity_darkening)
    return 0.15 * (15.0 + u) * (1.0 + g) / (3.0 - u)


def retr_tmptplandayynigh(
    tmptirra: float | np.ndarray,
    epsi: float | np.ndarray = 0.1,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Estimate the dayside and nightside planetary temperatures in Kelvin."""

    tmptdayy = np.asarray(tmptirra) * (2.0 / 3.0 - 5.0 / 12.0 * np.asarray(epsi)) ** 0.25  # [K]
    tmptnigh = np.asarray(tmptirra) * (np.asarray(epsi) / 4.0) ** 0.25  # [K]
    return tmptdayy, tmptnigh


def equilibrium_temperature(
    stellar_temperature_k: float | np.ndarray,
    stellar_radius_over_semimajor_axis: float | np.ndarray,
    albedo: float | np.ndarray = 0.0,
    recirculation_factor: float = 1.0,
) -> float | np.ndarray:
    """Calculate the planetary equilibrium temperature in Kelvin.

    Parameters
    ----------
    stellar_temperature_k : float or np.ndarray
        Effective temperature of the host star in Kelvin.
    stellar_radius_over_semimajor_axis : float or np.ndarray
        Dimensionless ratio R_* / a (stellar radius over semi-major axis).
    albedo : float or np.ndarray, optional
        Bond albedo (default 0.0).
    recirculation_factor : float, optional
        Heat redistribution factor: 1.0 for full uniform redistribution (4 pi),
        or 2.0 for dayside-only emission (2 pi) (default 1.0).

    Returns
    -------
    float or np.ndarray
        Equilibrium temperature in Kelvin.
    """
    rs2a = np.asarray(stellar_radius_over_semimajor_axis, dtype=float)
    teff = np.asarray(stellar_temperature_k, dtype=float)
    albedo_arr = np.asarray(albedo, dtype=float)
    teq = teff * np.sqrt(0.5 * rs2a) * (1.0 - albedo_arr) ** 0.25 * (recirculation_factor ** 0.25)
    if np.ndim(stellar_temperature_k) == 0 and np.ndim(stellar_radius_over_semimajor_axis) == 0:
        return float(teq)
    return teq


def irradiation_temperature(
    stellar_temperature_k: float | np.ndarray,
    stellar_radius_over_semimajor_axis: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the planetary irradiation temperature T_irr = T_* * sqrt(R_* / a) in Kelvin."""
    rs2a = np.asarray(stellar_radius_over_semimajor_axis, dtype=float)
    teff = np.asarray(stellar_temperature_k, dtype=float)
    tirr = teff * np.sqrt(rs2a)
    if np.ndim(stellar_temperature_k) == 0 and np.ndim(stellar_radius_over_semimajor_axis) == 0:
        return float(tirr)
    return tirr


def retr_esmm(
    tmptplanequi: float | np.ndarray,
    tmptstar: float | np.ndarray,
    radicomp: float | np.ndarray,
    radistar: float | np.ndarray,
    kmag: float | np.ndarray,
    recirculation_efficiency: float = 0.1,
) -> float | np.ndarray:
    """Calculate the Emission Spectroscopy Metric (Kempton et al. 2018)."""

    from .spectral import blackbody_flux_wavelength
    from .util import retr_factconv

    tmptplanirra = np.asarray(tmptplanequi) * np.sqrt(2.0)  # [K]
    tmptplandayy, _ = retr_tmptplandayynigh(tmptplanirra, recirculation_efficiency)
    solar_to_earth_radius = retr_factconv()["rsre"]
    radius_ratio = np.asarray(radicomp) / (np.asarray(radistar) * solar_to_earth_radius)
    flux_ratio = blackbody_flux_wavelength(tmptplandayy, 7.5) / blackbody_flux_wavelength(tmptstar, 7.5)
    scale_factor = 1.1e3
    return scale_factor * flux_ratio * radius_ratio**2 * 10.0 ** (-np.asarray(kmag) / 5.0)


def keplerian_semi_amplitude(
    period_days: float | np.ndarray | None = None,
    stellar_mass_solar: float | np.ndarray | None = None,
    companion_mass_solar: float | np.ndarray | None = None,
    inclination_degrees: float | np.ndarray = 90.0,
    eccentricity: float | np.ndarray = 0.0,
    *,
    pericomp: float | np.ndarray | None = None,
    massstar: float | np.ndarray | None = None,
    masscomp: float | np.ndarray | None = None,
    inclcomp: float | np.ndarray | None = None,
    eccecomp: float | np.ndarray | None = None,
) -> float | np.ndarray:
    """Calculate the Keplerian radial velocity semi-amplitude in m/s."""

    if period_days is None and pericomp is not None:
        period_days = pericomp
    if stellar_mass_solar is None and massstar is not None:
        stellar_mass_solar = massstar
    if companion_mass_solar is None and masscomp is not None:
        companion_mass_solar = masscomp
    if inclcomp is not None:
        inclination_degrees = inclcomp
    if eccecomp is not None:
        eccentricity = eccecomp

    gravitational_constant = 6.67430e-11  # [m^3 / (kg s^2)]
    solar_mass_kg = 1.98847e30  # [kg]
    seconds_per_day = 86400.0  # [s / day]

    period_seconds = np.asarray(period_days, dtype=float) * seconds_per_day  # [s]
    mass_star_kg = np.asarray(stellar_mass_solar, dtype=float) * solar_mass_kg  # [kg]
    mass_comp_kg = np.asarray(companion_mass_solar, dtype=float) * solar_mass_kg  # [kg]
    inclination_radians = np.deg2rad(np.asarray(inclination_degrees, dtype=float))
    ecc = np.asarray(eccentricity, dtype=float)

    semi_amplitude = (
        (2.0 * np.pi * gravitational_constant / period_seconds) ** (1.0 / 3.0)
        * mass_comp_kg
        * np.sin(inclination_radians)
        / (mass_star_kg + mass_comp_kg) ** (2.0 / 3.0)
        / np.sqrt(np.maximum(1.0 - ecc**2, 1e-12))
    )  # [m / s]
    if np.ndim(semi_amplitude) == 0:
        return float(semi_amplitude)
    return semi_amplitude


def keplerian_orbit_radial_velocity(
    time: np.ndarray,
    epoch_transit_days: float | np.ndarray | None = None,
    period_days: float | np.ndarray | None = None,
    companion_mass_solar: float | np.ndarray | None = None,
    stellar_mass_solar: float | np.ndarray | None = None,
    inclination_degrees: float | np.ndarray = 90.0,
    eccentricity: float | np.ndarray = 0.0,
    argument_periastron_degrees: float | np.ndarray = 90.0,
    *,
    epocmtracomp: float | np.ndarray | None = None,
    pericomp: float | np.ndarray | None = None,
    masscomp: float | np.ndarray | None = None,
    massstar: float | np.ndarray | None = None,
    inclcomp: float | np.ndarray | None = None,
    eccecomp: float | np.ndarray | None = None,
    arpacomp: float | np.ndarray | None = None,
) -> np.ndarray:
    """Calculate Keplerian stellar radial velocity from orbital transit parameters."""

    if epoch_transit_days is None and epocmtracomp is not None:
        epoch_transit_days = epocmtracomp
    if period_days is None and pericomp is not None:
        period_days = pericomp
    if companion_mass_solar is None and masscomp is not None:
        companion_mass_solar = masscomp
    if stellar_mass_solar is None and massstar is not None:
        stellar_mass_solar = massstar
    if inclcomp is not None:
        inclination_degrees = inclcomp
    if eccecomp is not None:
        eccentricity = eccecomp
    if arpacomp is not None:
        argument_periastron_degrees = arpacomp

    arg_periastron_rad = np.deg2rad(np.asarray(argument_periastron_degrees, dtype=float))
    true_anomaly_transit = np.pi / 2.0 - arg_periastron_rad
    ecc = np.asarray(eccentricity, dtype=float)
    eccentric_anomaly_transit = 2.0 * np.arctan2(
        np.sqrt(np.maximum(1.0 - ecc, 0.0)) * np.sin(true_anomaly_transit / 2.0),
        np.sqrt(1.0 + ecc) * np.cos(true_anomaly_transit / 2.0),
    )
    mean_anomaly_transit = eccentric_anomaly_transit - ecc * np.sin(eccentric_anomaly_transit)

    semi_amplitude = keplerian_semi_amplitude(
        period_days,
        stellar_mass_solar,
        companion_mass_solar,
        inclination_degrees,
        eccentricity,
    )
    return keplerian_radial_velocity(
        time,
        period_days,
        semi_amplitude,
        ecc,
        arg_periastron_rad,
        mean_anomaly_transit,
        epoch_transit_days,
    )


def retr_radifrommass(
    mass_earth: float | np.ndarray,
    strgtype: str = "mine",
) -> float | np.ndarray:
    """Estimate planetary radius in Earth radii from mass in Earth masses."""

    if strgtype != "mine":
        raise ValueError(f"Unsupported radius-mass model {strgtype!r}.")

    mass_arr = np.asarray(mass_earth, dtype=float)
    radius_arr = np.empty_like(mass_arr)

    mask1 = mass_arr <= 2.0
    radius_arr[mask1] = mass_arr[mask1] ** 0.28

    mask2 = (mass_arr > 2.0) & (mass_arr < 130.0)
    radius_arr[mask2] = 4.48 * (mass_arr[mask2] / 20.0) ** 0.57

    mask3 = (mass_arr >= 130.0) & (mass_arr < 2.66e4)
    radius_arr[mask3] = 10.1 * (mass_arr[mask3] / 1e5) ** (-0.04)

    mask4 = mass_arr >= 2.66e4
    radius_arr[mask4] = 20.0 * (mass_arr[mask4] / 5e4) ** 0.88

    if mass_arr.ndim == 0:
        return float(radius_arr)
    return radius_arr


def retr_massfromradi(
    radius_earth: float | np.ndarray,
    strgtype: str = "mine",
) -> float | np.ndarray:
    """Estimate planetary mass in Earth masses from radius in Earth radii."""

    if strgtype != "mine":
        raise ValueError(f"Unsupported radius-mass model {strgtype!r}.")

    radius_arr = np.asarray(radius_earth, dtype=float)
    mass_arr = np.empty_like(radius_arr)

    r_break1 = 2.0 ** 0.28  # ~1.21419
    r_break2 = 4.48 * (130.0 / 20.0) ** 0.57  # ~13.013

    mask1 = radius_arr <= r_break1
    mass_arr[mask1] = radius_arr[mask1] ** (1.0 / 0.28)

    mask2 = (radius_arr > r_break1) & (radius_arr <= r_break2)
    mass_arr[mask2] = 20.0 * (radius_arr[mask2] / 4.48) ** (1.0 / 0.57)

    mask3 = radius_arr > r_break2
    mass_arr[mask3] = 5e4 * (radius_arr[mask3] / 20.0) ** (1.0 / 0.88)

    if radius_arr.ndim == 0:
        return float(mass_arr)
    return mass_arr


def retr_reso(
    listperi: np.ndarray,
    maxmordr: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Find period ratios and closest integer orbital resonances."""

    periods = np.asarray(listperi, dtype=float)
    if periods.ndim == 1:
        periods = periods[None, :]
    if (periods <= 0.0).any() or not np.isfinite(periods).all():
        raise ValueError("Periods must be finite and positive.")

    numbsamp, numbcomp = periods.shape
    listratiperi = np.zeros((numbsamp, numbcomp, numbcomp), dtype=float)
    intgreso = np.zeros((numbcomp, numbcomp, 2), dtype=int)

    for j in range(numbcomp):
        for jj in range(numbcomp):
            if j >= jj:
                continue

            ratio = periods[:, j] / periods[:, jj]
            ratio_ordered = np.where(ratio < 1.0, 1.0 / ratio, ratio)
            listratiperi[:, j, jj] = ratio_ordered

            minmdiff = 1e100
            best_reso = (1, 1)
            for a in range(1, maxmordr):
                for aa in range(1, maxmordr):
                    diff = np.mean(np.abs(float(a) / aa - ratio_ordered))
                    if diff < minmdiff:
                        minmdiff = diff
                        best_reso = (a, aa)
            intgreso[j, jj, :] = best_reso

    return intgreso, listratiperi


def retr_liststrgcomp(number_companions: int) -> np.ndarray:
    """Return standard lowercase letter labels for exoplanet companions."""

    labels = ["b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]
    return np.array(labels[:number_companions])


def retr_listcolrcomp(number_companions: int) -> np.ndarray:
    """Return standard colors for exoplanet companions."""

    colors = ["magenta", "orange", "red", "green", "purple", "cyan"]
    return np.array(colors[:number_companions])


def solar_system_planets() -> dict[str, np.ndarray]:
    """Return dictionary of Solar System planet names, semi-major axes [AU], masses [M_Earth], and radii [R_Earth]."""
    radii_km = np.array([2440.0, 6052.0, 6371.0, 3390.0, 69911.0, 58232.0, 25362.0, 24622.0])
    return {
        "name": np.array(["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]),
        "smax": np.array([0.3870993, 0.723336, 1.0, 1.52371, 5.2029, 9.537, 19.189, 30.0699]),
        "mass": np.array([0.05527, 0.81500, 1.0, 0.10745, 317.83, 95.159, 14.500, 17.204]),
        "radi": radii_km / radii_km[2],
    }


retr_dictsols = solar_system_planets
retr_rvelsema = keplerian_semi_amplitude
retr_rvel = keplerian_orbit_radial_velocity


def companion_flux_dilution(
    target_temperature_k: float,
    companion_temperature_k: float,
    wavelength_min_microns: float = 0.6,
    wavelength_max_microns: float = 1.0,
    number_wavelengths: int = 50,
) -> float:
    """Calculate photometric dilution factor 1 - F_target / (F_target + F_companion) across a bandpass."""
    from .spectral import integrate_blackbody_flux_wavelength

    fluxtarg = integrate_blackbody_flux_wavelength(
        target_temperature_k, wavelength_min_microns, wavelength_max_microns, number_wavelengths
    )
    fluxcomp = integrate_blackbody_flux_wavelength(
        companion_temperature_k, wavelength_min_microns, wavelength_max_microns, number_wavelengths
    )

    return 1.0 - fluxtarg / (fluxtarg + fluxcomp)


def retr_dilu(target_temperature_k, companion_temperature_k, strgwlentype="tess"):
    """Compatibility wrapper for companion_flux_dilution."""
    if strgwlentype != "tess":
        raise ValueError(f"Unsupported wavelength band {strgwlentype!r}.")
    return companion_flux_dilution(target_temperature_k, companion_temperature_k, 0.6, 1.0)


def retr_rflxmodlrise(
    time: float | np.ndarray,
    timerise: float,
    coeflinerise: float,
    coefquadrise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate relative flux and excess flux for a quadratic post-rise profile."""
    time = np.asarray(time, dtype=float)
    timeoffs = time - timerise
    dflxrise = np.zeros_like(time)
    indxpost = np.where(timeoffs > 0.0)[0]
    dflxrise[indxpost] = coeflinerise * timeoffs[indxpost] + coefquadrise * timeoffs[indxpost]**2
    return 1.0 + dflxrise, dflxrise


def transit_epochs(
    time: float | np.ndarray,
    epoch_days: float,
    period_days: float,
    duration_hours: float | None = None,
    *,
    epoc: float | None = None,
    peri: float | None = None,
    duratrantotl: float | None = None,
    booldiag: bool = True,
) -> np.ndarray:
    """Return integer transit epoch numbers occurring within the time range."""
    if epoc is not None:
        epoch_days = epoc
    if peri is not None:
        period_days = peri
    if duratrantotl is not None:
        duration_hours = duratrantotl

    time_arr = np.asarray(time, dtype=float)
    if not np.isfinite(period_days) or period_days <= 0.0:
        return np.arange(1, dtype=int)
    duration_offset = (duration_hours / 48.0) if duration_hours is not None else 0.0
    min_epoch = int(np.ceil((np.amin(time_arr) - epoch_days - duration_offset) / period_days))
    max_epoch = int(np.ceil((np.amax(time_arr) - epoch_days - duration_offset) / period_days))
    return np.arange(min_epoch, max_epoch, dtype=int)


retr_indxtran = transit_epochs


def transit_times(
    time: float | np.ndarray,
    epoch_days: float,
    period_days: float,
    duration_hours: float | None = None,
    *,
    epoc: float | None = None,
    peri: float | None = None,
    duratrantotl: float | None = None,
) -> np.ndarray:
    """Return predicted transit times occurring within the time range."""
    epochs = transit_epochs(time, epoch_days, period_days, duration_hours, epoc=epoc, peri=peri, duratrantotl=duratrantotl)
    p = peri if peri is not None else period_days
    t0 = epoc if epoc is not None else epoch_days
    return t0 + epochs * p


retr_listepoctran = transit_times


def transit_time_indices(
    time: float | np.ndarray,
    epoch_days: float,
    period_days: float,
    duration_hours: float,
    duration_full_hours: float | None = None,
    phase_interval: str | None = None,
    individual_transits: bool = False,
    out_of_transit: bool = False,
    secondary: bool = False,
    *,
    epoc: float | None = None,
    peri: float | None = None,
    duratrantotl: float | None = None,
    duratranfull: float | None = None,
    typeineg: str | None = None,
    boolindi: bool = False,
    booloutt: bool = False,
    boolseco: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Return indices of timestamps falling in or out of transit windows."""
    if epoc is not None:
        epoch_days = epoc
    if peri is not None:
        period_days = peri
    if duratrantotl is not None:
        duration_hours = duratrantotl
    if duratranfull is not None:
        duration_full_hours = duratranfull
    if typeineg is not None:
        phase_interval = typeineg
    if boolindi:
        individual_transits = boolindi
    if booloutt:
        out_of_transit = booloutt
    if boolseco:
        secondary = boolseco

    time_arr = np.asarray(time, dtype=float)
    epochs = transit_epochs(time_arr, epoch_days, period_days, duration_hours)
    offset = 0.5 if secondary else 0.0

    half_duration_days = duration_hours / 48.0
    full_half_duration_days = (duration_full_hours / 48.0) if duration_full_hours is not None else None

    indices_list = []
    for n in epochs:
        center = epoch_days + (n + offset) * period_days
        t_start = center - half_duration_days
        t_end = center + half_duration_days
        if full_half_duration_days is not None:
            t_full_start = center - full_half_duration_days
            t_full_end = center + full_half_duration_days
            t_ingr_mid = 0.5 * (t_start + t_full_start)
            t_eggr_mid = 0.5 * (t_end + t_full_end)
            if phase_interval == "inge":
                idx = np.where(((time_arr > t_start) & (time_arr < t_full_start)) | ((time_arr > t_full_end) & (time_arr < t_end)))[0]
            elif phase_interval == "ingr":
                idx = np.where((time_arr > t_start) & (time_arr < t_full_start))[0]
            elif phase_interval == "eggr":
                idx = np.where((time_arr > t_full_end) & (time_arr < t_end))[0]
            elif phase_interval == "ingrinit":
                idx = np.where((time_arr > t_start) & (time_arr < t_ingr_mid))[0]
            elif phase_interval == "ingrfinl":
                idx = np.where((time_arr > t_ingr_mid) & (time_arr < t_full_start))[0]
            elif phase_interval == "eggrinit":
                idx = np.where((time_arr > t_full_end) & (time_arr < t_eggr_mid))[0]
            elif phase_interval == "eggrfinl":
                idx = np.where((time_arr > t_eggr_mid) & (time_arr < t_end))[0]
            else:
                idx = np.where((time_arr > t_start) & (time_arr < t_end))[0]
        else:
            idx = np.where((time_arr > t_start) & (time_arr < t_end))[0]

        if idx.size > 0:
            indices_list.append(idx)

    if individual_transits:
        return indices_list

    if len(indices_list) > 0:
        transit_indices = np.unique(np.concatenate(indices_list))
    else:
        transit_indices = np.array([], dtype=int)

    if out_of_transit:
        return np.setdiff1d(np.arange(time_arr.size), transit_indices)
    return transit_indices


retr_indxtimetran = transit_time_indices
