"""Reusable helpers for Pandeia calculation dictionaries and reports."""

import numpy as np


def build_nirspec_calculation(
    *,
    mode: str,
    number_integrations: int,
    number_groups: int,
    disperser: str,
    filter_name: str,
    subarray: str,
    temperature_kelvin: float,
    magnitude: float,
    zero_point_jansky: float,
    effective_wavelength_micrometer: float,
) -> dict:
    """Build a magnitude-normalized NIRSpec BOTS or acquisition calculation."""
    from pandeia.engine.calc_utils import build_default_calc

    calculation = build_default_calc("jwst", "nirspec", mode)
    instrument = {"filter": filter_name}
    if mode == "bots":
        instrument.update(aperture="s1600a1", disperser=disperser)
    calculation["configuration"]["instrument"].update(instrument)
    calculation["configuration"]["detector"].update(
        subarray=subarray,
        readout_pattern="nrsrapid",
        ngroup=number_groups,
        nint=number_integrations,
        nexp=1,
    )
    configure_blackbody_source(
        calculation,
        temperature_kelvin,
        magnitude,
        zero_point_jansky,
        effective_wavelength_micrometer,
    )
    return calculation


def configure_blackbody_source(
    calculation: dict,
    temperature_kelvin: float,
    magnitude: float,
    zero_point_jansky: float,
    effective_wavelength_micrometer: float,
) -> None:
    """Configure a magnitude-normalized blackbody source."""
    source = calculation["scene"][0]
    source["spectrum"]["sed"] = {
        "sed_type": "blackbody",
        "temp": temperature_kelvin,
    }
    source["spectrum"]["normalization"] = {
        "type": "at_lambda",
        "norm_wave": effective_wavelength_micrometer,
        "norm_waveunit": "microns",
        "norm_flux": zero_point_jansky * 1.0e3 * 10.0 ** (-0.4 * magnitude),
        "norm_fluxunit": "mjy",
    }


def summarize_saturation(report: dict, number_groups: int) -> dict:
    """Summarize signal-limit fraction and saturation-map flags."""
    flags, counts = np.unique(report["2d"]["saturation_unrotated"], return_counts=True)
    flag_counts = {int(flag): int(count) for flag, count in zip(flags, counts)}
    return {
        "number_groups": number_groups,
        "fraction_of_etc_signal_limit": float(report["scalar"]["fraction_saturation"]),
        "partially_saturated_pixels": flag_counts.get(1, 0),
        "fully_saturated_pixels": flag_counts.get(2, 0),
    }


def bin_depth_precision(
    report: dict,
    phase_duration_hours: float,
    bin_width_micrometer: float,
    wavelength_range_micrometer: tuple[float, float],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate photon-plus-detector transit-depth precision in fixed bins."""
    wavelength = np.asarray(report["1d"]["extracted_flux"][0])  # [micrometer]
    source_rate = np.asarray(report["1d"]["extracted_flux"][1])  # [electron s^-1]
    rate_uncertainty = np.asarray(report["1d"]["extracted_noise"][1])  # [electron s^-1]
    integration_time = report["scalar"]["total_exposure_time"]  # [s]
    number_integrations = int(phase_duration_hours * 3600.0 // integration_time)
    edges = np.append(
        np.arange(2.9, 5.1 + 0.01, bin_width_micrometer),
        wavelength_range_micrometer[1],
    )  # [micrometer]
    centers = 0.5 * (edges[:-1] + edges[1:])  # [micrometer]
    precision = np.full(centers.size, np.nan)  # [ppm]
    single_integration_precision = np.full(centers.size, np.nan)  # [ppm]
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (wavelength >= lower) & (wavelength < upper)
        if not np.any(selected):
            continue
        summed_rate = np.sum(source_rate[selected])  # [electron s^-1]
        summed_rate_uncertainty = np.sqrt(
            np.sum(rate_uncertainty[selected] ** 2)
        )  # [electron s^-1]
        phase_fractional_uncertainty = (
            summed_rate_uncertainty / summed_rate / np.sqrt(number_integrations)
        )
        single_integration_precision[index] = (
            summed_rate_uncertainty / summed_rate * 1.0e6
        )
        precision[index] = np.sqrt(2.0) * phase_fractional_uncertainty * 1.0e6
    return number_integrations, centers, single_integration_precision, precision


def summarize_transit_visit(
    report: dict,
    phase_duration_hours: float,
    bin_width_micrometer: float,
    wavelength_range_micrometer: tuple[float, float],
) -> dict:
    """Summarize equal-duration in- and out-of-transit exposure and precision."""
    integrations, wavelength, single_precision, depth_precision = bin_depth_precision(
        report,
        phase_duration_hours,
        bin_width_micrometer,
        wavelength_range_micrometer,
    )
    return {
        "phase_duration_hours": phase_duration_hours,
        "number_integrations_per_phase": integrations,
        "number_in_transit_integrations": integrations,
        "number_out_of_transit_integrations": integrations,
        "science_integrations": 2 * integrations,
        "science_exposure_hours": 2 * integrations * report["scalar"]["total_exposure_time"] / 3600.0,
        "wavelength_micrometer": wavelength.tolist(),
        "single_integration_relative_uncertainty_ppm": single_precision.tolist(),
        "depth_precision_ppm": depth_precision.tolist(),
        "median_depth_precision_ppm": float(np.nanmedian(depth_precision)),
        "minimum_depth_precision_ppm": float(np.nanmin(depth_precision)),
        "maximum_depth_precision_ppm": float(np.nanmax(depth_precision)),
    }