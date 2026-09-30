"""Reusable astronomy calculations for observation planning."""

from tdpy.verbosity import print

from datetime import datetime
from collections.abc import Mapping
import csv
import io
import itertools
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


def read_tap_csv_rows(
    endpoint: str,
    query: str,
    *,
    timeout_seconds: float = 60.0,
) -> list[dict[str, str]]:
    """Execute a Table Access Protocol query and return its CSV rows."""
    url = f"{endpoint}?{urlencode({'query': query, 'format': 'csv'})}"
    print(f"Reading from {url}...")
    with urlopen(url, timeout=timeout_seconds) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def datetime_to_julian_date(value: datetime) -> float:
    """Convert a timezone-aware UTC datetime to Julian Date."""
    if value.tzinfo is None:
        raise ValueError("value must include a timezone")
    return value.timestamp() / 86400.0 + 2440587.5


def periodic_event_numbers(
    epoch_julian_date: float,
    period_days: float,
    start_julian_date: float,
    end_julian_date: float,
) -> range:
    """Return integer epochs with event centers inside a closed interval."""
    if period_days <= 0.0:
        raise ValueError("period_days must be positive")
    first = int(np.ceil((start_julian_date - epoch_julian_date) / period_days))
    last = int(np.floor((end_julian_date - epoch_julian_date) / period_days))
    return range(first, last + 1)


def linear_ephemeris_uncertainties(
    epoch_uncertainty_days: float,
    period_uncertainty_days: float,
    event_number: int,
) -> tuple[float, float]:
    """Return diagonal uncertainty and the covariance-independent upper bound."""
    propagated = float(
        np.hypot(epoch_uncertainty_days, event_number * period_uncertainty_days)
    )  # [day]
    upper_bound = float(
        epoch_uncertainty_days + abs(event_number) * period_uncertainty_days
    )  # [day]
    return propagated, upper_bound


def centered_event_start_phase_range(
    duration_hours: float,
    period_days: float,
    range_minutes: float,
) -> tuple[float, float]:
    """Return a centered start-phase range one event duration before its center."""
    center = -duration_hours / (24.0 * period_days)
    half_width = 0.5 * range_minutes / (24.0 * 60.0 * period_days)
    return center - half_width, center + half_width


def periodic_event_overlaps(
    target_name: str,
    midpoint_julian_date: float,
    events: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, float | int | str]]:
    """Return periodic events intersecting a target's full event duration."""
    target = events[target_name]
    target_half_duration_days = target["duration_hours"] / 24.0
    overlaps = []
    for name, event in events.items():
        if name == target_name:
            continue
        nearest_event = int(
            np.rint(
                (midpoint_julian_date - event["epoch_bjd_tdb"])
                / event["period_days"]
            )
        )
        other_midpoint = event["epoch_bjd_tdb"] + nearest_event * event["period_days"]
        other_half_duration_days = event["duration_hours"] / 48.0
        if abs(other_midpoint - midpoint_julian_date) <= (
            target_half_duration_days + other_half_duration_days
        ):
            overlaps.append(
                {
                    "planet": name,
                    "event_number": nearest_event,
                    "midpoint_bjd_tdb": other_midpoint,
                    "offset_from_target_midpoint_hours": (
                        other_midpoint - midpoint_julian_date
                    )
                    * 24.0,
                }
            )
    return overlaps


def periodic_event_schedule(
    target_name: str,
    events: Mapping[str, Mapping[str, Any]],
    start_julian_date: float,
    end_julian_date: float,
) -> list[dict]:
    """Build event centers, timing bounds, baselines, and companion overlaps."""
    target = events[target_name]
    schedule = []
    for event_number in periodic_event_numbers(
        target["epoch_bjd_tdb"],
        target["period_days"],
        start_julian_date,
        end_julian_date,
    ):
        midpoint = target["epoch_bjd_tdb"] + event_number * target["period_days"]
        uncertainty, uncertainty_bound = linear_ephemeris_uncertainties(
            target["epoch_uncertainty_days"],
            target["period_uncertainty_days"],
            event_number,
        )
        schedule.append(
            {
                "event_number": event_number,
                "midpoint_bjd_tdb": midpoint,
                "timing_uncertainty_minutes": uncertainty * 24.0 * 60.0,
                "covariance_independent_timing_bound_minutes": uncertainty_bound * 24.0 * 60.0,
                "three_sigma_bound_pre_ingress_baseline_hours": (
                    0.5 * target["duration_hours"] - 3.0 * uncertainty_bound * 24.0
                ),
                "overlapping_transits": periodic_event_overlaps(
                    target_name, midpoint, events
                ),
            }
        )
    return schedule


def build_periodic_event_report(
    events: Mapping[str, Mapping[str, Any]],
    target_names: tuple[str, ...],
    start: datetime,
    end: datetime,
    *,
    phase_range_duration_minutes: float,
    initial_settling_minutes: float,
    required_clean_baseline_hours: Mapping[str, float],
) -> dict:
    """Build a serializable timing and overlap report for periodic events."""
    start_julian_date = datetime_to_julian_date(start)
    end_julian_date = datetime_to_julian_date(end)
    report = {
        "cycle_interval": {
            "start_utc": start.isoformat(),
            "end_utc_exclusive": end.isoformat(),
            "start_julian_date": start_julian_date,
            "end_julian_date": end_julian_date,
        },
        "time_standard": "Ephemerides are BJD_TDB; UTC is used only to select interval boundaries.",
        "uncertainty_equation": "sigma_T(N)^2 = sigma_T0^2 + N^2 sigma_P^2; covariance term unavailable",
        "covariance_independent_bound": "sigma_T(N) <= sigma_T0 + |N| sigma_P from |Cov(T0,P)| <= sigma_T0 sigma_P",
        "scheduling_requirement": "Regenerate event windows from posterior samples including cov(T0, P) and measured TTVs.",
        "duration_uncertainty_requirement": "Propagate event-duration uncertainty when defining minimum clean baselines.",
        "planet_ephemerides": events,
        "targets": {},
    }
    for name in target_names:
        event = events[name]
        schedule = periodic_event_schedule(
            name, events, start_julian_date, end_julian_date
        )
        minimum_baseline = min(
            item["three_sigma_bound_pre_ingress_baseline_hours"]
            for item in schedule
        )  # [hour]
        report["targets"][name] = {
            "number_of_cycle_events": len(schedule),
            "first_event": schedule[0],
            "last_event": schedule[-1],
            "minimum_three_sigma_bound_pre_ingress_baseline_hours": minimum_baseline,
            "science_start_phase_range": list(
                centered_event_start_phase_range(
                    event["duration_hours"],
                    event["period_days"],
                    phase_range_duration_minutes,
                )
            ),
            "phase_range_duration_minutes": phase_range_duration_minutes,
            "initial_settling_minutes": initial_settling_minutes,
            "minimum_clean_baseline_after_phase_and_settling_hours": (
                minimum_baseline
                - 0.5 * phase_range_duration_minutes / 60.0
                - initial_settling_minutes / 60.0
            ),
            "required_clean_baseline_hours": required_clean_baseline_hours[name],
            "events_with_known_planet_overlap": sum(
                bool(item["overlapping_transits"]) for item in schedule
            ),
            "overlap_free_calendar_events_before_visibility_screening": sum(
                not item["overlapping_transits"] for item in schedule
            ),
            "events": schedule,
        }
    return report


def transmission_spectroscopy_metric(
    radius_earth: float,
    temperature_kelvin: float,
    mass_earth: float,
    stellar_radius_solar: float,
    j_magnitude: float,
) -> float:
    """Return the Kempton et al. (2018) sub-Neptune metric."""
    scale_factor = 1.26 if radius_earth < 2.75 else 1.28
    return (
        scale_factor
        * radius_earth**3
        * temperature_kelvin
        / (mass_earth * stellar_radius_solar**2)
        * 10.0 ** (-j_magnitude / 5.0)
    )


def build_transiting_planet_pairs(
    rows: list[Mapping[str, Any]],
    replacement_systems: Mapping[str, list[Mapping[str, Any]]] | None = None,
    excluded_hosts: tuple[str, ...] = (),
) -> list[dict]:
    """Build every within-system pair and its transmission metrics."""
    systems = {}
    for row in rows:
        if row["hostname"] not in excluded_hosts:
            systems.setdefault(row["hostname"], []).append(row)
    systems.update(replacement_systems or {})
    pairs = []
    for hostname, planets in systems.items():
        for first, second in itertools.combinations(planets, 2):
            metrics = {
                planet["pl_name"]: transmission_spectroscopy_metric(
                    float(planet["pl_rade"]),
                    float(planet["pl_eqt"]),
                    float(planet["pl_bmasse"]),
                    float(planet["st_rad"]),
                    float(planet["sy_jmag"]),
                )
                for planet in (first, second)
            }
            pairs.append(
                {
                    "hostname": hostname,
                    "planets": sorted((first["pl_name"], second["pl_name"])),
                    "j_magnitude": float(first["sy_jmag"]),
                    "temperature_difference_kelvin": abs(
                        float(first["pl_eqt"]) - float(second["pl_eqt"])
                    ),
                    "radius_difference_earth": abs(
                        float(first["pl_rade"]) - float(second["pl_rade"])
                    ),
                    "transmission_spectroscopy_metrics": metrics,
                    "minimum_transmission_spectroscopy_metric": min(metrics.values()),
                }
            )
    return pairs


def select_transiting_planet_pair_comparisons(
    pairs: list[dict],
    target: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Select pair samples matching a target's radius, temperature, and merit."""
    substantial = [
        pair
        for pair in pairs
        if pair["radius_difference_earth"] >= target["radius_difference_earth"]
    ]
    bright_temperature_matches = [
        pair
        for pair in substantial
        if pair["temperature_difference_kelvin"]
        <= target["temperature_difference_kelvin"]
        and pair["j_magnitude"] <= target["j_magnitude"]
    ]
    joint_matches = [
        pair
        for pair in substantial
        if pair["temperature_difference_kelvin"]
        <= target["temperature_difference_kelvin"]
        and pair["minimum_transmission_spectroscopy_metric"]
        >= target["minimum_transmission_spectroscopy_metric"]
    ]
    return substantial, bright_temperature_matches, joint_matches


def plot_transiting_planet_pair_context(
    pairs: list[dict],
    target: dict,
    highlighted_pairs: list[dict],
    output_path: str | Path,
    *,
    target_label: str,
    highlighted_label: str,
    annotation: str,
    annotation_position: tuple[float, float],
    font_size: float = 10.0,  # [point]
    x_limits: tuple[float, float] = (0.0, 1250.0),  # [K]
) -> Path:
    """Plot temperature contrast against pair transmission merit."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    figure, axis = plt.subplots(figsize=(6.5, 1.5), facecolor="white")
    axis.scatter(
        [pair["temperature_difference_kelvin"] for pair in pairs],
        [pair["minimum_transmission_spectroscopy_metric"] for pair in pairs],
        s=18,
        color="#8A8A8A",
        alpha=0.55,
        edgecolors="none",
        label="other pairs",
    )
    upper_merit = 1.6 * max(
        pair["minimum_transmission_spectroscopy_metric"] for pair in pairs
    )
    axis.fill_between(
        [0.0, target["temperature_difference_kelvin"]],
        [target["minimum_transmission_spectroscopy_metric"]] * 2,
        [upper_merit] * 2,
        color="#DCEFEA",
        zorder=0,
    )
    axis.scatter(
        [pair["temperature_difference_kelvin"] for pair in highlighted_pairs],
        [
            pair["minimum_transmission_spectroscopy_metric"]
            for pair in highlighted_pairs
        ],
        s=55,
        marker="D",
        color="#B23A48",
        edgecolor="black",
        linewidth=0.6,
        zorder=4,
        label=highlighted_label,
    )
    axis.scatter(
        target["temperature_difference_kelvin"],
        target["minimum_transmission_spectroscopy_metric"],
        s=150,
        marker="*",
        color="#007360",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label=target_label,
    )
    axis.annotate(
        annotation,
        (
            target["temperature_difference_kelvin"],
            target["minimum_transmission_spectroscopy_metric"],
        ),
        xytext=annotation_position,
        arrowprops={"arrowstyle": "-", "color": "black"},
        color="black",
        fontsize=font_size,
    )
    axis.set_xlim(*x_limits)
    axis.set_yscale("log")
    axis.set_ylim(1.0, upper_merit)
    axis.set_xlabel(
        r"Pair temperature difference $|\Delta T_{\rm eq}|$ [K]",
        color="black",
        fontsize=font_size,
    )
    axis.set_ylabel("Minimum pair TSM", color="black", fontsize=font_size)
    axis.tick_params(colors="black", labelsize=font_size)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["bottom", "left"]].set_color("black")
    axis.legend(
        loc="upper right",
        ncol=3,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=font_size,
        borderpad=0.2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.34, top=0.82)
    print(f"Writing to {output_path}...")
    figure.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(figure)
    return output_path