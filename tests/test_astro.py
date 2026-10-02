from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tdpy.astro import (
    analyze_target_visibility,
    run_target_visibility_diagnostic,
    build_periodic_event_report,
    build_transiting_planet_pairs,
    centered_event_start_phase_range,
    datetime_to_julian_date,
    linear_ephemeris_uncertainties,
    periodic_event_overlaps,
    periodic_event_schedule,
    periodic_event_numbers,
    select_transiting_planet_pair_comparisons,
    transmission_spectroscopy_metric,
)


def test_target_visibility_samples_night_and_annual_darkness():
    result = analyze_target_visibility(
        right_ascension_degrees=186.574,  # [deg]
        declination_degrees=-51.363,  # [deg]
        latitude_degrees=36.824166,  # [deg]
        longitude_degrees=30.335555,  # [deg]
        height_meters=2500.0,  # [m]
        utc_offset_hours=3.0,  # [hour]
        night='2022-07-13 00:00:00',
        year_start='2022-01-01 00:00:00',
    )

    assert result.hours_from_midnight.size == 193
    assert result.days_from_year_start.size == 53
    assert (result.nightly_sun_altitude_degrees < -12.0).any()
    assert (result.annual_max_altitude_degrees > 0.0).any()


@pytest.mark.parametrize('format_name', ['png', 'pdf'])
def test_target_visibility_writes_figure(tmp_path, format_name):
    output_path = tmp_path / f'visibility.{format_name}'
    result = run_target_visibility_diagnostic(
        output_path=output_path,
        target_label='TOI-1233',
        observatory_label='TUG',
        right_ascension_degrees=186.574,  # [deg]
        declination_degrees=-51.363,  # [deg]
        latitude_degrees=36.824166,  # [deg]
        longitude_degrees=30.335555,  # [deg]
        height_meters=2500.0,  # [m]
        utc_offset_hours=3.0,  # [hour]
        night='2022-07-13 00:00:00',
        year_start='2022-01-01 00:00:00',
    )

    assert result.days_from_year_start.size == 53
    assert output_path.is_file()
    assert output_path.stat().st_size > 1000


def test_visibility_example_uses_tdpy_without_miletos():
    repository_path = Path(__file__).resolve().parents[1]
    script = repository_path / 'examples' / 'target_visibility' / 'run.py'
    notebook_path = script.with_name('TargetVisibility.ipynb')
    completed = subprocess.run(
        [sys.executable, str(script), '--help'],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert '--observatory' in completed.stdout
    print(f'Reading from {notebook_path}...')
    notebook = json.loads(notebook_path.read_text())
    sources = ''.join(''.join(cell['source']) for cell in notebook['cells'])
    assert 'from tdpy.astro import run_target_visibility_diagnostic' in sources
    assert 'miletos' not in sources.lower()


def test_datetime_to_julian_date_converts_unix_epoch():
    assert datetime_to_julian_date(
        datetime(1970, 1, 1, tzinfo=timezone.utc)
    ) == 2440587.5


def test_datetime_to_julian_date_requires_timezone():
    with pytest.raises(ValueError, match="timezone"):
        datetime_to_julian_date(datetime(1970, 1, 1))


def test_periodic_event_numbers_include_interval_endpoints():
    assert list(periodic_event_numbers(10.0, 2.0, 11.0, 16.0)) == [1, 2, 3]


def test_linear_ephemeris_uncertainties_returns_both_bounds():
    propagated, upper_bound = linear_ephemeris_uncertainties(0.2, 0.1, 2)
    assert propagated == pytest.approx((0.2**2 + (2 * 0.1) ** 2) ** 0.5)
    assert upper_bound == pytest.approx(0.4)


def test_centered_event_start_phase_range_has_requested_width():
    lower, upper = centered_event_start_phase_range(4.0, 20.0, 10.0)
    assert 20.0 * 24.0 * 60.0 * (upper - lower) == pytest.approx(10.0)
    assert 0.5 * (lower + upper) == pytest.approx(-4.0 / (24.0 * 20.0))


def test_periodic_event_overlaps_finds_intersecting_companion():
    events = {
        "target": {
            "epoch_bjd_tdb": 10.0,
            "period_days": 5.0,
            "duration_hours": 4.0,
        },
        "companion": {
            "epoch_bjd_tdb": 10.1,
            "period_days": 2.0,
            "duration_hours": 2.0,
        },
    }
    overlaps = periodic_event_overlaps("target", 10.0, events)
    assert overlaps[0]["planet"] == "companion"
    assert overlaps[0]["event_number"] == 0
    assert overlaps[0]["offset_from_target_midpoint_hours"] == pytest.approx(2.4)


def test_periodic_event_schedule_propagates_timing_and_overlap():
    events = {
        "target": {
            "epoch_bjd_tdb": 10.0,
            "epoch_uncertainty_days": 0.01,
            "period_days": 5.0,
            "period_uncertainty_days": 0.001,
            "duration_hours": 4.0,
        },
        "companion": {
            "epoch_bjd_tdb": 15.1,
            "period_days": 20.0,
            "duration_hours": 2.0,
        },
    }
    schedule = periodic_event_schedule("target", events, 14.0, 16.0)
    assert len(schedule) == 1
    assert schedule[0]["event_number"] == 1
    assert schedule[0]["timing_uncertainty_minutes"] == pytest.approx(
        (0.01**2 + 0.001**2) ** 0.5 * 24.0 * 60.0
    )
    assert schedule[0]["overlapping_transits"][0]["planet"] == "companion"


def test_build_periodic_event_report_summarizes_schedule():
    events = {
        "target": {
            "epoch_bjd_tdb": 2440587.5,
            "epoch_uncertainty_days": 0.01,
            "period_days": 1.0,
            "period_uncertainty_days": 0.001,
            "duration_hours": 4.0,
        }
    }
    report = build_periodic_event_report(
        events,
        ("target",),
        datetime(1970, 1, 1, tzinfo=timezone.utc),
        datetime(1970, 1, 2, tzinfo=timezone.utc),
        phase_range_duration_minutes=10.0,
        initial_settling_minutes=15.0,
        required_clean_baseline_hours={"target": 1.0},
    )
    target = report["targets"]["target"]
    assert target["number_of_cycle_events"] == 2
    assert target["overlap_free_calendar_events_before_visibility_screening"] == 2
    assert target["minimum_clean_baseline_after_phase_and_settling_hours"] < 2.0


def test_transmission_spectroscopy_metric_matches_reference_formula():
    metric = transmission_spectroscopy_metric(2.61, 707.0, 2.61, 0.876, 8.046)
    expected = 1.26 * 2.61**3 * 707.0 / (2.61 * 0.876**2) * 10.0 ** (-8.046 / 5.0)
    assert metric == pytest.approx(expected)


def test_build_transiting_planet_pairs_groups_rows_and_replaces_systems():
    planet = {
        "sy_jmag": "8.0",
        "pl_eqt": "700",
        "pl_bmasse": "5",
        "st_rad": "1",
    }
    rows = [
        {**planet, "hostname": "original", "pl_name": "b", "pl_rade": "2"},
        {**planet, "hostname": "original", "pl_name": "c", "pl_rade": "3", "pl_eqt": "600"},
        {**planet, "hostname": "excluded", "pl_name": "d", "pl_rade": "2"},
    ]
    replacements = {
        "adopted": [
            {**planet, "pl_name": "e", "pl_rade": "2"},
            {**planet, "pl_name": "f", "pl_rade": "2.5"},
        ]
    }
    pairs = build_transiting_planet_pairs(rows, replacements, ("excluded",))
    assert [pair["hostname"] for pair in pairs] == ["original", "adopted"]
    assert pairs[0]["planets"] == ["b", "c"]
    assert pairs[0]["temperature_difference_kelvin"] == 100.0
    assert pairs[1]["radius_difference_earth"] == 0.5


def test_select_transiting_planet_pair_comparisons_applies_target_thresholds():
    target = {
        "radius_difference_earth": 0.4,
        "temperature_difference_kelvin": 100.0,
        "j_magnitude": 8.0,
        "minimum_transmission_spectroscopy_metric": 50.0,
    }
    pairs = [
        target,
        {**target, "j_magnitude": 9.0, "minimum_transmission_spectroscopy_metric": 60.0},
        {**target, "radius_difference_earth": 0.2},
    ]
    substantial, bright, joint = select_transiting_planet_pair_comparisons(
        pairs, target
    )
    assert len(substantial) == 2
    assert bright == [target]
    assert joint == pairs[:2]