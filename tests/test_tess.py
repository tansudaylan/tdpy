from datetime import date
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import tdpy
import tdpy.tess as tess_module

from tdpy.tess import (
    TessToiAlert,
    animate_tess_sectors,
    plot_tess_sector_map,
    plot_tess_visibility,
    target_is_visible,
    tess_sector_dates,
    tess_sector_footprints,
    tess_target_visibility,
)
from tdpy.util import plot_lsst_ddf_tess_footprint


SECTOR_ONE_CCD_CENTER = (319.2026575157548, -41.18304654929803)  # [deg]


def test_tess_sector_footprints_return_all_ccds():
    footprints = tess_sector_footprints(1, samples_per_edge=8)

    assert len(footprints) == 16
    assert {(item.camera, item.ccd) for item in footprints} == {
        (camera, ccd) for camera in range(1, 5) for ccd in range(1, 5)
    }
    for footprint in footprints:
        assert footprint.right_ascension_deg.shape == footprint.declination_deg.shape
        assert np.all(np.isfinite(footprint.right_ascension_deg))
        assert np.all(np.isfinite(footprint.declination_deg))
        assert np.all((0.0 <= footprint.right_ascension_deg) & (footprint.right_ascension_deg < 360.0))
        assert np.all((-90.0 <= footprint.declination_deg) & (footprint.declination_deg <= 90.0))


def test_tess_sector_dates_come_from_pointing_table():
    assert tess_sector_dates(1) == ('2018-07-25', '2018-08-22')
    assert tess_sector_dates(121) == ('2027-08-23', '2027-09-19')


def test_planned_sector_dates_and_predicted_footprints():
    assert tess_sector_dates(122) == ('2027-09-19', '2027-10-17')
    assert tess_sector_dates(134) == ('2028-08-16', '2028-09-13')
    assert len(tess_sector_footprints(122, samples_per_edge=4)) == 16
    with pytest.raises(ValueError, match='published dates'):
        tess_sector_dates(135)


def test_tess_functions_are_exported_at_package_level():
    assert tdpy.tess_sector_footprints is tess_sector_footprints
    assert tdpy.plot_tess_sector_map is plot_tess_sector_map
    assert tdpy.animate_tess_sectors is animate_tess_sectors


def test_tess_target_visibility_uses_detector_geometry():
    targets = {
        "Sector 1 CCD center": SECTOR_ONE_CCD_CENTER,
        "North celestial pole": (0.0, 90.0),
    }

    assert target_is_visible(1, *SECTOR_ONE_CCD_CENTER)
    visibility = tess_target_visibility(targets, (1, 2))
    assert 1 in visibility["Sector 1 CCD center"]
    assert visibility["North celestial pole"] == ()


def test_tess_plots_and_animation_write_outputs(tmp_path):
    targets = {"Sector 1 CCD center": SECTOR_ONE_CCD_CENTER}
    visibility, map_path = plot_tess_sector_map(
        (1, 2), tmp_path / "sector_map", targets=targets
    )
    timeline_visibility, timeline_path = plot_tess_visibility(
        targets, (1, 2), tmp_path / "visibility"
    )
    animation_path = animate_tess_sectors(
        (1, 2), tmp_path / "sectors.gif", targets=targets, duration_ms=20
    )

    assert visibility == timeline_visibility
    assert map_path.is_file()
    assert timeline_path.is_file()
    assert animation_path.is_file()
    with Image.open(animation_path) as animation:
        assert animation.n_frames == 2


def test_animation_dates_and_highlights_planned_target(tmp_path, monkeypatch):
    titles = []
    text_labels = []
    render_frame = tess_module.figure_to_frame

    def capture_frame(figure):
        axis = figure.axes[0]
        titles.append(axis.get_title())
        text_labels.append([(item.get_text(), item.get_fontweight()) for item in axis.texts])
        return render_frame(figure)

    monkeypatch.setattr(tess_module, 'figure_to_frame', capture_frame)
    path = animate_tess_sectors(
        (1, 134), tmp_path / 'sequence.gif',
        targets={'TOI-1233': (186.574548, -51.362837)},
        highlighted_targets=('TOI-1233',),
        as_of=date(2026, 10, 1),
    )

    assert '2018-07-25 to 2018-08-22 (Past)' in titles[0]
    assert '2028-08-16 to 2028-09-13 (Planned)' in titles[1]
    assert all(any('TOI-1233' in text and weight == 'bold' for text, weight in labels)
               for labels in text_labels)
    with Image.open(path) as animation:
        assert animation.n_frames == 2


def test_toi_dots_start_at_alert_date_and_highlight_toi1233(tmp_path, monkeypatch):
    frames = []
    render_frame = tess_module.figure_to_frame

    def capture_frame(figure):
        axis = figure.axes[0]
        frames.append((axis.get_title(), [item.get_text() for item in axis.texts]))
        return render_frame(figure)

    monkeypatch.setattr(tess_module, 'figure_to_frame', capture_frame)
    alerts = (
        TessToiAlert('101.01', 319.2, -41.2, date(2018, 9, 5)),
        TessToiAlert('1233.01', 186.574, -51.363, date(2019, 8, 26)),
    )
    path = animate_tess_sectors(
        (1, 2, 14, 15, 134), tmp_path / 'toi_sequence.gif',
        toi_alerts=alerts, highlighted_toi='1233', as_of=date(2026, 10, 1),
    )

    assert '0 alerted TOIs' in frames[0][1]
    assert '1 alerted TOIs' in frames[1][1]
    assert not any('TOI-1233' in text for text in frames[2][1])
    assert '2 alerted TOIs' in frames[3][1]
    assert 'TOI-1233' in frames[3][1]
    assert '2 alerted TOIs' in frames[4][1]
    with Image.open(path) as animation:
        assert animation.n_frames == 5


def test_example_animation_defaults_to_every_published_sector(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / 'examples' / 'tess_visibility' / 'tess_visibility.py'
    specification = importlib.util.spec_from_file_location('tess_visibility_example', script)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)
    monkeypatch.setattr(example, 'VISUAL_PATH', tmp_path)
    selected = []

    def capture_animation(sectors, output_path, **kwargs):
        selected.append(tuple(sectors))
        return output_path

    monkeypatch.setattr(example, 'animate_tess_sectors', capture_animation)
    monkeypatch.setattr(example, 'load_toi_alerts', lambda: ())
    example.run_example(sectors=(1, 2))
    assert selected == [tuple(range(1, 135))] * 2
    assert 'TOI-1233' in example.TARGETS
    assert 'TOI-700' not in example.TARGETS


def test_example_uses_frozen_exofop_alert_dates():
    script = Path(__file__).resolve().parents[1] / 'examples' / 'tess_visibility' / 'tess_visibility.py'
    specification = importlib.util.spec_from_file_location('tess_visibility_example', script)
    example = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(example)

    alerts = example.load_toi_alerts()
    assert len(alerts) == 8148
    assert {alert.alerted_on for alert in alerts if alert.toi.startswith('1233.')} == {date(2019, 8, 26)}
    assert min(alert.alerted_on for alert in alerts) == date(2018, 8, 30)
    assert max(alert.alerted_on for alert in alerts) <= date(2026, 10, 1)


def test_legacy_lsst_wrapper_delegates_to_tess_submodule(tmp_path):
    target = [("Sector 1 CCD center", *SECTOR_ONE_CCD_CENTER)]
    visibility, output_path = plot_lsst_ddf_tess_footprint(
        1, tmp_path, fields=target
    )

    assert visibility == {"Sector 1 CCD center": [1]}
    assert Path(output_path).is_file()