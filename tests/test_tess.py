from pathlib import Path

import numpy as np
from PIL import Image
import tdpy

from tdpy.tess import (
    animate_tess_sectors,
    plot_tess_sector_map,
    plot_tess_visibility,
    target_is_visible,
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


def test_legacy_lsst_wrapper_delegates_to_tess_submodule(tmp_path):
    target = [("Sector 1 CCD center", *SECTOR_ONE_CCD_CENTER)]
    visibility, output_path = plot_lsst_ddf_tess_footprint(
        1, tmp_path, fields=target
    )

    assert visibility == {"Sector 1 CCD center": [1]}
    assert Path(output_path).is_file()