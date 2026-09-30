#!/usr/bin/env python3
"""Visualize TESS sector footprints and target visibility from mission WCS geometry."""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
from pathlib import Path

from tdpy.tess import (
    animate_tess_sectors,
    plot_tess_sector_map,
    plot_tess_sector_sequence,
    plot_tess_visibility,
)


EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"

# International Celestial Reference System coordinates [deg] from commonly
# adopted catalog positions. They mark real sky locations; the plotted TESS
# footprints come directly from tesswcs rather than simulated observations.
TARGETS = {
    "Beta Pictoris": (86.8212, -51.0665),
    "TOI-700": (97.1854, -65.5797),
    "Large Magellanic Cloud": (80.8942, -69.7561),
}


def run_example(sectors=range(1, 5), duration_ms=450):
    """Write individual, combined, visibility, and animated TESS sky products."""
    sectors = tuple(int(sector) for sector in sectors)
    sector_paths = plot_tess_sector_sequence(
        sectors,
        VISUAL_PATH,
        targets=TARGETS,
    )
    visibility, combined_path = plot_tess_sector_map(
        sectors,
        VISUAL_PATH / "tess_sectors_combined",
        targets=TARGETS,
    )
    _, visibility_path = plot_tess_visibility(
        TARGETS,
        sectors,
        VISUAL_PATH / "tess_target_visibility",
    )
    animation_path = animate_tess_sectors(
        sectors,
        VISUAL_PATH / "tess_sector_sequence.gif",
        targets=TARGETS,
        duration_ms=duration_ms,
    )
    return {
        "visibility": visibility,
        "sector_paths": sector_paths,
        "combined_path": combined_path,
        "visibility_path": visibility_path,
        "animation_path": animation_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sectors", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--duration-ms", type=int, default=450)
    arguments = parser.parse_args()
    products = run_example(arguments.sectors, arguments.duration_ms)
    for name, sectors in products["visibility"].items():
        print(f"{name}: sectors {list(sectors)}")


if __name__ == "__main__":
    main()