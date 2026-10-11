#!/usr/bin/env python3
"""Plot NASA-confirmed exoplanet discoveries and highlight selected targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tdpy.astro import read_tap_csv_rows
from tdpy.cli import add_plot_arguments
from tdpy.exoplanet import plot_exoplanet_discovery_timeline, plot_exoplanet_period_radius
from tdpy.paths import open_narr
from tdpy.verbosity import print


EXAMPLE_PATH = Path(__file__).resolve().parent
DATA_PATH = EXAMPLE_PATH / "confirmed_planets.csv"
VISUAL_PATH = EXAMPLE_PATH / "visuals"
ARCHIVE_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
ARCHIVE_QUERY = (
    "select top 10000 pl_name, hostname, discoverymethod, disc_year, pl_orbper, pl_rade "
    "from pscomppars"
)
DATA_COLUMNS = ("pl_name", "hostname", "discoverymethod", "disc_year", "pl_orbper", "pl_rade")
HIGHLIGHTED_TARGETS = {
    "HD 108147": "TOI-1233 / HD 108147",
    "WASP-39": "WASP-39",
    "WASP-121": "WASP-121",
    "TRAPPIST-1": "TRAPPIST-1",
    "TOI-270": "TOI-270",
    "TOI-700": "TOI-700",
}


def load_planets(refresh_data: bool = False) -> list[dict[str, str]]:
    """Load the bundled archive snapshot or refresh it from NASA's TAP service."""
    if refresh_data or not DATA_PATH.is_file():
        rows = read_tap_csv_rows(ARCHIVE_ENDPOINT, ARCHIVE_QUERY)
        with open_narr(DATA_PATH, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=DATA_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    with open_narr(DATA_PATH, newline="") as input_file:
        return list(csv.DictReader(input_file))


def run_example(refresh_data: bool = False, typefileplot: str = "png") -> tuple[Path, Path]:
    """Load the confirmed-planet catalog and write discovery and target-context plots."""
    planets = load_planets(refresh_data)
    return (
        plot_exoplanet_discovery_timeline(
            planets,
            HIGHLIGHTED_TARGETS,
            VISUAL_PATH / f"exoplanet_discovery_timeline.{typefileplot}",
        ),
        plot_exoplanet_period_radius(
            planets,
            HIGHLIGHTED_TARGETS,
            VISUAL_PATH / f"exoplanet_period_radius.{typefileplot}",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download a fresh confirmed-planet table from NASA's Exoplanet Archive.",
    )
    add_plot_arguments(parser)
    arguments = parser.parse_args()
    output_paths = run_example(arguments.refresh_data, arguments.typefileplot)
    for output_path in output_paths:
        print(f"Created {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())