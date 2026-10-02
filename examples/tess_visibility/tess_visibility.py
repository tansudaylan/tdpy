#!/usr/bin/env python3
"""Visualize TESS sector footprints and target visibility from mission WCS geometry."""

from __future__ import annotations

from tdpy.verbosity import print

import argparse
from datetime import date
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd

from tdpy.tess import (
    PLANNED_SECTORS,
    TessToiAlert,
    animate_tess_sectors,
    plot_tess_sector_map,
    plot_tess_sector_sequence,
    plot_tess_visibility,
)


EXAMPLE_PATH = Path(__file__).resolve().parent
VISUAL_PATH = EXAMPLE_PATH / "visuals"
TOI_ALERTS_PATH = EXAMPLE_PATH / 'toi_alerts.csv'
EXOFOP_TOI_URL = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'

# International Celestial Reference System coordinates [deg] from commonly
# adopted catalog positions. They mark real sky locations; the plotted TESS
# footprints come directly from tesswcs rather than simulated observations.
TARGETS = {
    "Beta Pictoris": (86.8212, -51.0665),
    "TOI-1233": (186.574548, -51.362837),
    "Large Magellanic Cloud": (80.8942, -69.7561),
}


def refresh_toi_alerts(source=EXOFOP_TOI_URL):
    """Record ExoFOP's TOI-alert dates and ICRS positions for offline reruns."""

    print(f'Reading from {source}...')
    table = pd.read_csv(source, dtype={'TOI': str}, low_memory=False)
    columns = ('TOI', 'RA', 'Dec', 'Date TOI Alerted (UTC)')
    if not set(columns).issubset(table.columns):
        raise ValueError('ExoFOP TOI table is missing identification, position, or alert date')
    table = table.dropna(subset=list(columns))
    coordinates = SkyCoord(table['RA'].astype(str).to_list(), table['Dec'].astype(str).to_list(),
                           unit=(u.hourangle, u.deg))
    alerts = pd.DataFrame({
        'toi': table['TOI'].to_numpy(),
        'ra_deg': coordinates.ra.deg,
        'dec_deg': coordinates.dec.deg,
        'alert_date': pd.to_datetime(table['Date TOI Alerted (UTC)'], utc=True).dt.date.astype(str).to_numpy(),
    })
    print(f'Writing to {TOI_ALERTS_PATH}...')
    alerts.sort_values('toi').to_csv(TOI_ALERTS_PATH, index=False)


def load_toi_alerts():
    """Read the frozen public TOI-alert snapshot; retain coincident TOI entries."""

    print(f'Reading from {TOI_ALERTS_PATH}...')
    table = pd.read_csv(TOI_ALERTS_PATH, dtype={'toi': str})
    return tuple(TessToiAlert(row.toi, row.ra_deg, row.dec_deg, date.fromisoformat(row.alert_date))
                 for row in table.itertuples(index=False))


def run_example(sectors=range(1, 5), duration_ms=450, animation_sectors=None, refresh_tois=False):
    """Write short static maps and a dated animation of all published sectors."""
    if refresh_tois:
        refresh_toi_alerts()
    sectors = tuple(int(sector) for sector in sectors)
    animation_sectors = tuple(range(1, max(PLANNED_SECTORS) + 1)) if animation_sectors is None else tuple(animation_sectors)
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
        animation_sectors,
        VISUAL_PATH / "tess_sector_sequence.gif",
        targets=TARGETS,
        duration_ms=duration_ms,
        highlighted_targets=('TOI-1233',),
    )
    toi_animation_path = animate_tess_sectors(
        animation_sectors,
        VISUAL_PATH / 'tess_sector_sequence_tois.gif',
        toi_alerts=load_toi_alerts(),
        highlighted_toi='1233',
        duration_ms=duration_ms,
    )
    return {
        "visibility": visibility,
        "sector_paths": sector_paths,
        "combined_path": combined_path,
        "visibility_path": visibility_path,
        "animation_path": animation_path,
        'toi_animation_path': toi_animation_path,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sectors", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--animation-sectors", type=int, nargs="+", help="Override the default full dated sector sequence.")
    parser.add_argument("--duration-ms", type=int, default=450)
    parser.add_argument('--refresh-tois', action='store_true', help='Update the local ExoFOP alert-date snapshot.')
    arguments = parser.parse_args()
    products = run_example(arguments.sectors, arguments.duration_ms, arguments.animation_sectors, arguments.refresh_tois)
    for name, sectors in products["visibility"].items():
        print(f"{name}: sectors {list(sectors)}")


if __name__ == "__main__":
    main()