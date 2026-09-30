"""TESS sector geometry, target visibility, and full-sky visualization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping
import warnings

from tdpy.verbosity import print

import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tesswcs

from .plotting import plot_background_colors, save_figure


DETECTOR_SHAPE = (2048, 2048)


@dataclass(frozen=True)
class TessCcdFootprint:
    """Sky boundary of one TESS CCD in the International Celestial Reference System."""

    sector: int
    camera: int
    ccd: int
    right_ascension_deg: np.ndarray
    declination_deg: np.ndarray


def _validate_sector(sector: int) -> int:
    sector = int(sector)
    if sector < 1:
        raise ValueError("sector must be a positive integer")
    return sector


def _normalize_sectors(sectors) -> tuple[int, ...]:
    if np.isscalar(sectors):
        sectors = (sectors,)
    sectors = tuple(_validate_sector(sector) for sector in sectors)
    if not sectors:
        raise ValueError("sectors must contain at least one sector")
    return sectors


@lru_cache(maxsize=None)
def _get_wcs(sector: int, camera: int, ccd: int):
    return tesswcs.WCS.from_sector(sector=sector, camera=camera, ccd=ccd)


def _detector_boundary(detector_shape, samples_per_edge):
    if len(detector_shape) != 2 or min(detector_shape) <= 0:
        raise ValueError("detector_shape must contain two positive pixel counts")
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least two")
    width, height = map(float, detector_shape)
    horizontal = np.linspace(0.0, width, samples_per_edge, endpoint=False)
    vertical = np.linspace(0.0, height, samples_per_edge, endpoint=False)
    x_pixel = np.concatenate(
        (horizontal, np.full_like(vertical, width), width - horizontal, np.zeros_like(vertical), [0.0])
    )
    y_pixel = np.concatenate(
        (np.zeros_like(horizontal), vertical, np.full_like(horizontal, height), height - vertical, [0.0])
    )
    return x_pixel, y_pixel


def tess_sector_footprints(
    sector: int,
    detector_shape=DETECTOR_SHAPE,
    samples_per_edge: int = 24,
) -> tuple[TessCcdFootprint, ...]:
    """Return the 16 camera/CCD sky boundaries for one TESS sector."""
    sector = _validate_sector(sector)
    x_pixel, y_pixel = _detector_boundary(detector_shape, samples_per_edge)
    footprints = []
    for camera in range(1, 5):
        for ccd in range(1, 5):
            sky = _get_wcs(sector, camera, ccd).pixel_to_world(x_pixel, y_pixel)
            footprints.append(
                TessCcdFootprint(
                    sector=sector,
                    camera=camera,
                    ccd=ccd,
                    right_ascension_deg=np.asarray(sky.ra.to_value(u.deg)),
                    declination_deg=np.asarray(sky.dec.to_value(u.deg)),
                )
            )
    return tuple(footprints)


def target_is_visible(
    sector: int,
    right_ascension_deg: float,
    declination_deg: float,
    detector_shape=DETECTOR_SHAPE,
) -> bool:
    """Return whether an ICRS coordinate falls on any CCD in a TESS sector."""
    sector = _validate_sector(sector)
    width, height = detector_shape
    coordinate = SkyCoord(right_ascension_deg * u.deg, declination_deg * u.deg, frame="icrs")
    for camera in range(1, 5):
        for ccd in range(1, 5):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="'WCS.all_world2pix' failed to converge.*",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                x_pixel, y_pixel = _get_wcs(sector, camera, ccd).world_to_pixel(coordinate)
            if (
                np.isfinite(x_pixel)
                and np.isfinite(y_pixel)
                and 0.0 <= x_pixel <= width
                and 0.0 <= y_pixel <= height
            ):
                return True
    return False


def tess_target_visibility(
    targets: Mapping[str, tuple[float, float]],
    sectors,
    detector_shape=DETECTOR_SHAPE,
) -> dict[str, tuple[int, ...]]:
    """Return visible TESS sectors for named ICRS targets in degrees."""
    sectors = _normalize_sectors(sectors)
    visibility = {}
    for name, coordinates in targets.items():
        if len(coordinates) != 2:
            raise ValueError(f"target {name!r} must contain right ascension and declination")
        right_ascension_deg, declination_deg = map(float, coordinates)
        visibility[name] = tuple(
            sector
            for sector in sectors
            if target_is_visible(
                sector,
                right_ascension_deg,
                declination_deg,
                detector_shape=detector_shape,
            )
        )
    return visibility


def _mollweide_longitude(right_ascension_deg):
    return np.deg2rad(((180.0 - np.asarray(right_ascension_deg)) % 360.0) - 180.0)


def _split_wrapped_path(longitude, latitude):
    split_indices = np.flatnonzero(np.abs(np.diff(longitude)) > np.pi / 2.0) + 1
    return zip(np.split(longitude, split_indices), np.split(latitude, split_indices))


def _draw_sector_map(axis, sectors, targets, typeplotback):
    background, foreground = plot_background_colors(typeplotback)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(sectors)))
    for sector, color in zip(sectors, colors):
        label_pending = True
        for footprint in tess_sector_footprints(sector):
            longitude = _mollweide_longitude(footprint.right_ascension_deg)
            latitude = np.deg2rad(footprint.declination_deg)
            for longitude_segment, latitude_segment in _split_wrapped_path(longitude, latitude):
                axis.plot(
                    longitude_segment,
                    latitude_segment,
                    color=color,
                    linewidth=0.8,
                    alpha=0.72,
                    label=f"Sector {sector}" if label_pending else None,
                )
                label_pending = False

    visibility = tess_target_visibility(targets, sectors) if targets else {}
    for name, (right_ascension_deg, declination_deg) in targets.items():
        visible = visibility[name]
        axis.scatter(
            _mollweide_longitude(right_ascension_deg),
            np.deg2rad(declination_deg),
            marker="*",
            s=65,
            color="#A51417" if visible else "0.55",
            edgecolor=foreground,
            linewidth=0.5,
            zorder=4,
        )
        label = f"{name} ({', '.join(map(str, visible)) if visible else 'not visible'})"
        axis.annotate(
            label,
            (_mollweide_longitude(right_ascension_deg), np.deg2rad(declination_deg)),
            xytext=(4, 5),
            textcoords="offset points",
            color=foreground,
            fontsize=8,
        )

    tick_degrees = np.arange(-150, 181, 30)
    axis.set_xticks(np.deg2rad(tick_degrees))
    axis.set_xticklabels([f"{int((360 - value) % 360)}°" for value in tick_degrees])
    axis.set_xlabel("Right ascension [deg]")
    axis.set_ylabel("Declination [deg]")
    axis.set_facecolor(background)
    axis.grid(False)
    if len(sectors) <= 8:
        axis.legend(loc="lower left", frameon=True, fancybox=True, framealpha=1.0, fontsize=8)
    return visibility


def plot_tess_sector_map(
    sectors,
    output_path,
    targets: Mapping[str, tuple[float, float]] | None = None,
    typefileplot: str = "png",
    typeplotback: str = "white",
) -> tuple[dict[str, tuple[int, ...]], Path]:
    """Plot TESS sector footprints and optional target visibility on the full sky."""
    sectors = _normalize_sectors(sectors)
    targets = {} if targets is None else dict(targets)
    figure = plt.figure(figsize=(11.0, 6.2))
    axis = figure.add_subplot(111, projection="mollweide")
    visibility = _draw_sector_map(axis, sectors, targets, typeplotback)
    sector_label = str(sectors[0]) if len(sectors) == 1 else f"{sectors[0]}-{sectors[-1]}"
    axis.set_title(f"TESS sector {sector_label} sky coverage")
    figure.tight_layout()
    path = Path(save_figure(figure, output_path, typefileplot, typeplotback, close_figure=True))
    return visibility, path


def plot_tess_visibility(
    targets: Mapping[str, tuple[float, float]],
    sectors,
    output_path,
    typefileplot: str = "png",
    typeplotback: str = "white",
) -> tuple[dict[str, tuple[int, ...]], Path]:
    """Plot a target-by-sector TESS visibility matrix."""
    sectors = _normalize_sectors(sectors)
    targets = dict(targets)
    if not targets:
        raise ValueError("targets must contain at least one target")
    visibility = tess_target_visibility(targets, sectors)
    matrix = np.array(
        [[sector in visibility[name] for sector in sectors] for name in targets],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(max(6.0, 0.32 * len(sectors)), max(2.5, 0.45 * len(targets))))
    shown = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(sectors)))
    axis.set_xticklabels(sectors, rotation=90 if len(sectors) > 15 else 0)
    axis.set_yticks(np.arange(len(targets)))
    axis.set_yticklabels(targets)
    axis.set_xlabel("TESS sector")
    axis.set_ylabel("Target")
    axis.set_title("TESS target visibility")
    axis.grid(False)
    colorbar = figure.colorbar(shown, ax=axis, ticks=(0.0, 1.0), shrink=0.8)
    colorbar.ax.set_yticklabels(("Outside CCDs", "On silicon"))
    figure.tight_layout()
    path = Path(save_figure(figure, output_path, typefileplot, typeplotback, close_figure=True))
    return visibility, path


def plot_tess_sector_sequence(
    sectors,
    output_directory,
    targets: Mapping[str, tuple[float, float]] | None = None,
    typefileplot: str = "png",
    typeplotback: str = "white",
) -> tuple[Path, ...]:
    """Write one full-sky footprint figure for every requested TESS sector."""
    sectors = _normalize_sectors(sectors)
    output_directory = Path(output_directory)
    paths = []
    for sector in sectors:
        _, path = plot_tess_sector_map(
            sector,
            output_directory / f"tess_sector_{sector:04d}",
            targets=targets,
            typefileplot=typefileplot,
            typeplotback=typeplotback,
        )
        paths.append(path)
    return tuple(paths)


def animate_tess_sectors(
    sectors,
    output_path,
    targets: Mapping[str, tuple[float, float]] | None = None,
    duration_ms: int = 500,
    typeplotback: str = "white",
) -> Path:
    """Write a full-sky GIF with one frame per TESS sector."""
    sectors = _normalize_sectors(sectors)
    if duration_ms < 1:
        raise ValueError("duration_ms must be positive")
    targets = {} if targets is None else dict(targets)
    background, _ = plot_background_colors(typeplotback)
    frames = []
    for sector in sectors:
        figure = plt.figure(figsize=(9.6, 5.4), facecolor=background)
        axis = figure.add_subplot(111, projection="mollweide")
        _draw_sector_map(axis, (sector,), targets, typeplotback)
        axis.set_title(f"TESS Sector {sector}")
        figure.tight_layout()
        figure.canvas.draw()
        frames.append(Image.fromarray(np.asarray(figure.canvas.buffer_rgba())).convert("RGB"))
        plt.close(figure)

    output_path = Path(output_path).with_suffix(".gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return output_path


__all__ = [
    "DETECTOR_SHAPE",
    "TessCcdFootprint",
    "animate_tess_sectors",
    "plot_tess_sector_map",
    "plot_tess_sector_sequence",
    "plot_tess_visibility",
    "target_is_visible",
    "tess_sector_footprints",
    "tess_target_visibility",
]