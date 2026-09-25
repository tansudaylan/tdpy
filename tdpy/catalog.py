"""Reusable synthetic catalog diagnostics for plotting workflows."""

from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .util import plot_catl


class SyntheticCatalog:
    """Deterministic source and candidate positions for a catalog diagnostic."""

    def __init__(self):
        self.numbpositext = 3
        self.indxsideyposdataflat = np.array([6.0, 18.0, 29.0])  # [pixel]
        self.indxsidexposdataflat = np.array([9.0, 25.0, 14.0])  # [pixel]
        self.indxdatascorsort = np.array([0, 1, 2])
        self.numbsideedge = 3
        self.datatype = "mock"
        self.indxsour = np.array([0, 1])
        self.indxsoursupn = np.array([2])
        self.trueypos = np.array(
            [[7.0, 19.0, 30.0], [7.4, 18.6, 30.3]]
        )  # [pixel]
        self.truexpos = np.array(
            [[10.0, 26.0, 15.0], [10.2, 25.7, 15.4]]
        )  # [pixel]
        self.truemagtmean = np.array([11.0, 12.0, 13.0])  # [mag]
        self.truemagtstdv = np.array([0.1, 0.2, 0.3])  # [mag]


def make_synthetic_field(
    catalog: SyntheticCatalog,
    side_pixels: int = 40,
    width_pixels: float = 1.4,
) -> np.ndarray:
    """Render the catalog as a deterministic Gaussian source field."""

    if side_pixels <= 0 or width_pixels <= 0.0:
        raise ValueError("Field size and source width must be positive.")

    y_grid, x_grid = np.mgrid[:side_pixels, :side_pixels]  # [pixel]
    field = np.zeros((side_pixels, side_pixels), dtype=float)
    source_x = np.mean(catalog.trueypos, axis=0)  # [pixel]
    source_y = np.mean(catalog.truexpos, axis=0)  # [pixel]
    relative_flux = 10.0 ** (-0.4 * (catalog.truemagtmean - catalog.truemagtmean.min()))
    for x_position, y_position, flux in zip(source_x, source_y, relative_flux):
        radius_squared = (x_grid - x_position) ** 2 + (y_grid - y_position) ** 2
        field += flux * np.exp(-0.5 * radius_squared / width_pixels**2)
    return field


def plot_synthetic_catalog_diagnostic(output_path: str | Path) -> Path:
    """Plot the simulated field before and after catalog annotation."""

    output_path = Path(output_path)
    catalog = SyntheticCatalog()
    field = make_synthetic_field(catalog)

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(9.5, 4.2),
        sharex=True,
        sharey=True,
        facecolor="white",
        constrained_layout=True,
    )
    images = []
    for axis, title in zip(axes, ("Simulated input", "Catalog diagnostic")):
        images.append(
            axis.imshow(
                field,
                origin="lower",
                cmap="gray_r",
                vmin=0.0,
                vmax=field.max(),
            )
        )
        axis.set_title(title)
        axis.set_xlabel("x position [pixel]")
        axis.grid(False)
    axes[0].set_ylabel("y position [pixel]")
    plot_catl(
        catalog,
        axes[1],
        indxsideyposoffs=1,
        indxsidexposoffs=2,
        annotation_alpha=0.9,
        label_offset=1.5,
    )
    legend_handles = [
        Line2D([], [], color="blue", marker="$0$", linestyle="None", label="Candidate rank"),
        Line2D([], [], color="#B8860B", marker="*", linestyle="None", label="Isolated source"),
        Line2D([], [], color="green", marker="*", linestyle="None", label="Blended source"),
    ]
    legend = axes[1].legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    figure.colorbar(images[-1], ax=axes, label="Relative intensity")
    figure.suptitle("Synthetic source-field catalog overlay", fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output_path