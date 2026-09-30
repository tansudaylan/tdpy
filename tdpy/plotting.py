"""Shared lightweight figure output and theme helpers."""

from pathlib import Path

from tdpy.verbosity import print

import matplotlib.pyplot as plt


def plot_file_path(path, typefileplot="png"):
    """Return a plot path with the requested supported extension."""
    if typefileplot not in {"png", "pdf"}:
        raise ValueError("typefileplot must be 'png' or 'pdf'")
    return str(Path(path).with_suffix("." + typefileplot))


def normalize_plot_background(typeplotback="norm"):
    """Return the canonical plot-background name for a supported alias."""
    aliases = {"norm": "norm", "white": "norm", "dark": "dark", "black": "dark"}
    try:
        return aliases[typeplotback]
    except KeyError as exception:
        raise ValueError(
            "typeplotback must be 'norm', 'white', 'dark', or 'black'"
        ) from exception


def plot_background_colors(typeplotback="norm"):
    """Return background and foreground colors for a supported plot theme."""
    typeplotback = normalize_plot_background(typeplotback)
    if typeplotback == "norm":
        return "white", "black"
    return "black", "white"


def save_figure(
    figure,
    path,
    typefileplot="png",
    typeplotback="norm",
    close_figure=False,
    **kwargs,
):
    """Save a figure with a consistent background, axes, and output format."""
    path = plot_file_path(path, typefileplot)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    color_background, color_foreground = plot_background_colors(typeplotback)
    figure.patch.set_facecolor(color_background)
    for axis in figure.axes:
        axis.set_facecolor(color_background)
        axis.grid(False)
        axis.tick_params(colors=color_foreground)
        for spine in axis.spines.values():
            spine.set_color(color_foreground)
        axis.xaxis.label.set_color(color_foreground)
        axis.yaxis.label.set_color(color_foreground)
        axis.title.set_color(color_foreground)

    arguments = {"facecolor": color_background, "bbox_inches": "tight"}
    if typefileplot == "png":
        arguments["dpi"] = 300
    arguments.update(kwargs)
    print(f"Writing to {path}...")
    figure.savefig(path, **arguments)
    if close_figure:
        plt.close(figure)
    return path


def save_current_figure(path, typefileplot="png", typeplotback="norm", **kwargs):
    """Save the current Matplotlib figure with the standard plot styling."""
    return save_figure(plt.gcf(), path, typefileplot, typeplotback, **kwargs)


__all__ = [
    "normalize_plot_background",
    "plot_background_colors",
    "plot_file_path",
    "save_current_figure",
    "save_figure",
]