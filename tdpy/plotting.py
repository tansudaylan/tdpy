"""Shared lightweight figure output and theme helpers."""

from pathlib import Path

from tdpy.verbosity import print

import matplotlib.pyplot as plt
import numpy as np


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


def figure_to_frame(figure, close_figure=True):
    """Render a Matplotlib figure into an RGB PIL image for an animation frame."""
    from PIL import Image

    figure.canvas.draw()
    frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())).convert("RGB")
    if close_figure:
        plt.close(figure)
    return frame


def write_animation(frames, path, duration_ms=200, loop=0):
    """Write RGB frames to an animated GIF with one shared palette and return its path.

    Frames of different sizes are padded with white to the largest size, and one adaptive
    palette built from all frames keeps colors identical from frame to frame.
    """
    from PIL import Image

    frames = [frame if frame.mode == "RGB" else frame.convert("RGB") for frame in frames]
    if not frames:
        raise ValueError("write_animation needs at least one frame")
    if duration_ms < 1:
        raise ValueError("duration_ms must be positive")
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    if all(frame.size == (width, height) for frame in frames):
        padded = frames
    else:
        padded = []
        for frame in frames:
            canvas = Image.new("RGB", (width, height), "white")
            canvas.paste(frame, (0, 0))
            padded.append(canvas)
    # sample the shared palette from across the sequence without building an
    # enormous strip for a long animation
    palette_frames = [padded[index] for index in np.unique(np.linspace(0, len(padded) - 1, min(len(padded), 8)).astype(int))]
    strip = Image.new("RGB", (width, height * len(palette_frames)))
    for index, frame in enumerate(palette_frames):
        strip.paste(frame, (0, index * height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    quantized = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in padded]
    path = Path(path).with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=duration_ms,
                      loop=loop, disposal=2, optimize=False)
    return path


__all__ = [
    "figure_to_frame",
    "normalize_plot_background",
    "plot_background_colors",
    "plot_file_path",
    "save_current_figure",
    "save_figure",
    "write_animation",
]