"""Shared lightweight figure output and theme helpers."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from inspect import Parameter, signature
from pathlib import Path

from tdpy.verbosity import print

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np

WASHU_RED = "#A51417"
WASHU_GREEN = "#007360"
WASHU_GRAY = "#6C7373"
WASHU_COLORS = {
    "red": WASHU_RED,
    "green": WASHU_GREEN,
    "gray": WASHU_GRAY,
}
_PLOT_BACKGROUND = ContextVar("tdpy_plot_background", default="white")


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


def plot_background_colors(typeplotback=None):
    """Return background and foreground colors for a supported plot theme."""
    typeplotback = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    typeplotback = normalize_plot_background(typeplotback)
    if typeplotback == "norm":
        return "white", "black"
    return "black", "white"


def plot_style_parameters(typeplotback="white"):
    """Return Matplotlib settings for white or dark plots without gridlines."""
    background, foreground = plot_background_colors(typeplotback)
    return {
        "figure.facecolor": background,
        "figure.edgecolor": background,
        "axes.facecolor": background,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "axes.titlecolor": foreground,
        "axes.grid": False,
        "text.color": foreground,
        "xtick.color": foreground,
        "xtick.labelcolor": foreground,
        "ytick.color": foreground,
        "ytick.labelcolor": foreground,
        "legend.facecolor": background,
        "legend.edgecolor": foreground,
        "legend.labelcolor": foreground,
        "legend.framealpha": 1.0,
        "savefig.facecolor": background,
        "savefig.edgecolor": background,
        "savefig.transparent": False,
    }


@contextmanager
def plot_style(typeplotback=None):
    """Create plots in a theme and restore the caller's settings on exit."""
    background = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    parameters = plot_style_parameters(background)
    token = _PLOT_BACKGROUND.set(background)
    try:
        with plt.rc_context(parameters):
            yield
    finally:
        _PLOT_BACKGROUND.reset(token)


def configure_plot_style(typeplotback="white"):
    """Set a script or notebook's plot theme and return its foreground color."""
    parameters = plot_style_parameters(typeplotback)
    plt.rcParams.update(parameters)
    _PLOT_BACKGROUND.set(typeplotback)
    return parameters["text.color"]


def themed_plot(function):
    """Scope an existing background option or add a keyword-only one."""
    original_signature = signature(function)
    parameters = list(original_signature.parameters.values())
    has_background = 'typeplotback' in original_signature.parameters
    unspecified = object()
    if not has_background:
        insertion = next((index for index, parameter in enumerate(parameters)
                          if parameter.kind == Parameter.VAR_KEYWORD), len(parameters))
        parameters.insert(insertion, Parameter('typeplotback', Parameter.KEYWORD_ONLY, default=None))

    @wraps(function)
    def render(*args, typeplotback=unspecified, **kwargs):
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure

        if has_background:
            if typeplotback is not unspecified:
                kwargs['typeplotback'] = typeplotback
            bound = original_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            selected = bound.arguments['typeplotback']
        else:
            selected = None if typeplotback is unspecified else typeplotback
        if selected is None and args:
            selected = getattr(args[0], '__dict__', {}).get('typeplotback')
        if selected is None:
            selected = function.__globals__.get('typeplotback')
        with plot_style(selected):
            previous = set(plt.get_fignums())
            result = function(*args, **kwargs)
            current = plt.gcf() if plt.get_fignums() else None
            figures = [plt.figure(number) for number in plt.get_fignums() if number not in previous]
            candidates = result if isinstance(result, (tuple, list)) else (result,)
            if isinstance(result, dict):
                candidates = (result.get('figure'),)
            for candidate in (*args, *kwargs.values(), *candidates):
                items = candidate.flat if isinstance(candidate, np.ndarray) and candidate.dtype == object else (candidate,)
                for item in items:
                    figure = item.figure if isinstance(item, Axes) else item
                    if not isinstance(figure, Figure):
                        figure = getattr(item, '__dict__', {}).get('fig')
                    if isinstance(figure, Figure) and figure not in figures:
                        figures.append(figure)
            for figure in figures:
                apply_plot_theme(figure)
            if current is not None:
                plt.figure(current.number)
            return result

    render.__signature__ = original_signature.replace(parameters=parameters)
    return render


def save_plot(figure, *args, typeplotback=None, **kwargs):
    """Theme a Matplotlib export while retaining its path, format, and options."""
    apply_plot_theme(figure, typeplotback)
    kwargs['facecolor'] = figure.get_facecolor()
    return figure.savefig(*args, **kwargs)


def apply_plot_theme(figure, typeplotback=None):
    """Style an existing figure without changing data colors or global settings."""
    typeplotback = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    background, foreground = plot_background_colors(typeplotback)
    figure.set_facecolor(background)
    figure.set_edgecolor(background)
    for text in figure.texts:
        if to_rgba(text.get_color())[:3] in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
            text.set_color(foreground)
    legends = list(figure.legends)
    for axis in figure.axes:
        axis.set_facecolor(background)
        axis.grid(False)
        axis.tick_params(colors=foreground, which="both")
        for text in axis.texts:
            if to_rgba(text.get_color())[:3] in ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)):
                text.set_color(foreground)
        if hasattr(axis, "coords"):
            for coordinate in axis.coords:
                coordinate.set_ticks(color=foreground)
                coordinate.set_ticklabel(color=foreground)
                coordinate.set_axislabel(coordinate.get_axislabel(), color=foreground)
        for spine in axis.spines.values():
            spine.set_color(foreground)
        for coordinate in (axis.xaxis, axis.yaxis, getattr(axis, "zaxis", None)):
            if coordinate is not None:
                coordinate.label.set_color(foreground)
                coordinate.get_offset_text().set_color(foreground)
                if hasattr(coordinate, "line"):
                    coordinate.line.set_color(foreground)
                if hasattr(coordinate, "set_pane_color"):
                    coordinate.set_pane_color(background)
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_color(foreground)
        if axis.get_legend() is not None:
            legends.append(axis.get_legend())
    for legend in legends:
        legend.get_frame().set_facecolor(background)
        legend.get_frame().set_edgecolor(foreground)
        legend.get_frame().set_alpha(1.0)
        for text in (*legend.get_texts(), legend.get_title()):
            text.set_color(foreground)
    return figure


def save_figure(
    figure,
    path,
    typefileplot="png",
    typeplotback=None,
    close_figure=False,
    **kwargs,
):
    """Save a figure with a consistent background, axes, and output format."""
    typeplotback = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    path = plot_file_path(path, typefileplot)
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    if getattr(figure, "axes", None) and getattr(figure, "get_layout_engine", lambda: None)() is None:
        figure.subplots_adjust(
            left=0.12,
            right=0.98,
            bottom=0.14,
            top=0.92,
            wspace=0.18,
            hspace=0.18,
        )

    color_background, _ = plot_background_colors(typeplotback)
    apply_plot_theme(figure, typeplotback)

    arguments = {"facecolor": color_background, "bbox_inches": "tight", "pad_inches": 0.02}
    if typefileplot == "png":
        arguments["dpi"] = 300
    arguments.update(kwargs)
    print(f"Writing to {path}...")
    figure.savefig(path, **arguments)
    if close_figure:
        plt.close(figure)
    return path


def save_current_figure(path, typefileplot="png", typeplotback=None, **kwargs):
    """Save the current Matplotlib figure with the standard plot styling."""
    return save_figure(plt.gcf(), path, typefileplot, typeplotback, **kwargs)


def show_figures(typeplotback=None, **kwargs):
    """Apply the selected theme to every open figure before displaying them."""
    current = plt.gcf() if plt.get_fignums() else None
    for number in plt.get_fignums():
        apply_plot_theme(plt.figure(number), typeplotback)
    if current is not None:
        plt.figure(current.number)
    plt.show(**kwargs)


def save_bokeh_html(figure, path, title=None, typeplotback=None):
    """Save a Bokeh layout as a self-contained interactive HTML document."""
    from bokeh.io import save
    from bokeh.models import Axis, ColorBar, Grid, Legend, Plot, Title
    from bokeh.resources import INLINE

    typeplotback = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    background, foreground = plot_background_colors(typeplotback)
    path = Path(path)
    if path.suffix.lower() != ".html":
        raise ValueError("Bokeh output path must use the .html extension")
    for plot in figure.select({"type": Plot}):
        plot.background_fill_color = background
        plot.border_fill_color = background
        plot.outline_line_color = foreground
    for axis in figure.select({"type": Axis}):
        axis.axis_line_color = foreground
        axis.major_tick_line_color = foreground
        axis.minor_tick_line_color = foreground
        axis.major_label_text_color = foreground
        axis.axis_label_text_color = foreground
    for grid in figure.select({"type": Grid}):
        grid.visible = False
    for text in figure.select({"type": Title}):
        text.text_color = foreground
    for legend in figure.select({"type": Legend}):
        legend.background_fill_color = background
        legend.background_fill_alpha = 1.0
        legend.border_line_color = foreground
        legend.label_text_color = foreground
        legend.title_text_color = foreground
    for colorbar in figure.select({"type": ColorBar}):
        colorbar.background_fill_color = background
        colorbar.major_label_text_color = foreground
        colorbar.major_tick_line_color = foreground
        colorbar.minor_tick_line_color = foreground
        colorbar.title_text_color = foreground
    path.parent.mkdir(parents=True, exist_ok=True)
    if title is None:
        title = path.stem
    print(f"Writing to {path}...")
    save(figure, filename=str(path), title=title, resources=INLINE)
    return path


def figure_to_frame(figure, close_figure=True, typeplotback=None):
    """Render a Matplotlib figure into an RGB PIL image for an animation frame."""
    from PIL import Image

    typeplotback = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    apply_plot_theme(figure, typeplotback)
    figure.canvas.draw()
    frame = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())).convert("RGB")
    if close_figure:
        plt.close(figure)
    return frame


def write_animation(frames, path, duration_ms=200, loop=0, typeplotback=None):
    """Write RGB frames to an animated GIF with one shared palette and return its path.

    Frames of different sizes are padded to the largest size in the requested background, and one adaptive
    palette built from all frames keeps colors identical from frame to frame.
    """
    from PIL import Image, ImageColor

    typeplotback = _PLOT_BACKGROUND.get() if typeplotback is None else typeplotback
    background, _ = plot_background_colors(typeplotback)
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
            canvas = Image.new("RGB", (width, height), background)
            canvas.paste(frame, (0, 0))
            padded.append(canvas)
    # sample the shared palette from across the sequence without building an
    # enormous strip for a long animation
    palette_frames = [padded[index] for index in np.unique(np.linspace(0, len(padded) - 1, min(len(padded), 8)).astype(int))]
    strip = Image.new("RGB", (width, height * len(palette_frames)))
    for index, frame in enumerate(palette_frames):
        strip.paste(frame, (0, index * height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    background_rgb = ImageColor.getrgb(background)
    palette.putpalette(palette.getpalette()[:765] + list(background_rgb))
    quantized = []
    for frame in padded:
        indexed_frame = frame.quantize(palette=palette, dither=Image.Dither.NONE)
        background_mask = Image.fromarray(np.all(np.asarray(frame) == background_rgb, axis=-1))
        indexed_frame.paste(255, mask=background_mask)
        quantized.append(indexed_frame)
    path = Path(path).with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {path}...")
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=duration_ms,
                      loop=loop, disposal=2, optimize=False)
    return path


__all__ = [
    "WASHU_COLORS",
    "WASHU_GRAY",
    "WASHU_GREEN",
    "WASHU_RED",
    "apply_plot_theme",
    "configure_plot_style",
    "figure_to_frame",
    "normalize_plot_background",
    "plot_background_colors",
    "plot_file_path",
    "plot_style",
    "plot_style_parameters",
    "save_bokeh_html",
    "save_current_figure",
    "save_figure",
    "save_plot",
    "show_figures",
    "themed_plot",
    "write_animation",
]