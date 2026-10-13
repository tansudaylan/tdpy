#!/usr/bin/env python3
"""Compare TDpy's explicit-parameter evaluator with its canonical double-King kernel."""

from tdpy.verbosity import print

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import tdpy
from tdpy.cli import add_plot_arguments


def evaluate_profiles() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the same double-King profile through both public APIs."""

    angular_deviation = np.linspace(0.0, 5.0, 501)
    core_fraction = 0.35
    core_width = 0.8
    core_index = 2.2
    tail_width = 1.7
    tail_index = 3.4

    parameters = np.asarray([core_width, core_index, tail_width, tail_index, core_fraction])[:, None, None]
    evaluated_profile = tdpy.evaluate_psf_profile(
        {}, np.array([0]), angular_deviation, 'doubking', psfp=parameters, indxpsfpinit=0,
    )[0, :, 0]
    tdpy_profile = tdpy.retr_doubking(
        angular_deviation,
        core_fraction,
        core_width,
        core_index,
        tail_width,
        tail_index,
    )
    return angular_deviation, evaluated_profile, tdpy_profile


def run_example(output_path: Path) -> float:
    """Plot the shared profile and numerical compatibility residual."""

    angular_deviation, evaluated_profile, tdpy_profile = evaluate_profiles()
    absolute_difference = np.abs(evaluated_profile - tdpy_profile)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        facecolor="white",
        gridspec_kw={"height_ratios": [3, 1]},
    )
    axes[0].plot(
        angular_deviation,
        tdpy_profile,
        color="#16697A",
        linewidth=2.4,
        label="TDpy canonical kernel",
    )
    axes[0].plot(
        angular_deviation,
        evaluated_profile,
        color="#A51417",
        linewidth=1.5,
        linestyle="--",
        label="TDpy explicit-parameter evaluator",
    )
    axes[0].set_ylabel("Double-King profile")
    axes[0].set_title("Explicit double-King parameters preserve the canonical profile")
    axes[0].legend(frameon=True, fancybox=True, framealpha=1.0)

    axes[1].plot(
        angular_deviation,
        absolute_difference,
        color="black",
        linewidth=1.8,
        label="Absolute difference",
    )
    axes[1].set_xlabel("Normalized angular deviation")
    axes[1].set_ylabel("Difference")
    axes[1].legend(frameon=True, fancybox=True, framealpha=1.0)
    for axis in axes:
        axis.grid(False)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return float(absolute_difference.max())


def run_bokeh_example(output_path: Path) -> Path:
    """Write an interactive profile comparison and compatibility residual."""
    from bokeh.layouts import column
    from bokeh.models import HoverTool
    from bokeh.plotting import figure
    from tdpy.plotting import save_bokeh_html

    angular_deviation, evaluated_profile, tdpy_profile = evaluate_profiles()
    profile_plot = figure(
        title="Double-King point-spread profile",
        width=720,
        height=340,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    profile_plot.line(angular_deviation, tdpy_profile, color="#16697A", line_width=2.4,
                      legend_label="TDpy canonical kernel")
    profile_plot.line(angular_deviation, evaluated_profile, color="#A51417",
                      line_width=1.5, line_dash="dashed",
                      legend_label="TDpy explicit-parameter evaluator")
    profile_plot.xaxis.axis_label = "Normalized angular deviation"
    profile_plot.yaxis.axis_label = "Double-King profile"

    difference_plot = figure(
        title="Absolute compatibility difference",
        width=720,
        height=240,
        x_range=profile_plot.x_range,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    difference_plot.line(
        angular_deviation,
        np.abs(evaluated_profile - tdpy_profile),
        color="black",
        line_width=1.8,
        legend_label="Absolute difference",
    )
    difference_plot.xaxis.axis_label = "Normalized angular deviation"
    difference_plot.yaxis.axis_label = "Difference"
    for plot in (profile_plot, difference_plot):
        plot.grid.visible = False
        plot.legend.location = "top_right"
        plot.add_tools(
            HoverTool(tooltips=[("Angular deviation", "$x"), ("Value", "$y")])
        )
    return save_bokeh_html(
        column(profile_plot, difference_plot, sizing_mode="stretch_width"),
        output_path,
        title="TDpy double-King profile",
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare TDpy's explicit PSF evaluator and double-King kernel."
    )
    add_plot_arguments(parser)
    parser.add_argument("--bokeh", action="store_true", help="also write interactive HTML")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"double_king_profile.{arguments.typefileplot}"
    )
    maximum_difference = run_example(output_path)
    if arguments.bokeh:
        run_bokeh_example(Path(__file__).with_name("double_king_profile.html"))
    print(f"Maximum absolute difference: {maximum_difference:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())