"""Shared command-line parsing for scientific diagnostic scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def add_plot_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the standard PNG-or-PDF option on an existing parser."""
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser


def parse_plot_arguments(
    description: str | None = None, arguments: Sequence[str] | None = None
) -> argparse.Namespace:
    """Parse the standard PNG-or-PDF diagnostic output option."""

    parser = add_plot_arguments(argparse.ArgumentParser(description=description))
    return parser.parse_args(arguments)