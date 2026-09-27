"""Shared command-line parsing for scientific diagnostic scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def parse_plot_arguments(
    description: str | None = None, arguments: Sequence[str] | None = None
) -> argparse.Namespace:
    """Parse the standard PNG-or-PDF diagnostic output option."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args(arguments)