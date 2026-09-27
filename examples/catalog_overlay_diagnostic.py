#!/usr/bin/env python3
"""Run TDpy's synthetic catalog-overlay diagnostic."""

from pathlib import Path

from tdpy.catalog import plot_synthetic_catalog_diagnostic
from tdpy.cli import parse_plot_arguments


def main() -> int:
    arguments = parse_plot_arguments(description=__doc__)
    output_path = Path(__file__).with_name(
        f"catalog_overlay_diagnostic.{arguments.typefileplot}"
    )
    plot_synthetic_catalog_diagnostic(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())