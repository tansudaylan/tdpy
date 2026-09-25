#!/usr/bin/env python3
"""Run TDpy's synthetic catalog-overlay diagnostic."""

import argparse
from pathlib import Path

from tdpy.catalog import plot_synthetic_catalog_diagnostic


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"catalog_overlay_diagnostic.{arguments.typefileplot}"
    )
    plot_synthetic_catalog_diagnostic(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())