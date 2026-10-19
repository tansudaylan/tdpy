"""Shared command-line parsing for scientific diagnostic scripts."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence


def dispatch_cli(namespace, arguments=None, forward_arguments=True):
    """Run a named callable from a module namespace using command-line arguments."""
    arguments = sys.argv[1:] if arguments is None else list(arguments)
    if not arguments:
        raise SystemExit('A command name is required.')

    command_name, *command_arguments = arguments
    command = namespace.get(command_name)
    if not callable(command) or command_name.startswith('_'):
        raise SystemExit('Unknown command: %s' % command_name)
    return command(*(command_arguments if forward_arguments else []))


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


def run_plot_cli(
    runner: Callable[[str], object],
    description: str | None = None,
    arguments: Sequence[str] | None = None,
) -> int:
    """Run a plot example with the shared command-line output-format option."""
    from .plotting import plot_style

    parser = add_plot_arguments(argparse.ArgumentParser(description=description))
    parser.add_argument(
        "--typeplotback", choices=("white", "dark", "norm", "black"), default="white"
    )
    options = parser.parse_args(arguments)
    with plot_style(options.typeplotback):
        runner(options.typefileplot)
    return 0
