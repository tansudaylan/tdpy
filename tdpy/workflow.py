"""Small helpers for reproducible command workflows."""

from tdpy.verbosity import print

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import subprocess


def run_command_steps(
    steps: Sequence[tuple[str, Sequence[str | Path]]],
    *,
    working_directory: str | Path,
    environment: Mapping[str, str] | None = None,
) -> None:
    """Run labeled commands sequentially and stop at the first failure."""
    command_environment = os.environ.copy()
    if environment is not None:
        command_environment.update(environment)
    for label, command in steps:
        print(f"\n{label}\n  {' '.join(map(str, command))}")
        subprocess.run(
            command,
            cwd=working_directory,
            env=command_environment,
            check=True,
        )