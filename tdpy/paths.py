"""Repository-local runtime paths for TDpy."""

import os
from pathlib import Path
from typing import IO, Any


PATH_ENV_VAR = "TDPY_PATH"


def open_narr(path: str | os.PathLike[str], mode: str = "r", **kwargs: Any) -> IO[Any]:
    """Open a normalized path after narrating whether it is read or written."""

    normalized_path = os.path.normpath(path)
    narration = "Reading from" if mode.startswith("r") and "+" not in mode else "Writing to"
    print(f"{narration} {normalized_path}...")

    return open(normalized_path, mode, **kwargs)


def get_repository_path() -> Path:
    """Return the repository path configured by TDPY_PATH."""

    path_value = os.environ.get(PATH_ENV_VAR)
    if not path_value or not path_value.strip():
        raise EnvironmentError(f"{PATH_ENV_VAR} is required and cannot be empty.")
    return Path(path_value).expanduser().resolve()


def get_data_path() -> Path:
    """Return the ignored repository-local data directory."""

    return get_repository_path() / "data"


def get_visuals_path() -> Path:
    """Return the ignored repository-local visualization directory."""

    return get_repository_path() / "visuals"