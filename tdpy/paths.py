"""Repository-local runtime paths for TDpy."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


PATH_ENV_VAR = "TDPY_PATH"


@dataclass(frozen=True)
class RepositoryPaths:
    """Resolve one repository's runtime directories from its environment variable."""

    environment_variable: str

    def get_repository_path(self) -> Path:
        """Return the configured repository path."""

        path_value = os.environ.get(self.environment_variable)
        if not path_value or not path_value.strip():
            raise EnvironmentError(
                f"{self.environment_variable} is required and cannot be empty."
            )
        return Path(path_value).expanduser().resolve()

    def get_data_path(self) -> Path:
        """Return the ignored repository-local data directory."""

        return self.get_repository_path() / "data"

    def get_visuals_path(self) -> Path:
        """Return the ignored repository-local visualization directory."""

        return self.get_repository_path() / "visuals"


def open_narr(path: str | os.PathLike[str], mode: str = "r", **kwargs: Any) -> IO[Any]:
    """Open a normalized path after narrating whether it is read or written."""

    normalized_path = os.path.normpath(path)
    narration = "Reading from" if mode.startswith("r") and "+" not in mode else "Writing to"
    print(f"{narration} {normalized_path}...")

    return open(normalized_path, mode, **kwargs)


def make_directory(path: str | os.PathLike[str]) -> None:
    """Create a directory and narrate the filesystem write."""

    print(f"Writing to {path}...")
    os.makedirs(path, exist_ok=True)


def make_symlink(
    pathorig: str | os.PathLike[str], pathlink: str | os.PathLike[str]
) -> None:
    """Replace a symbolic link and narrate the filesystem write."""

    print(f"Writing to {pathlink}...")
    if os.path.lexists(pathlink):
        os.unlink(pathlink)
    os.symlink(pathorig, pathlink)


_REPOSITORY_PATHS = RepositoryPaths(PATH_ENV_VAR)
get_repository_path = _REPOSITORY_PATHS.get_repository_path
get_data_path = _REPOSITORY_PATHS.get_data_path
get_visuals_path = _REPOSITORY_PATHS.get_visuals_path