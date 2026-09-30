"""Compact serialization and file-reading helpers."""

from tdpy.verbosity import print

from pathlib import Path
import json
from typing import Any

import numpy as np


def read_first_line(path: str | Path) -> str:
    """Read the first line of a text file with standard narration."""
    path = Path(path)
    print(f"Reading from {path}...")
    return path.read_text(encoding="utf-8").splitlines()[0]


def read_json(path: str | Path) -> Any:
    """Read a JSON file with standard narration."""
    path = Path(path)
    print(f"Reading from {path}...")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    """Write indented JSON with standard narration and a trailing newline."""
    path = Path(path)
    print(f"Writing to {path}...")
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def summarize_json_value(value: Any) -> Any:
    """Return compact JSON-safe metadata without retaining large arrays."""
    if isinstance(value, np.ndarray):
        finite = value[np.isfinite(value)]
        return {
            "shape": list(value.shape),
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: summarize_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [summarize_json_value(item) for item in value]
    return value