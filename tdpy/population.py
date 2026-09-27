"""Shared helpers for population dictionaries."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np


def subset_population(
    populations: MutableMapping[str, dict[str, list[Any]]],
    source_name: str,
    subset_name: str,
    indices: Any,
    sample_counts: MutableMapping[str, int] | None = None,
    sample_indices: MutableMapping[str, dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """Copy indexed ``[values, unit]`` features into a named subpopulation."""

    normalized_indices = (
        np.array([], dtype=int)
        if indices is None
        else np.atleast_1d(np.asarray(indices, dtype=int))
    )
    subset: dict[str, list[Any]] = {}
    for feature_name, feature in populations[source_name].items():
        if not isinstance(feature, list) or len(feature) != 2:
            raise ValueError(
                f"Population feature {feature_name!r} must be a [values, unit] list."
            )
        values, unit = feature
        if normalized_indices.size and np.ndim(values) == 0:
            raise ValueError(f"Population feature {feature_name!r} values must be indexable.")
        subset[feature_name] = [
            values[normalized_indices] if normalized_indices.size else np.array([]),
            unit,
        ]
    populations[subset_name] = subset

    if sample_indices is not None:
        sample_indices.setdefault(source_name, {})[subset_name] = normalized_indices
        sample_indices.setdefault(subset_name, {})
    if sample_counts is not None:
        sample_counts[subset_name] = normalized_indices.size

    return normalized_indices