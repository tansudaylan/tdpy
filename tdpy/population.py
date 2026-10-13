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


def partition_classified_population(
    relevant: Any,
    irrelevant: Any,
    positive: Any,
    negative: Any,
) -> dict[str, Any]:
    """Group target indices by true relevance and inferred classification.

    Parameters
    ----------
    relevant : array_like
        Indices of truly relevant targets.
    irrelevant : array_like
        Indices of truly irrelevant targets.
    positive : array_like
        Indices classified as positive.
    negative : array_like
        Indices classified as negative.

    Returns
    -------
    dict
        Dictionary containing arrays or collections of:
        - 're': relevant
        - 'ir': irrelevant
        - 'po': positive
        - 'ne': negative
        - 'trpo': true positives
        - 'trne': true negatives
        - 'flpo': false positives
        - 'flne': false negatives
    """
    return {
        "re": relevant,
        "ir": irrelevant,
        "po": positive,
        "ne": negative,
        "trpo": np.intersect1d(positive, relevant),
        "trne": np.intersect1d(negative, irrelevant),
        "flpo": np.intersect1d(positive, irrelevant),
        "flne": np.intersect1d(negative, relevant),
    }
