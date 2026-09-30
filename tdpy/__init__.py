"""Shared numerical and visualization utilities for the active astrophysics stack.

This package is the canonical shared foundation for reusable scientific helpers
and plotting routines across the ecosystem. New shared numerical functionality
should live here unless there is a compelling repository-specific exception.
"""

from importlib import import_module
import sys

import numpy as np

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from .astro import *
from .numerics import *
from .pandeia import *
from .serialization import *
from .paths import RepositoryPaths, get_data_path, get_repository_path, get_visuals_path, open_narr

__all__ = [
    name for name in globals()
    if not name.startswith('_') and name not in {'import_module', 'np', 'sys'}
]


def __getattr__(name):
    """Load legacy utility exports only when callers request them."""
    utility = import_module('.util', __name__)
    try:
        return getattr(utility, name)
    except AttributeError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error

