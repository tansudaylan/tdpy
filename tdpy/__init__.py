"""Numerical, catalog, and figure-output utilities for astrophysics.

The public exports include orbital geometry, numerical transforms, archive
queries, point-spread profiles, catalog annotations, and themed figure output.
Additional utility exports are loaded on demand.
"""

from importlib import import_module
import sys

import numpy as np

from .astro import *
from .cli import dispatch_cli
from .exoplanet import *
from .mast import quer_mast, xmat_tici, retr_listtablobsv, retr_rascdeclfromstrgmast, setp_coorstrgmast
from .numerics import *
from .pandeia import *
from .plotting import *
from .population import *
from .psf import evaluate_psf_profile
from .serialization import *
from .spectral import *
from .tess import *
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

