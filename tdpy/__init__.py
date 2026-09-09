"""Shared numerical and visualization utilities for the active astrophysics stack.

This package is the canonical shared foundation for reusable scientific helpers
and plotting routines across the ecosystem. New shared numerical functionality
should live here unless there is a compelling repository-specific exception.
"""

import sys
import numpy as np

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from .util import *

__all__ = [
    name for name in globals()
    if not name.startswith('_') and name not in {'sys', 'np'}
]

