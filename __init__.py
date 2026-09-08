"""Compatibility shim for legacy repository-root imports.

Older project scripts and tests import this module as ``from __init__ import *``.
The package itself lives under ``tdpy/``, so re-export the public package API
from there to keep both import styles working.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

# Re-export the actual package implementation. The root-level ``__init__.py`` may
# be imported as a top-level module by legacy scripts, in which case a relative
# import is unavailable. Fall back to the real package path in that case.
try:
    from .tdpy import *  # noqa: F401,F403
except ImportError:
    pkg_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(pkg_root))
    import tdpy as _tdpy  # type: ignore

    for _name in dir(_tdpy):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_tdpy, _name)

    __all__ = [name for name in dir(_tdpy) if not name.startswith('_')]
