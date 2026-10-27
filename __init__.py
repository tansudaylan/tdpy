"""Compatibility shim for legacy repository-root imports.

Older project scripts and tests import this module as ``from __init__ import *``.
The package itself lives under ``tdpy/``, so re-export the public package API
from there to keep both import styles working.
"""

from __future__ import annotations

import sys
from pathlib import Path
from importlib import import_module, util as import_util

import numpy as np

pkg_root = Path(__file__).resolve().parent
if __name__ == 'tdpy':
    package_directory = pkg_root / 'tdpy'
    specification = import_util.spec_from_file_location(
        'tdpy', package_directory / '__init__.py',
        submodule_search_locations=[str(package_directory)],
    )
    _tdpy = import_util.module_from_spec(specification)
    sys.modules['tdpy'] = _tdpy
    specification.loader.exec_module(_tdpy)
else:
    sys.path.insert(0, str(pkg_root))
    _tdpy = import_module('tdpy')
    for _name in dir(_tdpy):
        if not _name.startswith('_'):
            globals()[_name] = getattr(_tdpy, _name)

    __all__ = [name for name in dir(_tdpy) if not name.startswith('_')]


def __getattr__(name):
    """Forward lazy legacy exports to the package implementation."""
    return getattr(_tdpy, name)
