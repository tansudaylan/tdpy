"""Process-wide switch for the console output of the pipelines.

Console output is off by default so that pipeline runs emit no text. Set the environment
variable TDPY_VERBOSITY to a positive integer to restore progress and file narration.
Modules opt in with ``from tdpy.verbosity import print``, which shadows the built-in.
"""

import builtins
import os
import sys


def retr_boolverb():
    """Return True when console output is enabled through TDPY_VERBOSITY."""
    return os.environ.get('TDPY_VERBOSITY', '0').strip() not in ('', '0')


def print(*args, **kwargs):
    """Built-in print that stays silent on the console unless TDPY_VERBOSITY is positive.

    Writes to explicit files other than the console always go through.
    """
    file = kwargs.get('file')
    if (file is not None and file is not sys.stdout and file is not sys.stderr) or retr_boolverb():
        builtins.print(*args, **kwargs)


def tqdm(*args, **kwargs):
    """Progress bar that is disabled unless TDPY_VERBOSITY is positive."""
    from tqdm import tqdm as tqdmbase

    kwargs.setdefault('disable', not retr_boolverb())
    return tqdmbase(*args, **kwargs)
