"""Compatibility entry point for the legacy ``tdpy.mcmc`` import path.

The historical project API imports the MCMC functionality as a top-level module
named ``mcmc``. The implementation currently lives in ``mcmc_depr.py``; expose
that implementation under the expected public module name without changing the
underlying algorithms.
"""

from .mcmc_depr import *  # noqa: F401,F403
