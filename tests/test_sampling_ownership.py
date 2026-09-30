"""Enforce TDpy's numerical-utility boundary around sampler ownership."""

import importlib.util

import tdpy


def test_tdpy_does_not_expose_sampling_entry_points():
    for name in (
        "mcmc",
        "samp",
        "sample",
        "sample_fixed",
        "sample_fixed_chains",
        "sample_posterior",
    ):
        assert not hasattr(tdpy, name)


def test_tdpy_has_no_mcmc_submodule():
    assert importlib.util.find_spec("tdpy.mcmc") is None