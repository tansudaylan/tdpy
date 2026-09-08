import numpy as np

import nicomedia
import tdpy


def test_kernel_functions_match_nicomedia_scalar():
    scaldevi = np.array([0.0, 0.2, 0.8, 1.5])
    frac = 0.35
    sigc = 0.8
    gamc = 2.2
    sigt = 1.7
    gamt = 3.4

    assert np.allclose(tdpy.retr_singgaus(scaldevi, sigc), nicomedia.retr_singgaus(scaldevi, sigc))
    assert np.allclose(tdpy.retr_singking(scaldevi, sigc, gamc), nicomedia.retr_singking(scaldevi, sigc, gamc))
    assert np.allclose(tdpy.retr_doubgaus(scaldevi, frac, sigc, sigt), nicomedia.retr_doubgaus(scaldevi, frac, sigc, sigt))
    assert np.allclose(tdpy.retr_gausking(scaldevi, frac, sigc, sigt, gamt), nicomedia.retr_gausking(scaldevi, frac, sigc, sigt, gamt))
    assert np.allclose(tdpy.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt), nicomedia.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt))


def test_kernel_functions_match_nicomedia_broadcast():
    scaldevi = np.linspace(0.0, 2.0, 5)[None, :, None]
    frac = np.array([0.25, 0.75])[:, None, None]
    sigc = np.array([0.7, 1.0])[:, None, None]
    gamc = np.array([2.0, 2.5])[:, None, None]
    sigt = np.array([1.2, 1.8])[:, None, None]
    gamt = np.array([3.0, 4.0])[:, None, None]

    assert np.allclose(tdpy.retr_singgaus(scaldevi, sigc), nicomedia.retr_singgaus(scaldevi, sigc))
    assert np.allclose(tdpy.retr_singking(scaldevi, sigc, gamc), nicomedia.retr_singking(scaldevi, sigc, gamc))
    assert np.allclose(tdpy.retr_doubgaus(scaldevi, frac, sigc, sigt), nicomedia.retr_doubgaus(scaldevi, frac, sigc, sigt))
    assert np.allclose(tdpy.retr_gausking(scaldevi, frac, sigc, sigt, gamt), nicomedia.retr_gausking(scaldevi, frac, sigc, sigt, gamt))
    assert np.allclose(tdpy.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt), nicomedia.retr_doubking(scaldevi, frac, sigc, gamc, sigt, gamt))
